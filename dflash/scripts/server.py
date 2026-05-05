"""
OpenAI-compatible HTTP server on top of test_dflash.

    pip install fastapi uvicorn transformers
    python3 scripts/server.py                 # serves on :8000

    curl http://localhost:8000/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"luce-dflash","messages":[{"role":"user","content":"hi"}],"stream":true}'

Drop-in for Open WebUI / LM Studio / Cline by setting
  OPENAI_API_BASE=http://localhost:8000/v1  OPENAI_API_KEY=sk-any

Streams tokens as Server-Sent Events using the OpenAI delta format.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware          # FIX 1: add CORS
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool
from transformers import AutoTokenizer

from _prefill_hook import (
    PrefillConfig, add_cli_flags, config_from_args,
    compress_text_via_daemon,
)
from prefix_cache import DaemonStdoutBus, PrefixCache


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = Path(os.environ.get(
    "DFLASH_TARGET",
    str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),
))
DEFAULT_DRAFT_ROOT = ROOT / "models" / "draft"
DEFAULT_BIN = ROOT / "build" / ("test_dflash" + (".exe" if sys.platform == "win32" else ""))
DEFAULT_BUDGET = 22
MODEL_NAME = "luce-dflash"

_ALLOWED_TEMPLATE_KWARGS = frozenset({"enable_thinking", "tools", "add_generation_prompt"})


def resolve_draft(root: Path) -> Path:
    for st in root.rglob("model.safetensors"):
        return st
    raise FileNotFoundError(f"no model.safetensors under {root}")


_QWEN35_FAMILY_TOKENIZERS = {
    "Qwen3.5-27B": "Qwen/Qwen3.5-27B",
    "Qwen3.6-27B": "Qwen/Qwen3.6-27B",
}


def _tokenizer_id_from_gguf(gguf_path: Path) -> str:
    default = "Qwen/Qwen3.5-27B"
    try:
        from gguf import GGUFReader  # type: ignore
        r = GGUFReader(str(gguf_path))
        for key in ("general.basename", "general.name"):
            f = r.fields.get(key)
            if f is None or not f.data:
                continue
            import numpy as np
            p = f.parts[f.data[0]]
            if not isinstance(p, np.ndarray):
                continue
            try:
                val = bytes(p).decode("utf-8", errors="replace")
            except Exception:
                continue
            for known, repo in _QWEN35_FAMILY_TOKENIZERS.items():
                if known.lower() in val.lower():
                    return repo
    except Exception:
        pass
    return default


# FIX 2: _content_to_str helper used for BOTH OpenAI and Anthropic message
# content fields (str | list[dict]). Previously OpenAI list[dict] content
# was passed raw to the tokenizer and caused a crash.
def _content_to_str(content: "str | list[dict]") -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


class ChatMessage(BaseModel):
    role: str
    # FIX 2 cont: accept list[dict] in the model but always stringify it
    content: str | list[dict]


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = 512
    temperature: float | None = None   # accepted, ignored (greedy-only)
    top_p: float | None = None         # accepted, ignored
    top_k: int | None = None           # accepted, ignored
    stop: list[str] | str | None = None  # FIX 3: accept stop field (Open WebUI sends it)
    chat_template_kwargs: dict | None = None


class AnthropicMessage(BaseModel):
    role: str
    content: str | list[dict]


class AnthropicMessagesRequest(BaseModel):
    model: str = MODEL_NAME
    max_tokens: int
    messages: list[AnthropicMessage]
    system: str | list[dict] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None
    chat_template_kwargs: dict | None = None


def build_app(target: Path, draft: Path, bin_path: Path, budget: int, max_ctx: int,
              tokenizer: AutoTokenizer, stop_ids: set[int],
              prefill_cfg: PrefillConfig | None = None,
              drafter_tokenizer: AutoTokenizer | None = None,
              prefix_cache_slots: int = 4,
              prefill_cache_slots: int = 4) -> FastAPI:
    import asyncio
    app = FastAPI(title="Luce DFlash OpenAI server")

    # FIX 1: CORS middleware so Open WebUI / browser frontends on other ports
    # can reach this server without being blocked by the browser.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    daemon_lock = asyncio.Lock()

    r_pipe, w_pipe = os.pipe()
    if sys.platform == "win32":
        import msvcrt
        os.set_inheritable(w_pipe, True)
        stream_fd_val = int(msvcrt.get_osfhandle(w_pipe))
    else:
        stream_fd_val = w_pipe

    bin_abs = str(Path(bin_path).resolve())
    dll_dir = str(Path(bin_abs).parent / "bin")
    env = {**os.environ}
    if sys.platform == "win32":
        env["PATH"] = dll_dir + os.pathsep + str(Path(bin_abs).parent) + os.pathsep + env.get("PATH", "")

    cmd = [bin_abs, str(target), str(draft), "--daemon",
           "--fast-rollback", "--ddtree", f"--ddtree-budget={budget}",
           f"--max-ctx={max_ctx}",
           f"--stream-fd={stream_fd_val}"]
    if sys.platform == "win32":
        daemon_proc = subprocess.Popen(cmd, close_fds=False, env=env,
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, bufsize=0)
    else:
        daemon_proc = subprocess.Popen(cmd, pass_fds=(w_pipe,), env=env,
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, bufsize=0)
    os.close(w_pipe)

    bus = DaemonStdoutBus(daemon_proc.stdout)

    def _resolve_kv_k_type():
        kv = "q8_0"
        if os.environ.get("DFLASH27B_KV_F16", "0") != "0":
            kv = "f16"
        if os.environ.get("DFLASH27B_KV_Q4", "0") != "0":
            kv = "q4_0"
        if os.environ.get("DFLASH27B_KV_TQ3", "0") != "0":
            kv = "tq3_0"
        if os.environ.get("DFLASH27B_KV_K"):
            kv = os.environ["DFLASH27B_KV_K"].lower()
        return kv

    _fa_window = int(os.environ.get("DFLASH27B_FA_WINDOW", 2048))
    prefix_cache = PrefixCache(
        daemon_stdin=daemon_proc.stdin,
        await_reply=bus.await_reply,
        daemon_lock=daemon_lock,
        tokenizer=tokenizer,
        kv_k_type=_resolve_kv_k_type(),
        fa_window=_fa_window,
        cap=prefix_cache_slots,
    )
    if prefill_cfg is not None and prefill_cache_slots > 0:
        prefix_cache.init_full_cache(prefill_cache_slots)

    @app.on_event("startup")
    async def _startup():
        bus.start(asyncio.get_running_loop())
        await prefix_cache.startup_sync()

    # FIX 4: /health endpoint — Open WebUI and many clients ping this before
    # sending requests. Without it they show a permanent "disconnected" badge.
    @app.get("/health")
    def health():
        alive = daemon_proc.poll() is None
        if not alive:
            return JSONResponse({"status": "error", "detail": "daemon exited"}, status_code=503)
        return {"status": "ok"}

    # FIX 5: richer /v1/models response — Open WebUI uses `context_length` and
    # `created` to populate the model picker and context-bar correctly.
    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "luce",
                "created": 1700000000,
                "context_length": max_ctx,          # shown in Open WebUI header
                "max_context_length": max_ctx,
            }],
        }

    def _ids_to_bin(ids: list[int]) -> Path:
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as f:
            for t in ids:
                f.write(struct.pack("<i", int(t)))
        return Path(path)

    def _render_messages(msgs_list: list[dict],
                         template_kwargs: dict | None = None
                         ) -> tuple[Path, list[int], str]:
        """Apply chat template to msgs_list and return (bin path, ids, raw prompt).

        The raw prompt is returned for spec-prefill: when compression fires we
        re-tokenise it with the drafter vocab.

        ``template_kwargs`` is passed through to ``apply_chat_template`` so callers
        can toggle template knobs like ``enable_thinking`` per-request.
        """
        tpl_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
        tpl_kwargs.update(
            {k: v for k, v in (template_kwargs or {}).items() if k in _ALLOWED_TEMPLATE_KWARGS}
        )
        prompt = tokenizer.apply_chat_template(msgs_list, **tpl_kwargs)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        return _ids_to_bin(ids), ids, prompt

    # FIX 2 applied: always call _content_to_str on message content
    def _tokenize_prompt(req: ChatRequest) -> tuple[Path, list[int], list[dict]]:
        msgs = [{"role": m.role, "content": _content_to_str(m.content)}
                for m in req.messages]
        path, ids, _prompt = _render_messages(msgs, req.chat_template_kwargs)
        return path, ids, msgs

    def _maybe_compress(msgs: list[dict], prompt_bin: Path, prompt_ids: list[int],
                        template_kwargs: dict | None = None
                        ) -> tuple[Path, list[int]]:
        if not prefill_cfg or not prefill_cfg.enabled:
            return prompt_bin, prompt_ids
        if not prefill_cfg.should_compress(len(prompt_ids)):
            return prompt_bin, prompt_ids
        if drafter_tokenizer is None:
            return prompt_bin, prompt_ids

        last_user_idx = next((i for i in range(len(msgs) - 1, -1, -1)
                              if msgs[i]["role"] == "user"), None)
        if last_user_idx is None:
            return prompt_bin, prompt_ids
        long_text = msgs[last_user_idx]["content"]

        compressed_text = compress_text_via_daemon(
            daemon_stdin=daemon_proc.stdin,
            r_pipe=r_pipe,
            drafter_tokenizer=drafter_tokenizer,
            cfg=prefill_cfg,
            prompt_text=long_text,
        )

        new_msgs = list(msgs)
        new_msgs[last_user_idx] = {"role": "user", "content": compressed_text}
        new_bin, new_ids, _ = _render_messages(new_msgs, template_kwargs)
        try:
            prompt_bin.unlink()
        except Exception:
            pass
        return new_bin, new_ids

    def _token_stream(r, n_gen):
        generated = 0
        hit_stop = False
        while True:
            b = os.read(r, 4)
            if not b or len(b) < 4:
                break
            tok_id = struct.unpack("<i", b)[0]
            if tok_id == -1:
                break
            if hit_stop:
                continue
            if tok_id in stop_ids:
                hit_stop = True
                continue
            generated += 1
            yield tok_id
            if generated >= n_gen:
                hit_stop = True

    # FIX 6: _collect_tokens_sync — non-streaming paths previously called
    # list(_token_stream(...)) directly (blocking the event loop) or used
    # an async comprehension over _astream_tokens inside daemon_lock
    # (risking a deadlock if the threadpool stalled). Using run_in_executor
    # offloads the blocking os.read loop to a thread without holding any
    # asyncio primitive across the thread boundary.
    async def _collect_tokens_sync(r, n_gen) -> list[int]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: list(_token_stream(r, n_gen)))

    async def _astream_tokens(r, n_gen):
        generated = 0
        hit_stop = False
        loop = asyncio.get_running_loop()
        while True:
            b = await loop.run_in_executor(None, os.read, r, 4)
            if not b or len(b) < 4:
                break
            tok_id = struct.unpack("<i", b)[0]
            if tok_id == -1:
                break
            if hit_stop:
                continue
            if tok_id in stop_ids:
                hit_stop = True
                continue
            generated += 1
            yield tok_id
            if generated >= n_gen:
                hit_stop = True

    # FIX 7: _write_cmd helper — centralises stdin write+flush and guards
    # against a dead daemon so callers get a clean 503 instead of a hang.
    def _write_cmd(cmd_line: str):
        if daemon_proc.poll() is not None:
            raise RuntimeError("dflash daemon has exited unexpectedly")
        daemon_proc.stdin.write(cmd_line.encode("utf-8"))
        daemon_proc.stdin.flush()

    def _build_cmd_line(cur_bin, cur_ids, gen_len, prefix_cache,
                        prompt_ids, full_snap_prep_ref: list,
                        compression_fired: bool):
        """
        FIX 8: extracted cmd_line construction so both streaming and
        non-streaming paths share identical logic and can't diverge.
        Returns (cmd_line, snap_prep).
        full_snap_prep_ref is a 1-element list used as an out-param.
        """
        if compression_fired:
            full_snap_prep = prefix_cache.prepare_full_snap(prompt_ids)
            full_snap_prep_ref[0] = full_snap_prep
            if full_snap_prep is not None:
                fslot, _ = full_snap_prep
                return f"{cur_bin} {gen_len} snap={len(cur_ids)}:{fslot}\n", None
            else:
                return f"{cur_bin} {gen_len}\n", None
        else:
            full_snap_prep_ref[0] = None
            hit = prefix_cache.lookup(cur_ids)
            snap_prep = prefix_cache.prepare_inline_snap(cur_ids)
            if hit:
                slot, _prefix_len = hit
                cmd_line = f"RESTORE {slot} {cur_bin} {gen_len}"
            else:
                cmd_line = f"{cur_bin} {gen_len}"
            if snap_prep:
                cmd_line += f" snap={snap_prep[1]}:{snap_prep[0]}"
            return cmd_line + "\n", snap_prep

    def _gen_len_for(prompt_len: int, max_tokens: int) -> int:
        return min(max_tokens, max_ctx - prompt_len - 20)

    # ── /v1/chat/completions ────────────────────────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        prompt_bin, prompt_ids, raw_msgs = _tokenize_prompt(req)
        completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        if req.stream:
            async def sse() -> AsyncIterator[str]:
                async with daemon_lock:
                    full_snap_prep_ref = [None]
                    snap_prep = None

                    full_hit = prefix_cache.lookup_full(prompt_ids)
                    if full_hit is not None:
                        slot, cached_cur_bin, cached_cur_ids_len = full_hit
                        cur_bin = Path(cached_cur_bin)
                        prompt_len = cached_cur_ids_len
                        gen_len = _gen_len_for(prompt_len, req.max_tokens)
                        if gen_len <= 0:
                            try: prompt_bin.unlink()
                            except Exception: pass
                            err = {"id": completion_id, "object": "chat.completion.chunk",
                                   "created": created, "model": MODEL_NAME,
                                   "choices": [{"index": 0, "delta": {},
                                                "finish_reason": "length"}]}
                            yield f"data: {json.dumps(err)}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        cmd_line = f"RESTORE {slot} {cached_cur_bin} {gen_len}\n"
                    else:
                        cur_bin, cur_ids = await asyncio.to_thread(
                            _maybe_compress, raw_msgs, prompt_bin, prompt_ids,
                            req.chat_template_kwargs)
                        prompt_len = len(cur_ids)
                        gen_len = _gen_len_for(prompt_len, req.max_tokens)
                        if gen_len <= 0:
                            try: cur_bin.unlink()
                            except Exception: pass
                            err = {"id": completion_id, "object": "chat.completion.chunk",
                                   "created": created, "model": MODEL_NAME,
                                   "choices": [{"index": 0, "delta": {},
                                                "finish_reason": "length"}]}
                            yield f"data: {json.dumps(err)}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        compression_fired = (cur_bin != prompt_bin)
                        cmd_line, snap_prep = _build_cmd_line(
                            cur_bin, cur_ids, gen_len, prefix_cache,
                            prompt_ids, full_snap_prep_ref, compression_fired)

                    # FIX 7: guard against dead daemon
                    try:
                        _write_cmd(cmd_line)
                    except RuntimeError as e:
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    head = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": MODEL_NAME,
                        "choices": [{"index": 0,
                                     "delta": {"role": "assistant"},
                                     "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(head)}\n\n"

                    try:
                        async for tok_id in _astream_tokens(r_pipe, gen_len):
                            chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created, "model": MODEL_NAME,
                                "choices": [{"index": 0,
                                             "delta": {"content": tokenizer.decode([tok_id])},
                                             "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                    finally:
                        if full_hit is None:
                            try: cur_bin.unlink()
                            except Exception: pass
                        else:
                            try: prompt_bin.unlink()
                            except Exception: pass

                    full_snap_prep = full_snap_prep_ref[0]
                    if full_snap_prep is not None:
                        fslot, _ = full_snap_prep
                        prefix_cache.confirm_full_snap(fslot, prompt_ids, cur_bin, len(cur_ids))
                    elif snap_prep:
                        prefix_cache.confirm_inline_snap(*snap_prep, cur_ids)

                    tail = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": MODEL_NAME,
                        "choices": [{"index": 0, "delta": {},
                                     "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(tail)}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        # Non-streaming
        async with daemon_lock:
            full_snap_prep_ref = [None]
            snap_prep = None

            full_hit = prefix_cache.lookup_full(prompt_ids)
            if full_hit is not None:
                slot, cached_cur_bin, cached_cur_ids_len = full_hit
                cur_bin = Path(cached_cur_bin)
                cur_ids = None
                prompt_len = cached_cur_ids_len
                gen_len = _gen_len_for(prompt_len, req.max_tokens)
                if gen_len <= 0:
                    try: prompt_bin.unlink()
                    except Exception: pass
                    return JSONResponse(
                        {"detail": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"},
                        status_code=400)
                cmd_line = f"RESTORE {slot} {cached_cur_bin} {gen_len}\n"
            else:
                cur_bin, cur_ids = await asyncio.to_thread(
                    _maybe_compress, raw_msgs, prompt_bin, prompt_ids,
                            req.chat_template_kwargs)
                prompt_len = len(cur_ids)
                gen_len = _gen_len_for(prompt_len, req.max_tokens)
                if gen_len <= 0:
                    try: cur_bin.unlink()
                    except Exception: pass
                    return JSONResponse(
                        {"detail": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"},
                        status_code=400)
                compression_fired = (cur_bin != prompt_bin)
                cmd_line, snap_prep = _build_cmd_line(
                    cur_bin, cur_ids, gen_len, prefix_cache,
                    prompt_ids, full_snap_prep_ref, compression_fired)

            try:
                _write_cmd(cmd_line)
            except RuntimeError as e:
                return JSONResponse({"detail": str(e)}, status_code=503)

            # FIX 6: use run_in_executor instead of list() blocking event loop
            tokens = await _collect_tokens_sync(r_pipe, gen_len)

            full_snap_prep = full_snap_prep_ref[0]
            if full_snap_prep is not None:
                fslot, _ = full_snap_prep
                prefix_cache.confirm_full_snap(fslot, prompt_ids, cur_bin, len(cur_ids))
            elif snap_prep:
                prefix_cache.confirm_inline_snap(*snap_prep, cur_ids)

        if full_hit is None:
            try: cur_bin.unlink()
            except Exception: pass
        else:
            try: prompt_bin.unlink()
            except Exception: pass

        text = tokenizer.decode(tokens, skip_special_tokens=True)
        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": prompt_len,
                      "completion_tokens": len(tokens),
                      "total_tokens": prompt_len + len(tokens)},
        })

    # ── Anthropic Messages API ──────────────────────────────────────────────

    def _tokenize_anthropic(req: AnthropicMessagesRequest
                            ) -> tuple[Path, list[int], list[dict]]:
        msgs = []
        system_text = _content_to_str(req.system) if req.system else None
        if system_text:
            msgs.append({"role": "system", "content": system_text})
        for m in req.messages:
            msgs.append({"role": m.role, "content": _content_to_str(m.content)})
        path, ids, _prompt = _render_messages(msgs, req.chat_template_kwargs)
        return path, ids, msgs

    @app.post("/v1/messages")
    async def anthropic_messages(req: AnthropicMessagesRequest):
        prompt_bin, prompt_ids, raw_msgs = _tokenize_anthropic(req)
        msg_id = "msg_" + uuid.uuid4().hex[:24]

        if req.stream:
            async def sse() -> AsyncIterator[str]:
                async with daemon_lock:
                    full_snap_prep_ref = [None]
                    snap_prep = None

                    full_hit = prefix_cache.lookup_full(prompt_ids)
                    if full_hit is not None:
                        slot, cached_cur_bin, cached_cur_ids_len = full_hit
                        cur_bin = Path(cached_cur_bin)
                        cur_ids = None
                        prompt_len = cached_cur_ids_len
                        gen_len = min(req.max_tokens, max_ctx - prompt_len - 20)
                        if gen_len <= 0:
                            try: prompt_bin.unlink()
                            except Exception: pass
                            err = {"type": "error",
                                   "error": {"type": "invalid_request_error",
                                             "message": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"}}
                            yield f"event: error\ndata: {json.dumps(err)}\n\n"
                            return
                        cmd_line = f"RESTORE {slot} {cached_cur_bin} {gen_len}\n"
                    else:
                        cur_bin, cur_ids = await asyncio.to_thread(
                            _maybe_compress, raw_msgs, prompt_bin, prompt_ids,
                            req.chat_template_kwargs)
                        prompt_len = len(cur_ids)
                        gen_len = min(req.max_tokens, max_ctx - prompt_len - 20)
                        if gen_len <= 0:
                            try: cur_bin.unlink()
                            except Exception: pass
                            err = {"type": "error",
                                   "error": {"type": "invalid_request_error",
                                             "message": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"}}
                            yield f"event: error\ndata: {json.dumps(err)}\n\n"
                            return
                        compression_fired = (cur_bin != prompt_bin)
                        cmd_line, snap_prep = _build_cmd_line(
                            cur_bin, cur_ids, gen_len, prefix_cache,
                            prompt_ids, full_snap_prep_ref, compression_fired)

                    message_start = {
                        "type": "message_start",
                        "message": {
                            "id": msg_id, "type": "message", "role": "assistant",
                            "model": req.model or MODEL_NAME,
                            "content": [], "stop_reason": None, "stop_sequence": None,
                            "usage": {"input_tokens": prompt_len, "output_tokens": 0},
                        },
                    }
                    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

                    try:
                        _write_cmd(cmd_line)
                    except RuntimeError as e:
                        yield f"event: error\ndata: {json.dumps({'type':'error','error':{'type':'server_error','message':str(e)}})}\n\n"
                        return

                    out_tokens = 0
                    try:
                        async for tok_id in _astream_tokens(r_pipe, gen_len):
                            out_tokens += 1
                            delta = {
                                "type": "content_block_delta", "index": 0,
                                "delta": {"type": "text_delta",
                                          "text": tokenizer.decode([tok_id])},
                            }
                            yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"
                    finally:
                        if full_hit is None:
                            try: cur_bin.unlink()
                            except Exception: pass
                        else:
                            try: prompt_bin.unlink()
                            except Exception: pass

                    full_snap_prep = full_snap_prep_ref[0]
                    if full_snap_prep is not None:
                        fslot, _ = full_snap_prep
                        prefix_cache.confirm_full_snap(fslot, prompt_ids, cur_bin, len(cur_ids))
                    elif snap_prep:
                        prefix_cache.confirm_inline_snap(*snap_prep, cur_ids)

                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                    msg_delta = {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": out_tokens},
                    }
                    yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"
                    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        # Non-streaming Anthropic
        async with daemon_lock:
            full_snap_prep_ref = [None]
            snap_prep = None

            full_hit = prefix_cache.lookup_full(prompt_ids)
            if full_hit is not None:
                slot, cached_cur_bin, cached_cur_ids_len = full_hit
                cur_bin = Path(cached_cur_bin)
                cur_ids = None
                prompt_len = cached_cur_ids_len
                gen_len = min(req.max_tokens, max_ctx - prompt_len - 20)
                if gen_len <= 0:
                    try: prompt_bin.unlink()
                    except Exception: pass
                    return JSONResponse(
                        {"type": "error", "error": {"type": "invalid_request_error",
                         "message": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"}},
                        status_code=400)
                cmd_line = f"RESTORE {slot} {cached_cur_bin} {gen_len}\n"
            else:
                cur_bin, cur_ids = await asyncio.to_thread(
                    _maybe_compress, raw_msgs, prompt_bin, prompt_ids,
                            req.chat_template_kwargs)
                prompt_len = len(cur_ids)
                gen_len = min(req.max_tokens, max_ctx - prompt_len - 20)
                if gen_len <= 0:
                    try: cur_bin.unlink()
                    except Exception: pass
                    return JSONResponse(
                        {"type": "error", "error": {"type": "invalid_request_error",
                         "message": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"}},
                        status_code=400)
                compression_fired = (cur_bin != prompt_bin)
                cmd_line, snap_prep = _build_cmd_line(
                    cur_bin, cur_ids, gen_len, prefix_cache,
                    prompt_ids, full_snap_prep_ref, compression_fired)

            try:
                _write_cmd(cmd_line)
            except RuntimeError as e:
                return JSONResponse({"type": "error", "error": {"type": "server_error",
                                     "message": str(e)}}, status_code=503)

            # FIX 6: use run_in_executor — same fix as OpenAI non-streaming path
            tokens = await _collect_tokens_sync(r_pipe, gen_len)

            full_snap_prep = full_snap_prep_ref[0]
            if full_snap_prep is not None:
                fslot, _ = full_snap_prep
                prefix_cache.confirm_full_snap(fslot, prompt_ids, cur_bin, len(cur_ids))
            elif snap_prep:
                prefix_cache.confirm_inline_snap(*snap_prep, cur_ids)

        if full_hit is None:
            try: cur_bin.unlink()
            except Exception: pass
        else:
            try: prompt_bin.unlink()
            except Exception: pass

        text = tokenizer.decode(tokens, skip_special_tokens=True)
        return JSONResponse({
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": req.model or MODEL_NAME,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": prompt_len,
                      "output_tokens": len(tokens)},
        })

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--draft",  type=Path, default=DEFAULT_DRAFT_ROOT)
    ap.add_argument("--bin",    type=Path, default=DEFAULT_BIN)
    ap.add_argument("--budget", type=int,  default=DEFAULT_BUDGET)
    default_ctx = 16384
    ap.add_argument("--max-ctx", type=int, default=default_ctx,
                    help=f"Maximum context length (default: {default_ctx}; "
                         "oversizing this — e.g. 131072 on short prompts — "
                         "can slow attention 20×+ until issue #10 is fixed)")
    ap.add_argument("--kv-f16", action="store_true",
                    help="Force F16 KV cache.")
    ap.add_argument("--cache-type-k", "--ctk", dest="cache_type_k", default=None,
                    choices=["f16","bf16","q4_0","q4_1","q5_0","q5_1","q8_0","tq3_0"])
    ap.add_argument("--cache-type-v", "--ctv", dest="cache_type_v", default=None,
                    choices=["f16","bf16","q4_0","q4_1","q5_0","q5_1","q8_0","tq3_0"])
    ap.add_argument("--fa-window", type=int, default=None,
                    help="Sliding window for FA layers. 0 = full attention.")
    ap.add_argument("--tokenizer", type=str, default=None)
    ap.add_argument("--prefix-cache-slots", type=int, default=4)
    ap.add_argument("--prefill-cache-slots", type=int, default=4)
    ap.add_argument("--daemon", action="store_true")
    add_cli_flags(ap)
    args = ap.parse_args()
    prefill_cfg = config_from_args(args)

    if args.cache_type_k:
        os.environ["DFLASH27B_KV_K"] = args.cache_type_k
    if args.cache_type_v:
        os.environ["DFLASH27B_KV_V"] = args.cache_type_v
    if args.max_ctx > 6144 and not args.kv_f16 and not args.cache_type_k and not args.cache_type_v:
        os.environ.setdefault("DFLASH27B_KV_TQ3", "1")

    if args.fa_window is not None:
        os.environ["DFLASH27B_FA_WINDOW"] = str(args.fa_window)

    if args.prefill_compression != "off":
        os.environ.setdefault("DFLASH27B_LM_HEAD_FIX", "0")
        os.environ.setdefault("DFLASH27B_FA_WINDOW", "0")
        os.environ.setdefault("DFLASH_FP_USE_BSA", "1")
        os.environ.setdefault("DFLASH_FP_ALPHA",   "0.85")

    if not args.bin.is_file():
        raise SystemExit(f"binary not found at {args.bin}")
    if not args.target.is_file():
        raise SystemExit(f"target GGUF not found at {args.target}")
    draft = resolve_draft(args.draft) if args.draft.is_dir() else args.draft
    if not draft.is_file():
        raise SystemExit(f"draft safetensors not found at {args.draft}")

    tokenizer_id = args.tokenizer or _tokenizer_id_from_gguf(args.target)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    stop_ids = set()
    for s in ("<|im_end|>", "<|endoftext|>"):
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids: stop_ids.add(ids[0])

    drafter_tokenizer = None
    if prefill_cfg.enabled:
        drafter_tokenizer = AutoTokenizer.from_pretrained(
            prefill_cfg.drafter_tokenizer_id, trust_remote_code=True)

    app = build_app(args.target, draft, args.bin, args.budget, args.max_ctx,
                    tokenizer, stop_ids,
                    prefill_cfg=prefill_cfg if prefill_cfg.enabled else None,
                    drafter_tokenizer=drafter_tokenizer,
                    prefix_cache_slots=args.prefix_cache_slots,
                    prefill_cache_slots=args.prefill_cache_slots)

    import uvicorn
    print(f"Luce DFlash OpenAI server on http://{args.host}:{args.port}")
    print(f"  target    = {args.target}")
    print(f"  draft     = {draft}")
    print(f"  bin       = {args.bin}")
    print(f"  budget    = {args.budget}")
    print(f"  max_ctx   = {args.max_ctx}")
    print(f"  tokenizer = {tokenizer_id}")
    if prefill_cfg.enabled:
        print(f"  pflash    = {prefill_cfg.mode} · threshold={prefill_cfg.threshold} "
              f"keep={prefill_cfg.keep_ratio} drafter={prefill_cfg.drafter_gguf}")
    else:
        print("  pflash    = off")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
