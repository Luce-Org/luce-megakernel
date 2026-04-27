"""
OpenAI-compatible HTTP server on top of test_dflash, with tool-calling support.

    uv run dflash-server                      # serves on :1236

    curl http://localhost:1236/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"Qwen3.5-27B-Q4_K_M","messages":[{"role":"user","content":"hi"}],"stream":true}'

Drop-in for Open WebUI / LM Studio / Cline by setting
  OPENAI_API_BASE=http://localhost:1236/v1  OPENAI_API_KEY=sk-any

Supports OpenAI /v1/chat/completions with tools and Anthropic /v1/messages.
Model stays resident in daemon mode (default); context default is 128K.
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

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool
from transformers import AutoTokenizer

from .parsing import (
    TOOL_OPEN_TAG, THINK_OPEN_TAG, THINK_CLOSE_TAG,
    normalize_stop, first_stop_match,
    parse_tool_calls, parse_reasoning,
)
from .schemas import ChatRequest, AnthropicMessagesRequest


ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_TARGET = ROOT / "models" / "Qwen3.5-27B-Q4_K_M.gguf"
DEFAULT_DRAFT_ROOT = ROOT / "models" / "draft"
DEFAULT_BIN = ROOT / "build" / ("test_dflash" + (".exe" if sys.platform == "win32" else ""))
DEFAULT_BUDGET = 22

_QWEN35_FAMILY_TOKENIZERS = {
    "Qwen3.5-27B": "Qwen/Qwen3.5-27B",
    "Qwen3.6-27B": "Qwen/Qwen3.6-27B",
}


def resolve_draft(root: Path) -> Path:
    for st in root.rglob("model.safetensors"):
        return st
    raise FileNotFoundError(f"no model.safetensors under {root}")


def _tokenizer_id_from_gguf(gguf_path: Path) -> str:
    """Infer the HuggingFace tokenizer repo from a GGUF target file."""
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


def build_app(target: Path, draft: Path, bin_path: Path, budget: int,
              max_ctx: int, tokenizer: AutoTokenizer, stop_ids: set[int],
              model_name: str = "") -> FastAPI:
    import asyncio
    app = FastAPI(title="Luce DFlash OpenAI server")
    daemon_lock = asyncio.Lock()

    r_pipe, w_pipe = os.pipe()
    if sys.platform == "win32":
        import msvcrt
        os.set_inheritable(w_pipe, True)
        stream_fd_val = int(msvcrt.get_osfhandle(w_pipe))
    else:
        stream_fd_val = w_pipe

    bin_abs = str(Path(bin_path).resolve())
    env = {**os.environ}
    if sys.platform == "win32":
        dll_dir = str(Path(bin_abs).parent / "bin")
        env["PATH"] = dll_dir + os.pathsep + str(Path(bin_abs).parent) + os.pathsep + env.get("PATH", "")

    cmd = [bin_abs, str(target), str(draft), "--daemon",
           "--fast-rollback", "--ddtree", f"--ddtree-budget={budget}",
           f"--max-ctx={max_ctx}", f"--stream-fd={stream_fd_val}"]
    if sys.platform == "win32":
        daemon_proc = subprocess.Popen(cmd, close_fds=False, env=env, stdin=subprocess.PIPE)
    else:
        daemon_proc = subprocess.Popen(cmd, pass_fds=(w_pipe,), env=env, stdin=subprocess.PIPE)
    os.close(w_pipe)

    @app.get("/v1/models")
    def list_models():
        return {"object": "list",
                "data": [{"id": model_name, "object": "model", "owned_by": "luce"}]}

    def _tokenize_prompt(req: ChatRequest) -> tuple[Path, bool]:
        """Tokenize a ChatRequest to a prompt .bin file.

        Returns (path, started_in_thinking) where started_in_thinking is True
        when the chat template prefilled <think> — streaming begins in reasoning mode.
        """
        import re
        msgs: list[dict] = []
        for m in req.messages:
            d: dict = {"role": m.role}
            if m.content is not None:
                d["content"] = m.content
            if m.name is not None:
                d["name"] = m.name
            if m.tool_call_id is not None:
                d["tool_call_id"] = m.tool_call_id
            if m.tool_calls is not None:
                d["tool_calls"] = []
                for tc in m.tool_calls:
                    args = tc.function.arguments
                    if isinstance(args, str):
                        try:
                            args_obj = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            args_obj = {"_raw": args}
                    else:
                        args_obj = args
                    d["tool_calls"].append({
                        "id": tc.id, "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": args_obj},
                    })
            msgs.append(d)

        kwargs: dict = dict(tokenize=False, add_generation_prompt=True)
        if req.tools:
            kwargs["tools"] = [t.model_dump() for t in req.tools]
        if req.chat_template_kwargs:
            kwargs.update(req.chat_template_kwargs)
        prompt = tokenizer.apply_chat_template(msgs, **kwargs)
        started_in_thinking = bool(re.search(r"<think>\s*$", prompt))
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        fd, path = tempfile.mkstemp(suffix=".bin")
        tmp = Path(path)
        with os.fdopen(fd, "wb") as f:
            for t in ids:
                f.write(struct.pack("<i", int(t)))
        return tmp, started_in_thinking

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

    async def _astream_tokens(r, n_gen):
        generated = 0
        hit_stop = False
        while True:
            b = await asyncio.to_thread(os.read, r, 4)
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

    # ── OpenAI /v1/chat/completions ────────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        prompt_bin, started_in_thinking = _tokenize_prompt(req)
        prompt_len = prompt_bin.stat().st_size // 4
        available_gen = max_ctx - prompt_len - 20
        gen_len = min(req.max_tokens, available_gen)
        if gen_len <= 0:
            try: prompt_bin.unlink()
            except Exception: pass
            return JSONResponse(
                {"detail": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"},
                status_code=400)

        completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        if req.stream:
            return await _stream_response(req, prompt_bin, gen_len,
                                          completion_id, created,
                                          started_in_thinking, daemon_lock)

        async with daemon_lock:
            daemon_proc.stdin.write(f"{prompt_bin} {gen_len}\n".encode())
            daemon_proc.stdin.flush()
            tokens = list(_token_stream(r_pipe, gen_len))
        try: prompt_bin.unlink()
        except Exception: pass

        text = tokenizer.decode(tokens, skip_special_tokens=True)
        stops = normalize_stop(req.stop)
        if stops:
            i = first_stop_match(text, stops)
            if i != -1:
                text = text[:i]
        thinking_enabled = True
        if req.chat_template_kwargs:
            thinking_enabled = req.chat_template_kwargs.get("enable_thinking", True)
        cleaned, tool_calls = parse_tool_calls(text, tools=req.tools)
        cleaned, reasoning = parse_reasoning(cleaned, thinking_enabled=thinking_enabled)

        msg: dict = {"role": "assistant"}
        finish_reason = "stop"
        if reasoning:
            msg["reasoning_content"] = reasoning
        if tool_calls:
            msg["content"] = cleaned if cleaned else None
            msg["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        else:
            msg["content"] = cleaned

        return JSONResponse({
            "id": completion_id, "object": "chat.completion",
            "created": created, "model": model_name,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": prompt_len,
                      "completion_tokens": len(tokens),
                      "total_tokens": prompt_len + len(tokens)},
        })

    async def _stream_response(req, prompt_bin, gen_len, completion_id, created,
                                started_in_thinking, lock):
        prompt_len = prompt_bin.stat().st_size // 4
        include_usage = bool(req.stream_options and req.stream_options.get("include_usage"))

        def chunk(delta_obj, finish=None):
            return {"id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model_name,
                    "choices": [{"index": 0, "delta": delta_obj, "finish_reason": finish}]}

        def usage_chunk(completion_tokens):
            return {"id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model_name, "choices": [],
                    "usage": {"prompt_tokens": prompt_len,
                              "completion_tokens": completion_tokens,
                              "total_tokens": prompt_len + completion_tokens}}

        async def sse() -> AsyncIterator[str]:
            async with lock:
                daemon_proc.stdin.write(f"{prompt_bin} {gen_len}\n".encode())
                daemon_proc.stdin.flush()
                yield f"data: {json.dumps(chunk({'role': 'assistant'}))}\n\n"

                # mode ∈ {'reasoning', 'content', 'tool_buffer'}
                mode = "reasoning" if started_in_thinking else "content"
                window = ""
                tool_buffer = ""
                stops = normalize_stop(req.stop)
                HOLDBACK = max(
                    len(THINK_OPEN_TAG), len(THINK_CLOSE_TAG), len(TOOL_OPEN_TAG),
                    *(len(s) for s in stops),
                    0,
                )
                completion_tokens = 0
                stop_hit = False

                def emit(text, kind):
                    if not text:
                        return None
                    return f"data: {json.dumps(chunk({kind: text}))}\n\n"

                try:
                    async for tok_id in iterate_in_threadpool(_token_stream(r_pipe, gen_len)):
                        completion_tokens += 1
                        window += tokenizer.decode([tok_id])

                        if stops and mode != "tool_buffer":
                            si = first_stop_match(window, stops)
                            if si != -1:
                                out = emit(window[:si], "reasoning_content" if mode == "reasoning" else "content")
                                if out: yield out
                                window = ""
                                stop_hit = True
                                break

                        while True:
                            if mode == "tool_buffer":
                                tool_buffer += window; window = ""; break

                            if mode == "reasoning":
                                idx = window.find(THINK_CLOSE_TAG)
                                if idx != -1:
                                    out = emit(window[:idx], "reasoning_content")
                                    if out: yield out
                                    window = window[idx + len(THINK_CLOSE_TAG):]
                                    mode = "content"; continue
                                if len(window) > HOLDBACK:
                                    out = emit(window[:-HOLDBACK], "reasoning_content")
                                    if out: yield out
                                    window = window[-HOLDBACK:]
                                break

                            else:  # content
                                think_idx = window.find(THINK_OPEN_TAG)
                                tool_idx  = window.find(TOOL_OPEN_TAG)
                                hits = sorted((i, t) for i, t in
                                              ((think_idx, "think"), (tool_idx, "tool")) if i != -1)
                                if hits:
                                    idx, which = hits[0]
                                    out = emit(window[:idx], "content")
                                    if out: yield out
                                    if which == "think":
                                        window = window[idx + len(THINK_OPEN_TAG):]
                                        mode = "reasoning"
                                    else:
                                        tool_buffer = window[idx:]; window = ""; mode = "tool_buffer"
                                    continue
                                if len(window) > HOLDBACK:
                                    out = emit(window[:-HOLDBACK], "content")
                                    if out: yield out
                                    window = window[-HOLDBACK:]
                                break

                    finish_reason = "stop"
                    if stop_hit:
                        yield f"data: {json.dumps(chunk({}, finish='stop'))}\n\n"
                        if include_usage: yield f"data: {json.dumps(usage_chunk(completion_tokens))}\n\n"
                        yield "data: [DONE]\n\n"
                        try: prompt_bin.unlink()
                        except Exception: pass
                        return

                    if mode == "reasoning" and window:
                        out = emit(window, "reasoning_content")
                        if out: yield out
                    elif mode == "content" and window:
                        out = emit(window, "content")
                        if out: yield out
                    elif mode == "tool_buffer":
                        tool_buffer += window

                    if mode == "tool_buffer":
                        cleaned_after, tool_calls = parse_tool_calls(tool_buffer, tools=req.tools)
                        if tool_calls:
                            if cleaned_after:
                                out = emit(cleaned_after, "content")
                                if out: yield out
                            tc_delta = [
                                {"index": i, "id": tc["id"], "type": "function",
                                 "function": {"name": tc["function"]["name"],
                                              "arguments": tc["function"]["arguments"]}}
                                for i, tc in enumerate(tool_calls)
                            ]
                            yield f"data: {json.dumps(chunk({'tool_calls': tc_delta}))}\n\n"
                            finish_reason = "tool_calls"
                        else:
                            out = emit(tool_buffer, "content")
                            if out: yield out

                finally:
                    try: prompt_bin.unlink()
                    except Exception: pass

                yield f"data: {json.dumps(chunk({}, finish=finish_reason))}\n\n"
                if include_usage: yield f"data: {json.dumps(usage_chunk(completion_tokens))}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    # ── Anthropic /v1/messages ─────────────────────────────────────

    def _anthropic_text(content) -> str:
        if isinstance(content, str):
            return content
        return "".join(b.get("text", "") for b in content
                       if isinstance(b, dict) and b.get("type") == "text")

    def _tokenize_anthropic(req: AnthropicMessagesRequest) -> tuple[Path, int]:
        msgs = []
        if req.system:
            system_text = _anthropic_text(req.system)
            if system_text:
                msgs.append({"role": "system", "content": system_text})
        for m in req.messages:
            msgs.append({"role": m.role, "content": _anthropic_text(m.content)})
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        fd, path = tempfile.mkstemp(suffix=".bin")
        tmp = Path(path)
        with os.fdopen(fd, "wb") as f:
            for t in ids:
                f.write(struct.pack("<i", int(t)))
        return tmp, len(ids)

    @app.post("/v1/messages")
    async def anthropic_messages(req: AnthropicMessagesRequest):
        prompt_bin, prompt_len = _tokenize_anthropic(req)
        available_gen = max_ctx - prompt_len - 20
        gen_len = min(req.max_tokens, available_gen)
        if gen_len <= 0:
            try: prompt_bin.unlink()
            except Exception: pass
            return JSONResponse(
                {"type": "error", "error": {"type": "invalid_request_error",
                 "message": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"}},
                status_code=400)

        msg_id = "msg_" + uuid.uuid4().hex[:24]

        if req.stream:
            async def sse() -> AsyncIterator[str]:
                async with daemon_lock:
                    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': req.model or model_name, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': prompt_len, 'output_tokens': 0}}})}\n\n"
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    daemon_proc.stdin.write(f"{prompt_bin} {gen_len}\n".encode())
                    daemon_proc.stdin.flush()
                    out_tokens = 0
                    try:
                        async for tok_id in _astream_tokens(r_pipe, gen_len):
                            out_tokens += 1
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': tokenizer.decode([tok_id])}})}\n\n"
                    finally:
                        try: prompt_bin.unlink()
                        except Exception: pass
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': out_tokens}})}\n\n"
                    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        async with daemon_lock:
            daemon_proc.stdin.write(f"{prompt_bin} {gen_len}\n".encode())
            daemon_proc.stdin.flush()
            tokens = [t async for t in _astream_tokens(r_pipe, gen_len)]
        try: prompt_bin.unlink()
        except Exception: pass
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        return JSONResponse({
            "id": msg_id, "type": "message", "role": "assistant",
            "model": req.model or model_name,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": prompt_len, "output_tokens": len(tokens)},
        })

    return app


def main():
    ap = argparse.ArgumentParser(description="Luce DFlash OpenAI-compatible server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=1236)
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--draft",  type=Path, default=DEFAULT_DRAFT_ROOT)
    ap.add_argument("--bin",    type=Path, default=DEFAULT_BIN)
    ap.add_argument("--budget", type=int,  default=DEFAULT_BUDGET)
    ap.add_argument("--max-ctx", type=int, default=131072,
                    help="Maximum context length (default: 131072)")
    ap.add_argument("--kv-f16", action="store_true",
                    help="Force F16 KV cache (default: TQ3_0 when --max-ctx > 6144)")
    ap.add_argument("--fa-window", type=int, default=None,
                    help="Sliding window for FA layers; 0 = full attention")
    ap.add_argument("--tokenizer", type=str, default=None,
                    help="HuggingFace tokenizer repo (default: auto-detect from GGUF)")
    ap.add_argument("--daemon", action="store_true",
                    help="(accepted, ignored — daemon is always on)")
    args = ap.parse_args()

    if args.max_ctx > 6144 and not args.kv_f16:
        os.environ.setdefault("DFLASH27B_KV_TQ3", "1")
    if args.fa_window is not None:
        os.environ["DFLASH27B_FA_WINDOW"] = str(args.fa_window)

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

    model_name = args.target.stem
    app = build_app(args.target, draft, args.bin, args.budget, args.max_ctx,
                    tokenizer, stop_ids, model_name=model_name)

    import uvicorn
    print(f"Luce DFlash OpenAI server on http://{args.host}:{args.port}")
    print(f"  target    = {args.target}")
    print(f"  draft     = {draft}")
    print(f"  bin       = {args.bin}")
    print(f"  budget    = {args.budget}")
    print(f"  max_ctx   = {args.max_ctx}")
    print(f"  tokenizer = {tokenizer_id}")
    print(f"  model     = {model_name}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
