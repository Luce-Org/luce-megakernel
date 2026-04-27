"""
OpenAI-compatible HTTP server on top of test_dflash.

    uv run scripts/server.py                  # serves on :1236

    curl http://localhost:8000/v1/chat/completions \\
        -H 'Content-Type: application/json' \\
        -d '{"model":"luce-dflash","messages":[{"role":"user","content":"hi"}],"stream":true}'

Drop-in for Open WebUI / LM Studio / Cline by setting
  OPENAI_API_BASE=http://localhost:8000/v1  OPENAI_API_KEY=sk-any

Streams tokens as Server-Sent Events using the OpenAI delta format.
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

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = ROOT / "models" / "Qwen3.5-27B-Q4_K_M.gguf"
DEFAULT_DRAFT_ROOT = ROOT / "models" / "draft"
DEFAULT_BIN = ROOT / "build" / ("test_dflash" + (".exe" if sys.platform == "win32" else ""))
DEFAULT_BUDGET = 22


def resolve_draft(root: Path) -> Path:
    for st in root.rglob("model.safetensors"):
        return st
    raise FileNotFoundError(f"no model.safetensors under {root}")


# Models known to share the qwen35 GGUF arch + vocab. Verified via
# tokenizer.ggml.pre == "qwen35" and identical eos/pad/bos token IDs.
_QWEN35_FAMILY_TOKENIZERS = {
    "Qwen3.5-27B": "Qwen/Qwen3.5-27B",
    "Qwen3.6-27B": "Qwen/Qwen3.6-27B",
}


def _tokenizer_id_from_gguf(gguf_path: Path) -> str:
    """Infer the HuggingFace tokenizer repo from a GGUF target file.

    The GGUF file encodes its own tokenizer so in principle we could use that
    directly, but `test_dflash` drives generation through the HF tokenizer for
    chat-template application. We match on `general.basename` / `general.name`
    metadata; if anything goes wrong we fall back to the historical default
    (Qwen/Qwen3.5-27B) so existing setups don't break.
    """
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


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict]


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = 512
    temperature: float | None = None  # noted + ignored (greedy-only)
    top_p: float | None = None


class AnthropicMessage(BaseModel):
    role: str
    # Anthropic allows either a plain string or a list of content blocks.
    content: str | list[dict]


class AnthropicMessagesRequest(BaseModel):
    model: str = ""
    max_tokens: int
    messages: list[AnthropicMessage]
    system: str | list[dict] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None


def build_app(target: Path, draft: Path, bin_path: Path, budget: int, max_ctx: int,
              tokenizer: AutoTokenizer, stop_ids: set[int],
              model_name: str = "") -> FastAPI:
    import asyncio
    app = FastAPI(title="Luce DFlash OpenAI server")
    MODEL_NAME = model_name
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
                                       stdin=subprocess.PIPE)
    else:
        daemon_proc = subprocess.Popen(cmd, pass_fds=(w_pipe,), env=env,
                                       stdin=subprocess.PIPE)
    os.close(w_pipe)

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "luce"}],
        }

    def _tokenize_prompt(req: ChatRequest) -> Path:
        msgs = [{"role": m.role, "content": _anthropic_text_from_content(m.content)
                 if isinstance(m.content, list) else m.content}
                for m in req.messages]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        # mkstemp returns (fd, path). The previous code kept only the
        # path and discarded fd, leaking 1 file descriptor per request.
        # os.fdopen() takes ownership of the fd and closes it on __exit__.
        fd, path = tempfile.mkstemp(suffix=".bin")
        tmp = Path(path)
        with os.fdopen(fd, "wb") as f:
            for t in ids:
                f.write(struct.pack("<i", int(t)))
        return tmp

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

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        prompt_bin = _tokenize_prompt(req)
        
        # Clamp max_tokens to available headroom
        prompt_len = prompt_bin.stat().st_size // 4
        # Safety buffer for the dflash block_size (16)
        available_gen = max_ctx - prompt_len - 20
        gen_len = min(req.max_tokens, available_gen)
        if gen_len <= 0:
            return JSONResponse({"detail": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"}, status_code=400)

        completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        if req.stream:
            async def sse() -> AsyncIterator[str]:
                async with daemon_lock:
                    cmd_line = f"{prompt_bin} {gen_len}\n"
                    daemon_proc.stdin.write(cmd_line.encode("utf-8"))
                    daemon_proc.stdin.flush()
                    head = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": MODEL_NAME,
                        "choices": [{"index": 0,
                                      "delta": {"role": "assistant"},
                                      "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(head)}\n\n"
                    try:
                        # Offload blocking os.read in _token_stream to a thread so
                        # SSE chunks flush progressively instead of after generation ends.
                        async for tok_id in iterate_in_threadpool(_token_stream(r_pipe, gen_len)):
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
                        try: prompt_bin.unlink()
                        except Exception: pass
                    tail = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": MODEL_NAME,
                        "choices": [{"index": 0, "delta": {},
                                      "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(tail)}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        # Non-streaming: collect all tokens, return one response
        async with daemon_lock:
            cmd_line = f"{prompt_bin} {gen_len}\n"
            daemon_proc.stdin.write(cmd_line.encode("utf-8"))
            daemon_proc.stdin.flush()
            tokens = list(_token_stream(r_pipe, gen_len))
            
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
            "usage": {"prompt_tokens": 0,  # not tracked yet
                      "completion_tokens": len(tokens),
                      "total_tokens": len(tokens)},
        })

    # ── Anthropic Messages API ──────────────────────────────────────
    # Mirrors the OpenAI endpoint but formatted for the Anthropic SDK.
    # `?beta=true` (or any other query params) are accepted and ignored.

    def _anthropic_text_from_content(content) -> str:
        if isinstance(content, str):
            return content
        # list of blocks — concatenate the text blocks, ignore images/tools
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
        return "".join(parts)

    def _tokenize_anthropic(req: AnthropicMessagesRequest) -> tuple[Path, int]:
        msgs = []
        system_text = _anthropic_text_from_content(req.system) if req.system else None
        if system_text:
            msgs.append({"role": "system", "content": system_text})
        for m in req.messages:
            msgs.append({"role": m.role,
                         "content": _anthropic_text_from_content(m.content)})
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        # mkstemp returns (fd, path); discarding fd leaks 1 per request (#15).
        fd, path = tempfile.mkstemp(suffix=".bin")
        tmp = Path(path)
        with os.fdopen(fd, "wb") as f:
            for t in ids:
                f.write(struct.pack("<i", int(t)))
        return tmp, len(ids)

    async def _astream_tokens(r, n_gen):
        """Yields one token at a time without blocking the event loop.
        Each 4-byte pipe read is dispatched to a worker thread."""
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

    @app.post("/v1/messages")
    async def anthropic_messages(req: AnthropicMessagesRequest):
        prompt_bin, prompt_len = _tokenize_anthropic(req)

        available_gen = max_ctx - prompt_len - 20
        gen_len = min(req.max_tokens, available_gen)
        if gen_len <= 0:
            try: prompt_bin.unlink()
            except Exception: pass
            return JSONResponse(
                {"type": "error",
                 "error": {"type": "invalid_request_error",
                           "message": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"}},
                status_code=400)

        msg_id = "msg_" + uuid.uuid4().hex[:24]

        if req.stream:
            async def sse() -> AsyncIterator[str]:
                # Hold the lock across the ENTIRE read cycle so concurrent
                # requests don't interleave tokens through the shared pipe.
                async with daemon_lock:
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

                    cb_start = {
                        "type": "content_block_start", "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(cb_start)}\n\n"

                    cmd_line = f"{prompt_bin} {gen_len}\n"
                    daemon_proc.stdin.write(cmd_line.encode("utf-8"))
                    daemon_proc.stdin.flush()

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
                        try: prompt_bin.unlink()
                        except Exception: pass

                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                    msg_delta = {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": out_tokens},
                    }
                    yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"
                    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        # Non-streaming
        async with daemon_lock:
            cmd_line = f"{prompt_bin} {gen_len}\n"
            daemon_proc.stdin.write(cmd_line.encode("utf-8"))
            daemon_proc.stdin.flush()
            tokens = [t async for t in _astream_tokens(r_pipe, gen_len)]

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
    ap.add_argument("--port", type=int, default=1236)
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--draft",  type=Path, default=DEFAULT_DRAFT_ROOT)
    ap.add_argument("--bin",    type=Path, default=DEFAULT_BIN)
    ap.add_argument("--budget", type=int,  default=DEFAULT_BUDGET)
    default_ctx = 131072
    ap.add_argument("--max-ctx", type=int, default=default_ctx,
                    help=f"Maximum context length (default: {default_ctx})")
    ap.add_argument("--kv-f16", action="store_true",
                    help="Force F16 KV cache. When --max-ctx > 6144 the server "
                         "auto-enables TQ3_0 KV to fit; pass --kv-f16 to opt out.")
    ap.add_argument("--fa-window", type=int, default=None,
                    help="Sliding window for FA layers (KV positions). 0 = full "
                         "attention. Default 2048 (set in C++); only kicks in "
                         "once kv_cache > window. Trades attention range for "
                         "long-context decode speed.")
    ap.add_argument("--tokenizer", type=str, default=None,
                    help="HuggingFace tokenizer repo ID (default: auto-detect "
                         "from target GGUF basename; falls back to Qwen/Qwen3.5-27B)")
    ap.add_argument("--daemon", action="store_true", help="Run with persistent model daemon (now default)")
    args = ap.parse_args()

    # Auto-enable TQ3_0 KV cache when the requested context exceeds what F16 fits.
    # Clients like Claude Code routinely send 10k+ token system prompts, so
    # 6144 is too tight for real-world use. setdefault so an explicit user
    # DFLASH27B_KV_TQ3=0 still wins.
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


if __name__ == "__main__":
    main()
