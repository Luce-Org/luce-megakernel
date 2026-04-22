"""
OpenAI-compatible HTTP server on top of test_dflash.

    pip install fastapi uvicorn transformers
    python3 scripts/server.py                 # serves on :8000

    curl http://localhost:8000/v1/chat/completions \\
        -H 'Content-Type: application/json' \\
        -d '{"model":"luce-dflash","messages":[{"role":"user","content":"hi"}],"stream":true}'

Drop-in for Open WebUI / LM Studio / Cline by setting
  OPENAI_API_BASE=http://localhost:8000/v1  OPENAI_API_KEY=sk-any

Streams tokens as Server-Sent Events using the OpenAI delta format.
Model reloads per request (~10 s first-token latency). A daemon-mode
binary that keeps the model resident is a planned follow-up.
"""
import argparse
import json
import os
import struct
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = ROOT / "models" / "Qwen3.5-27B-Q4_K_M.gguf"
DEFAULT_DRAFT_ROOT = ROOT / "models" / "draft"
DEFAULT_BIN = ROOT / "build" / "test_dflash"
DEFAULT_BUDGET = 22
MODEL_NAME = "luce-dflash"


def resolve_draft(root: Path) -> Path:
    for st in root.rglob("model.safetensors"):
        return st
    raise FileNotFoundError(f"no model.safetensors under {root}")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = 512
    temperature: float | None = None  # noted + ignored (greedy-only)
    top_p: float | None = None


def build_app(target: Path, draft: Path, bin_path: Path, budget: int, max_ctx: int,
              tokenizer: AutoTokenizer, stop_ids: set[int]) -> FastAPI:
    import asyncio
    app = FastAPI(title="Luce DFlash OpenAI server")
    daemon_lock = asyncio.Lock()

    r_pipe, w_pipe = os.pipe()
    cmd = [str(bin_path), str(target), str(draft), "--daemon",
           "--fast-rollback", "--ddtree", f"--ddtree-budget={budget}",
           f"--max-ctx={max_ctx}",
           f"--stream-fd={w_pipe}"]
    daemon_proc = subprocess.Popen(cmd, pass_fds=(w_pipe,),
                                   stdin=subprocess.PIPE)
    os.close(w_pipe)

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "luce"}],
        }

    def _tokenize_prompt(req: ChatRequest) -> Path:
        msgs = [m.model_dump() for m in req.messages]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        tmp = Path(tempfile.mkstemp(suffix=".bin")[1])
        with open(tmp, "wb") as f:
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
                        for tok_id in _token_stream(r_pipe, gen_len):
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

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--draft",  type=Path, default=DEFAULT_DRAFT_ROOT)
    ap.add_argument("--bin",    type=Path, default=DEFAULT_BIN)
    ap.add_argument("--budget", type=int,  default=DEFAULT_BUDGET)
    default_ctx = 131072 if os.environ.get("DFLASH27B_KV_Q4") == "1" else 6144
    ap.add_argument("--max-ctx", type=int, default=default_ctx, help="Maximum context length")
    ap.add_argument("--daemon", action="store_true", help="Run with persistent model daemon (now default)")
    args = ap.parse_args()

    if not args.bin.is_file():
        raise SystemExit(f"binary not found at {args.bin}")
    if not args.target.is_file():
        raise SystemExit(f"target GGUF not found at {args.target}")
    draft = resolve_draft(args.draft) if args.draft.is_dir() else args.draft
    if not draft.is_file():
        raise SystemExit(f"draft safetensors not found at {args.draft}")

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-27B", trust_remote_code=True)
    stop_ids = set()
    for s in ("<|im_end|>", "<|endoftext|>"):
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids: stop_ids.add(ids[0])

    app = build_app(args.target, draft, args.bin, args.budget, args.max_ctx,
                    tokenizer, stop_ids)

    import uvicorn
    print(f"Luce DFlash OpenAI server on http://{args.host}:{args.port}")
    print(f"  target = {args.target}")
    print(f"  draft  = {draft}")
    print(f"  bin    = {args.bin}")
    print(f"  budget = {args.budget}")
    print(f"  max_ctx= {args.max_ctx}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
