"""End-to-end DFlash prefill backend benchmark.

Runs server_tools.py against the same long prompt with:
  - prefill off
  - daemon PFlash
  - Python Mega PFlash
  - native Mega PFlash

The script disables prefix and full-prefill caches so every request exercises
the selected prefill path. It prints wall time and TTFT for each run.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PY = Path(os.environ.get("DFLASH_PYTHON", sys.executable))
SERVER = ROOT / "scripts/server_tools.py"


def build_prompt(repeats: int) -> str:
    passage = (
        "Mega PFlash end-to-end DFlash A/B test. The passage says the "
        "important code is 741923 and the backend being tested is prompt "
        "compression. We repeat this content to force the prefill compression "
        "path through the server, daemon process, tokenizer, and generation "
        "endpoint. "
    )
    return (
        "Below is a repeated technical passage. Preserve the important details "
        "and answer briefly.\n\n"
        + passage * repeats
        + "\nQuestion: What is the important code? Answer with only the code."
    )


def post_stream(port: int, prompt: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": "luce-dflash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    t0 = time.perf_counter()
    first = None
    chunks = 0
    text_parts: list[str] = []
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                text = delta.get("content") or delta.get("reasoning_content") or ""
                if text:
                    if first is None:
                        first = time.perf_counter()
                    chunks += 1
                    text_parts.append(text)
    return {
        "wall": time.perf_counter() - t0,
        "ttft": (first - t0) if first else 0.0,
        "chunks": chunks,
        "text": "".join(text_parts),
    }


def wait_ready(proc: subprocess.Popen, port: int, log_path: Path, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(log_path.read_text(errors="replace")[-8000:])
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2).read()
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("timeout waiting for server\n" + log_path.read_text(errors="replace")[-8000:])


def run_backend(args: argparse.Namespace, name: str, port: int, extra: list[str]) -> list[dict]:
    log_path = Path(tempfile.gettempdir()) / f"dflash_prefill_{name}_{port}.log"
    env = os.environ.copy()
    env.setdefault("DFLASH27B_KV_TQ3", "1")
    env["DFLASH27B_FA_WINDOW"] = str(args.fa_window)
    env.setdefault("DFLASH27B_LM_HEAD_FIX", "0")
    env.setdefault("DFLASH_FP_USE_BSA", "1")
    env.setdefault("DFLASH_FP_ALPHA", "0.85")
    if args.show_timing:
        env["DFLASH_SERVER_SHOW_TIMING"] = "1"
        env["DFLASH_FP_PROFILE"] = "1"

    cmd = [
        str(PY), "-u", str(SERVER),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--target", str(args.target),
        "--draft", str(args.draft),
        "--bin", str(args.bin),
        "--budget", str(args.budget),
        "--max-ctx", str(args.max_ctx),
        "--prefix-cache-slots", "0",
        "--prefill-cache-slots", "0",
        *extra,
    ]

    print(f"[{name}] starting port={port}", flush=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
    try:
        wait_ready(proc, port, log_path, args.startup_timeout)
        print(f"[{name}] ready log={log_path}", flush=True)
        rows = []
        for i in range(args.runs):
            row = post_stream(port, args.prompt, args.max_tokens)
            rows.append(row)
            print(
                f"[{name}] run {i + 1}: wall={row['wall']:.3f}s "
                f"ttft={row['ttft']:.3f}s chunks={row['chunks']} "
                f"text={row['text'][:80]!r}",
                flush=True,
            )
        return rows
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def summarize(name: str, rows: list[dict]) -> None:
    if not rows:
        print(f"{name}: no successful runs")
        return
    mean_wall = sum(r["wall"] for r in rows) / len(rows)
    mean_ttft = sum(r["ttft"] for r in rows) / len(rows)
    runs = ",".join(f"{r['wall']:.3f}/{r['ttft']:.3f}" for r in rows)
    print(f"{name}: mean_wall={mean_wall:.3f}s mean_ttft={mean_ttft:.3f}s runs={runs}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, default=Path("/home/jake/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q5_K_XL.gguf"))
    ap.add_argument("--draft", type=Path, default=Path("/home/jake/models/Qwen3.6-27B-DFlash-safetensors/model.safetensors"))
    ap.add_argument("--bin", type=Path, default=ROOT / "build-luce-sm120/test_dflash")
    ap.add_argument("--drafter", type=Path, default=Path("/home/jake/models/Qwen3-0.6B-GGUF/Qwen3-0.6B-BF16.gguf"))
    ap.add_argument("--mega-native-model", type=Path,
                    default=Path("/home/jake/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"))
    ap.add_argument("--daemon-keep-ratio", type=float, default=0.10,
                    help="Keep ratio for stock daemon PFlash backend.")
    ap.add_argument("--daemon-no-park", action="store_true",
                    help="Run stock daemon PFlash with compress-nopark.")
    ap.add_argument("--prefill-park-policy", choices=["full", "draft", "none"],
                    default="full",
                    help="Parking policy passed to native prefill backends.")
    ap.add_argument("--budget", type=int, default=22)
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--fa-window", type=int, default=0,
                    help="Target sliding FA window. 0 means full attention.")
    ap.add_argument("--mega-max-seq-len", type=int, default=0,
                    help="Override --prefill-mega-max-seq-len for mega backends.")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--prompt-repeats", type=int, default=80)
    ap.add_argument("--base-port", type=int, default=18190)
    ap.add_argument("--startup-timeout", type=float, default=360)
    ap.add_argument("--backends", default="off,daemon,mega,mega-native",
                    help="Comma-separated backend set to run.")
    ap.add_argument("--show-timing", action="store_true")
    args = ap.parse_args()
    if args.daemon_no_park:
        args.prefill_park_policy = "none"
    args.prompt = build_prompt(args.prompt_repeats)

    requested = {name.strip() for name in args.backends.split(",") if name.strip()}
    configs = [
        ("off", [
            "--prefill-compression", "off",
        ]),
        ("daemon", [
            "--prefill-compression", "always",
            "--prefill-threshold", "1",
            "--prefill-backend", "daemon",
            "--prefill-keep-ratio", str(args.daemon_keep_ratio),
            "--prefill-drafter", str(args.drafter),
            "--prefill-park-policy", args.prefill_park_policy,
            *(["--prefill-no-park"] if args.prefill_park_policy == "none" else []),
        ]),
        ("mega", [
            "--prefill-compression", "always",
            "--prefill-threshold", "1",
            "--prefill-backend", "mega",
            "--prefill-keep-ratio", "0.05",
            *(["--prefill-mega-max-seq-len", str(args.mega_max_seq_len)]
              if args.mega_max_seq_len else []),
        ]),
        ("mega-native", [
            "--prefill-compression", "always",
            "--prefill-threshold", "1",
            "--prefill-backend", "mega-native",
            "--prefill-keep-ratio", "0.05",
            "--prefill-mega-native-model", str(args.mega_native_model),
            "--prefill-park-policy", args.prefill_park_policy,
            *(["--prefill-no-park"] if args.prefill_park_policy == "none" else []),
            *(["--prefill-mega-max-seq-len", str(args.mega_max_seq_len)]
              if args.mega_max_seq_len else []),
        ]),
    ]
    configs = [(name, extra) for name, extra in configs if name in requested]
    if not configs:
        raise SystemExit(f"no matching backends selected from: {args.backends}")

    all_rows: dict[str, list[dict]] = {}
    for offset, (name, extra) in enumerate(configs):
        try:
            all_rows[name] = run_backend(args, name, args.base_port + offset, extra)
        except Exception as exc:
            print(f"[{name}] FAILED: {exc}", flush=True)
            all_rows[name] = []
        time.sleep(2)

    print("\nSUMMARY")
    for name, _ in configs:
        summarize(name, all_rows[name])


if __name__ == "__main__":
    main()
