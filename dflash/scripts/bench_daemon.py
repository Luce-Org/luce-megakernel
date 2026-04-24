"""Daemon-mode HE bench. Hits /v1/chat/completions with the same 10 HE
prompts as bench_he.py and reports mean tok/s.

Streams the response and reports two numbers per prompt:

  * wall    — total HTTP time (tokenize + prefill + decode + HTTP / JSON)
  * decode  — first-token → last-token elapsed, matching bench_he.py's
              tok/s (excludes prefill + setup)

Compare `decode` against bench_he.py to verify the C++ decode path is as
fast under the daemon as under a one-shot test_dflash invocation.

Start the server first (same config the published numbers use):
    DFLASH27B_KV_TQ3=1 python3 scripts/server_tools.py \\
        --budget 22 --max-ctx 16384 --port 8000

Then:
    python3 scripts/bench_daemon.py --url http://localhost:8000 --n-gen 256
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path
import sys

# Reuse the exact same 10 HE prompts bench_he.py uses.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_he import PROMPTS


def run(url: str, prompt: str, n_gen: int) -> tuple[int, float, float]:
    """POST to /v1/chat/completions with stream=true. Return (n_tok, wall_secs,
    decode_secs) where decode_secs starts at the first streamed token (after
    prefill) and ends at the last token."""
    body = json.dumps({
        "model": "luce-dflash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_gen,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json",
                 "Accept": "text/event-stream"},
    )
    t0 = time.perf_counter()
    t_first = 0.0
    t_last = 0.0
    n_tok = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            # Count tokens by content / reasoning deltas. Tool-call deltas
            # aren't counted — they arrive as a single final chunk.
            if delta.get("content") or delta.get("reasoning_content"):
                if n_tok == 0:
                    t_first = time.perf_counter()
                n_tok += 1
                t_last = time.perf_counter()
    wall = time.perf_counter() - t0
    decode = (t_last - t_first) if n_tok > 1 else 0.0
    return n_tok, wall, decode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000",
                    help="Base URL of the running server (no /v1 suffix)")
    ap.add_argument("--n-gen", type=int, default=256)
    ap.add_argument("--warmup", action="store_true",
                    help="Run the first prompt once before timing to discard "
                         "cold-start effects (model is already resident, but "
                         "the first request allocates the decode VMM chunks).")
    args = ap.parse_args()

    if args.warmup:
        print("[bench] warmup...", flush=True)
        run(args.url, PROMPTS[0][1], args.n_gen)

    print(f"[bench] daemon API  n_gen={args.n_gen}  url={args.url}", flush=True)
    print(f"{'prompt':28s}  {'n_tok':>5s} {'wall_s':>7s} {'dec_s':>7s} "
          f"{'wall_tps':>9s} {'dec_tps':>9s}")
    print("-" * 72)
    wall_tps_list: list[float] = []
    dec_tps_list: list[float] = []
    total_tok = 0
    total_wall = 0.0
    total_decode = 0.0
    for name, text in PROMPTS:
        try:
            n_tok, wall, decode = run(args.url, text, args.n_gen)
        except Exception as e:
            print(f"  {name:26s}  FAILED: {e}", flush=True)
            continue
        if n_tok == 0:
            print(f"  {name:26s}  {n_tok:5d} {wall:7.2f}    --         --        -- "
                  "  (empty — daemon likely OOM'd)", flush=True)
            continue
        wall_tps = n_tok / wall
        dec_tps = (n_tok - 1) / decode if decode > 0 else 0.0
        wall_tps_list.append(wall_tps)
        if dec_tps > 0:
            dec_tps_list.append(dec_tps)
            total_decode += decode
        total_tok += n_tok
        total_wall += wall
        print(f"  {name:26s}  {n_tok:5d} {wall:7.2f} {decode:7.2f} "
              f"{wall_tps:9.2f} {dec_tps:9.2f}", flush=True)

    print("-" * 72)
    if wall_tps_list:
        print(f"wall tok/s mean:       {sum(wall_tps_list)/len(wall_tps_list):7.2f}  "
              f"(HTTP + tokenize + prefill + decode)")
        if dec_tps_list:
            print(f"decode tok/s mean:     {sum(dec_tps_list)/len(dec_tps_list):7.2f}  "
                  f"(first-token → last-token, matches bench_he.py's number)")
            agg_dec = (total_tok - len(dec_tps_list)) / total_decode if total_decode > 0 else 0.0
            print(f"decode tok/s aggregate:{agg_dec:7.2f}")
            print(f"decode tok/s range:    {min(dec_tps_list):.2f} - {max(dec_tps_list):.2f}")
    else:
        print("no successful runs")


if __name__ == "__main__":
    main()
