"""
OpenAI-compat server throughput benchmark.

Streams each prompt through /v1/chat/completions and records:
  TTFT   — time to first content token (ms)
  tok/s  — decode throughput (completion_tokens / decode_wall_time)
  n_tok  — completion token count (from usage chunk)
  p_chars — prompt length in chars (replay mode only)

Two modes:

  Default — short hand-written code-agent prompts (~50–200 tokens).
            Measures isolated short-prompt decode tok/s.

  Replay  — recorded production prompts from a sessions.jsonl file.
            Realistic prompt sizes (5K–12K chars typical), measures the
            same throughput numbers under prefill-dominant load.

Usage:
    # dflash (default short-prompt mode)
    uv run scripts/bench_server.py

    # LM Studio or any other endpoint
    uv run scripts/bench_server.py --url http://localhost:1234 --model <model-id>

    # Repeat each prompt N times for stable averages
    uv run scripts/bench_server.py --repeat 3 --n-gen 256

    # Replay real prompts from a ddx sessions.jsonl
    uv run scripts/bench_server.py --replay ~/Projects/ddx/.ddx/agent-logs/sessions.jsonl
    uv run scripts/bench_server.py --replay file.jsonl --max-prompts 10
"""
import argparse
import json
import time
import sys
import urllib.request
import urllib.error
from dataclasses import dataclass, field


# ── Prompt corpus ─────────────────────────────────────────────────
# Typical single-turn code-agent requests: short system + focused user ask.
# Kept concise so prompt tokens stay small and decode throughput dominates.

SYSTEM = (
    "You are a senior software engineer. "
    "Reply with working code and brief inline comments. "
    "No preamble, no markdown fences unless the output is a complete file."
)

PROMPTS = [
    (
        "refactor_function",
        "Refactor this Python function so it handles empty input gracefully "
        "and uses a list comprehension instead of the explicit loop:\n\n"
        "def doubled(nums):\n    result = []\n    for n in nums:\n        result.append(n * 2)\n    return result",
    ),
    (
        "write_tests",
        "Write pytest unit tests for this function:\n\n"
        "def clamp(value, lo, hi):\n    return max(lo, min(hi, value))\n\n"
        "Cover: normal range, below lo, above hi, lo==hi, negative values.",
    ),
    (
        "fix_bug",
        "Fix the off-by-one error in this binary search and explain the change:\n\n"
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr)\n"
        "    while lo < hi:\n        mid = (lo + hi) // 2\n"
        "        if arr[mid] == target: return mid\n"
        "        elif arr[mid] < target: lo = mid\n"
        "        else: hi = mid\n    return -1",
    ),
    (
        "explain_code",
        "Explain what this decorator does and when you'd use it:\n\n"
        "import functools\n"
        "def retry(n=3):\n    def decorator(fn):\n        @functools.wraps(fn)\n"
        "        def wrapper(*args, **kwargs):\n            for i in range(n):\n"
        "                try: return fn(*args, **kwargs)\n"
        "                except Exception:\n                    if i == n-1: raise\n"
        "        return wrapper\n    return decorator",
    ),
    (
        "add_feature",
        "Add an optional `timeout` parameter (default 30s) to this HTTP GET helper. "
        "Raise a clear TimeoutError with the URL in the message:\n\n"
        "import urllib.request\ndef fetch(url: str) -> bytes:\n"
        "    with urllib.request.urlopen(url) as r:\n        return r.read()",
    ),
    (
        "code_review",
        "Review this Go snippet for correctness and idiomatic style. "
        "Give concrete suggestions, not general advice:\n\n"
        "func sum(nums []int) int {\n    total := 0\n    for i := 0; i < len(nums); i++ {\n"
        "        total = total + nums[i]\n    }\n    return total\n}",
    ),
    (
        "shell_command",
        "Give me a single bash one-liner to find all Python files modified in the last "
        "7 days under ./src, print their paths, and count them at the end.",
    ),
    (
        "generate_schema",
        "Write a Pydantic v2 model for a REST API response that contains: "
        "a string `id`, an ISO-8601 `created_at` datetime, a list of `tags` (strings), "
        "and an optional `metadata` dict. Add field validators where useful.",
    ),
]


# ── Streaming SSE reader ───────────────────────────────────────────

@dataclass
class ProbResult:
    name: str
    ttft_ms: float
    decode_tok_s: float  # n_tok / (t_end - t_first)
    overall_tok_s: float  # n_tok / (t_end - t0)
    n_tok: int
    error: str = ""


def _stream_prompt(url: str, model: str, system: str, user: str,
                   n_gen: int) -> ProbResult:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": n_gen,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()

    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    t_first: float | None = None
    n_tok = 0
    buf = b""

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            while True:
                chunk = resp.read(256)
                if not chunk:
                    break
                buf += chunk
                while b"\n\n" in buf:
                    line, buf = buf.split(b"\n\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload_str = line[5:].strip()
                    if payload_str == b"[DONE]":
                        break
                    try:
                        ev = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue

                    # Usage chunk (choices=[])
                    if not ev.get("choices") and ev.get("usage"):
                        n_tok = ev["usage"].get("completion_tokens", n_tok)
                        continue

                    choices = ev.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})

                    # First real token → record TTFT (any text, including reasoning)
                    if t_first is None and (
                        delta.get("content") or delta.get("reasoning_content")
                        or delta.get("tool_calls")
                    ):
                        t_first = time.perf_counter()

    except urllib.error.URLError as e:
        return ProbResult(name="", ttft_ms=0, decode_tok_s=0, overall_tok_s=0,
                          n_tok=0, error=str(e))
    except Exception as e:  # HTTPError, ConnectionResetError, etc. mid-stream
        return ProbResult(name="", ttft_ms=0, decode_tok_s=0, overall_tok_s=0,
                          n_tok=0, error=str(e))

    t_end = time.perf_counter()

    if t_first is None:
        t_first = t_end  # no content token received

    ttft_ms = (t_first - t0) * 1000
    decode_wall = t_end - t_first
    total_wall = t_end - t0
    decode_tok_s = n_tok / decode_wall if decode_wall > 0 else 0.0
    overall_tok_s = n_tok / total_wall if total_wall > 0 else 0.0

    return ProbResult(name="", ttft_ms=ttft_ms, decode_tok_s=decode_tok_s,
                      overall_tok_s=overall_tok_s, n_tok=n_tok)


# ── Replay corpus loader ───────────────────────────────────────────

def _load_replay_corpus(path: str, min_chars: int, max_prompts: int) -> list[tuple[str, str]]:
    """Load prompts from a ddx-style sessions.jsonl (one whole-session record per line).

    Each row is expected to have a top-level `prompt` string field. Sessions
    without a usable prompt are skipped silently. Sessions whose prompt is
    shorter than min_chars are also skipped (they are not representative of
    realistic agentic load).

    Returns [(label, prompt_text), ...] in file order, capped at max_prompts.
    Label is the session id when present, falling back to "row{n}".
    """
    out: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or len(prompt) < min_chars:
                continue
            label = str(row.get("id") or f"row{n}")
            out.append((label, prompt))
            if max_prompts and len(out) >= max_prompts:
                break
    return out


def _bucket(prompt_chars: int) -> str:
    if prompt_chars < 2000:   return "small"
    if prompt_chars < 8000:   return "medium"
    return "large"


# ── Main ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="OpenAI-compat server throughput benchmark")
    ap.add_argument("--url",    default="http://localhost:1236",
                    help="Base URL of the OpenAI-compat server (default: %(default)s)")
    ap.add_argument("--model",  default="",
                    help="Model id to send in the request (default: auto-detect from /v1/models)")
    ap.add_argument("--n-gen",  type=int, default=512,
                    help="max_tokens per request (default: %(default)s)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="Repeat each prompt N times and average (default: %(default)s)")
    ap.add_argument("--prompts", nargs="+", metavar="NAME",
                    help="(default mode) Run only named prompts")
    ap.add_argument("--replay", metavar="PATH", default=None,
                    help="Replay prompts from a ddx-style sessions.jsonl. Each row must have "
                         "a top-level `prompt` string. Disables the default code-agent corpus.")
    ap.add_argument("--replay-min-chars", type=int, default=500,
                    help="(replay) Skip sessions with prompts shorter than this (default: %(default)s)")
    ap.add_argument("--max-prompts", type=int, default=0,
                    help="(replay) Cap number of prompts to run (0 = no cap)")
    args = ap.parse_args()

    # Auto-detect model id if not given
    model = args.model
    if not model:
        try:
            with urllib.request.urlopen(f"{args.url}/v1/models", timeout=10) as r:
                data = json.loads(r.read())
            model = data["data"][0]["id"]
            print(f"model: {model} (auto-detected)", file=sys.stderr)
        except Exception as e:
            sys.exit(f"Cannot reach {args.url}/v1/models: {e}\nPass --model explicitly.")

    if args.replay:
        corpus = _load_replay_corpus(args.replay, args.replay_min_chars, args.max_prompts)
        if not corpus:
            sys.exit(f"No usable prompts (>= {args.replay_min_chars} chars) in {args.replay}")
        mode_desc = f"replay={args.replay} prompts={len(corpus)}"
        system_prompt = ""  # replay prompts already contain their own framing
    else:
        corpus = [(n, p) for n, p in PROMPTS
                  if not args.prompts or n in args.prompts]
        if not corpus:
            sys.exit(f"No prompts matched {args.prompts}")
        mode_desc = f"corpus=code-agent prompts={len(corpus)}"
        system_prompt = SYSTEM

    print(f"\nBenchmarking {args.url}  model={model}  {mode_desc}  n_gen={args.n_gen}  repeat={args.repeat}")
    print(f"  dec t/s = tok / (t_end - t_first)   ovr t/s = tok / total_wall")
    if args.replay:
        print(f"{'prompt':<22}  {'p_chars':>7}  {'TTFT ms':>8}  {'dec t/s':>8}  {'ovr t/s':>8}  {'n_tok':>6}")
        print("-" * 76)
    else:
        print(f"{'prompt':<25}  {'TTFT ms':>8}  {'dec t/s':>8}  {'ovr t/s':>8}  {'n_tok':>6}")
        print("-" * 66)

    all_decode_tok_s:  list[float] = []
    all_overall_tok_s: list[float] = []
    all_ttft:          list[float] = []
    by_bucket: dict[str, list[tuple[float, float, float]]] = {"small": [], "medium": [], "large": []}

    for name, user in corpus:
        run_decode:   list[float] = []
        run_overall:  list[float] = []
        run_ttft:     list[float] = []
        run_ntok:     list[int]   = []
        err = ""
        for _ in range(args.repeat):
            r = _stream_prompt(args.url, model, system_prompt, user, args.n_gen)
            if r.error:
                err = r.error
                break
            run_decode.append(r.decode_tok_s)
            run_overall.append(r.overall_tok_s)
            run_ttft.append(r.ttft_ms)
            run_ntok.append(r.n_tok)

        if err:
            if args.replay:
                print(f"{name[:22]:<22}  {len(user):>7}  ERROR: {err}")
            else:
                print(f"{name:<25}  ERROR: {err}")
            continue

        avg_decode  = sum(run_decode)  / len(run_decode)
        avg_overall = sum(run_overall) / len(run_overall)
        avg_ttft    = sum(run_ttft)    / len(run_ttft)
        avg_ntok    = sum(run_ntok)    // len(run_ntok)
        all_decode_tok_s.append(avg_decode)
        all_overall_tok_s.append(avg_overall)
        all_ttft.append(avg_ttft)
        if args.replay:
            by_bucket[_bucket(len(user))].append((avg_ttft, avg_decode, avg_overall))
            print(f"{name[:22]:<22}  {len(user):>7}  {avg_ttft:>8.0f}  {avg_decode:>8.1f}  {avg_overall:>8.1f}  {avg_ntok:>6}")
        else:
            print(f"{name:<25}  {avg_ttft:>8.0f}  {avg_decode:>8.1f}  {avg_overall:>8.1f}  {avg_ntok:>6}")

    if not all_decode_tok_s:
        return

    if args.replay:
        print("-" * 76)
        print(f"{'mean':<22}  {'':>7}  {sum(all_ttft)/len(all_ttft):>8.0f}  "
              f"{sum(all_decode_tok_s)/len(all_decode_tok_s):>8.1f}  "
              f"{sum(all_overall_tok_s)/len(all_overall_tok_s):>8.1f}")
        print(f"{'min/max dec t/s':<22}  {'':>7}  {'':>8}  "
              f"{min(all_decode_tok_s):>5.1f} / {max(all_decode_tok_s):.1f}")
        print()
        print(f"{'bucket':<10}  {'n':>3}  {'TTFT ms':>8}  {'dec t/s':>8}  {'ovr t/s':>8}")
        for b in ("small", "medium", "large"):
            rows = by_bucket[b]
            if not rows:
                continue
            mean_ttft    = sum(r[0] for r in rows) / len(rows)
            mean_decode  = sum(r[1] for r in rows) / len(rows)
            mean_overall = sum(r[2] for r in rows) / len(rows)
            print(f"{b:<10}  {len(rows):>3}  {mean_ttft:>8.0f}  {mean_decode:>8.1f}  {mean_overall:>8.1f}")
    else:
        print("-" * 66)
        print(f"{'mean':<25}  {sum(all_ttft)/len(all_ttft):>8.0f}  "
              f"{sum(all_decode_tok_s)/len(all_decode_tok_s):>8.1f}  "
              f"{sum(all_overall_tok_s)/len(all_overall_tok_s):>8.1f}")


if __name__ == "__main__":
    main()
