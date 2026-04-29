"""
OpenAI-compat server throughput benchmark.

Two modes:

  Default    — short hand-written code-agent prompts (~50–200 tokens),
               one request each. Measures isolated short-prompt
               decode tok/s.

  Transcript — replay a recorded Claude Code session transcript
               (~/.claude/projects/<workspace>/<uuid>.jsonl) call by
               call. Each assistant turn becomes one timed LLM call:
               we send the message prefix that was actually sent at
               that point in the session, measure TTFT / decode rate,
               then advance state with the recorded next turn. Every
               server under test sees the exact same per-call input
               distribution, including real tool I/O accumulating
               turn-over-turn — which is the only honest way to
               measure agentic workload.

Per-call metrics:
  TTFT     — time to first token (ms); covers prefill + first decode
  dec t/s  — n_tok / (t_end − t_first)
  in_chars — total chars in the OpenAI message array sent for this call

Usage:
    # default short-prompt mode
    uv run scripts/bench_server.py

    # transcript replay (the real workload)
    uv run scripts/bench_server.py \\
        --transcript ~/.claude/projects/<workspace>/<uuid>.jsonl \\
        --max-calls 30

    # multiple transcripts, against a remote endpoint
    uv run scripts/bench_server.py --url http://hel:1234 --model qwen/qwen3.6-27b \\
        --transcript a.jsonl --transcript b.jsonl
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


# ── Streaming chat (with tool-call assembly) ───────────────────────


@dataclass
class CallResult:
    ttft_ms: float
    decode_tok_s: float
    overall_tok_s: float
    n_tok_out: int
    content: str
    tool_calls: list  # [{"id","name","arguments_str"}]
    finish_reason: str
    error: str = ""


def _stream_chat(url: str, model: str, messages: list,
                 n_gen: int, tools: list | None = None) -> CallResult:
    """One streaming /v1/chat/completions call. Assembles content + tool_calls."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": n_gen,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    body = json.dumps(payload).encode()

    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    t_first: float | None = None
    n_tok_out = 0
    content_parts: list[str] = []
    tool_call_buf: dict[int, dict] = {}  # index → {id, name, args}
    finish_reason = ""
    buf = b""

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
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

                    if not ev.get("choices") and ev.get("usage"):
                        n_tok_out = ev["usage"].get("completion_tokens", n_tok_out)
                        continue

                    choices = ev.get("choices", [])
                    if not choices:
                        continue
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish_reason = fr
                    delta = choices[0].get("delta", {})

                    if t_first is None and (
                        delta.get("content")
                        or delta.get("reasoning_content")
                        or delta.get("tool_calls")
                    ):
                        t_first = time.perf_counter()

                    if delta.get("content"):
                        content_parts.append(delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        slot = tool_call_buf.setdefault(
                            idx, {"id": "", "name": "", "args": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["args"] += fn["arguments"]
    except urllib.error.URLError as e:
        return CallResult(0, 0, 0, 0, "", [], "", error=str(e))
    except Exception as e:
        return CallResult(0, 0, 0, 0, "", [], "", error=str(e))

    t_end = time.perf_counter()
    if t_first is None:
        t_first = t_end
    ttft_ms = (t_first - t0) * 1000
    decode_wall = max(t_end - t_first, 1e-9)
    total_wall = max(t_end - t0, 1e-9)

    tool_calls = []
    for idx in sorted(tool_call_buf):
        s = tool_call_buf[idx]
        tool_calls.append({
            "id": s["id"] or f"call_{idx}",
            "name": s["name"],
            "arguments_str": s["args"],
        })

    return CallResult(
        ttft_ms=ttft_ms,
        decode_tok_s=n_tok_out / decode_wall,
        overall_tok_s=n_tok_out / total_wall,
        n_tok_out=n_tok_out,
        content="".join(content_parts),
        tool_calls=tool_calls,
        finish_reason=finish_reason,
    )


def _messages_chars(messages: list) -> int:
    """Total chars across all messages — proxy for prompt size in tokens.

    Roughly chars/3.5 ≈ tokens for English/code. We track chars (not
    estimated tokens) so the metric is independent of tokenizer choice.
    """
    n = 0
    for m in messages:
        n += len(m.get("content") or "")
        for tc in m.get("tool_calls") or []:
            n += len(tc.get("function", {}).get("arguments") or "")
            n += len(tc.get("function", {}).get("name") or "")
    return n


# ── Claude Code transcript replay ──────────────────────────────────
#
# Claude Code stores per-session transcripts at
#   ~/.claude/projects/<workspace>/<session_uuid>.jsonl
#
# Each line is one event in Anthropic Messages format. A real LLM "turn"
# (one /v1/messages API call's response) can span MULTIPLE jsonl records:
# typically a `thinking` block, a `text` block, and one or more `tool_use`
# blocks all share a single API response but get serialized as separate
# rows. Same for user turns: each `tool_result` is its own row.
#
# To replay against an OpenAI-compat server, we:
#   1. Group consecutive same-role records into single turns.
#   2. Convert each turn to OpenAI message format.
#   3. For every assistant turn, the prefix that was originally sent to
#      the API = system prompt + all preceding user/assistant turns. We
#      send that prefix to the bench server, measure TTFT + decode tok/s,
#      then DISCARD the response and advance state with the recorded
#      assistant turn. This way every server under test sees the exact
#      same per-call input distribution.
#
# Tool definitions are NOT sent. Real Claude Code requests include a
# 30-tool catalog (~5–10 KB of stable prefix). The transcript doesn't
# capture it, so per-call input here is slightly smaller than the real
# request — comparable across servers but not bit-exact to the original.


def _load_claude_transcript(path: str) -> list[dict]:
    """Parse a Claude Code .jsonl into ordered (role, blocks) turns.

    Coalesces consecutive same-role records (since one logical turn can
    span multiple records). Skips attachment / queue-operation / ai-title.
    """
    turns: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") not in ("user", "assistant"):
                continue
            msg = rec.get("message") or {}
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(content, list):
                blocks = content
            elif isinstance(content, str):
                blocks = [{"type": "text", "text": content}]
            else:
                continue
            if turns and turns[-1]["role"] == role:
                turns[-1]["blocks"].extend(blocks)
            else:
                turns.append({"role": role, "blocks": blocks})
    return turns


def _to_openai_messages(turns: list[dict]) -> list[dict]:
    """Convert a list of Anthropic-format turns into an OpenAI messages array."""
    out: list[dict] = []
    for turn in turns:
        role = turn["role"]
        blocks = turn["blocks"]
        if role == "user":
            text_parts: list[str] = []
            for blk in blocks:
                t = blk.get("type")
                if t == "text":
                    text_parts.append(blk.get("text") or "")
                elif t == "tool_result":
                    tc_id = blk.get("tool_use_id") or ""
                    raw = blk.get("content")
                    if isinstance(raw, list):
                        text = "".join(
                            c.get("text", "") for c in raw
                            if isinstance(c, dict) and c.get("type") == "text"
                        )
                    else:
                        text = str(raw) if raw else ""
                    out.append({"role": "tool", "tool_call_id": tc_id, "content": text})
            if text_parts:
                out.append({"role": "user", "content": "\n".join(text_parts)})
        else:  # assistant
            text_parts = []
            tool_calls: list[dict] = []
            for blk in blocks:
                t = blk.get("type")
                if t == "text":
                    text_parts.append(blk.get("text") or "")
                elif t == "tool_use":
                    tool_calls.append({
                        "id": blk.get("id") or "",
                        "type": "function",
                        "function": {
                            "name": blk.get("name") or "",
                            "arguments": json.dumps(blk.get("input") or {}),
                        },
                    })
                # `thinking` blocks dropped — Anthropic-only, no OpenAI equivalent
            asst: dict = {"role": "assistant"}
            if text_parts:
                asst["content"] = "\n".join(text_parts)
            elif not tool_calls:
                asst["content"] = ""
            if tool_calls:
                asst["tool_calls"] = tool_calls
            out.append(asst)
    return out


@dataclass
class TranscriptResult:
    label: str
    per_call: list  # one dict per replayed LLM call
    wall_s: float
    max_in_chars: int
    finish: str  # "complete" | "error: ..." | "max-calls"


def _replay_transcript(url: str, model: str, transcript_path: str, label: str,
                       n_gen: int, max_calls: int) -> TranscriptResult:
    """Walk a Claude Code transcript, replay each LLM call against the server."""
    turns = _load_claude_transcript(transcript_path)
    if not turns:
        return TranscriptResult(label=label, per_call=[], wall_s=0,
                                max_in_chars=0, finish="empty")

    # Each assistant turn is a replay point — we send everything before it.
    asst_indices = [i for i, t in enumerate(turns) if t["role"] == "assistant"]
    if max_calls:
        asst_indices = asst_indices[:max_calls]

    per_call: list = []
    max_in = 0
    t_session = time.perf_counter()
    finish = "complete"

    for n, idx in enumerate(asst_indices, start=1):
        prefix = _to_openai_messages(turns[:idx])
        in_chars = _messages_chars(prefix)
        max_in = max(max_in, in_chars)
        r = _stream_chat(url, model, prefix, n_gen, tools=None)
        per_call.append({
            "call": n,
            "in_chars": in_chars,
            "n_tok_out": r.n_tok_out,
            "ttft_ms": r.ttft_ms,
            "decode_tok_s": r.decode_tok_s,
            "error": r.error,
        })
        if r.error:
            finish = f"error: {r.error[:80]}"
            break
    else:
        if max_calls and len(asst_indices) == max_calls:
            finish = "max-calls"

    return TranscriptResult(
        label=label,
        per_call=per_call,
        wall_s=time.perf_counter() - t_session,
        max_in_chars=max_in,
        finish=finish,
    )


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
    ap.add_argument("--transcript", metavar="PATH", action="append", default=None,
                    help="Replay a Claude Code session transcript "
                         "(~/.claude/projects/<workspace>/<uuid>.jsonl). "
                         "Each assistant turn becomes one timed LLM call against the bench "
                         "server. Repeatable to replay multiple transcripts.")
    ap.add_argument("--max-calls", type=int, default=0,
                    help="(transcript) Cap LLM calls per transcript (0 = all)")
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

    if args.transcript:
        return _run_transcripts(args, model)

    # ── default short-prompt mode ──────────────────────────────────
    corpus = [(n, p) for n, p in PROMPTS
              if not args.prompts or n in args.prompts]
    if not corpus:
        sys.exit(f"No prompts matched {args.prompts}")

    print(f"\nBenchmarking {args.url}  model={model}  corpus=code-agent prompts={len(corpus)}  n_gen={args.n_gen}  repeat={args.repeat}")
    print(f"  dec t/s = tok / (t_end - t_first)   ovr t/s = tok / total_wall")
    print(f"{'prompt':<25}  {'TTFT ms':>8}  {'dec t/s':>8}  {'ovr t/s':>8}  {'n_tok':>6}")
    print("-" * 66)

    all_decode_tok_s:  list[float] = []
    all_overall_tok_s: list[float] = []
    all_ttft:          list[float] = []

    for name, user in corpus:
        run_decode:   list[float] = []
        run_overall:  list[float] = []
        run_ttft:     list[float] = []
        run_ntok:     list[int]   = []
        err = ""
        for _ in range(args.repeat):
            r = _stream_prompt(args.url, model, SYSTEM, user, args.n_gen)
            if r.error:
                err = r.error
                break
            run_decode.append(r.decode_tok_s)
            run_overall.append(r.overall_tok_s)
            run_ttft.append(r.ttft_ms)
            run_ntok.append(r.n_tok)

        if err:
            print(f"{name:<25}  ERROR: {err}")
            continue

        avg_decode  = sum(run_decode)  / len(run_decode)
        avg_overall = sum(run_overall) / len(run_overall)
        avg_ttft    = sum(run_ttft)    / len(run_ttft)
        avg_ntok    = sum(run_ntok)    // len(run_ntok)
        all_decode_tok_s.append(avg_decode)
        all_overall_tok_s.append(avg_overall)
        all_ttft.append(avg_ttft)
        print(f"{name:<25}  {avg_ttft:>8.0f}  {avg_decode:>8.1f}  {avg_overall:>8.1f}  {avg_ntok:>6}")

    if not all_decode_tok_s:
        return
    print("-" * 66)
    print(f"{'mean':<25}  {sum(all_ttft)/len(all_ttft):>8.0f}  "
          f"{sum(all_decode_tok_s)/len(all_decode_tok_s):>8.1f}  "
          f"{sum(all_overall_tok_s)/len(all_overall_tok_s):>8.1f}")


def _run_transcripts(args, model: str) -> None:
    """Replay each Claude Code transcript against the bench server."""
    transcripts = args.transcript or []

    print(f"\nBenchmarking {args.url}  model={model}  transcripts={len(transcripts)}  "
          f"n_gen={args.n_gen}  max_calls={args.max_calls or 'all'}")
    print(f"  dec t/s = tok / (t_end - t_first)   in_chars = OpenAI message-array size at the moment of the call")

    print(f"\n{'transcript':<28}  {'calls':>5}  {'wall s':>7}  {'max_in_chars':>13}  "
          f"{'mean TTFT':>10}  {'mean dec t/s':>13}  finish")
    print("-" * 110)

    results: list[TranscriptResult] = []
    for path in transcripts:
        # Use the parent dir name as a label — it encodes the workspace
        # and bead id in Claude Code's path scheme.
        from pathlib import Path
        label = Path(path).parent.name or Path(path).stem
        tr = _replay_transcript(args.url, model, path, label,
                                args.n_gen, args.max_calls)
        ok = [c for c in tr.per_call if not c.get("error")]
        mean_ttft = (sum(c["ttft_ms"] for c in ok) / len(ok)) if ok else 0.0
        mean_dec  = (sum(c["decode_tok_s"] for c in ok) / len(ok)) if ok else 0.0
        results.append(tr)
        print(f"{label[:28]:<28}  {len(tr.per_call):>5}  {tr.wall_s:>7.1f}  "
              f"{tr.max_in_chars:>13,}  {mean_ttft:>10.0f}  {mean_dec:>13.1f}  {tr.finish[:40]}")

    # ── aggregate per-call distribution ───────────────────────────
    all_calls = [c for r in results for c in r.per_call if not c.get("error")]
    if not all_calls:
        return
    in_chars = sorted(c["in_chars"] for c in all_calls)
    ttfts    = sorted(c["ttft_ms"] for c in all_calls)
    dec_tps  = sorted(c["decode_tok_s"] for c in all_calls)

    def pct(xs, p):
        return xs[min(len(xs) - 1, int(len(xs) * p))]

    print("\n=== per-call distribution (all transcripts, ok calls only) ===")
    print(f"  n_calls       = {len(all_calls)}")
    print(f"  in_chars      p50 = {pct(in_chars, 0.50):>8,}   "
          f"p95 = {pct(in_chars, 0.95):>8,}   max = {in_chars[-1]:>8,}")
    print(f"  TTFT ms       p50 = {pct(ttfts, 0.50):>8.0f}   "
          f"p95 = {pct(ttfts, 0.95):>8.0f}   max = {ttfts[-1]:>8.0f}")
    print(f"  dec t/s       p50 = {pct(dec_tps, 0.50):>8.1f}   "
          f"p05 = {pct(dec_tps, 0.05):>8.1f}   min = {dec_tps[0]:>8.1f}")

    # ── input-size buckets ─────────────────────────────────────────
    buckets: dict[str, list] = {"small": [], "medium": [], "large": [], "huge": []}
    for c in all_calls:
        ch = c["in_chars"]
        b = "small" if ch < 4000 else "medium" if ch < 16000 else "large" if ch < 48000 else "huge"
        buckets[b].append(c)
    print("\n=== by input-size bucket ===")
    print(f"  {'bucket':<8}  {'range chars':<14}  {'n':>4}  {'TTFT ms':>9}  {'dec t/s':>9}")
    for name, label, _hi in (("small", "<4K", 4000), ("medium", "4K-16K", 16000),
                              ("large", "16K-48K", 48000), ("huge", ">48K", 10**9)):
        rows = buckets[name]
        if not rows:
            continue
        m_ttft = sum(c["ttft_ms"] for c in rows) / len(rows)
        m_dec  = sum(c["decode_tok_s"] for c in rows) / len(rows)
        print(f"  {name:<8}  {label:<14}  {len(rows):>4}  {m_ttft:>9.0f}  {m_dec:>9.1f}")


if __name__ == "__main__":
    main()
