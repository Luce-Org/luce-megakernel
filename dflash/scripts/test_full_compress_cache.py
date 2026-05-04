"""Option 3 integration test: full-compress-result cache.

Sends an identical ~40K-token NIAH prompt 3 times to a server with both
pFlash compression AND the full-compress-result cache enabled.

Expected behaviour:
  - Turn 1: cold compress + cold prefill + cache registration.
  - Turn 2: full-cache hit — skips BOTH the drafter dance AND the prefill.
            Latency should be < 30% of Turn 1.
  - Turn 3: same full-cache hit, same speedup.

Correctness: all three replies must be identical.

Skipped automatically if any prerequisite is missing:
  - target GGUF
  - draft (drafter) safetensors dir or GGUF
  - Qwen3-0.6B-BF16 drafter GGUF
  - test_dflash binary
"""
import os
import sys
import time
import signal
import subprocess
import urllib.request
import urllib.error
import json
import random
from pathlib import Path

# ─── Prerequisites ────────────────────────────────────────────────────

ROOT          = Path(__file__).resolve().parent.parent.parent
TARGET        = Path.home() / "models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf"
DRAFT         = Path.home() / "models/qwen3.6-27b-dflash"
DRAFTER_GGUF  = Path.home() / "models/Qwen3-0.6B-BF16.gguf"
BIN           = ROOT / "dflash/build/test_dflash"
SERVER_SCRIPT = ROOT / "dflash/scripts/server.py"

for p, label in [
    (TARGET,       "target GGUF"),
    (DRAFT,        "draft dir/GGUF"),
    (DRAFTER_GGUF, "drafter GGUF"),
    (BIN,          "test_dflash binary"),
]:
    if not p.exists():
        print(f"SKIP: {label} missing at {p}")
        sys.exit(0)

# ─── NIAH prompt builder ──────────────────────────────────────────────

def build_long_prompt(target_tokens: int, seed: int = 42) -> tuple[str, str]:
    """Needle-In-A-Haystack prompt for the PFlash compression path."""
    rng = random.Random(seed)
    key   = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=8))
    value = "".join(rng.choices("0123456789", k=7))
    filler = ("The grass is green. The sky is blue. The sun is yellow. "
              "Here we go. There and back again. ")
    target_chars = int(target_tokens * 4.0)
    body = (filler * (target_chars // len(filler) + 1))[:target_chars]
    insert = rng.randint(target_chars // 4, 3 * target_chars // 4)
    needle = f"The special magic {key} number is: {value}."
    body = body[:insert] + " " + needle + " " + body[insert:]
    prompt = (
        "Below is a long passage. Answer the question at the end "
        "based ONLY on information in the passage.\n\n"
        f"{body}\n\nQuestion: What is the special magic {key} number? "
        "Answer in one short sentence.\nAnswer:"
    )
    return prompt, value


PROMPT, ANSWER = build_long_prompt(target_tokens=40000)
print(f"Built NIAH prompt: ~{len(PROMPT) // 4} tokens (target ~40000), "
      f"needle={ANSWER!r}\n", flush=True)

# ─── Server launch ────────────────────────────────────────────────────

PORT       = 18184
SERVER_LOG = open("/tmp/test_full_compress_cache_server.log", "w")

# --prefix-cache-slots 4: prefix-cache pool (slots 0-3, unused when
#   compression fires because compressed tokens lack chat-template markers).
# --prefill-cache-slots 4: full-cache pool (slots 4-7).
# Total = 8 == daemon hard cap (PREFIX_CACHE_SLOTS in test_dflash.cpp).
proc = subprocess.Popen(
    [
        sys.executable, "-u", str(SERVER_SCRIPT),
        "--target",             str(TARGET),
        "--draft",              str(DRAFT),
        "--bin",                str(BIN),
        "--max-ctx",            "8192",
        "--port",               str(PORT),
        "--prefix-cache-slots", "4",
        "--prefill-cache-slots","4",
        "--prefill-compression","auto",
        "--prefill-threshold",  "32000",
        "--prefill-keep-ratio", "0.05",
        "--prefill-drafter",    str(DRAFTER_GGUF),
    ],
    stdout=SERVER_LOG,
    stderr=subprocess.STDOUT,
    bufsize=1,
)


def cleanup():
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


import atexit
atexit.register(cleanup)

# ─── Wait for server ──────────────────────────────────────────────────

print("Waiting for server (full-compress-cache test)...", flush=True)
deadline = time.time() + 240
ready = False
while time.time() < deadline:
    if proc.poll() is not None:
        SERVER_LOG.flush()
        print("SERVER DIED. Tail of /tmp/test_full_compress_cache_server.log:")
        print(open("/tmp/test_full_compress_cache_server.log").read()[-4000:])
        sys.exit(2)
    try:
        urllib.request.urlopen(
            f"http://127.0.0.1:{PORT}/v1/models", timeout=1
        ).read()
        ready = True
        break
    except (urllib.error.URLError, ConnectionResetError, TimeoutError):
        time.sleep(1)

if not ready:
    print("Server didn't come up within 240 s", flush=True)
    sys.exit(2)
print("Server up.\n", flush=True)

# ─── Request helper ───────────────────────────────────────────────────

def chat(max_tokens: int = 16) -> tuple[float, str]:
    payload = {
        "model": "luce-dflash",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    out = urllib.request.urlopen(req, timeout=900).read()
    dt = time.time() - t0
    txt = json.loads(out)["choices"][0]["message"]["content"]
    return dt, txt


# ─── Run 3 turns ─────────────────────────────────────────────────────

print("=== Turn 1 (cold: compress 40K -> ~2K, cold prefill, cache registration) ===")
t1, r1 = chat()
print(f"latency={t1:.2f}s  reply={r1!r}\n", flush=True)

print("=== Turn 2 (full-cache hit: skip compress + skip prefill) ===")
t2, r2 = chat()
print(f"latency={t2:.2f}s  reply={r2!r}\n", flush=True)

print("=== Turn 3 (full-cache hit again) ===")
t3, r3 = chat()
print(f"latency={t3:.2f}s  reply={r3!r}\n", flush=True)

# ─── Server log scan for hit markers ─────────────────────────────────

SERVER_LOG.flush()
log_text = open("/tmp/test_full_compress_cache_server.log").read()
# Turn 2 and 3 should both produce a [pc] full-cache hit line.
full_cache_hits = log_text.count("[pc] full-cache hit")

# ─── Verdict ─────────────────────────────────────────────────────────

ok_identical   = (r1 == r2 == r3)
ok_speedup_t2  = (t2 < t1 * 0.30)
ok_speedup_t3  = (t3 < t1 * 0.30)
ok_log_hits    = (full_cache_hits >= 2)
ok_needle_t1   = (ANSWER in r1)

print("=== Verdict ===")
print(f"t1={t1:.2f}s  t2={t2:.2f}s  t3={t3:.2f}s")
print(f"  ratio_2/1={t2/t1:.3f}  ratio_3/1={t3/t1:.3f}")
print(f"replies_identical:         {ok_identical}")
print(f"t2 < 30% of t1:            {ok_speedup_t2}")
print(f"t3 < 30% of t1:            {ok_speedup_t3}")
print(f"log [pc] full-cache hits:  {full_cache_hits} (need >= 2): {ok_log_hits}")
print(f"needle retrieved turn 1:   {ok_needle_t1}  (looking for {ANSWER!r})")

# needle retrieval is informational, not gated. The needle is a quality
# property of pFlash's compression (depends on --prefill-keep-ratio and the
# importance-scoring picking the right blocks), not of the cache. Failing
# the test on needle absence would couple cache correctness to compression
# quality. Tracked separately; for the cache test, identity-of-replies +
# log-confirmed cache hits is the right gate.
passed = ok_identical and ok_speedup_t2 and ok_speedup_t3 and ok_log_hits
if not passed:
    print("\nFAILED — see /tmp/test_full_compress_cache_server.log for details")
else:
    print("\nPASSED")
sys.exit(0 if passed else 1)
