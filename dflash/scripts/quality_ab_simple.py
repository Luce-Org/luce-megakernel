#!/usr/bin/env python3
"""
Simple A/B quality check: do prefix-cache + pflash compression degrade replies
vs baseline (everything off)? Hand-picked prompts that don't need extended
reasoning, no thinking budget burned. One server per config kept alive.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROMPTS = [
    # (id, category, prompt) — only fast-completing prompts; the model thinks regardless
    ("p1",  "factual",      "What is the capital of France?"),
    ("p2",  "factual",      "Who wrote 'Pride and Prejudice'?"),
    ("p3",  "translation",  "Translate the sentence 'Good morning, how are you?' into Spanish."),
    ("p4",  "list",         "List three primary colors."),
    ("p5",  "definition",   "What does the acronym HTTP stand for?"),
    ("p6",  "haiku",        "Write a haiku about autumn leaves. Just the poem, no intro."),
    ("p7",  "code-short",   "Write a Python one-liner that returns the sum of integers from 1 to 100."),
]

CONFIGS = {
    "baseline":         {  # everything off — reference run
        "flags": ["--prefix-cache-slots", "0", "--prefill-compression", "off"],
        "env":   {"DFLASH_FP_USE_BSA": "0"},
    },
    "baseline_2":       {  # IDENTICAL to baseline — sanity check for determinism
        "flags": ["--prefix-cache-slots", "0", "--prefill-compression", "off"],
        "env":   {"DFLASH_FP_USE_BSA": "0"},
    },
    "prefix_on":        {  # only prefix cache enabled (default tq3_0 K-cache)
        "flags": ["--prefix-cache-slots", "4", "--prefill-compression", "off"],
        "env":   {"DFLASH_FP_USE_BSA": "0"},
    },
    "baseline_f16":     {  # f16 K-cache, no caches — proper reference for prefix_on_f16
        "flags": ["--prefix-cache-slots", "0", "--prefill-compression", "off", "--kv-f16"],
        "env":   {"DFLASH_FP_USE_BSA": "0"},
    },
    "prefix_on_f16":    {  # prefix cache + f16 K-cache (tests KV-quantization hypothesis)
        "flags": ["--prefix-cache-slots", "4", "--prefill-compression", "off", "--kv-f16"],
        "env":   {"DFLASH_FP_USE_BSA": "0"},
    },
    "compression_on":   {  # only pflash compression (no prefix cache, no BSA)
        "flags": ["--prefix-cache-slots", "0", "--prefill-compression", "always"],
        "env":   {"DFLASH_FP_USE_BSA": "0"},
    },
    "bsa_on":           {  # only BSA in pflash (no prefix cache)
        "flags": ["--prefix-cache-slots", "0", "--prefill-compression", "off"],
        "env":   {"DFLASH_FP_USE_BSA": "1"},
    },
    "all_on":           {  # prefix + pflash compression + BSA
        "flags": ["--prefix-cache-slots", "4", "--prefill-compression", "always"],
        "env":   {"DFLASH_FP_USE_BSA": "1"},
    },
}

# Suppress thinking via system prompt (works regardless of chat_template_kwargs plumbing)
SYSTEM_NO_THINK = "/no_think"

PORT = 8765
TARGET = os.environ.get("PFLASH_TARGET", "/home/peppi/models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf")
DRAFT  = os.environ.get("PFLASH_DRAFT",  "/home/peppi/models/qwen3.6-27b-dflash/model.safetensors")
BIN    = os.environ.get("PFLASH_BIN",    "dflash/build/test_dflash")
DRAFTER = os.environ.get("PFLASH_DRAFTER", str(Path.home() / "models/Qwen3-0.6B-BF16.gguf"))


def chat_post(payload, timeout=120):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def wait_ready(proc, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            data = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2).read()
            # Verify the response belongs to OUR server, not a stale/foreign process on the port.
            parsed = json.loads(data)
            ids = [m.get("id", "") for m in parsed.get("data", [])]
            if any("luce-dflash" in mid for mid in ids):
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def spawn(name, cfg, log_path):
    flags = list(cfg["flags"])
    if "--prefill-compression" in flags:
        idx = flags.index("--prefill-compression")
        if flags[idx + 1] != "off" and DRAFTER:
            flags += ["--prefill-drafter", DRAFTER]
    cmd = [
        sys.executable, "-u", "dflash/scripts/server.py",
        "--target", TARGET, "--draft", DRAFT, "--bin", BIN,
        "--port", str(PORT), *flags,
    ]
    env = {**os.environ, **cfg.get("env", {})}
    print(f"[{name}] spawn: {' '.join(cmd)}", flush=True)
    log = open(log_path, "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def run(name, cfg):
    log = f"/tmp/ab_results/server_{name}.log"
    # Kill any stale/foreign process occupying the port before spawning our server.
    subprocess.run(["fuser", "-k", f"{PORT}/tcp"], capture_output=True)
    time.sleep(0.5)
    proc = spawn(name, cfg, log)
    try:
        if not wait_ready(proc):
            print(f"[{name}] server failed to start — see {log}", flush=True)
            return None
        print(f"[{name}] ready", flush=True)
        rows = []
        for pid_, cat, text in PROMPTS:
            t0 = time.time()
            try:
                reply = chat_post({
                    "model": "luce-dflash",
                    "messages": [
                        {"role": "system", "content": SYSTEM_NO_THINK},
                        {"role": "user", "content": text},
                    ],
                    "max_tokens": 256,
                    "stream": False,
                })
                err = None
            except Exception as e:
                reply = ""
                err = str(e)
            dur = time.time() - t0
            print(f"  [{name}] {pid_} ({cat}) {dur:.1f}s — {reply[:60]!r}", flush=True)
            rows.append({"id": pid_, "category": cat, "prompt": text, "reply": reply, "err": err, "secs": round(dur, 2)})
        return rows
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()


def diff_replies(a, b):
    if a == b:
        return ("EXACT", 1.0, len(a))
    n = min(len(a), len(b))
    lcp = 0
    while lcp < n and a[lcp] == b[lcp]:
        lcp += 1
    ratio = lcp / max(1, max(len(a), len(b)))
    return ("DIFF", round(ratio, 3), lcp)


def main():
    os.makedirs("/tmp/ab_results", exist_ok=True)
    results = {}
    for name, cfg in CONFIGS.items():
        rows = run(name, cfg)
        if rows is None:
            print(f"ABORT: {name} did not run")
            sys.exit(1)
        Path(f"/tmp/ab_results/simple_{name}.json").write_text(json.dumps(rows, indent=2))
        results[name] = rows

    # Each config is compared against the matching baseline:
    #   - configs ending in _f16 use baseline_f16
    #   - everything else uses baseline
    print()
    for other in [c for c in CONFIGS if c != "baseline"]:
        ref_name = "baseline_f16" if other.endswith("_f16") and "baseline_f16" in results else "baseline"
        if other == ref_name:
            continue
        ref = {r["id"]: r for r in results[ref_name]}
        test = {r["id"]: r for r in results[other]}
        print(f"=== A/B COMPARISON ({ref_name} vs {other}) ===")
        print(f"{'id':6} {'category':14} {'status':6} {'lcp_ratio':10} {'lcp_chars':10}")
        exact = 0
        compared = 0
        skipped = 0
        for pid_ in sorted(ref):
            ref_row, test_row = ref[pid_], test[pid_]
            if ref_row.get("err") or test_row.get("err"):
                # Both (or one) errored — empty string would look like an exact match;
                # skip to avoid inflating the match score and hiding real failures.
                skipped += 1
                print(f"{pid_:6} {ref_row['category']:14} SKIP   (err: ref={ref_row.get('err')!r} test={test_row.get('err')!r})")
                continue
            a, b = ref_row["reply"], test_row["reply"]
            status, ratio, lcp = diff_replies(a, b)
            if status == "EXACT":
                exact += 1
            compared += 1
            print(f"{pid_:6} {ref_row['category']:14} {status:6} {ratio:<10} {lcp}")
        if compared:
            print(f"Exact match: {exact}/{compared}  ({100*exact/compared:.0f}%)  [{skipped} skipped due to errors]\n")
        else:
            print(f"Exact match: N/A  (all {skipped} prompts errored — no valid comparison)\n")

    # Sanity-check verdict
    base = {r["id"]: r for r in results["baseline"]}
    base2 = {r["id"]: r for r in results["baseline_2"]}
    # Exclude errored prompts from the sanity count to avoid false exact-matches.
    valid_pids = [pid_ for pid_ in base if not base[pid_].get("err") and not base2[pid_].get("err")]
    sanity_exact = sum(1 for pid_ in valid_pids if base[pid_]["reply"] == base2[pid_]["reply"])
    n = len(valid_pids)
    print(f"=== SANITY: baseline vs baseline_2 → {sanity_exact}/{n} exact ===")
    if sanity_exact == n:
        print(">>> Server is deterministic. Any other DIFF is caused by the cache/feature being tested.")
    else:
        print(f">>> WARNING: Server is NONDETERMINISTIC ({n - sanity_exact}/{n} differ even between identical runs).")
        print(">>> A/B comparisons against baseline cannot distinguish cache effects from intrinsic noise.")


if __name__ == "__main__":
    main()
