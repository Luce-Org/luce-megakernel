#!/usr/bin/env python3
"""NIAH (Needle In A Haystack) driver for Laguna + PFlash on dflash.

Flow per test point (context_len in tokens):
  1. Build a haystack of `context_len` Laguna tokens of repeated filler text.
     Insert the needle string at the configured depth.
  2. Tokenize the haystack into BOTH:
        - drafter (Qwen3) tokens   -> drafter_ids.bin
        - target (Laguna) tokens   -> laguna_ids.bin
  3. Spawn pflash_daemon (drafter) once, send `compress` cmd over stdin
     pointing at drafter_ids.bin, capture compressed Qwen3 IDs over stream-fd.
  4. Detokenize compressed Qwen3 IDs back to text via Qwen3 tokenizer.
  5. Re-tokenize the compressed text with the Laguna tokenizer ->
     compressed_laguna_ids.bin.
  6. Spawn test_laguna_daemon (target) once, send `generate compressed_laguna_ids.bin
     <n_gen> out_ids.bin`. Read out_ids.bin.
  7. Detokenize output via Laguna tokenizer, check if needle answer is in text.
  8. Report TTFT, decode tok/s, NIAH retrieval result.

Cross-tokenizer is byte-level BPE bijective (both Qwen3 and Laguna use
GPT-2 byte-level pre-tokenizer), so detok-then-retok preserves info.

Usage:
    python3 laguna_pflash_niah.py \\
        --target /path/to/laguna-xs2-Q4_K_M.gguf \\
        --drafter /path/to/Qwen3-0.6B-BF16.gguf \\
        --laguna-tok /path/to/Laguna-XS.2 \\
        --drafter-tok /path/to/Qwen3-0.6B \\
        --pflash-bin /path/to/pflash_daemon \\
        --laguna-bin /path/to/test_laguna_daemon \\
        --ctx 16384 --depth 0.5 --keep 0.10
"""
from __future__ import annotations
import argparse, os, struct, subprocess, sys, tempfile, time
from pathlib import Path

NEEDLE = "The magic password for the Lucebox vault is BLUEHORIZON-7421."
NEEDLE_QUERY = "What is the magic password for the Lucebox vault?"
NEEDLE_ANSWER_KEY = "BLUEHORIZON-7421"

# Boilerplate filler. Repeated to reach the desired context length.
FILLER = (
    "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions. "
    "It extends from the Arctic Ocean in the north to the Antarctic in the south, "
    "bounded by Asia and Oceania in the west and the Americas in the east. "
)

def write_counted_i32(path: Path, ids: list[int]) -> None:
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(ids)))
        for tok in ids:
            f.write(struct.pack("<i", int(tok)))

def read_counted_i32(path: Path) -> list[int]:
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 0)[0]
    return list(struct.unpack_from(f"<{n}i", data, 4))

def build_haystack(target_tok, ctx: int, depth_frac: float,
                    filler_text: str | None = None) -> tuple[str, list[int]]:
    # Build text of ~ctx tokens of filler, insert needle near depth_frac.
    # When filler_text is provided (--filler-file), use that instead of the
    # synthetic FILLER constant. The user's filler is repeated to reach the
    # required char budget so a short source file still produces a long
    # haystack — but a long source (e.g. concatenated source code or a long
    # markdown doc) gives a non-uniform filler distribution that is closer to
    # what real prompts look like, instead of the worst-case uniform repeat.
    needed_chars = max(1024, ctx * 5)
    base = filler_text if filler_text else FILLER
    if not base:
        base = FILLER
    text = (base * (needed_chars // len(base) + 1))[:needed_chars]
    insert_at = int(len(text) * depth_frac)
    text = text[:insert_at] + "\n" + NEEDLE + "\n" + text[insert_at:]
    # Truncate by tokens.
    ids = target_tok.encode(text, add_special_tokens=False)
    if len(ids) > ctx:
        ids = ids[:ctx]
    truncated_text = target_tok.decode(ids, skip_special_tokens=False)
    # Wrap in chat template: system, user with haystack + question, then start
    # of assistant. Laguna template emits BOS at the beginning.
    messages = [
        {"role": "user", "content": truncated_text + "\n\n" + NEEDLE_QUERY},
    ]
    chat_text = target_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    chat_ids = target_tok.encode(chat_text, add_special_tokens=False)
    return chat_text, list(chat_ids)

def tokenize_for_drafter(drafter_tok, text: str) -> list[int]:
    return drafter_tok.encode(text, add_special_tokens=False)

def detok_drafter(drafter_tok, ids: list[int]) -> str:
    return drafter_tok.decode(ids, skip_special_tokens=False)

# ------------------------------------------------------------------
# Cross-tokenizer round-trip helpers (PFlash chunk-boundary recovery).
#
# pflash_daemon returns a subset of drafter token IDs (preserving order)
# but DROPS tokens that fall in low-importance chunks. When a needle word
# spans a chunk boundary, the drafter may keep the leading token (e.g.
# "BLUEH") and drop the trailing tokens ("ORIZON-7421"). Decoding the
# subset alone yields a truncated word.
#
# Fix: recover the kept-token positions by greedy subsequence match,
# group consecutive positions into runs, then expand each run outward
# until both endpoints sit on whitespace. Concat the expanded spans and
# decode once. The expanded set is a SUPERSET of the kept tokens, so the
# semantic compression is preserved while word integrity is restored.
# ------------------------------------------------------------------

def _recover_kept_indices(full_ids: list[int], kept_ids: list[int]) -> list[int]:
    """Greedy subsequence match: returns positions in full_ids that align with kept_ids in order."""
    out: list[int] = []
    j = 0
    for i, t in enumerate(full_ids):
        if j < len(kept_ids) and t == kept_ids[j]:
            out.append(i)
            j += 1
    if j != len(kept_ids):
        raise RuntimeError(f"kept_ids not a subsequence of full_ids ({j}/{len(kept_ids)} matched)")
    return out


def _group_runs(idxs: list[int]) -> list[tuple[int, int]]:
    """Group consecutive integers into [start,end] inclusive runs."""
    if not idxs:
        return []
    runs: list[tuple[int, int]] = []
    a = b = idxs[0]
    for x in idxs[1:]:
        if x == b + 1:
            b = x
        else:
            runs.append((a, b))
            a = b = x
    runs.append((a, b))
    return runs


def _expand_run_to_word_boundaries(full_ids: list[int], r0: int, r1: int,
                                    tok, decode_cache: dict[int, str],
                                    max_extend: int = 24) -> tuple[int, int]:
    """Extend [r0,r1] outward until both ends sit on a whitespace boundary.

    A token's text "starts on whitespace" if its decoded form begins with a
    space, newline or tab (Qwen3/Laguna byte-level BPE: a leading 'Ġ' becomes
    a leading space after decode). max_extend caps growth per side to keep
    the compression ratio close to the drafter's intent.
    """
    def _txt(idx: int) -> str:
        if idx not in decode_cache:
            decode_cache[idx] = tok.decode([full_ids[idx]], skip_special_tokens=False)
        return decode_cache[idx]

    a, b = r0, r1
    # Extend left: keep prepending tokens until the FIRST token in the span
    # itself begins with whitespace (so the word is complete on the left).
    steps = 0
    while a > 0 and steps < max_extend:
        cur = _txt(a)
        if cur and cur[0] in " \n\t\r":
            break
        a -= 1
        steps += 1
    # Extend right: keep appending tokens until the NEXT token (b+1) begins
    # with whitespace, indicating the current span ends on a word boundary.
    steps = 0
    while b + 1 < len(full_ids) and steps < max_extend:
        nxt = _txt(b + 1)
        if nxt and nxt[0] in " \n\t\r":
            break
        b += 1
        steps += 1
    return a, b


def _merge_overlapping(runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not runs:
        return []
    runs = sorted(runs)
    out = [runs[0]]
    for a, b in runs[1:]:
        pa, pb = out[-1]
        if a <= pb + 1:
            out[-1] = (pa, max(pb, b))
        else:
            out.append((a, b))
    return out


def cross_tok_compressed(drafter_full_ids: list[int], compressed_drafter: list[int],
                          drafter_tok, target_tok) -> tuple[str, list[int]]:
    """Decode pflash-compressed drafter IDs to text -> retokenize as target IDs.

    Recovers chunk-boundary words by expanding each contiguous run of kept
    drafter tokens outward to the surrounding whitespace.
    """
    kept_idx = _recover_kept_indices(drafter_full_ids, compressed_drafter)
    runs = _group_runs(kept_idx)
    cache: dict[int, str] = {}
    expanded = [_expand_run_to_word_boundaries(drafter_full_ids, r0, r1, drafter_tok, cache)
                for r0, r1 in runs]
    expanded = _merge_overlapping(expanded)
    expanded_ids: list[int] = []
    for a, b in expanded:
        expanded_ids.extend(drafter_full_ids[a:b + 1])
    text = drafter_tok.decode(expanded_ids, skip_special_tokens=False)
    target_ids = target_tok.encode(text, add_special_tokens=False)
    return text, target_ids

class PflashDaemon:
    """Wraps pflash_daemon stdin/stdout protocol with a streamfd pipe."""
    def __init__(self, bin_path: Path, drafter_gguf: Path):
        r, w = os.pipe()
        self.r = r
        env = os.environ.copy()
        env.setdefault("DFLASH_FP_USE_BSA", "1")
        env.setdefault("DFLASH_FP_ALPHA", "0.85")
        cmd = [str(bin_path), str(drafter_gguf), f"--stream-fd={w}"]
        self.proc = subprocess.Popen(cmd, env=env, pass_fds=(w,),
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      bufsize=0)
        os.close(w)
        # Wait for ready line.
        for _ in range(100):
            line = self.proc.stdout.readline().decode(errors="replace")
            if line.startswith("[pflash-daemon] ready"):
                print("  drafter:", line.strip(), file=sys.stderr)
                return
            print("  drafter init:", line.rstrip(), file=sys.stderr)
        raise RuntimeError("pflash_daemon did not become ready")

    def compress(self, drafter_ids: list[int], keep_x1000: int,
                 lookahead: int, chunk: int, pool: int) -> list[int]:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            in_path = Path(f.name)
        write_counted_i32(in_path, drafter_ids)
        cmd = f"compress {keep_x1000} {lookahead} {chunk} {pool} {in_path}\n"
        self.proc.stdin.write(cmd.encode())
        self.proc.stdin.flush()
        out = []
        # Read int32 stream until -1 sentinel.
        while True:
            buf = b""
            while len(buf) < 4:
                chunk_b = os.read(self.r, 4 - len(buf))
                if not chunk_b: raise RuntimeError("stream-fd EOF")
                buf += chunk_b
            (val,) = struct.unpack("<i", buf)
            if val == -1: break
            out.append(val)
        in_path.unlink(missing_ok=True)
        return out

    def close(self):
        try:
            self.proc.stdin.write(b"quit\n"); self.proc.stdin.flush()
        except Exception: pass
        try: self.proc.wait(timeout=5)
        except Exception: self.proc.kill()
        os.close(self.r)

class LagunaDaemon:
    def __init__(self, bin_path: Path, laguna_gguf: Path, max_ctx: int, kv: str, chunk: int):
        cmd = [str(bin_path), str(laguna_gguf),
               "--max-ctx", str(max_ctx), "--kv", kv, "--chunk", str(chunk)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      bufsize=0)
        for _ in range(100):
            line = self.proc.stdout.readline().decode(errors="replace")
            if line.startswith("[laguna-daemon] ready"):
                print("  laguna:", line.strip(), file=sys.stderr)
                return
            print("  laguna init:", line.rstrip(), file=sys.stderr)
        raise RuntimeError("test_laguna_daemon did not become ready")

    def generate(self, laguna_ids: list[int], n_gen: int) -> tuple[list[int], dict]:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            in_path = Path(f.name)
        out_path = in_path.with_suffix(".out.bin")
        write_counted_i32(in_path, laguna_ids)
        cmd = f"generate {in_path} {n_gen} {out_path}\n"
        self.proc.stdin.write(cmd.encode())
        self.proc.stdin.flush()
        # Read response line.
        line = self.proc.stdout.readline().decode(errors="replace").strip()
        if not line.startswith("ok"):
            raise RuntimeError(f"laguna daemon: {line}")
        # Parse stats.
        stats = {}
        for kv in line.split()[1:]:
            if "=" in kv:
                k, v = kv.split("=", 1)
                stats[k] = v
        out = read_counted_i32(out_path)
        in_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)
        return out, stats

    def close(self):
        try:
            self.proc.stdin.write(b"quit\n"); self.proc.stdin.flush()
        except Exception: pass
        try: self.proc.wait(timeout=5)
        except Exception: self.proc.kill()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, type=Path, help="Laguna GGUF")
    ap.add_argument("--drafter", required=True, type=Path, help="Qwen3-0.6B drafter GGUF")
    ap.add_argument("--laguna-tok", required=True, type=Path, help="Laguna HF dir with tokenizer.json")
    ap.add_argument("--drafter-tok", required=True, type=Path, help="Qwen3 HF dir with tokenizer.json")
    ap.add_argument("--pflash-bin", required=True, type=Path)
    ap.add_argument("--laguna-bin", required=True, type=Path)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--keep", type=float, default=0.10)
    ap.add_argument("--lookahead", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=32, help="drafter chunk size for scoring")
    ap.add_argument("--pool", type=int, default=13)
    ap.add_argument("--max-gen", type=int, default=64)
    ap.add_argument("--target-chunk", type=int, default=2048, help="target prefill chunk size")
    ap.add_argument("--target-kv", type=str, default="q4_0")
    ap.add_argument("--no-compress", action="store_true",
                    help="Skip drafter compression; feed full haystack to target.")
    ap.add_argument("--filler-file", type=Path, default=None,
                    help="Read filler text from this file/directory instead of "
                         "the synthetic FILLER constant. Directories are "
                         "recursively scanned and the contents of every "
                         "regular file (text-decodable as UTF-8) are "
                         "concatenated. Use this for real-prompt NIAH on code "
                         "or doc corpora.")
    args = ap.parse_args()

    # Use HF `transformers.AutoTokenizer` for chat-template support. Path must
    # NOT contain dots (transformers parses 'Laguna-XS.2' as repo_id) -- caller
    # should pass a symlinked dir without dots.
    from transformers import AutoTokenizer
    print(f"[niah] loading tokenizers...", file=sys.stderr)
    target_tok = AutoTokenizer.from_pretrained(str(args.laguna_tok), trust_remote_code=True)
    drafter_tok = AutoTokenizer.from_pretrained(str(args.drafter_tok), trust_remote_code=True)

    filler_text: str | None = None
    if args.filler_file is not None:
        if args.filler_file.is_dir():
            chunks: list[str] = []
            for p in sorted(args.filler_file.rglob("*")):
                if not p.is_file():
                    continue
                try:
                    chunks.append(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
            filler_text = "\n\n".join(chunks)
        else:
            filler_text = args.filler_file.read_text(encoding="utf-8")
        if not filler_text.strip():
            raise SystemExit(f"--filler-file {args.filler_file} produced empty text")
        print(f"[niah] filler from {args.filler_file} ({len(filler_text)} chars)",
              file=sys.stderr)

    print(f"[niah] building haystack ctx={args.ctx} depth={args.depth} "
          f"filler={'real' if filler_text else 'synthetic'}...", file=sys.stderr)
    text, target_full_ids = build_haystack(target_tok, args.ctx, args.depth, filler_text)
    print(f"[niah] haystack: {len(target_full_ids)} target tokens, {len(text)} chars", file=sys.stderr)

    drafter_full_ids = drafter_tok.encode(text, add_special_tokens=False)
    print(f"[niah] drafter tokens: {len(drafter_full_ids)}", file=sys.stderr)

    if args.no_compress:
        print("[niah] --no-compress: skipping drafter, feeding full haystack to target", file=sys.stderr)
        compressed_drafter = drafter_full_ids
        drafter_s = 0.0
        laguna_ids = target_full_ids
    else:
        # ---- Phase 1: drafter compress ----
        print("[niah] starting drafter daemon ...", file=sys.stderr)
        pf = PflashDaemon(args.pflash_bin, args.drafter)
        keep_x1000 = int(args.keep * 1000)
        t0 = time.time()
        compressed_drafter = pf.compress(drafter_full_ids, keep_x1000, args.lookahead, args.chunk, args.pool)
        drafter_s = time.time() - t0
        pf.close()
        print(f"[niah] drafter compressed {len(drafter_full_ids)} -> {len(compressed_drafter)} "
              f"({len(compressed_drafter)/len(drafter_full_ids):.4f}) in {drafter_s:.2f}s", file=sys.stderr)

        # ---- Phase 2: cross-tokenizer round-trip Qwen3 -> text -> Laguna ----
        # Word-boundary recovery: expand each contiguous kept run to the
        # nearest whitespace before decoding, so words split across chunk
        # boundaries (e.g. needle 'BLUEHORIZON-7421' tokenized as several
        # Qwen3 tokens) are preserved instead of being truncated to their
        # first kept fragment.
        compressed_text, laguna_ids = cross_tok_compressed(
            drafter_full_ids, compressed_drafter, drafter_tok, target_tok)
        print(f"[niah] cross-tok: {len(compressed_drafter)} qwen3 (kept) -> "
              f"{len(laguna_ids)} laguna tokens (after word-boundary recovery)",
              file=sys.stderr)

    # ---- Phase 3: laguna prefill + decode ----
    # The target only ever sees the compressed laguna_ids (or the full haystack
    # when --no-compress is passed). Sizing max_ctx to the compressed length
    # avoids OOM at huge --ctx values where the original haystack would force a
    # KV cache allocation that doesn't fit on a 24GB GPU even though the target
    # never receives that many tokens.
    max_ctx = len(laguna_ids) + args.max_gen + 64
    print(f"[niah] starting laguna daemon (max_ctx={max_ctx}, kv={args.target_kv}) ...", file=sys.stderr)
    lg = LagunaDaemon(args.laguna_bin, args.target, max_ctx, args.target_kv, args.target_chunk)
    t0 = time.time()
    gen_ids, stats = lg.generate(laguna_ids, args.max_gen)
    target_total_s = time.time() - t0
    lg.close()

    # ---- Phase 4: detok + needle check ----
    gen_text = target_tok.decode(gen_ids, skip_special_tokens=False)
    found = NEEDLE_ANSWER_KEY.lower() in gen_text.lower()

    print()
    print("=" * 80)
    print(f"NIAH @ ctx={args.ctx} depth={args.depth} keep={args.keep}")
    print("=" * 80)
    print(f"haystack tokens (target): {len(target_full_ids)}")
    print(f"haystack tokens (drafter): {len(drafter_full_ids)}")
    print(f"compressed drafter tokens: {len(compressed_drafter)} (ratio {len(compressed_drafter)/len(drafter_full_ids):.4f})")
    print(f"compressed laguna tokens:  {len(laguna_ids)}")
    print(f"drafter time: {drafter_s:.2f}s")
    print(f"target prefill_s={stats.get('prefill_s')} decode_s={stats.get('decode_s')} decode_tok_s={stats.get('decode_tok_s')}")
    print(f"target total wall: {target_total_s:.2f}s")
    ttft = drafter_s + float(stats.get('prefill_s', '0'))
    print(f"end-to-end TTFT (drafter+target prefill): {ttft:.2f}s")
    print(f"generated {len(gen_ids)} tokens:")
    print("  " + repr(gen_text)[:200])
    print()
    print(f"NIAH RESULT: {'PASS' if found else 'FAIL'} (needle '{NEEDLE_ANSWER_KEY}' {'found' if found else 'NOT found'})")
    sys.exit(0 if found else 1)

if __name__ == "__main__":
    main()
