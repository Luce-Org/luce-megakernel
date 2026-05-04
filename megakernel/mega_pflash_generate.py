"""Generate with Qwen3.5 megakernel-native PFlash compression."""

from __future__ import annotations

import argparse

from mega_pflash import MegaPFlashCompressor
from model import Decoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--pflash-threshold", type=int, default=1024)
    ap.add_argument("--keep-ratio", type=float, default=0.05)
    ap.add_argument("--lookahead", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--smooth-kernel", type=int, default=13)
    ap.add_argument("--reduction", choices=("mean", "max"), default="mean")
    ap.add_argument("--no-preserve-edges", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    dec = Decoder(max_seq_len=args.max_seq_len)
    pflash = MegaPFlashCompressor(
        dec,
        keep_ratio=args.keep_ratio,
        lookahead=args.lookahead,
        chunk_size=args.chunk_size,
        smooth_kernel=args.smooth_kernel,
        preserve_edges=not args.no_preserve_edges,
        reduction=args.reduction,
    )

    if args.stats:
        probe_ids = dec.tokenizer.encode(args.prompt, add_special_tokens=True)
        if len(probe_ids) >= args.pflash_threshold:
            result = pflash.compress(args.prompt)
            print(
                f"[mega-pflash] tokens {result.input_tokens} -> {result.output_tokens} "
                f"keep_ratio={result.keep_ratio:.3f} chunk_size={result.chunk_size}")
            prompt = result.text
            print(dec.generate(prompt, max_tokens=args.max_tokens))
            return

    print(dec.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        pflash=pflash,
        pflash_threshold=args.pflash_threshold,
    ))


if __name__ == "__main__":
    main()
