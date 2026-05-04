"""Generate with Qwen3.5-0.8B megakernel plus no-park PFlash compression."""

from __future__ import annotations

import argparse
from pathlib import Path

from model import Decoder
from pflash import PFlashCompressor

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--pflash-bin", type=Path,
                    default=ROOT / "dflash/build-luce-sm120/pflash_daemon")
    ap.add_argument("--pflash-drafter", type=Path,
                    default=Path("/home/jake/models/Qwen3-0.6B-GGUF/Qwen3-0.6B-BF16.gguf"))
    ap.add_argument("--pflash-threshold", type=int, default=1024)
    ap.add_argument("--keep-ratio", type=float, default=0.05)
    args = ap.parse_args()

    dec = Decoder(max_seq_len=args.max_seq_len)
    with PFlashCompressor(
        daemon_bin=args.pflash_bin,
        drafter_gguf=args.pflash_drafter,
        keep_ratio=args.keep_ratio,
    ) as pflash:
        print(dec.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            pflash=pflash,
            pflash_threshold=args.pflash_threshold,
        ))


if __name__ == "__main__":
    main()
