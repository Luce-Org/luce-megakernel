"""Pflash speculative-prefill helper for the dflash OpenAI servers.

The dflash daemon already exposes the C++/CUDA spec-prefill pipeline via its
stdin protocol: ``compress <ids.bin> <keep_x1000> <drafter.gguf>`` runs the
in-process Qwen3-0.6B drafter + FlashPrefill scoring (BSA), then emits the
compressed token-id stream. ``free drafter`` releases drafter weights + KV +
BSA scratch, and ``park`` / ``unpark`` cycle target/draft weights through VRAM.

This module wraps that protocol so server.py and server_tools.py can fold
``--prefill-*`` flags into the existing request flow without duplicating the
plumbing. The drafter and target use *different* tokenizers (Qwen3-0.6B vs
Qwen3.5/3.6-27B), so the pipeline is:

    target_text  ──▶  drafter_tokenizer.encode  ──▶  daemon.compress
                                                          │
    target_tokenizer.encode  ◀──  drafter_tokenizer.decode ┘

The result is a shorter span of text (re-tokenised on the target side) that
the existing ``generate`` path can prefill in a fraction of the time.
"""
from __future__ import annotations
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ─── pipe / stdin helpers shared with both servers ─────────────────────

def _drain_until_sentinel(r_pipe: int) -> list[int]:
    """Read int32 LE values from r_pipe until -1 sentinel. Returns the list."""
    out: list[int] = []
    while True:
        b = os.read(r_pipe, 4)
        if not b or len(b) < 4:
            break
        v = struct.unpack("<i", b)[0]
        if v == -1:
            break
        out.append(v)
    return out


def _send_and_ack(daemon_stdin, r_pipe: int, line: str) -> None:
    """Write a daemon command and consume the trailing -1 ack."""
    daemon_stdin.write(line.encode("utf-8"))
    daemon_stdin.flush()
    _drain_until_sentinel(r_pipe)


# ─── public configuration block ────────────────────────────────────────

@dataclass(frozen=True)
class PrefillConfig:
    """Parsed --prefill-* flags. ``mode == "off"`` disables compression."""
    mode: str                                          # "off" | "auto" | "always"
    threshold: int                                     # token threshold for "auto"
    keep_ratio: float                                  # 0.015..0.125
    drafter_gguf: Optional[Path]                       # drafter weights (Qwen3-0.6B BF16 GGUF)
    drafter_tokenizer_id: str                          # HF repo ID for drafter vocab

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def should_compress(self, prompt_token_count: int) -> bool:
        if self.mode == "always":
            return True
        if self.mode == "auto":
            return prompt_token_count >= self.threshold
        return False


def add_cli_flags(ap) -> None:
    """Attach --prefill-* flags to an argparse.ArgumentParser."""
    ap.add_argument("--prefill-compression",
                    choices=["off", "auto", "always"], default="off",
                    help="Speculative-prefill mode. 'auto' compresses when the "
                         "prompt token count reaches --prefill-threshold; "
                         "'always' compresses every request.")
    ap.add_argument("--prefill-threshold", type=int, default=32000,
                    help="Token threshold above which 'auto' mode triggers "
                         "compression (default 32000).")
    ap.add_argument("--prefill-keep-ratio", type=float, default=0.05,
                    help="Fraction of source tokens to keep after compression "
                         "(default 0.05; bench setting).")
    ap.add_argument("--prefill-drafter", type=Path, default=None,
                    help="Path to the drafter Qwen3-0.6B BF16 GGUF used by "
                         "the daemon's compress command. Required when "
                         "--prefill-compression != off.")
    ap.add_argument("--prefill-drafter-tokenizer", default="Qwen/Qwen3-0.6B",
                    help="HF repo ID for the drafter tokenizer "
                         "(default Qwen/Qwen3-0.6B).")


def config_from_args(args) -> PrefillConfig:
    if args.prefill_compression != "off" and args.prefill_drafter is None:
        raise SystemExit(
            "--prefill-compression != off requires --prefill-drafter "
            "(path to Qwen3-0.6B BF16 GGUF used by the daemon's compress).")
    if args.prefill_compression != "off" and not args.prefill_drafter.is_file():
        raise SystemExit(f"prefill drafter not found at {args.prefill_drafter}")
    if not 0.0 < args.prefill_keep_ratio <= 1.0:
        raise SystemExit("--prefill-keep-ratio must be in (0.0, 1.0]")
    return PrefillConfig(
        mode=args.prefill_compression,
        threshold=args.prefill_threshold,
        keep_ratio=args.prefill_keep_ratio,
        drafter_gguf=args.prefill_drafter,
        drafter_tokenizer_id=args.prefill_drafter_tokenizer,
    )


# ─── compress dance ────────────────────────────────────────────────────

def compress_text_via_daemon(
    *,
    daemon_stdin,
    r_pipe: int,
    drafter_tokenizer,
    cfg: PrefillConfig,
    prompt_text: str,
) -> str:
    """Run the daemon's compress + memory dance, return the compressed text.

    Caller holds the daemon lock for the full duration. After this returns,
    the daemon has its target restored and is ready for ``generate``. If
    ``DFLASH_PFLASH_SKIP_DRAFT_RELOAD=1`` is set, the draft remains parked and
    the daemon must use its target-only decode fallback for that request.
    """
    # 1) drafter-tokenize the prompt
    drafter_ids = drafter_tokenizer(prompt_text, return_tensors=None,
                                    add_special_tokens=False)["input_ids"]
    if isinstance(drafter_ids[0], list):  # some tokenizers return [[...]]
        drafter_ids = drafter_ids[0]

    # 2) write drafter ids to a tempfile
    fd, path = tempfile.mkstemp(suffix=".bin")
    try:
        with os.fdopen(fd, "wb") as f:
            for t in drafter_ids:
                f.write(struct.pack("<i", int(t)))

        # 3) park target + draft so drafter has VRAM headroom on a 24 GB card.
        # Experimental SM75 path: keep target resident and only park the DFlash
        # draft. This avoids reloading the 27B target after compression, but can
        # OOM on tighter contexts, so it is opt-in via env.
        keep_target = os.environ.get("DFLASH_PFLASH_KEEP_TARGET", "0") not in ("", "0")
        if not keep_target:
            _send_and_ack(daemon_stdin, r_pipe, "park target\n")
        _send_and_ack(daemon_stdin, r_pipe, "park draft\n")

        # 4) compress: drafter loads, FlashPrefill scoring, emit compressed ids, drafter held
        keep_x1000 = int(round(cfg.keep_ratio * 1000))
        daemon_stdin.write(
            f"compress {path} {keep_x1000} {cfg.drafter_gguf}\n".encode("utf-8"))
        daemon_stdin.flush()
        compressed_ids = _drain_until_sentinel(r_pipe)

        # 5) free drafter weights + BSA scratch, then restore target + draft.
        # For short outputs, reloading the DFlash draft can dominate the whole
        # PFlash path; the matching daemon env falls back to target-only decode.
        skip_draft_reload = os.environ.get("DFLASH_PFLASH_SKIP_DRAFT_RELOAD", "0") not in ("", "0")
        _send_and_ack(daemon_stdin, r_pipe, "free drafter\n")
        if not keep_target:
            _send_and_ack(daemon_stdin, r_pipe, "unpark target\n")
        if not skip_draft_reload:
            _send_and_ack(daemon_stdin, r_pipe, "unpark draft\n")
    finally:
        try: os.unlink(path)
        except Exception: pass

    # 6) decode compressed drafter ids back to text for re-tokenisation by target
    return drafter_tokenizer.decode(compressed_ids, skip_special_tokens=True)
