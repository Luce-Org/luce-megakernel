"""Pflash speculative-prefill helper for the dflash OpenAI servers.

The dflash daemon already exposes the C++/CUDA spec-prefill pipeline via its
stdin protocol: ``compress <ids.bin> <keep_x1000> <drafter.gguf>`` runs the
in-process Qwen3-0.6B drafter + FlashPrefill scoring (BSA), then emits the
compressed token-id stream. ``free drafter`` releases drafter weights + KV +
BSA scratch, and ``park`` / ``unpark`` cycle target/draft weights through VRAM.

This module wraps that protocol so server.py and server_tools.py can fold
``--prefill-*`` flags into the existing request flow without duplicating the
plumbing. The legacy daemon backend drafter and target use *different*
tokenizers (Qwen3-0.6B vs Qwen3.5/3.6-27B), so the pipeline is:

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
import time
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
    backend: str                                       # "daemon" | "mega" | "mega-native"
    threshold: int                                     # token threshold for "auto"
    keep_ratio: float                                  # 0.015..0.125
    drafter_gguf: Optional[Path]                       # drafter weights (Qwen3-0.6B BF16 GGUF)
    drafter_tokenizer_id: str                          # HF repo ID for drafter vocab
    no_park: bool                                      # use daemon compress-nopark protocol
    park_policy: str                                   # "full" | "draft" | "none"
    mega_path: Path                                    # lucebox-hub/megakernel directory
    mega_model: str                                    # Qwen3.5 megakernel model id
    mega_native_model: Optional[Path]                  # Qwen3.5-0.8B safetensors dir/file
    mega_max_seq_len: int                              # megakernel compression window
    mega_lookahead: int                                # captured FA Q tail length
    mega_chunk_size: int                               # selected span granularity
    mega_smooth_kernel: int                            # score smoothing kernel
    mega_reduction: str                                # "mean" | "max"

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
    ap.add_argument("--prefill-backend", choices=["daemon", "mega", "mega-native"],
                    default="daemon",
                    help="Compression backend. 'daemon' uses the Qwen3-0.6B GGUF "
                         "PFlash path; 'mega' uses the Python megakernel proof; "
                         "'mega-native' uses the in-daemon C++/CUDA Qwen3.5 PFlash.")
    ap.add_argument("--prefill-drafter", type=Path, default=None,
                    help="Path to the drafter Qwen3-0.6B BF16 GGUF used by "
                         "the daemon's compress command. Required when "
                         "--prefill-compression != off.")
    ap.add_argument("--prefill-drafter-tokenizer", default="Qwen/Qwen3-0.6B",
                    help="HF repo ID for the drafter tokenizer "
                         "(default Qwen/Qwen3-0.6B).")
    ap.add_argument("--prefill-no-park", action="store_true",
                    help="Use compress-nopark and skip explicit park/unpark. "
                         "Only use when target + draft + PFlash fit in VRAM.")
    ap.add_argument("--prefill-park-policy", choices=["full", "draft", "none"],
                    default="full",
                    help="Which daemon weights to park during native prefill "
                         "compression. 'draft' keeps the target resident while "
                         "freeing draft/speculative state for PFlash headroom.")
    default_mega_path = Path(__file__).resolve().parents[2] / "megakernel"
    ap.add_argument("--prefill-mega-path", type=Path, default=default_mega_path,
                    help="Path to lucebox-hub/megakernel for --prefill-backend=mega.")
    ap.add_argument("--prefill-mega-model", default="Qwen/Qwen3.5-0.8B",
                    help="HF model id for megakernel-native PFlash.")
    ap.add_argument("--prefill-mega-native-model", type=Path, default=None,
                    help="Path to the Qwen3.5-0.8B safetensors file or snapshot "
                         "directory used by --prefill-backend=mega-native.")
    ap.add_argument("--prefill-mega-max-seq-len", type=int, default=0,
                    help="Megakernel-native PFlash max compression window. "
                         "0 means use --max-ctx when available.")
    ap.add_argument("--prefill-mega-lookahead", type=int, default=8,
                    help="Megakernel-native PFlash Q-tail lookahead.")
    ap.add_argument("--prefill-mega-chunk-size", type=int, default=32,
                    help="Megakernel-native PFlash selected chunk size.")
    ap.add_argument("--prefill-mega-smooth-kernel", type=int, default=13,
                    help="Megakernel-native PFlash score smoothing kernel.")
    ap.add_argument("--prefill-mega-reduction", choices=["mean", "max"], default="mean",
                    help="Megakernel-native PFlash layer/head score reduction.")


def config_from_args(args) -> PrefillConfig:
    mega_max_seq_len = args.prefill_mega_max_seq_len
    if mega_max_seq_len == 0:
        mega_max_seq_len = int(getattr(args, "max_ctx", 2048))
    if (args.prefill_compression != "off"
            and args.prefill_backend == "daemon"
            and args.prefill_drafter is None):
        raise SystemExit(
            "--prefill-compression != off requires --prefill-drafter "
            "(path to Qwen3-0.6B BF16 GGUF used by the daemon's compress).")
    if (args.prefill_compression != "off"
            and args.prefill_backend == "daemon"
            and not args.prefill_drafter.is_file()):
        raise SystemExit(f"prefill drafter not found at {args.prefill_drafter}")
    if args.prefill_compression != "off" and args.prefill_backend == "mega":
        if not args.prefill_mega_path.is_dir():
            raise SystemExit(f"megakernel path not found at {args.prefill_mega_path}")
    if args.prefill_compression != "off" and args.prefill_backend == "mega-native":
        if args.prefill_mega_native_model is None:
            raise SystemExit(
                "--prefill-backend=mega-native requires --prefill-mega-native-model "
                "(Qwen3.5-0.8B safetensors file or snapshot directory).")
        if not args.prefill_mega_native_model.exists():
            raise SystemExit(
                f"native Mega PFlash model path not found at {args.prefill_mega_native_model}")
    if args.prefill_compression != "off" and args.prefill_backend in ("mega", "mega-native"):
        if mega_max_seq_len <= 0:
            raise SystemExit("--prefill-mega-max-seq-len must be positive, or 0 with --max-ctx")
        if args.prefill_mega_lookahead <= 0:
            raise SystemExit("--prefill-mega-lookahead must be positive")
        if args.prefill_mega_chunk_size <= 0:
            raise SystemExit("--prefill-mega-chunk-size must be positive")
        if args.prefill_mega_smooth_kernel <= 0:
            raise SystemExit("--prefill-mega-smooth-kernel must be positive")
    if not 0.0 < args.prefill_keep_ratio <= 1.0:
        raise SystemExit("--prefill-keep-ratio must be in (0.0, 1.0]")
    return PrefillConfig(
        mode=args.prefill_compression,
        backend=args.prefill_backend,
        threshold=args.prefill_threshold,
        keep_ratio=args.prefill_keep_ratio,
        drafter_gguf=args.prefill_drafter,
        drafter_tokenizer_id=args.prefill_drafter_tokenizer,
        no_park=args.prefill_no_park,
        park_policy=("none" if args.prefill_no_park else args.prefill_park_policy),
        mega_path=args.prefill_mega_path,
        mega_model=args.prefill_mega_model,
        mega_native_model=args.prefill_mega_native_model,
        mega_max_seq_len=mega_max_seq_len,
        mega_lookahead=args.prefill_mega_lookahead,
        mega_chunk_size=args.prefill_mega_chunk_size,
        mega_smooth_kernel=args.prefill_mega_smooth_kernel,
        mega_reduction=args.prefill_mega_reduction,
    )


# ─── compress dance ────────────────────────────────────────────────────

_MEGA_COMPRESSOR = None


def _park_for_prefill(cfg: PrefillConfig, daemon_stdin, r_pipe: int) -> tuple[bool, bool]:
    """Apply the configured compression parking policy.

    Returns (target_parked, draft_parked) for symmetric restore.
    """
    parked_target = False
    parked_draft = False
    if cfg.park_policy == "full":
        _send_and_ack(daemon_stdin, r_pipe, "park target\n")
        parked_target = True
        _send_and_ack(daemon_stdin, r_pipe, "park draft\n")
        parked_draft = True
    elif cfg.park_policy == "draft":
        _send_and_ack(daemon_stdin, r_pipe, "park draft\n")
        parked_draft = True
    return parked_target, parked_draft


def _unpark_after_prefill(daemon_stdin, r_pipe: int,
                          parked_target: bool, parked_draft: bool) -> None:
    if parked_target:
        _send_and_ack(daemon_stdin, r_pipe, "unpark target\n")
    if parked_draft:
        _send_and_ack(daemon_stdin, r_pipe, "unpark draft\n")


def _compress_text_via_mega(cfg: PrefillConfig, prompt_text: str) -> str:
    """Run Qwen3.5 megakernel-native PFlash in this Python process."""
    global _MEGA_COMPRESSOR
    if _MEGA_COMPRESSOR is None:
        import sys
        mega_path = str(cfg.mega_path)
        if mega_path not in sys.path:
            sys.path.insert(0, mega_path)
        from mega_pflash import MegaPFlashCompressor
        from model import Decoder

        decoder = Decoder(
            model_name=cfg.mega_model,
            verbose=False,
            max_seq_len=cfg.mega_max_seq_len,
            decode_blocks=0,
        )
        _MEGA_COMPRESSOR = MegaPFlashCompressor(
            decoder,
            keep_ratio=cfg.keep_ratio,
            lookahead=cfg.mega_lookahead,
            chunk_size=cfg.mega_chunk_size,
            smooth_kernel=cfg.mega_smooth_kernel,
            reduction=cfg.mega_reduction,
        )
    t0 = time.perf_counter()
    try:
        result = _MEGA_COMPRESSOR.compress(prompt_text)
    except ValueError as exc:
        if "exceeds max_seq_len" not in str(exc):
            raise
        print(
            f"[mega-pflash] skip compression: {exc}",
            flush=True,
        )
        return prompt_text
    dt = time.perf_counter() - t0
    print(
        f"[mega-pflash] compressed {result.input_tokens} -> {result.output_tokens} "
        f"tokens in {dt:.3f}s keep_ratio={result.keep_ratio:.3f}",
        flush=True,
    )
    return result.text

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
    the daemon has its target + draft restored and is ready for ``generate``.
    """
    if cfg.backend == "mega":
        return _compress_text_via_mega(cfg, prompt_text)

    # Native Mega PFlash uses the target Qwen3.5 tokenizer directly and runs
    # the Qwen3.5-0.8B weights in-process inside test_dflash.
    if cfg.backend == "mega-native":
        if drafter_tokenizer is None or cfg.mega_native_model is None:
            return prompt_text
        ids = drafter_tokenizer(prompt_text, return_tensors=None,
                                add_special_tokens=False)["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if len(ids) > cfg.mega_max_seq_len:
            print(
                f"[mega-pflash] skip native compression: {len(ids)} tokens exceeds "
                f"max_seq_len={cfg.mega_max_seq_len}",
                flush=True,
            )
            return prompt_text
        fd, path = tempfile.mkstemp(suffix=".bin")
        parked_target = False
        parked_draft = False
        compressed_ids: list[int] = []
        try:
            with os.fdopen(fd, "wb") as f:
                for t in ids:
                    f.write(struct.pack("<i", int(t)))
            parked_target, parked_draft = _park_for_prefill(
                cfg, daemon_stdin, r_pipe)
            keep_x1000 = int(round(cfg.keep_ratio * 1000))
            reserve_seq_len = len(ids)
            daemon_stdin.write(
                f"compress-mega {path} {keep_x1000} {cfg.mega_native_model} "
                f"{reserve_seq_len}\n".encode("utf-8"))
            daemon_stdin.flush()
            compressed_ids = _drain_until_sentinel(r_pipe)
        finally:
            try:
                _send_and_ack(daemon_stdin, r_pipe, "free mega-pflash\n")
            except Exception:
                pass
            if parked_target or parked_draft:
                try:
                    _unpark_after_prefill(
                        daemon_stdin, r_pipe, parked_target, parked_draft)
                except Exception:
                    pass
            try: os.unlink(path)
            except Exception: pass
        if not compressed_ids:
            print("[mega-pflash] native compression returned empty output; keeping prompt",
                  flush=True)
            return prompt_text
        return drafter_tokenizer.decode(compressed_ids, skip_special_tokens=True)

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

        # 3) Park according to policy. For draft-only/none we must use
        # compress-nopark below so test_dflash does not auto-park target.
        parked_target, parked_draft = _park_for_prefill(
            cfg, daemon_stdin, r_pipe)

        # 4) compress: drafter loads, FlashPrefill scoring, emit compressed ids, drafter held
        keep_x1000 = int(round(cfg.keep_ratio * 1000))
        cmd = "compress" if cfg.park_policy == "full" else "compress-nopark"
        daemon_stdin.write(
            f"{cmd} {path} {keep_x1000} {cfg.drafter_gguf}\n".encode("utf-8"))
        daemon_stdin.flush()
        compressed_ids = _drain_until_sentinel(r_pipe)

        # 5) free drafter weights + BSA scratch, then restore target + draft
        _send_and_ack(daemon_stdin, r_pipe, "free drafter\n")
        _unpark_after_prefill(daemon_stdin, r_pipe, parked_target, parked_draft)
    finally:
        try: os.unlink(path)
        except Exception: pass

    # 6) decode compressed drafter ids back to text for re-tokenisation by target
    if not compressed_ids:
        print("[pflash] daemon compression returned empty output; keeping prompt",
              flush=True)
        return prompt_text
    return drafter_tokenizer.decode(compressed_ids, skip_special_tokens=True)
