"""NIAH bench for the Strix Halo path: prompt compression + llama.cpp CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pflash import LlamaCliRunner, PromptCompressor, detect_accelerator


def _is_correct(answer: str, output: str) -> bool:
    return answer.strip() in output


def _fmt_s(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--llama-cli", required=True,
                    help="path to llama.cpp's llama-cli built with GGML_HIP=ON")
    ap.add_argument("--model", required=True, help="target GGUF for llama.cpp")
    ap.add_argument("--drafter-model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--drafter-tokenizer", default=None)
    ap.add_argument("--keep-ratio", type=float, default=0.05)
    ap.add_argument("--ctx-size", type=int, default=32768)
    ap.add_argument("--n-gen", type=int, default=64)
    ap.add_argument("--n-gpu-layers", default="all")
    ap.add_argument("--flash-attn", choices=["on", "off", "auto"], default="on")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--threads-batch", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--ubatch-size", type=int, default=512)
    ap.add_argument("--head-tokens", type=int, default=256)
    ap.add_argument("--tail-tokens", type=int, default=768)
    ap.add_argument("--chunk-tokens", type=int, default=384)
    ap.add_argument("--query-tail-tokens", type=int, default=512)
    ap.add_argument("--min-chunks", type=int, default=4)
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    accel = detect_accelerator()
    print(
        f"[init] backend={accel.backend} device={accel.torch_device} "
        f"name={accel.name or 'cpu'} gfx={accel.gfx_target or 'n/a'}",
        flush=True,
    )

    compressor = PromptCompressor(
        model_name=args.drafter_model,
        tokenizer_name=args.drafter_tokenizer,
        head_tokens=args.head_tokens,
        tail_tokens=args.tail_tokens,
        query_tail_tokens=args.query_tail_tokens,
        chunk_tokens=args.chunk_tokens,
        min_chunks=args.min_chunks,
    )
    runner = LlamaCliRunner(
        binary=args.llama_cli,
        model=args.model,
        ctx_size=args.ctx_size,
        n_predict=args.n_gen,
        n_gpu_layers=args.n_gpu_layers,
        flash_attn=args.flash_attn,
        threads=args.threads,
        threads_batch=args.threads_batch,
        batch_size=args.batch_size,
        ubatch_size=args.ubatch_size,
        log_disable=True,
    )

    with open(args.cases) as f:
        cases = [json.loads(line) for line in f][:args.n]

    baseline_correct = 0
    compressed_correct = 0
    baseline_ttft: list[float] = []
    baseline_wall: list[float] = []
    compressed_ttft: list[float] = []
    compressed_wall: list[float] = []
    compression_wall: list[float] = []
    compressed_e2e_wall: list[float] = []

    for idx, case in enumerate(cases):
        prompt = case["prompt"]
        print(f"[case {idx}] source chars={len(prompt)}", flush=True)

        if not args.skip_baseline:
            base = runner.generate(prompt)
            if base.returncode != 0:
                raise RuntimeError(f"baseline llama-cli failed:\n{base.stderr}")
            ok = _is_correct(case["answer"], base.stdout)
            baseline_correct += int(ok)
            if base.ttft_s is not None:
                baseline_ttft.append(base.ttft_s)
            baseline_wall.append(base.wall_s)
            print(
                f"[case {idx}] baseline ttft={_fmt_s(base.ttft_s)} wall={base.wall_s:.2f}s ok={ok}",
                flush=True,
            )

        t0 = time.perf_counter()
        compressed = compressor.compress(prompt, keep_ratio=args.keep_ratio)
        compress_s = time.perf_counter() - t0
        compression_wall.append(compress_s)
        print(
            f"[case {idx}] compressed tokens={compressed.compressed_tokens}/"
            f"{compressed.original_tokens} keep={compressed.keep_ratio:.3f} "
            f"ratio={compressed.compression_ratio:.2f}x compress={compress_s:.2f}s",
            flush=True,
        )

        result = runner.generate(compressed.compressed_prompt)
        if result.returncode != 0:
            raise RuntimeError(f"compressed llama-cli failed:\n{result.stderr}")
        ok = _is_correct(case["answer"], result.stdout)
        compressed_correct += int(ok)
        if result.ttft_s is not None:
            compressed_ttft.append(result.ttft_s)
        compressed_wall.append(result.wall_s)
        compressed_e2e = compress_s + result.wall_s
        compressed_e2e_wall.append(compressed_e2e)
        print(
            f"[case {idx}] compressed ttft={_fmt_s(result.ttft_s)} wall={result.wall_s:.2f}s "
            f"e2e={compressed_e2e:.2f}s ok={ok}",
            flush=True,
        )

    print("\nsummary:", flush=True)
    if baseline_ttft:
        mean_base = sum(baseline_ttft) / len(baseline_ttft)
        mean_base_wall = sum(baseline_wall) / len(baseline_wall)
        print(
            f"  baseline accuracy={baseline_correct}/{len(cases)} "
            f"mean_ttft={mean_base:.2f}s mean_wall={mean_base_wall:.2f}s",
            flush=True,
        )
    mean_comp_ttft = sum(compressed_ttft) / max(len(compressed_ttft), 1)
    mean_comp_wall = sum(compressed_wall) / max(len(compressed_wall), 1)
    mean_comp_compress = sum(compression_wall) / max(len(compression_wall), 1)
    mean_comp_e2e = sum(compressed_e2e_wall) / max(len(compressed_e2e_wall), 1)
    print(
        f"  compressed accuracy={compressed_correct}/{len(cases)} "
        f"mean_ttft={mean_comp_ttft:.2f}s mean_wall={mean_comp_wall:.2f}s "
        f"mean_compress={mean_comp_compress:.2f}s mean_e2e={mean_comp_e2e:.2f}s",
        flush=True,
    )
    if baseline_ttft and mean_comp_ttft > 0:
        mean_base = sum(baseline_ttft) / len(baseline_ttft)
        print(f"  ttft speedup={mean_base / mean_comp_ttft:.2f}x", flush=True)
    if baseline_wall and mean_comp_e2e > 0:
        mean_base_wall = sum(baseline_wall) / len(baseline_wall)
        print(f"  e2e wall speedup={mean_base_wall / mean_comp_e2e:.2f}x", flush=True)


if __name__ == "__main__":
    main()
