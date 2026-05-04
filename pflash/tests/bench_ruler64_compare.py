"""Compare null, approximate, and exact pflash paths on NIAH/RULER-style cases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pflash import LlamaCliRunner, PromptCompressor
from pflash.dflash_client import DflashClient


def _contains_term(output: str, term: str) -> bool:
    return term.lower() in output.lower()


def _score_case(case: dict, output: str) -> float:
    answer = case["answer"]
    if isinstance(answer, list):
        if not answer:
            return 1.0
        expected = [str(item).strip() for item in answer]
        forbidden = [str(item).strip() for item in case.get("forbidden", [])]
        tp = sum(1 for item in expected if _contains_term(output, item))
        fp = sum(1 for item in forbidden if _contains_term(output, item))
        if forbidden:
            precision = tp / max(tp + fp, 1)
            recall = tp / len(expected)
            return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return tp / len(expected)
    return float(str(answer).strip() in output)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _load_done(path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((row["mode"], int(row["case"])))
    return done


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _summarize(path: Path) -> str:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    modes = ["null", "approx", "exact"]
    lines = [
        "| mode | n | avg score | avg total s | avg compress s | avg target s | avg kept/source | avg compression x |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in modes:
        ms = [r for r in rows if r["mode"] == mode]
        if not ms:
            continue
        lines.append(
            "| {mode} | {n} | {score:.1f}% | {total:.2f} | {comp:.2f} | {target:.2f} | {keep:.4f} | {ratio:.2f} |".format(
                mode=mode,
                n=len(ms),
                score=100.0 * _mean([float(r["score"]) for r in ms]),
                total=_mean([float(r["total_s"]) for r in ms]),
                comp=_mean([float(r.get("compress_s", 0.0)) for r in ms]),
                target=_mean([float(r.get("target_s", 0.0)) for r in ms]),
                keep=_mean([float(r.get("keep_ratio", 1.0)) for r in ms]),
                ratio=_mean([float(r.get("compression_ratio", 1.0)) for r in ms]),
            )
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="null,approx,exact")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--llama-cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--exact-bin", required=True)
    ap.add_argument("--draft-spec", required=True)
    ap.add_argument("--drafter-gguf", required=True)
    ap.add_argument("--drafter-arch", default="qwen3-0.6b",
                    choices=["qwen3-0.6b", "qwen35-0.8b"],
                    help="exact C++ drafter architecture selector")
    ap.add_argument("--target-tokenizer", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--drafter-tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--approx-model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--keep-ratio", type=float, default=0.10)
    ap.add_argument("--ctx-size", type=int, default=65536)
    ap.add_argument("--exact-max-ctx", type=int, default=16384)
    ap.add_argument("--n-gen", type=int, default=64)
    args = ap.parse_args()

    cases = [json.loads(line) for line in Path(args.cases).read_text().splitlines() if line.strip()][: args.n]
    out_path = Path(args.out)
    done = _load_done(out_path)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    runner = None
    compressor = None
    dflash = None
    target_tok = None
    drafter_tok = None

    try:
        if "null" in modes or "approx" in modes or "exact" in modes:
            runner = LlamaCliRunner(
                args.llama_cli,
                args.model,
                ctx_size=args.ctx_size,
                n_predict=args.n_gen,
                n_gpu_layers="all",
                flash_attn="on",
                batch_size=2048,
                ubatch_size=512,
                log_disable=True,
                extra_args=["-no-cnv", "--reasoning", "off", "--no-show-timings"],
            )

        if "approx" in modes:
            compressor = PromptCompressor(
                model_name=args.approx_model,
                tokenizer_name=args.drafter_tokenizer,
                query_tail_tokens=512,
                chunk_tokens=32,
                tail_keep_chunks=1,
                anchor_context_chunks=1,
            )

        if "exact" in modes:
            from transformers import AutoTokenizer

            target_tok = AutoTokenizer.from_pretrained(args.target_tokenizer)
            drafter_tok = AutoTokenizer.from_pretrained(args.drafter_tokenizer)
            dflash = DflashClient(args.exact_bin, args.model, args.draft_spec, max_ctx=args.exact_max_ctx)

        for i, case in enumerate(cases):
            prompt = case["prompt"]
            answer = case["answer"]

            if "null" in modes and ("null", i) not in done:
                assert runner is not None
                res = runner.generate(prompt)
                if res.returncode != 0:
                    raise RuntimeError(f"null llama-cli failed on case {i}:\n{res.stderr}")
                row = {
                    "mode": "null",
                    "case": i,
                    "score": _score_case(case, res.stdout),
                    "source_tokens": case.get("n_tokens"),
                    "kept_tokens": case.get("n_tokens"),
                    "keep_ratio": 1.0,
                    "compression_ratio": 1.0,
                    "compress_s": 0.0,
                    "target_s": res.wall_s,
                    "total_s": res.wall_s,
                    "output": res.stdout,
                    "answer": answer,
                }
                _append(out_path, row)
                print(f"[null {i}] score={row['score']} total={row['total_s']:.2f}s", flush=True)

            if "approx" in modes and ("approx", i) not in done:
                assert runner is not None and compressor is not None
                t0 = time.perf_counter()
                comp = compressor.compress(prompt, keep_ratio=args.keep_ratio)
                comp_s = time.perf_counter() - t0
                res = runner.generate(comp.compressed_prompt)
                if res.returncode != 0:
                    raise RuntimeError(f"approx llama-cli failed on case {i}:\n{res.stderr}")
                row = {
                    "mode": "approx",
                    "case": i,
                    "score": _score_case(case, res.stdout),
                    "source_tokens": comp.original_tokens,
                    "kept_tokens": comp.compressed_tokens,
                    "keep_ratio": comp.keep_ratio,
                    "compression_ratio": comp.compression_ratio,
                    "compress_s": comp_s,
                    "target_s": res.wall_s,
                    "total_s": comp_s + res.wall_s,
                    "output": res.stdout,
                    "answer": answer,
                }
                _append(out_path, row)
                print(f"[approx {i}] score={row['score']} keep={row['keep_ratio']:.3f} total={row['total_s']:.2f}s", flush=True)

            if "exact" in modes and ("exact", i) not in done:
                assert dflash is not None and target_tok is not None and drafter_tok is not None and runner is not None
                ids = drafter_tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
                t0 = time.perf_counter()
                compressed_ids = dflash.compress(ids, args.keep_ratio, args.drafter_gguf, args.drafter_arch)
                comp_s = time.perf_counter() - t0
                comp_text = drafter_tok.decode(compressed_ids, skip_special_tokens=True)
                dflash.free_drafter()
                dflash.park_target()
                t1 = time.perf_counter()
                exact_prompt = comp_text + "\n\nAnswer the question above. Give only the answer. /no_think\nAnswer:"
                res = runner.generate(exact_prompt)
                gen_s = time.perf_counter() - t1
                if res.returncode != 0:
                    raise RuntimeError(f"exact llama-cli target failed on case {i}:\n{res.stderr}")
                out_text = res.stdout
                row = {
                    "mode": "exact",
                    "case": i,
                    "score": _score_case(case, out_text),
                    "source_tokens": len(ids),
                    "kept_tokens": len(compressed_ids),
                    "keep_ratio": len(compressed_ids) / max(len(ids), 1),
                    "compression_ratio": len(ids) / max(len(compressed_ids), 1),
                    "compress_s": comp_s,
                    "target_s": gen_s,
                    "total_s": comp_s + gen_s,
                    "output": out_text,
                    "answer": answer,
                }
                _append(out_path, row)
                print(f"[exact {i}] score={row['score']} keep={row['keep_ratio']:.3f} total={row['total_s']:.2f}s", flush=True)
    finally:
        if dflash is not None:
            dflash.close()

    print("\n" + _summarize(out_path), flush=True)


if __name__ == "__main__":
    main()
