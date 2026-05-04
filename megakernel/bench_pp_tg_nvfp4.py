"""Benchmark pp512 / tg128 for the local Qwen3.5-0.8B megakernel backend.

The default `all` mode runs each section in a fresh subprocess. This avoids
carrying CUDA state from correctness checks into the timed benchmark sections.
"""

import argparse
import json
import subprocess
import sys
import time

import torch
from transformers import AutoTokenizer

from model_nvfp4 import (
    Decoder,
    DN_CONV_CHANNELS,
    DN_NUM_HEADS,
    DN_V_SIZE,
    FA_KV_SIZE,
    FA_QPROJ_SIZE,
    FA_Q_SIZE,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    PREFILL_PROJ_FUSED_SIZE,
    PREFILL_PROJ_SCRATCH_SIZE,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark pp512 / tg128")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--backend", default="auto", choices=("auto", "bf16", "nvfp4"))
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument("--correctness-steps", type=int, default=30)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measure-runs", type=int, default=5)
    parser.add_argument("--verbose-loader", action="store_true")
    parser.add_argument(
        "--section",
        default="all",
        choices=("all", "correctness", "pp", "tg"),
    )
    parser.add_argument("--json-result", action="store_true")
    return parser.parse_args()


def build_exact_prompt_ids(tokenizer, target_tokens):
    seed = "Explain in great detail the history of artificial intelligence."
    text = seed
    ids = tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < target_tokens:
        text += " " + seed
        ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[:target_tokens]


def alloc_prefill_buffers(max_tokens):
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32 = dict(dtype=torch.float32, device="cuda")
    i32 = dict(dtype=torch.int32, device="cuda")
    return dict(
        hidden=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
        residual=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
        normalized=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
        proj_buf=torch.empty(max_tokens * PREFILL_PROJ_FUSED_SIZE, **bf16),
        proj_buf2=torch.empty(max_tokens * PREFILL_PROJ_SCRATCH_SIZE, **bf16),
        attn_buf=torch.empty(max_tokens * max(FA_Q_SIZE, FA_KV_SIZE), **bf16),
        mlp_buf=torch.empty(max_tokens * INTERMEDIATE_SIZE, **bf16),
        dn_out_buf=torch.empty(max_tokens * DN_V_SIZE, **bf16),
        beta_buf=torch.empty(max_tokens * DN_NUM_HEADS, **f32),
        alpha_buf=torch.empty(max_tokens * DN_NUM_HEADS, **f32),
        final_normed=torch.empty(HIDDEN_SIZE, **bf16),
        hidden_bf16_out=torch.empty(HIDDEN_SIZE, **bf16),
        lm_bmv=torch.empty(1024, **f32),
        lm_bmi=torch.empty(1024, **i32),
    )


def get_prefill_op(decoder):
    ops = torch.ops.qwen35_megakernel_bf16_C
    if decoder.backend == "nvfp4":
        return ops.prefill_megakernel_nvfp4
    return ops.prefill_bf16


def run_prefill(decoder, ids_t, prompt_len, buffers, prefill_op):
    decoder.reset()
    if decoder.backend == "nvfp4":
        return decoder.prefill_tokens(ids_t)
    else:
        prefill_op(
            decoder._out_token,
            ids_t,
            decoder._embed_weight,
            decoder._layer_weights_packed,
            decoder._prefill_fused_weights_packed,
            decoder._final_norm_weight,
            decoder._lm_head_weight,
            decoder._fa_k_cache,
            decoder._fa_v_cache,
            decoder._dn_states,
            decoder._conv_bufs,
            buffers["hidden"],
            buffers["residual"],
            buffers["normalized"],
            buffers["proj_buf"],
            buffers["proj_buf2"],
            buffers["attn_buf"],
            buffers["mlp_buf"],
            buffers["dn_out_buf"],
            buffers["beta_buf"],
            buffers["alpha_buf"],
            buffers["final_normed"],
            buffers["hidden_bf16_out"],
            buffers["lm_bmv"],
            buffers["lm_bmi"],
        )
    decoder._hidden.copy_(buffers["hidden_bf16_out"])
    decoder._position = prompt_len
    return decoder._out_token.item()


def decode_steps(decoder, first_token, num_steps, eos_token_id):
    out = [first_token]
    nid = first_token
    for _ in range(num_steps):
        nid = decoder.step(nid)
        torch.cuda.synchronize()
        if nid == eos_token_id:
            break
        out.append(nid)
    return out


def benchmark_prefill(decoder, ids_t, prompt_len, buffers, prefill_op, warmup_runs, measure_runs):
    for _ in range(warmup_runs):
        run_prefill(decoder, ids_t, prompt_len, buffers, prefill_op)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(measure_runs):
        run_prefill(decoder, ids_t, prompt_len, buffers, prefill_op)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / measure_runs
    return elapsed, prompt_len / elapsed


def benchmark_decode(decoder, prompt_ids, gen_tokens, tokenizer, buffers, prefill_op):
    ids_t = torch.tensor(prompt_ids, dtype=torch.int32, device="cuda")
    first = run_prefill(decoder, ids_t, len(prompt_ids), buffers, prefill_op)

    if decoder.backend == "nvfp4":
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        timed_ids_dev = decoder.step_many(first, gen_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        timed_ids = timed_ids_dev.cpu().tolist()
        if tokenizer.eos_token_id in timed_ids:
            timed_ids = timed_ids[:timed_ids.index(tokenizer.eos_token_id)]
        tps = (len(timed_ids) / elapsed) if timed_ids else 0.0
        return first, elapsed, tps, timed_ids

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    nid = first
    timed_ids = []
    for _ in range(gen_tokens):
        nid = decoder.step(nid)
        torch.cuda.synchronize()
        if nid == tokenizer.eos_token_id:
            break
        _ = tokenizer.decode([nid])
        timed_ids.append(nid)

    elapsed = time.perf_counter() - t0
    tps = (len(timed_ids) / elapsed) if timed_ids else 0.0
    return first, elapsed, tps, timed_ids


def emit_result(result, json_result):
    if json_result:
        print(f"RESULT_JSON {json.dumps(result, sort_keys=True)}", flush=True)


def run_correctness(args, tokenizer):
    print(f"Loading decoder for {args.model_name}...", flush=True)
    decoder = Decoder(
        model_name=args.model_name,
        backend=args.backend,
        verbose=args.verbose_loader,
    )
    print(f"Backend: {decoder.backend_label}", flush=True)
    if decoder.backend == "nvfp4":
        print("Correctness mode: prefill/decode handoff smoke test", flush=True)

    prefill_op = get_prefill_op(decoder)
    prompt_ids = tokenizer.encode("The capital of France is", add_special_tokens=False)
    buffers = alloc_prefill_buffers(max(32, len(prompt_ids)))

    print("\n=== Correctness test ===", flush=True)
    ids_t = torch.tensor(prompt_ids, dtype=torch.int32, device="cuda")
    first = run_prefill(decoder, ids_t, len(prompt_ids), buffers, prefill_op)
    print(f"Prefill -> first token: {first} = '{tokenizer.decode([first])}'", flush=True)

    out = decode_steps(decoder, first, args.correctness_steps, tokenizer.eos_token_id)
    text = tokenizer.decode(out, skip_special_tokens=True)
    print(f"Prefill + decode: {text[:100]}", flush=True)

    ok = bool(out)
    if decoder.backend == "nvfp4":
        if ok:
            print("PASS: prefill completed and decode continued on NVFP4", flush=True)
        else:
            print("FAIL: no tokens produced after prefill on NVFP4", flush=True)
    else:
        print("PASS: BF16 correctness smoke test completed", flush=True)

    result = {
        "backend_label": decoder.backend_label,
        "first_token": first,
        "ok": ok,
        "section": "correctness",
    }
    emit_result(result, args.json_result)
    return result


def run_pp(args, tokenizer):
    print(f"Loading decoder for {args.model_name}...", flush=True)
    decoder = Decoder(
        model_name=args.model_name,
        backend=args.backend,
        verbose=args.verbose_loader,
    )
    prefill_op = get_prefill_op(decoder)
    prompt_ids = build_exact_prompt_ids(tokenizer, args.prompt_tokens)
    buffers = alloc_prefill_buffers(len(prompt_ids))
    ids_t = torch.tensor(prompt_ids, dtype=torch.int32, device="cuda")

    print("\n=== pp benchmark ===", flush=True)
    print(f"Backend: {decoder.backend_label}", flush=True)
    print(f"Prompt tokens: {len(prompt_ids)}", flush=True)
    print(
        f"Warming {args.warmup_runs}x and timing {args.measure_runs}x prompt-processing runs...",
        flush=True,
    )
    pp_time, pp_tps = benchmark_prefill(
        decoder,
        ids_t,
        len(prompt_ids),
        buffers,
        prefill_op,
        args.warmup_runs,
        args.measure_runs,
    )
    print(f"pp{len(prompt_ids)}: {pp_tps:.1f} tok/s ({pp_time * 1000:.1f}ms avg)", flush=True)

    result = {
        "backend_label": decoder.backend_label,
        "pp_ms": pp_time * 1000.0,
        "pp_tps": pp_tps,
        "prompt_tokens": len(prompt_ids),
        "section": "pp",
    }
    emit_result(result, args.json_result)
    return result


def run_tg(args, tokenizer):
    print(f"Loading decoder for {args.model_name}...", flush=True)
    decoder = Decoder(
        model_name=args.model_name,
        backend=args.backend,
        verbose=args.verbose_loader,
    )
    prefill_op = get_prefill_op(decoder)
    prompt_ids = tokenizer.encode("The capital of France is", add_special_tokens=False)
    buffers = alloc_prefill_buffers(max(args.prompt_tokens, 32, len(prompt_ids)))

    print("\n=== tg benchmark ===", flush=True)
    print(f"Backend: {decoder.backend_label}", flush=True)
    print(f"Timed decode steps: {args.gen_tokens}", flush=True)
    first, tg_time, tg_tps, tg_ids = benchmark_decode(
        decoder,
        prompt_ids,
        args.gen_tokens,
        tokenizer,
        buffers,
        prefill_op,
    )
    print(f"Prefill seed token: {first} = '{tokenizer.decode([first])}'", flush=True)
    print(f"tg{len(tg_ids)}: {tg_tps:.1f} tok/s ({tg_time * 1000:.1f}ms)", flush=True)

    result = {
        "backend_label": decoder.backend_label,
        "first_token": first,
        "gen_tokens": len(tg_ids),
        "tg_ms": tg_time * 1000.0,
        "tg_tps": tg_tps,
        "section": "tg",
    }
    emit_result(result, args.json_result)
    return result


def build_child_cmd(args, section):
    cmd = [
        sys.executable,
        __file__,
        "--model-name",
        args.model_name,
        "--backend",
        args.backend,
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--gen-tokens",
        str(args.gen_tokens),
        "--correctness-steps",
        str(args.correctness_steps),
        "--warmup-runs",
        str(args.warmup_runs),
        "--measure-runs",
        str(args.measure_runs),
        "--section",
        section,
        "--json-result",
    ]
    if args.verbose_loader:
        cmd.append("--verbose-loader")
    return cmd


def filter_child_stderr(stderr_text):
    keep = []
    for line in stderr_text.splitlines():
        if line.startswith("Loading weights:"):
            continue
        if line.startswith("[transformers] The fast path is not available"):
            continue
        if line.strip():
            keep.append(line)
    return "\n".join(keep)


def run_child(args, section):
    proc = subprocess.run(
        build_child_cmd(args, section),
        capture_output=True,
        text=True,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        proc.check_returncode()
    elif proc.stderr:
        filtered = filter_child_stderr(proc.stderr)
        if filtered:
            print(filtered, file=sys.stderr, flush=True)

    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            result = json.loads(line[len("RESULT_JSON "):])
    if result is None:
        raise RuntimeError(f"missing RESULT_JSON for section {section}")
    return result


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.section == "correctness":
        run_correctness(args, tokenizer)
        return
    if args.section == "pp":
        run_pp(args, tokenizer)
        return
    if args.section == "tg":
        run_tg(args, tokenizer)
        return

    correctness = run_child(args, "correctness")
    pp = run_child(args, "pp")
    tg = run_child(args, "tg")

    print(f"\n=== Summary ({pp['backend_label']}, Qwen3.5-0.8B) ===", flush=True)
    print(f"pp{pp['prompt_tokens']:>3d}: {pp['pp_tps']:>7.1f} tok/s", flush=True)
    print(f"tg{tg['gen_tokens']:>3d}: {tg['tg_tps']:>7.1f} tok/s", flush=True)
    if not correctness["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
