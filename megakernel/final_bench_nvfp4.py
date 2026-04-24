"""Final benchmark for Qwen3.5-0.8B on local CUDA backends.

This script compares:
1. Megakernel prefill + decode
2. Hugging Face eager baseline

On GB10, the megakernel path may use BF16 prefill plus NVFP4 decode.
The script is intentionally progress-heavy so long GPU runs do not look hung.
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    parser = argparse.ArgumentParser(description="Benchmark megakernel vs HF baseline")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--backend", default="auto", choices=("auto", "bf16", "nvfp4"))
    parser.add_argument("--prompt-tokens", type=int, default=520)
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument("--our-warmup-runs", type=int, default=3)
    parser.add_argument("--our-pp-runs", type=int, default=5)
    parser.add_argument("--hf-warmup-runs", type=int, default=2)
    parser.add_argument("--hf-pp-runs", type=int, default=3)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--verbose-loader", action="store_true")
    parser.add_argument("--prefill-mode", default="eager",
                        choices=("eager", "mega"),
                        help="'eager' = cuBLAS+launches prefill (prefill_bf16); "
                             "'mega' = single-dispatch persistent megakernel "
                             "(prefill_bf16_mega)")
    return parser.parse_args()


def build_exact_prompt_ids(tokenizer, target_tokens):
    seed = (
        "Explain in great detail the history of artificial intelligence, machine learning, "
        "deep learning, and neural networks."
    )
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


def run_prefill(decoder, ids_t, prompt_len, buffers, prefill_op, use_mega=False):
    decoder.reset()
    if decoder.backend == "nvfp4":
        return decoder.prefill_tokens(ids_t)
    elif use_mega:
        prefill_op(
            decoder._out_token,
            ids_t,
            decoder._embed_weight,
            decoder._layer_weights_packed,
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


def benchmark_megakernel(decoder, tokenizer, prompt_ids, args):
    use_mega = getattr(args, "prefill_mode", "eager") == "mega" and decoder.backend != "nvfp4"
    if use_mega:
        prefill_op = torch.ops.qwen35_megakernel_bf16_C.prefill_bf16_mega
    else:
        prefill_op = get_prefill_op(decoder)
    prompt_len = len(prompt_ids)
    if use_mega:
        padded_len = ((prompt_len + 31) // 32) * 32
        ids_t = torch.zeros(padded_len, dtype=torch.int32, device="cuda")
        ids_t[:prompt_len] = torch.tensor(prompt_ids, dtype=torch.int32, device="cuda")
        ids_t = ids_t[:prompt_len]  # pass actual length; kernel reads only 0..prompt_len
        buffers = alloc_prefill_buffers(padded_len)
    else:
        ids_t = torch.tensor(prompt_ids, dtype=torch.int32, device="cuda")
        buffers = alloc_prefill_buffers(prompt_len)

    print(f"Backend: {decoder.backend_label}", flush=True)
    print(
        f"Warming megakernel prefill {args.our_warmup_runs}x and timing {args.our_pp_runs}x...",
        flush=True,
    )

    for _ in range(args.our_warmup_runs):
        run_prefill(decoder, ids_t, prompt_len, buffers, prefill_op, use_mega=use_mega)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.our_pp_runs):
        run_prefill(decoder, ids_t, prompt_len, buffers, prefill_op, use_mega=use_mega)
        torch.cuda.synchronize()
    our_pp_ms = (time.perf_counter() - t0) / args.our_pp_runs * 1000.0
    our_pp_tps = prompt_len / our_pp_ms * 1000.0

    print(f"Running megakernel decode for {args.gen_tokens} timed steps...", flush=True)
    del decoder
    torch.cuda.empty_cache()
    print("Reloading decoder for megakernel decode benchmark...", flush=True)
    decoder = Decoder(
        model_name=args.model_name,
        backend=args.backend,
        verbose=args.verbose_loader,
    )
    first = run_prefill(decoder, ids_t, prompt_len, buffers, prefill_op, use_mega=use_mega)
    decoded_ids = [first]
    eos = tokenizer.eos_token_id

    if decoder.backend == "nvfp4":
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        timed_ids_dev = decoder.step_many(first, args.gen_tokens)
        torch.cuda.synchronize()
        our_tg_ms = (time.perf_counter() - t0) * 1000.0
        timed_ids = timed_ids_dev.cpu().tolist()
        if eos in timed_ids:
            timed_ids = timed_ids[:timed_ids.index(eos)]
    else:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        nid = first
        timed_ids = []
        for _ in range(args.gen_tokens):
            nid = decoder.step(nid)
            if nid == eos:
                break
            timed_ids.append(nid)
        torch.cuda.synchronize()
        our_tg_ms = (time.perf_counter() - t0) * 1000.0

    decoded_ids.extend(timed_ids)
    our_tg_tps = (len(timed_ids) / our_tg_ms * 1000.0) if timed_ids else 0.0
    our_text = tokenizer.decode(decoded_ids, skip_special_tokens=True)

    return {
        "pp_ms": our_pp_ms,
        "pp_tps": our_pp_tps,
        "tg_ms": our_tg_ms,
        "tg_tps": our_tg_tps,
        "tg_count": len(timed_ids),
        "text": our_text,
    }


def benchmark_hf(model_name, tokenizer, prompt_ids, args):
    print("\n=== PyTorch HuggingFace ===", flush=True)
    print("Loading HF baseline model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    input_ids = torch.tensor([prompt_ids], device="cuda")

    print(
        f"Warming HF forward {args.hf_warmup_runs}x and timing {args.hf_pp_runs}x...",
        flush=True,
    )
    with torch.inference_mode():
        for _ in range(args.hf_warmup_runs):
            _ = model(input_ids)
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.hf_pp_runs):
            _ = model(input_ids)
            torch.cuda.synchronize()
        pt_pp_ms = (time.perf_counter() - t0) / args.hf_pp_runs * 1000.0
        pt_pp_tps = len(prompt_ids) / pt_pp_ms * 1000.0

        print(f"Running HF decode for {args.gen_tokens} timed steps...", flush=True)
        out = model(input_ids, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1:].argmax(-1)
        decoded_ids = [next_id.item()]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        timed_ids = []
        current = next_id
        for _ in range(args.gen_tokens):
            out = model(current, past_key_values=past, use_cache=True)
            past = out.past_key_values
            current = out.logits[:, -1:].argmax(-1)
            token = current.item()
            if token == tokenizer.eos_token_id:
                break
            timed_ids.append(token)
        torch.cuda.synchronize()

    decoded_ids.extend(timed_ids)
    pt_tg_ms = (time.perf_counter() - t0) * 1000.0
    pt_tg_tps = (len(timed_ids) / pt_tg_ms * 1000.0) if timed_ids else 0.0
    pt_text = tokenizer.decode(decoded_ids, skip_special_tokens=True)

    del model, input_ids, past, out
    torch.cuda.empty_cache()

    return {
        "pp_ms": pt_pp_ms,
        "pp_tps": pt_pp_tps,
        "tg_ms": pt_tg_ms,
        "tg_tps": pt_tg_tps,
        "tg_count": len(timed_ids),
        "text": pt_text,
    }


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"Loading megakernel decoder for {args.model_name}...", flush=True)
    decoder = Decoder(
        model_name=args.model_name,
        backend=args.backend,
        verbose=args.verbose_loader,
    )

    prompt_ids = build_exact_prompt_ids(tokenizer, args.prompt_tokens)
    print(f"Prompt: {len(prompt_ids)} tokens", flush=True)
    backend_label = decoder.backend_label

    print("\n=== Our Megakernel ===", flush=True)
    our = benchmark_megakernel(decoder, tokenizer, prompt_ids, args)

    print(
        f"pp{len(prompt_ids)}: {our['pp_tps']:.0f} tok/s ({our['pp_ms']:.1f}ms avg)",
        flush=True,
    )
    print(
        f"tg{our['tg_count']}: {our['tg_tps']:.0f} tok/s ({our['tg_ms']:.1f}ms total)",
        flush=True,
    )
    print(f"Completion: {our['text'][:120]}", flush=True)

    del decoder
    torch.cuda.empty_cache()

    hf = None
    if not args.skip_hf:
        try:
            hf = benchmark_hf(args.model_name, tokenizer, prompt_ids, args)
            print(
                f"pp{len(prompt_ids)}: {hf['pp_tps']:.0f} tok/s ({hf['pp_ms']:.1f}ms avg)",
                flush=True,
            )
            print(
                f"tg{hf['tg_count']}: {hf['tg_tps']:.0f} tok/s ({hf['tg_ms']:.1f}ms total)",
                flush=True,
            )
            print(f"Completion: {hf['text'][:120]}", flush=True)
        except Exception as exc:
            hf = {"error": str(exc)}
            print(f"HF baseline failed: {exc}", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"FINAL RESULTS — Qwen3.5-0.8B, {backend_label}", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"{'Method':<25} {'pp'+str(len(prompt_ids)):>12} {'tg':>16}", flush=True)
    print(f"{'-' * 55}", flush=True)
    print(
        f"{'Our megakernel':<25} {our['pp_tps']:>9.0f} t/s {our['tg_tps']:>9.0f} t/s",
        flush=True,
    )
    if hf is None:
        print(f"{'PyTorch HF':<25} {'(skipped)':>24}", flush=True)
    elif "error" in hf:
        print(f"{'PyTorch HF':<25} {'(failed)':>24}", flush=True)
    else:
        print(
            f"{'PyTorch HF':<25} {hf['pp_tps']:>9.0f} t/s {hf['tg_tps']:>9.0f} t/s",
            flush=True,
        )
    print(f"{'llama.cpp BF16':<25} {'(run separately)':>24}", flush=True)

    if hf is not None and "error" not in hf:
        print("", flush=True)
        print(
            f"Megakernel vs PyTorch:  pp {our['pp_tps'] / hf['pp_tps']:.1f}x  "
            f"tg {our['tg_tps'] / hf['tg_tps']:.1f}x",
            flush=True,
        )
        print("", flush=True)
        print("=== Completions ===", flush=True)
        print(f"Ours:    {our['text'][:100]}", flush=True)
        print(f"PyTorch: {hf['text'][:100]}", flush=True)


if __name__ == "__main__":
    main()
