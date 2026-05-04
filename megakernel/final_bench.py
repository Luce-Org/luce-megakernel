"""Final benchmark: pp520 tg128 — Our megakernel vs PyTorch naive.
Both properly warmed. Saves completions for verification.

Supports --backend {auto,bf16,nvfp4}. Default is auto: Blackwell (sm_12+)
dispatches to final_bench_nvfp4.py; everything else runs the bf16 path
below unchanged from upstream.
"""
import argparse as _argparse, os as _os, sys as _sys
import torch as _torch

_p = _argparse.ArgumentParser(add_help=False)
_p.add_argument("--backend", default="auto", choices=("auto", "bf16", "nvfp4"))
_a, _rest = _p.parse_known_args()
_backend = _a.backend
if _backend == "auto":
    _backend = "nvfp4" if (_torch.cuda.is_available() and _torch.cuda.get_device_capability()[0] >= 12) else "bf16"
if _backend == "nvfp4":
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _os.execv(_sys.executable, [_sys.executable, _os.path.join(_here, "final_bench_nvfp4.py"), *_rest])

import time, torch
import _phase2_variant  # noqa: F401 — prints "[megakernel] DN phase2 variant = scalar|wmma"
from model import Decoder, HIDDEN_SIZE, INTERMEDIATE_SIZE, FA_QPROJ_SIZE, FA_Q_SIZE, FA_KV_SIZE
from model import DN_CONV_CHANNELS, DN_V_SIZE, DN_NUM_HEADS, MAX_SEQ_LEN, _half_dtype
import qwen35_megakernel_bf16_C
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

# ============================================================
# Build a 520-token prompt
# ============================================================
long_text = "Explain in great detail the history of artificial intelligence, machine learning, deep learning, and neural networks. " * 40
prompt_ids = tok.encode(long_text, add_special_tokens=False)[:520]
print(f"Prompt: {len(prompt_ids)} tokens")

# ============================================================
# 1. Our megakernel (prefill cuBLAS + decode megakernel)
# ============================================================
_dtype_label = "FP16" if _half_dtype() == torch.float16 else "BF16"
_gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown GPU"

print(f"\n=== Our {_dtype_label} Megakernel ===")
dec = Decoder(verbose=False)
_pf = torch.ops.qwen35_megakernel_bf16_C.prefill_bf16

S = 520
bf16 = dict(dtype=_half_dtype(), device="cuda")
f32 = dict(dtype=torch.float32, device="cuda")
i32 = dict(dtype=torch.int32, device="cuda")
mx = max(DN_CONV_CHANNELS, FA_QPROJ_SIZE, INTERMEDIATE_SIZE)
b = dict(
    hidden=torch.empty(S*HIDDEN_SIZE, **bf16), residual=torch.empty(S*HIDDEN_SIZE, **bf16),
    normalized=torch.empty(S*HIDDEN_SIZE, **bf16),
    proj_buf=torch.empty(S*mx, **bf16), proj_buf2=torch.empty(S*mx, **bf16),
    attn_buf=torch.empty(S*max(FA_Q_SIZE, FA_KV_SIZE), **bf16),
    mlp_buf=torch.empty(S*INTERMEDIATE_SIZE, **bf16),
    dn_out_buf=torch.empty(S*DN_V_SIZE, **bf16),
    beta_buf=torch.empty(S*DN_NUM_HEADS, **f32), alpha_buf=torch.empty(S*DN_NUM_HEADS, **f32),
    final_normed=torch.empty(HIDDEN_SIZE, **bf16), hidden_bf16_out=torch.empty(HIDDEN_SIZE, **bf16),
    lm_bmv=torch.empty(1024, **f32), lm_bmi=torch.empty(1024, **i32),
)
b.update(dec.alloc_prefill_scratch(S))
ids_t = torch.tensor(prompt_ids, dtype=torch.int32, device="cuda")

def our_prefill():
    dec.reset()
    _pf(dec._out_token, ids_t, dec._embed_weight, dec._layer_weights_packed,
        dec._final_norm_weight, dec._lm_head_weight,
        dec._fa_k_cache, dec._fa_v_cache, dec._dn_states, dec._conv_bufs,
        b['hidden'], b['residual'], b['normalized'],
        b['proj_buf'], b['proj_buf2'], b['attn_buf'], b['mlp_buf'],
        b['dn_out_buf'], b['beta_buf'], b['alpha_buf'],
        b['dn_pre_qkv'],
        b['dn_u_scratch'], b['dn_w_scratch'], b['dn_cs_scratch'],
        dec._fused_fa_qkv, dec._fused_gate_up,
        b['final_normed'], b['hidden_bf16_out'], b['lm_bmv'], b['lm_bmi'],
        dec.max_seq_len)
    dec._hidden.copy_(b['hidden_bf16_out'])
    dec._position = len(prompt_ids)
    return dec._out_token.item()

# Warmup 10x
for _ in range(10): our_prefill(); torch.cuda.synchronize()

# PP benchmark (20 runs)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(20): our_prefill(); torch.cuda.synchronize()
our_pp_ms = (time.perf_counter() - t0) / 20 * 1000
our_pp_tps = len(prompt_ids) / our_pp_ms * 1000

# TG benchmark + completion
first = our_prefill()
torch.cuda.synchronize()
out_ids = [first]; nid = first
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(128):
    nid = dec.step(nid)
    if nid == tok.eos_token_id: break
    out_ids.append(nid)
torch.cuda.synchronize()
our_tg_ms = (time.perf_counter() - t0) * 1000
our_tg_tps = len(out_ids) / our_tg_ms * 1000
our_text = tok.decode(out_ids, skip_special_tokens=True)

print(f"pp{len(prompt_ids)}: {our_pp_tps:.0f} tok/s ({our_pp_ms:.1f}ms)")
print(f"tg{len(out_ids)}: {our_tg_tps:.0f} tok/s ({our_tg_ms:.1f}ms)")
print(f"Completion: {our_text[:120]}")

del dec
torch.cuda.empty_cache()

# ============================================================
# 2. PyTorch naive (HuggingFace)
# ============================================================
print("\n=== PyTorch HuggingFace ===")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=_half_dtype(), device_map="cuda")
model.eval()
input_ids = torch.tensor([prompt_ids], device="cuda")

# Warmup 5x
with torch.no_grad():
    for _ in range(5): _ = model(input_ids); torch.cuda.synchronize()

# PP benchmark (10 runs)
with torch.no_grad():
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(10): _ = model(input_ids); torch.cuda.synchronize()
    pt_pp_ms = (time.perf_counter() - t0) / 10 * 1000
    pt_pp_tps = len(prompt_ids) / pt_pp_ms * 1000

# TG benchmark + completion
with torch.no_grad():
    out = model(input_ids, use_cache=True)
    past = out.past_key_values
    next_id = out.logits[:, -1:].argmax(-1)
    pt_out_ids = [next_id.item()]

    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(128):
        out = model(next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1:].argmax(-1)
        if next_id.item() == tok.eos_token_id: break
        pt_out_ids.append(next_id.item())
    torch.cuda.synchronize()
    pt_tg_ms = (time.perf_counter() - t0) * 1000
    pt_tg_tps = len(pt_out_ids) / pt_tg_ms * 1000
    pt_text = tok.decode(pt_out_ids, skip_special_tokens=True)

print(f"pp{len(prompt_ids)}: {pt_pp_tps:.0f} tok/s ({pt_pp_ms:.1f}ms)")
print(f"tg{len(pt_out_ids)}: {pt_tg_tps:.0f} tok/s ({pt_tg_ms:.1f}ms)")
print(f"Completion: {pt_text[:120]}")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"FINAL RESULTS — Qwen3.5-0.8B {_dtype_label}, {_gpu_name}")
print(f"{'='*60}")
print(f"{'Method':<25} {'pp'+str(len(prompt_ids)):>8} {'tg128':>10}")
print(f"{'-'*45}")
print(f"{'Our megakernel':<25} {our_pp_tps:>7.0f} t/s {our_tg_tps:>8.0f} t/s")
print(f"{'PyTorch HF':<25} {pt_pp_tps:>7.0f} t/s {pt_tg_tps:>8.0f} t/s")
print(f"{'llama.cpp ' + _dtype_label:<25} {'(run separately)':>19}")
print(f"")
print(f"Megakernel vs PyTorch:  pp {our_pp_tps/pt_pp_tps:.1f}x  tg {our_tg_tps/pt_tg_tps:.1f}x")
print(f"")
print(f"=== Completions ===")
print(f"Ours:    {our_text[:100]}")
print(f"PyTorch: {pt_text[:100]}")
