"""Diagnostic from issue #5 (dknos): is prefill landing on cuBLAS MAGMA
(slow CUDA-core fallback) or CUTLASS tensor-core kernels?

Runs one warmup + one profiled prefill of a 520-token prompt and prints
the top CUDA kernels by self_cuda_time_total.

Decision rule:
  magma_*                       -> CUDA-core fallback, ~10x slower than tensor cores
  cutlass_*  / *gemm_*tensor_op -> tensor cores; slowdown is elsewhere
"""
import time
import torch
from model import Decoder, HIDDEN_SIZE, INTERMEDIATE_SIZE, FA_QPROJ_SIZE, FA_Q_SIZE, FA_KV_SIZE
from model import DN_CONV_CHANNELS, DN_V_SIZE, DN_NUM_HEADS, MAX_SEQ_LEN, _half_dtype
import qwen35_megakernel_bf16_C
from transformers import AutoTokenizer

print(f"GPU: {torch.cuda.get_device_name(0)} (cap {torch.cuda.get_device_capability()})")
print(f"torch: {torch.__version__}  CUDA: {torch.version.cuda}")

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
dec = Decoder(verbose=False)
_pf = torch.ops.qwen35_megakernel_bf16_C.prefill_bf16

S_MAX = 512
bf16 = dict(dtype=_half_dtype(), device="cuda")
f32 = dict(dtype=torch.float32, device="cuda")
i32 = dict(dtype=torch.int32, device="cuda")
mx = max(DN_CONV_CHANNELS, FA_QPROJ_SIZE, INTERMEDIATE_SIZE)
bufs = dict(
    hidden=torch.empty(S_MAX*HIDDEN_SIZE, **bf16),
    residual=torch.empty(S_MAX*HIDDEN_SIZE, **bf16),
    normalized=torch.empty(S_MAX*HIDDEN_SIZE, **bf16),
    proj_buf=torch.empty(S_MAX*mx, **bf16),
    proj_buf2=torch.empty(S_MAX*mx, **bf16),
    attn_buf=torch.empty(S_MAX*max(FA_Q_SIZE, FA_KV_SIZE), **bf16),
    mlp_buf=torch.empty(S_MAX*INTERMEDIATE_SIZE, **bf16),
    dn_out_buf=torch.empty(S_MAX*DN_V_SIZE, **bf16),
    beta_buf=torch.empty(S_MAX*DN_NUM_HEADS, **f32),
    alpha_buf=torch.empty(S_MAX*DN_NUM_HEADS, **f32),
    final_normed=torch.empty(HIDDEN_SIZE, **bf16),
    hidden_bf16_out=torch.empty(HIDDEN_SIZE, **bf16),
    lm_bmv=torch.empty(1024, **f32),
    lm_bmi=torch.empty(1024, **i32),
)
bufs.update(dec.alloc_prefill_scratch(S_MAX))


def prefill(ids):
    ids_t = torch.tensor(ids, dtype=torch.int32, device="cuda")
    _pf(dec._out_token, ids_t,
        dec._embed_weight, dec._layer_weights_packed,
        dec._final_norm_weight, dec._lm_head_weight,
        dec._fa_k_cache, dec._fa_v_cache, dec._dn_states, dec._conv_bufs,
        bufs['hidden'], bufs['residual'], bufs['normalized'],
        bufs['proj_buf'], bufs['proj_buf2'],
        bufs['attn_buf'], bufs['mlp_buf'],
        bufs['dn_out_buf'], bufs['beta_buf'], bufs['alpha_buf'],
        bufs['dn_pre_qkv'],
        bufs['dn_u_scratch'], bufs['dn_w_scratch'], bufs['dn_cs_scratch'],
        dec._fused_fa_qkv, dec._fused_gate_up,
        bufs['final_normed'], bufs['hidden_bf16_out'],
        bufs['lm_bmv'], bufs['lm_bmi'], dec.max_seq_len)
    dec._hidden.copy_(bufs['hidden_bf16_out'])
    dec._position = len(ids)
    return dec._out_token.item()


# Build a 520-ish token prompt
prompt = "The quick brown fox jumps over the lazy dog. " * 60
ids = tok.encode(prompt, add_special_tokens=False)[:512]
print(f"\nPrompt length: {len(ids)} tokens")

# Warmup (matches bench_pp_tg.py — one untimed warmup)
dec.reset()
prefill(ids)
torch.cuda.synchronize()

# Timed-only run (no profiler) for tok/s reference
dec.reset()
t0 = time.perf_counter()
prefill(ids)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
print(f"Untimed-warmup prefill ({len(ids)} tok): {dt*1000:.2f} ms  ->  {len(ids)/dt:,.0f} tok/s")

# Profiler run
dec.reset()
print("\n=== Profiling one prefill call ===")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
    record_shapes=False,
) as prof:
    prefill(ids)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

# Verdict
events = prof.key_averages()
gemm_like = [e for e in events
             if any(k in e.key.lower()
                    for k in ("gemm", "magma", "cutlass", "tensorop", "mm_out", "ampere", "wmma"))]
print("\n=== Verdict ===")
if not gemm_like:
    print("No GEMM-like kernels found in top events. Inspect the table above.")
else:
    for e in gemm_like:
        print(f"  {e.key}   self_cuda={e.self_cuda_time_total/1000:.2f} ms")
    bad = any("magma" in e.key.lower() for e in gemm_like)
    good = any(k in e.key.lower() for e in gemm_like for k in ("cutlass", "tensorop", "ampere", "wmma"))
    if bad and not good:
        print("\n[FAIL] Dominant GEMM is MAGMA — cuBLAS fell back to CUDA-core path.")
        print("       Apply dknos's fix: replace cublas_bf16_gemm with at::mm_out.")
        print("       PR: https://github.com/dknos/luce-megakernel/pull/1")
    elif good and not bad:
        print("\n[PASS] Dominant GEMM is on tensor cores. Slowdown is elsewhere.")
        print("       Investigate DVFS / thermal throttling / driver / launch-overhead.")
    else:
        print("\n[MIXED] Both magma and tensor-core kernels present. Look at relative time totals.")
