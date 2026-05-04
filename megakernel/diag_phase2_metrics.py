"""Deep CUPTI instrumentation on pf_dn_chunk_phase2.

Mirrors diag_prefill_kernels.py setup, then adds:
  - torch.profiler with record_shapes, profile_memory, with_stack
  - timeline gap analysis (phase1 -> phase2 -> next-prep)
  - Chrome trace export for inspection

Run from megakernel/:
    python3 diag_phase2_metrics.py 2>&1 | tail -150

Decision target: is phase2 compute-bound, memory-bound, sync-bound, or block-starved?
"""
import time
import statistics
import torch
import torch.profiler as prof_mod
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
f32  = dict(dtype=torch.float32,  device="cuda")
i32  = dict(dtype=torch.int32,    device="cuda")
mx   = max(DN_CONV_CHANNELS, FA_QPROJ_SIZE, INTERMEDIATE_SIZE)

bufs = dict(
    hidden           = torch.empty(S_MAX * HIDDEN_SIZE, **bf16),
    residual         = torch.empty(S_MAX * HIDDEN_SIZE, **bf16),
    normalized       = torch.empty(S_MAX * HIDDEN_SIZE, **bf16),
    proj_buf         = torch.empty(S_MAX * mx, **bf16),
    proj_buf2        = torch.empty(S_MAX * mx, **bf16),
    attn_buf         = torch.empty(S_MAX * max(FA_Q_SIZE, FA_KV_SIZE), **bf16),
    mlp_buf          = torch.empty(S_MAX * INTERMEDIATE_SIZE, **bf16),
    dn_out_buf       = torch.empty(S_MAX * DN_V_SIZE, **bf16),
    beta_buf         = torch.empty(S_MAX * DN_NUM_HEADS, **f32),
    alpha_buf        = torch.empty(S_MAX * DN_NUM_HEADS, **f32),
    final_normed     = torch.empty(HIDDEN_SIZE, **bf16),
    hidden_bf16_out  = torch.empty(HIDDEN_SIZE, **bf16),
    lm_bmv           = torch.empty(1024, **f32),
    lm_bmi           = torch.empty(1024, **i32),
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


# Build a 512-token prompt
prompt = "The quick brown fox jumps over the lazy dog. " * 60
ids    = tok.encode(prompt, add_special_tokens=False)[:512]
print(f"\nPrompt length: {len(ids)} tokens")

# ── Warmup ──────────────────────────────────────────────────────────────────
dec.reset()
prefill(ids)
torch.cuda.synchronize()

# ── Timed reference ─────────────────────────────────────────────────────────
dec.reset()
t0 = time.perf_counter()
prefill(ids)
torch.cuda.synchronize()
print(f"Wall time: {(time.perf_counter()-t0)*1000:.2f} ms  ({len(ids)/(time.perf_counter()-t0):,.0f} tok/s)")

# ── Two profiler runs ────────────────────────────────────────────────────────
for run_idx in range(2):
    dec.reset()
    with prof_mod.profile(
        activities=[prof_mod.ProfilerActivity.CUDA, prof_mod.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as p:
        with prof_mod.record_function("prefill_outer"):
            prefill(ids)
            torch.cuda.synchronize()

    print(f"\n{'='*60}  Run {run_idx}")
    print(p.key_averages(group_by_input_shape=True).table(
        sort_by="self_cuda_time_total", row_limit=25))
    # Export trace (useful for Chrome tracing; skip if disk is constrained)
    p.export_chrome_trace(f"/tmp/phase2_trace_run{run_idx}.json")
    print(f"  [trace saved -> /tmp/phase2_trace_run{run_idx}.json]")

# ── Phase2-specific timeline analysis ───────────────────────────────────────
# Use the second run (run_idx=1) — p still holds that run.
print("\n" + "="*60)
print("  PHASE2 DETAILED TIMELINE ANALYSIS (run 1)")
print("="*60)

events = p.events()

# Filter CUDA events (device_type == DeviceType.CUDA in newer torch,
# or check device_index >= 0)
cuda_events = [e for e in events if e.device_type == prof_mod.DeviceType.CUDA]
phase2_events = [e for e in cuda_events if "pf_dn_chunk_phase2" in e.name]
phase1_events = [e for e in cuda_events if "pf_dn_chunk_phase1" in e.name]
phase1_events.sort(key=lambda e: e.time_range.start)
phase2_events.sort(key=lambda e: e.time_range.start)

print(f"\nphase1 launches found: {len(phase1_events)}")
print(f"phase2 launches found: {len(phase2_events)}")

if phase2_events:
    # FunctionEvent stores CUDA time in self_cuda_time_total (microseconds)
    # For individual kernel events (not aggregated) cuda_time is per-event
    p2_durations_us = [e.cuda_time for e in phase2_events]
    p2_med  = statistics.median(p2_durations_us)
    p2_mean = statistics.mean(p2_durations_us)
    p2_p99  = sorted(p2_durations_us)[int(0.99 * len(p2_durations_us))]
    print(f"\nphase2 self-CUDA time per launch:")
    print(f"  count  : {len(p2_durations_us)}")
    print(f"  mean   : {p2_mean:.2f} µs")
    print(f"  median : {p2_med:.2f} µs")
    print(f"  P99    : {p2_p99:.2f} µs")
    print(f"  total  : {sum(p2_durations_us)/1000:.3f} ms")
else:
    print("\n[WARN] No pf_dn_chunk_phase2 CUDA events found in profiler trace.")
    print("       Kernel may be fused or named differently. Checking all CUDA events:")
    dn_related = [e for e in cuda_events if "dn" in e.name.lower() or "delta" in e.name.lower() or "phase" in e.name.lower()]
    for e in dn_related[:15]:
        print(f"  {e.name}  dur={e.duration:.1f}µs")
    if not dn_related:
        print("  (none matching 'dn/delta/phase')")
        print("\n  Top 15 CUDA events by duration:")
        top = sorted(cuda_events, key=lambda e: e.duration, reverse=True)[:15]
        for e in top:
            print(f"  {e.name:60s} dur={e.duration:8.1f}µs")

# ── Gap analysis: phase1→phase2 and phase2→next ──────────────────────────────
# FunctionEvent gap analysis using time_range (nanoseconds in torch 2.x)
# time_range.start and time_range.end are in microseconds for CUDA events
if phase1_events and phase2_events:
    print("\n--- Phase1→Phase2 and Phase2→next-prep gaps ---")
    gaps_p1_p2 = []
    for p1, p2 in zip(phase1_events, phase2_events):
        # cuda_time_range is available; fall back to time_range
        try:
            gap_us = p2.time_range.start - p1.time_range.end
        except AttributeError:
            gap_us = 0
        if abs(gap_us) < 1e6:
            gaps_p1_p2.append(gap_us)
    if gaps_p1_p2:
        print(f"  phase1→phase2 gap  median={statistics.median(gaps_p1_p2):.1f}µs  "
              f"mean={statistics.mean(gaps_p1_p2):.1f}µs  "
              f"max={max(gaps_p1_p2):.1f}µs")
    else:
        print("  (time_range not available or gaps out of range)")

# ── All kernel totals breakdown ──────────────────────────────────────────────
print("\n--- CUDA time breakdown (top 10 by total cuda_time) ---")
name_to_total: dict[str, float] = {}
for e in cuda_events:
    name_to_total[e.name] = name_to_total.get(e.name, 0.0) + e.cuda_time
grand_total = sum(name_to_total.values())
for name, t in sorted(name_to_total.items(), key=lambda x: -x[1])[:10]:
    pct = 100 * t / grand_total if grand_total > 0 else 0
    print(f"  {name:60s}  {t/1000:8.3f} ms  ({pct:.1f}%)")

print(f"\n  Grand total tracked CUDA time: {grand_total/1000:.3f} ms")

if phase2_events:
    p2_total = sum(e.cuda_time for e in phase2_events)
    print(f"  phase2 share: {100*p2_total/grand_total:.1f}%  ({p2_total/1000:.3f} ms)")
