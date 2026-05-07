# Benchmarks

## Reference numbers (RTX 3090, from upstream repo)

### DFlash spec decode
| Task | AR tok/s | DFlash+DDTree tok/s | Speedup |
|------|:--------:|:-------------------:|:-------:|
| HumanEval | 37.78 | 129.52 | 3.43× |
| Math500 | 37.71 | 110.51 | 2.93× |
| GSM8K | 37.65 | 96.15 | 2.55× |

### PFlash spec prefill (128K context, RTX 3090)
| Context | TTFT dflash | TTFT llama.cpp baseline | Speedup | NIAH |
|---------|:-----------:|:-----------------------:|:-------:|:----:|
| 64K | 13.5 s | 134.95 s | 10.0× | ✅ |
| 128K | 24.8 s | ~257 s | ~10.4× | ✅ |

Settings: Qwen3.6-27B Q4_K_M target, Qwen3-0.6B drafter, `DFLASH_FP_USE_BSA=1`, `DFLASH_FP_ALPHA=0.85`, `keep_ratio=0.05`

---

## Strix Halo baselines (to measure)

Hardware: Ryzen AI Max+ 395, 128 GB unified memory, RDNA3.5 iGPU (gfx1151)

### AR baseline (dflash test_generate, ROCm)
| Metric | Expected (RTX 3090) | Measured | Date |
|--------|:-------------------:|----------|------|
| Decode tok/s (Qwen3.5-27B Q4_K_M) | 37-40 | **11.70** | 2026-05-06 |
| Prompt processing 512 tok | TBD | TBD | - |
| Prompt processing 32K tok | TBD | TBD | - |
| Prompt processing 128K tok | TBD | TBD | - |

Notes on 11.70 tok/s:
- 63965 MiB VRAM visible (64 GB slice of 128 GB unified)
- Model loaded: 14.91 GiB on GPU
- Strix Halo iGPU bandwidth ~256 GB/s vs RTX 3090's 936 GB/s; 40 × (256/936) ≈ 11 tok/s expected
- test_generate uses single-token decode (no batched prefill) — same as spec decode step overhead
- Raw prompt (no chat template); output was degenerate (repeated newline token) but timing is valid

### DFlash spec decode (Strix Halo)
| Task | AR tok/s | DFlash tok/s | Speedup | Date |
|------|:--------:|:------------:|:-------:|------|
| Coding (quicksort prompt, 128 tok, 3 runs) | **11.36** | **29.23 avg** | **2.57×** | 2026-05-06 |
| HumanEval (10 prompts, n_gen=128, ddtree=16) | **11.70** | **29.20 avg** | **2.49×** | 2026-05-07 |
| Math500 | TBD | TBD | TBD | - |

HumanEval per-prompt breakdown (tok/s / acceptance %):
| Prompt | tok/s | AL | pct% |
|--------|:-----:|:--:|:----:|
| has_close_elements | 14.86 | 4.57 | 28.6 |
| separate_paren_groups | 21.07 | 6.40 | 40.0 |
| truncate_number | 19.60 | 5.82 | 36.4 |
| below_zero | 28.92 | 6.10 | 38.1 |
| mean_absolute_deviation | 24.53 | 5.12 | 32.0 |
| intersperse | 22.76 | 4.74 | 29.6 |
| parse_nested_parens | **46.80** | 9.85 | 61.5 |
| filter_by_substring | 31.21 | 6.40 | 40.0 |
| sum_product | **43.79** | 9.14 | 57.1 |
| rolling_max | 38.42 | 8.00 | 50.0 |
| **MEAN** | **29.20** | **6.61** | **41.3** |

Notes: tok/s variance reflects drafting quality per problem. Highly structured outputs
(parse_nested_parens, sum_product) get 9+ AL; less predictable outputs get 4-5 AL.
AR baseline 11.70 tok/s gives 2.49× mean speedup. RTX 3090 ref: 3.43× on HumanEval.
Gap vs RTX 3090: bandwidth-limited (256 vs 936 GB/s → ~3.7× floor). Acceptance rates
are hardware-independent and match expectations.

Settings: Qwen3.5-27B Q4_K_M target, z-lab/Qwen3.5-27B-DFlash draft (safetensors),
`--ddtree --ddtree-budget=16`, 36.4% per-step acceptance, avg 5.82 tokens/step.
Warmup: 2 runs before measurement. 3 measurement runs (29.89, 29.00, 28.80 tok/s).

Hardware: Radeon 8060S (gfx1151), 63965 MiB reported VRAM, Wave32, model 14.91 GiB on GPU.
Context: coding prompt with Qwen3.5 chat template (36 tokens prompt, 128 generated).

Note: RTX 3090 achieved 3.43× on HumanEval (37.78→129.52 tok/s). Lower speedup here
reflects lower absolute bandwidth (256 vs 936 GB/s) — spec decode overhead is a larger
fraction of each step. Acceptance rate should be similar on proper coding tasks.

### PFlash spec prefill (Strix Halo, Phase 2 — rocWMMA, no BSA)

Metric measured: drafter scoring time (FP step) and total compress latency.
TTFT = compress time + target prefill on compressed tokens + first generated token.
Phase 1 used ggml q8 FA fallback; Phase 2 uses custom rocWMMA sparse kernels.

| Context | P1 FP (q8) | P2 FP (WMMA) | FP speedup | P1 compress | P2 compress | Compress speedup | Date |
|---------|:----------:|:------------:|:----------:|:-----------:|:-----------:|:----------------:|------|
| 8K      | 2.45 s     | 1.02 s       | **2.40×**  | 4.63 s      | 3.19 s      | **1.45×**        | 2026-05-06 |
| 16K     | 8.47 s     | 4.83 s       | **1.75×**  | 11.67 s     | 7.96 s      | **1.47×**        | 2026-05-06 |
| 32K     | 31.54 s    | 34.78 s      | 0.91×      | 36.81 s     | 40.05 s     | ~1.0×            | 2026-05-06 |
| 64K     | TBD        | TBD          | TBD        | TBD         | TBD         | TBD              | - |
| 128K    | TBD        | TBD          | TBD        | TBD         | TBD         | TBD              | - |

Settings: Qwen3.5-27B Q4_K_M target, Qwen3-0.6B BF16 drafter, keep_ratio=0.05.
Hardware: Radeon 8060S gfx1151, Wave32. Timings post-warmup (1 warmup run, 2 measure runs averaged).

Notes:
- At ≤16K, the WMMA sparse forward (kernel 4) savings outweigh scoring overhead (kernels 1-3): 1.45–2.4× win.
- At 32K, kernel 2 (O(M²) block scoring) overhead matches the kernel 4 sparsity savings: ~breakeven.
- BSA (Phase 3) would replace kernel 2 with a more efficient approximate scoring path.

### PFlash spec prefill (Strix Halo, Phase 3 — BSA launcher)

Phase 3 ports BSA to HIP by wrapping our rocWMMA kernel 4 (`bsa_launcher_hip.cu`).
Both paths (BSA and non-BSA) use the same underlying kernel — no new FA kernel was needed.

NIAH results at 8K (3 cases, keep_ratio=0.05/0.15, bsa=1, alpha=0.85):
| keep_ratio | NIAH | Observation |
|:----------:|:----:|-------------|
| 0.05 | 0/3 | Needle absent from compressed text |
| 0.15 | 0/3 | Needle still absent — model says "no numbers in text" |

Root cause: drafter token importance scoring does not rank random-token NIAH needles
(8-char key + 7-digit value) higher than repetitive filler. This affects CUDA build too —
not a HIP-specific kernel quality issue. Fix requires changes to token scoring, not FA kernel.

Timing at 8K with BSA=1 (bsa_launcher_hip.cu active, fp-prof per layer):
| Step | Time (ms) |
|------|:---------:|
| mean-K | 0.23 |
| block-score | 3.48 |
| block-select | 0.01 |
| sparse-FA forward | ~12 |

No timing regression vs Phase 2 non-BSA path (same kernel).

### PFlash spec prefill (Strix Halo, Phase 4 — rocWMMA GEMM block scoring)

Phase 4 replaces kernel 2's O(M²) scalar loop with a rocWMMA tiled GEMM:
  `score[m,n,h] = mean_Q[m,h,:] · mean_K[n,kh,:] * scale`
Adds a mean_Q computation step (kernel 1 reused for Q) and eliminates the O(M²) bottleneck.

Per-step kernel timing (DFLASH_FP_PROFILE=1, H=16, Hk=8, D=128, B=1):
| S     | mean (ms) | score (ms) | Phase 3 score | Score speedup |
|-------|:---------:|:----------:|:-------------:|:-------------:|
| 8K    | 0.21      | **0.67**   | 3.48          | **5.2×**      |
| 16K   | 0.84      | **1.73**   | ~14 (est.)    | **~8×**       |
| 32K   | 2.00      | **5.60**   | ~56 (est.)    | **~10×**      |

Phase 3 8K score was measured during NIAH testing (BSA=1, alpha=0.85, keep=0.15).
Phase 4 values from bench_score (alpha=0.12, no BSA); e2e compress timing requires full pipeline run.

Impact on the 32K regression:
- Old kernel 2 overhead at 32K: ~56ms/layer × 28 layers = 1.57s of the total FP time
- New kernel 2 overhead at 32K: ~5.6ms × 28 = 0.16s  (-1.4s recovered)
- Kernel 4 at 32K is still ~5% slower than q8 FA fallback; full compress benchmark needed

### End-to-end compress timing (Strix Halo, Phase 3+4 combined, bench_niah_cpp.py)

Settings: keep_ratio=0.05, BSA=1, alpha=0.85 (Phase 4 includes both Phase 3 BSA and Phase 4 GEMM score).
Phase 1/2 compress from Phase 2 benchmarks (no BSA). Phase 4 measured 2026-05-06.

| Context | Phase 1 compress | Phase 2 compress | Phase 4 compress | vs Phase 2 |
|---------|:----------------:|:----------------:|:----------------:|:----------:|
| 8K      | 4.63 s           | 3.19 s           | **~1.9 s**       | **1.7×**   |
| 32K     | 36.81 s          | 40.05 s          | **~14.5 s**      | **2.76×**  |

Equivalent FP step (drafter custom-kernel time only):
| Context | Phase 1 FP | Phase 2 FP | Phase 4 FP | FP vs Phase 2 |
|---------|:----------:|:----------:|:----------:|:-------------:|
| 8K      | 2.45 s     | 1.02 s     | **0.55 s** | **1.85×**     |
| 32K     | 31.54 s    | 34.78 s    | **9.5 s**  | **3.66×**     |

The 32K gain is dominated by BSA (Phase 3): forward kernel at 32K now takes ~9.3s (avg 331ms/layer)
vs estimated ~33s without BSA. Phase 4 GEMM scoring saves an additional ~1.4s at 32K.

Per-step breakdown at 32K (Phase 4, DFLASH_FP_PROFILE=1, 28 layers averaged):
| Step | ms/layer | Total |
|------|:--------:|:-----:|
| mean_K | 1.80 | 50 ms |
| mean_Q + GEMM score | 5.57 | 156 ms |
| block_select | 0.17 | 4.8 ms |
| sparse FA forward (BSA) | ~331 avg | ~9.3 s |

NIAH accuracy with Phase 4 (same failure mode as Phase 3):
| S   | NIAH | Notes |
|-----|:----:|-------|
| 8K  | 0/3  | Needle absent from compressed text — GEMM scoring doesn't change needle ranking |
| 32K | 0/2  | Same root cause; fix requires scoring approach that ranks random-token needles higher |

### PFlash spec prefill (Strix Halo, Phase 4 — 64K context)

Measured 2026-05-06 with bench_niah_cpp.py --n 2, keep_ratio=0.05, BSA=1, alpha=0.85.

Per-layer kernel timing (S=65531, H=16, Hk=8, D=128, B=1, averaged over 28 layers):
| Step | ms/layer | vs 32K |
|------|:--------:|:------:|
| mean_K | ~4.4 | 2.4× |
| GEMM score | ~15.7 | 2.8× |
| block_select | ~0.74 | 4.4× |
| sparse FA forward | varies | — |

End-to-end timing (2 cases averaged):
| Metric | Value |
|--------|-------|
| FP step (kernels only) | ~41 s |
| Tail-score | ~2.85 s |
| Total compress (drafter) | ~49.5 s |
| Target prefill (3260 tok) | ~15.5 s |
| TTFT (wall clock) | **~73 s** |
| Compression ratio | 20.1× (65531→3260 tokens) |
| NIAH | 0/2 (same root cause as 8K/32K) |

RTX 3090 reference TTFT at 64K: 13.5 s → Strix Halo is ~5.4× slower (bandwidth ratio 3.7× explains most of gap).

NIAH root cause (confirmed 2026-05-06): RoPE distance decay in the tail-score. The tail-score
computes dense attention from Q_last (positions ~8160-8187) to all K. Attention decays with
relative distance Δ = Q_last_pos - K_pos. Recent chunks (Δ<200) and attention sinks always rank
top; the needle at Δ≈4000 scores below these regardless of content. Verified via chunk debug:
- n_lookahead=32 (question tokens with "qahftrxc" in Q_last): no improvement
- BSA=0 (dense FA): same failure
- Band-based selection (DFLASH_FP_BANDS=4): still fails — distance dominates within bands too
Fix requires NoPE-based scoring or hidden-state norm importance.

### PFlash NIAH (Strix Halo, Phase 6b — NoPE tail-score fix)

Fix: store pre-RoPE K (K_norope_v) and use it in the tail-score instead of post-RoPE K.
Content similarity (not positional distance) now dominates chunk ranking.
`enable_thinking=False` added to bench prompt template so model answers directly.

Settings: 8K, BSA=1, alpha=0.85, n_lookahead=64 (question tokens in Q_last).

| keep_ratio | n_keep | NIAH | Notes |
|:----------:|:------:|:----:|-------|
| 0.05       | 12     | 0/3  | Needle chunk 133 is rank 19 — just outside top 12 |
| 0.15       | 38     | 2/3  | Cases 0+1 pass; case 2 needle ranked >38 |

Chunk ranking change for case 0 needle (chunk 133, pos 4256-4287):
- Before NoPE fix (RoPE): not in top 17 (score ~0.002, overwhelmed by recent/sink chunks)
- After NoPE fix (NoPE):  rank 19 (score 0.003177 — uniform with filler chunks, +0 distance bias)
- The needle is content-matched but scored equally to generic filler due to uniform embeddings

Additional semantic-needle test (Eiffel Tower / fictional city / fictional compound, 8K, keep=0.05 and 0.15):
| keep_ratio | NIAH | Notes |
|:----------:|:----:|-------|
| 0.05 | 0/3 | Needle chunk not in top 17 |
| 0.15 | 0/3 | Needle chunk not in top 43 |

Unexpected finding: semantic needles fail WORSE than random-key needles with NoPE fix.
Root cause: "qahftrxc" is OOV-style (unusual subword tokens) → distinctive hidden states → high
K·Q score in NoPE scoring. "Eiffel Tower" is within-distribution → similar hidden states to filler.
The NoPE fix benefits needles with unusual TOKEN STATISTICS, not semantic meaning.
Hidden-state norm scoring (||h[j]||₂) would likely handle this better — rare proper nouns
tend to have higher norms than common-word filler.

### PFlash spec prefill (Strix Halo, Phase 8 — 128K context)

Measured 2026-05-06 with bench_niah_cpp.py --n 2, keep_ratio=0.05, BSA=1, alpha=0.85,
n_lookahead=64, NoPE tail-score fix active.

Forward pass breakdown (S=131068, case 0):
| Step | Time |
|------|------|
| drafter attention (A) | ~3.9 s |
| sparse FA forward (FP) | ~114-120 s |
| drafter output (B) | ~6.0 s |
| tail-score | ~8.5 s |
| total compress | ~134-140 s |

End-to-end timing (2 cases, case 0 / case 1):
| Metric | Case 0 | Case 1 |
|--------|--------|--------|
| Total compress | ~135 s | ~141 s |
| Target prefill (6524 tok) | ~33.2 s | ~32.8 s |
| TTFT (wall clock) | **~169 s** | **~176 s** |
| Compression ratio | 20.1× (131K→6524 tokens) | 20.1× |
| NIAH | ✅ ok | ✅ ok |

**accuracy: 2/2** — both cases pass NIAH at 128K with keep_ratio=0.05 + NoPE fix.

RTX 3090 reference TTFT at 128K: 24.8 s → Strix Halo is **~7× slower** (avg ~173s).
Bandwidth ratio 3.7× accounts for most gap; remainder is kernel efficiency.

FP kernel scaling from 64K→128K: ~3× longer (114s vs ~41s at 64K), consistent with O(S) sparse FA.
Tail-score scaling: 8.5s vs 2.85s at 64K ≈ 3× (O(S) dense attention).

### PFlash kernel 4 — K/V transpose optimization (Phase 9)

Root cause of kernel 4 inefficiency: K/V layout was `[B, S, Hk, D]`, so loading a K tile for
one head required stride-`Hk*D` (2 KB) jumps between consecutive tokens. Each warp loaded
separate 2 KB cache-line regions per token row instead of a contiguous tile.

Fix (2026-05-07): pre-transpose K/V from `[B, S, Hk, D]` → `[B, Hk, S, D]` in a new
`launch_transpose_kv_bf16` kernel before calling the sparse FA kernel. With transposed layout,
token stride = D = 128 elements = 256 bytes. Consecutive token rows fall in the same 256 B
region → single coalesced load per warp instead of N scattered 2KB loads.

bench_score throughput (B=1, H=16, Hk=8, D=128, block_size=128, alpha=0.12):
| S     | Dense (no BSA) | BSA + transpose | Speedup |
|-------|:--------------:|:---------------:|:-------:|
| 8K    | 35.4 ms        | 29.0 ms         | 1.2×    |
| 16K   | 177.0 ms       | 111.0 ms        | 1.6×    |
| 32K   | 1347.5 ms      | 433.1 ms        | **3.1×** |
| 64K   | 7244.1 ms      | 1711.0 ms       | **4.2×** |

End-to-end FP step (28 drafter layers, alpha=0.85 ≈ 69/256 blocks selected):
| S     | Phase 4 FP | Phase 9 FP | Speedup |
|-------|:----------:|:----------:|:-------:|
| 32K   | 9.5 s      | **5.0 s**  | **1.9×** |

Total compress (drafter forward + tail-score):
| S     | Phase 4    | Phase 9    | Speedup |
|-------|:----------:|:----------:|:-------:|
| 32K   | ~14.5 s    | **~9.8 s** | **1.5×** |

Speedup scales with S because: (a) BSA drops 73% of blocks, (b) transpose eliminates
cache-line waste within each remaining block. Both effects compound at larger S.
bench_score (alpha=0.12, 31 blocks) shows 3.1–4.2× because fewer blocks are selected,
so the cache-unfriendly strided loads are a larger fraction of total work.
e2e at alpha=0.85 (69 blocks) shows 1.9× — still meaningful at larger S.
test_flashprefill_kernels numerics PASS (max diff = 0.00053, unchanged).
Persistent kv_buf_K/V device buffers: allocated once per sequence length, reused across all
28 layer calls (hipMalloc overhead per call was ~1ms total, not the bottleneck).

End-to-end (Strix Halo, Phase 9, 64K context, bench_niah_cpp.py, 2 cases, 2026-05-07):
| Metric | Phase 4 | Phase 9 | Speedup |
|--------|:-------:|:-------:|:-------:|
| FP step (drafter forward) | ~41 s | **~19.5 s** | **2.1×** |
| Total compress | ~49.5 s | **~23.7 s** | **2.1×** |
| Target prefill (3298 tok) | ~15.5 s | **~15.7 s** | ~same |
| TTFT | ~73 s | **~43 s** | **1.7×** |
| NIAH | 0/2 ❌ | **2/2 ✅** | — |
| vs RTX 3090 (13.5 s) | 5.4× slower | **3.2× slower** | — |

End-to-end (Strix Halo, Phase 9, 128K context, bench_niah_cpp.py, 2 cases, 2026-05-07):
| Metric | Phase 8 | Phase 9 | Speedup |
|--------|:-------:|:-------:|:-------:|
| FP step (drafter forward) | ~114–120 s | **~49–50 s** | **~2.3×** |
| Tail-score | ~8.5 s | **~8.5 s** | same |
| Total compress | ~135–141 s | **~59 s** | **~2.3×** |
| Target prefill (6539 tok) | ~33 s | **~33 s** | same |
| TTFT | ~173 s | **~95–96 s** | **1.8×** |
| NIAH | 2/2 ✅ | **2/2 ✅** | — |
| vs RTX 3090 (24.8 s) | 7× slower | **~3.9× slower** | — |

Phase 9 cut the gap to RTX 3090 roughly in half at both contexts. The remaining gap (~3–4×)
is close to the raw bandwidth ratio (256 vs 936 GB/s = 3.7×) — kernel efficiency is near
the memory-bandwidth ceiling.

Estimated llama.cpp prefill baseline (from target-prefill rate in Phase 9 runs: ~200 tok/s):
| Context | Estimated llama.cpp TTFT | Phase 9 dflash TTFT | Estimated speedup |
|---------|:------------------------:|:-------------------:|:-----------------:|
| 64K | ~313 s | **43 s** | **~7.3×** |
| 128K | ~640 s | **96 s** | **~6.7×** |

Note: llama.cpp baseline is estimated from the compressed-target prefill rate (~200 tok/s)
applied to full uncompressed context. Actual speedup may be higher (llama.cpp degrades at
long context without optimized sparse FA). NIAH at 64K now passes (NoPE fix included in
Phase 9 build) where Phase 4 failed.

---

## Notes

- Strix Halo has 128 GB unified vs 24 GB VRAM on RTX 3090. All models fit simultaneously.
  This eliminates the park/unpark latency overhead, so TTFT numbers may differ from reference.
- Prompt processing on Strix Halo is ~5× slower than DGX Spark (339 vs 1723 tok/s),
  so PFlash's prefill compression is proportionally MORE valuable here.
- DFlash decode speedup should be similar to RTX 3090 — the memory bandwidth bottleneck
  is the same dynamic regardless of GPU vendor.
