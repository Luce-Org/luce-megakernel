# dflash spec-prefill, daemon-side build & tunables

In-process speculative-prefill + speculative-decode daemon (C++/CUDA only,
no Python, no Triton, no PyTorch at runtime).

This doc is the build / runtime / tunables reference for the C++ daemon
path described in [`pflash/README.md`](../../pflash/README.md) and on the
[blog post](https://lucebox.com/blog/pflash):

- **Drafter** (Qwen3-0.6B) loaded via a custom forward (`qwen3_0p6b_*`)
  with the FlashPrefill block-sparse attention kernel for long-context
  scoring.
- **Target** (Qwen3.6-27B Q4_K_M) loaded directly via ggml.
- **Speculative decode** between draft + target with rollback / DDTree.

Both models live in the same process, the same ggml allocator, on a
single RTX 3090 (24 GB). No PyTorch at runtime.

## Build

```
git submodule update --init --recursive
mkdir build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=86 -DDFLASH27B_ENABLE_BSA=ON ..
cmake --build . --target test_dflash test_flashprefill_kernels -- -j8
```

Required:
- CUDA Toolkit 12.0+ (sm_80+ for BSA path; sm_86 RTX 3090 is the
  reference target).
- `git submodule update --init --recursive` to pull
  `deps/llama.cpp` (ggml only) and `deps/Block-Sparse-Attention` (with
  cutlass).

CMake options:
- `DFLASH27B_ENABLE_BSA=ON` (default) — build the Block-Sparse-Attention
  kernel for sparse FA forward. Required for the long-context perf claim.
  Turn OFF only on sm<80.
- `DFLASH27B_FA_ALL_QUANTS=ON` (default) — compile ggml-cuda fattn for
  all KV-quant pairs (needed for asymmetric Q4_0 K + Q8_0 V cache). Off
  cuts build time ~3x but breaks the 128K target gen path.

## Runtime tunables

```
DFLASH_FP_USE_BSA=1    # dispatch sparse FA forward through BSA (sm_80+)
DFLASH_FP_ALPHA=0.85   # block-selection threshold (default 0.12);
                       # higher = stricter = fewer K-blocks per Q-row.
DFLASH_FP_PROFILE=1    # log mean/score/select/forward stage timings
```

See `src/flashprefill.h` for the full list and defaults.

PFlash compression is tuned separately from the sparse-attention kernel:

```
DFLASH_PFLASH_LOOKAHEAD=8       # override tail Q count used for scoring
DFLASH_PFLASH_KEEP_TARGET=0     # experimental: keep target resident during
                                # compression; caller should park draft first
DFLASH_PFLASH_SKIP_DRAFT_RELOAD=0
                                # experimental: after PFlash compression, keep
                                # DFlash draft parked for TTFT-only fallback
DFLASH_PFLASH_CHUNK_RADIUS=0    # keep neighbor chunks around score winners
DFLASH_PFLASH_QUERY_TAIL=128    # tail tokens scanned for lexical rescue
DFLASH_PFLASH_RARE_MAX_FREQ=4   # only rare tail tokens can rescue chunks
DFLASH_PFLASH_QUERY_MIN_HITS=2  # rare-token hits required in a body chunk
DFLASH_PFLASH_QUERY_RADIUS=1    # keep neighbor chunks around lexical hits
DFLASH_PFLASH_DUMP_CHUNKS=/tmp/pflash_chunks.csv
```

The lexical rescue path is a quality guard for long-prompt retrieval cases:
if the final question contains rare tokens also present near the answer, those
body chunks are kept even when the score-only top-K misses them. It only scans
body chunks before the query tail, so the query itself does not spuriously
rescue neighboring tail chunks. Set `DFLASH_PFLASH_RARE_MAX_FREQ=0` to disable
it for controlled ablations.

## SM75 / RTX 2080 Ti

SM75 does not have BF16 tensor cores and cannot run the BSA backend
(`DFLASH27B_ENABLE_BSA=ON` auto-disables when the build arch includes 75).
For RTX 2080 Ti, build the local path with FP16 drafter compute:

```
cmake -S . -B build-sm75-f16 -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=75 -DDFLASH27B_ENABLE_BSA=OFF
cmake --build build-sm75-f16 --target \
      test_dflash test_flashprefill_kernels test_pflash_chunk_select -j
```

Local benchmark setup: RTX 2080 Ti 22 GB, driver 595.58.03, CUDA 12.0,
power limit 280 W, persistence mode enabled, Qwen3.6-27B Q4_K_M target,
Qwen3-0.6B FP16 GGUF drafter, same 16K NIAH qtail prompt, model load excluded.

| Case | Request time | Speedup | Notes |
|------|-------------:|--------:|-------|
| no PFlash | 50.35 s | 1.00x | original 16K prompt |
| current PFlash hook | 26.11 s | 1.93x | parks and reloads target + draft |
| `DFLASH_PFLASH_KEEP_TARGET=1` | 14.19 s | 3.55x | target stays resident, draft reloads |
| `KEEP_TARGET=1` + `SKIP_DRAFT_RELOAD=1` | 4.13 s | 12.21x | TTFT-only fallback; no speculative decode |

The last row is a TTFT / very short-output fallback, not a replacement for
full DFlash speculative decode. The daemon skips `migrate_prefill_cache` when
the DFlash draft remains parked because rollback tensors are only needed by
speculative verification. This fallback should not be used as a decode
performance claim.

The candidate quality smoke is retrieval-only: the original 16K NIAH prompt
and a 5-position synthetic 16K NIAH sweep retained the key and answer. Broader
long-prompt QA should pass before enabling this by default. Keeping the 0.6B
PFlash drafter resident across requests was rejected locally because the lower
second compression time was offset by slower target prefill.

## Performance

NIAH single-needle end-to-end on RTX 3090 (Qwen3.6-27B Q4_K_M target,
Qwen3-0.6B drafter, in-process daemon, `DFLASH_FP_USE_BSA=1`,
`DFLASH_FP_ALPHA=0.85`, `keep_ratio=0.05`):

| Source S | dflash TTFT | llama.cpp baseline | Speedup | NIAH |
|----------|------------:|-------------------:|--------:|:----:|
| 64K      | **13.5 s**  | 134.95 s (FA off, dense) | **10.0×** | ✅ |
| 128K     | **24.8 s**  | ~257 s (FA on, Q4_0 KV)  | **~10.4×** | ✅ |

NIAH needle retrieved (accuracy 1/1) at every measured context. The
runtime is C++/CUDA only — the headline number is the dflash binary on
its own, no Python or Triton in the loop.

## Repo layout

```
src/
  flashprefill.{h,cpp}         FlashPrefill C++ entry + dispatcher
  flashprefill_kernels.cu       4 CUDA kernels (mean_K, score, select, sparse_fwd)
  flashprefill_select.cpp       Host fallback for block_select (rarely used)
  pflash_chunk_select.{h,cpp}   CPU chunk top-K + lexical rescue + span merge
  bsa_launcher.cu               BSA launcher: blockmask conversion + Flash_fwd_params
  bsa_fwd_inst.cu               Single-TU instantiation of BSA's hdim128 kernel
  qwen3_0p6b_loader.cpp         GGUF → Qwen3-0.6B BF16 weight tensors
  qwen3_0p6b_graph.cpp          Custom Qwen3-0.6B forward (per-layer A/FP/B graphs)
  qwen3_drafter.{h,cpp}         drafter_score_and_compress() entry point
  qwen35_target_graph.cpp       Qwen3.5/3.6 target graph (ggml)
  qwen3_dflash_graph.cpp        DFlash speculative draft head
  kv_cache.cpp / kv_quant.cpp   Q4_0 KV cache + asymmetric quant
test/
  test_dflash.cpp               daemon executable; supports
                                  `compress / generate / park / unpark / free drafter`
  test_flashprefill_kernels.cpp parity tests for the 4 FP kernels
  test_pflash_chunk_select.cpp  regression for rare-query lexical rescue
  smoke_qwen3_0p6b_forward.cpp  drafter forward smoke at S=8K-128K
deps/
  llama.cpp/                    submodule (ggml only; libllama not built)
  Block-Sparse-Attention/       submodule (BSA + cutlass)
  bsa_stubs/                    PyTorch ATen/c10 header shims (see its README)
```
