# RTX 5090 Benchmark Sheet

This sheet records local 5090 Lucebox benchmark runs. Add new rows rather than
overwriting old ones.

## Speed Benchmarks

| Date | Commit | Build | Target | Draft | Context / KV / FA | Bench | Samples | AR tok/s | DFlash tok/s | Speedup | AL | Notes |
|---|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| 2026-05-03 | `69ebfb9` | `build-luce-sm120`, CUDA arch `120a` | `Qwen3.6-27B-UD-Q5_K_XL.gguf` | `Qwen3.6-27B-DFlash model.safetensors` | prompt auto-fit, FP16/FP16 KV, FA window 4096 | `bench_he.py --n-gen 128 --ddtree-budget 22` | 10 HumanEval prompts | n/a | 203.42 | n/a | 7.25 | Accept 45.3%, tok/s range 149.0-252.9. |
| 2026-05-03 | `69ebfb9` | `build-luce-sm120`, CUDA arch `120a` | `Qwen3.6-27B-UD-Q5_K_XL.gguf` | `Qwen3.6-27B-DFlash model.safetensors` | prompt auto-fit, FP16/FP16 KV, FA window 4096 | `bench_llm.py` HumanEval | 10 | 56.55 | 211.68 | 3.74x | 7.12 | Full sweep JSON: `/tmp/dflash_bench/bench_llm_results.json`. |
| 2026-05-03 | `69ebfb9` | `build-luce-sm120`, CUDA arch `120a` | `Qwen3.6-27B-UD-Q5_K_XL.gguf` | `Qwen3.6-27B-DFlash model.safetensors` | prompt auto-fit, FP16/FP16 KV, FA window 4096 | `bench_llm.py` GSM8K | 10 | 56.65 | 175.71 | 3.10x | 5.88 | Speed-only run; no answer accuracy scored. |
| 2026-05-03 | `69ebfb9` | `build-luce-sm120`, CUDA arch `120a` | `Qwen3.6-27B-UD-Q5_K_XL.gguf` | `Qwen3.6-27B-DFlash model.safetensors` | prompt auto-fit, FP16/FP16 KV, FA window 4096 | `bench_llm.py` Math500 | 10 | 56.95 | 217.23 | 3.81x | 7.31 | Speed-only run; no answer accuracy scored. |
| 2026-05-03 | `69ebfb9` | `build-luce-sm120`, CUDA arch `120a` | `Qwen3.6-27B-UD-Q5_K_XL.gguf` | `Qwen3.6-27B-DFlash model.safetensors` | prompt auto-fit, FP16/FP16 KV, FA window 4096 | `bench_llm.py` HumanEval | 10 | 58.25 | 218.23 | 3.75x | 7.12 | Clean reproduction for proposed `dflash/RESULTS.md` entry; log: `reports/rtx5090_base_build/bench_llm_qwen36_q5xl_budget22.log`. |
| 2026-05-03 | `69ebfb9` | `build-luce-sm120`, CUDA arch `120a` | `Qwen3.6-27B-UD-Q5_K_XL.gguf` | `Qwen3.6-27B-DFlash model.safetensors` | prompt auto-fit, FP16/FP16 KV, FA window 4096 | `bench_llm.py` GSM8K | 10 | 58.39 | 179.07 | 3.07x | 5.88 | Clean reproduction for proposed `dflash/RESULTS.md` entry; speed-only run. |
| 2026-05-03 | `69ebfb9` | `build-luce-sm120`, CUDA arch `120a` | `Qwen3.6-27B-UD-Q5_K_XL.gguf` | `Qwen3.6-27B-DFlash model.safetensors` | prompt auto-fit, FP16/FP16 KV, FA window 4096 | `bench_llm.py` Math500 | 10 | 57.57 | 219.06 | 3.80x | 7.31 | Clean reproduction for proposed `dflash/RESULTS.md` entry; speed-only run. |
| 2026-05-03 | `8dec2ec` dirty | `build-luce-sm120`, CUDA arch `120a` | `Qwen3.6-27B-UD-Q5_K_XL.gguf` | `Qwen3.6-27B-DFlash model.safetensors` | prompt auto-fit, FP16/FP16 KV, FA window 4096, GPU DDTree top-K + rollback default | `bench_llm.py` HumanEval | 10 | 57.76 | 230.95 | 4.00x | 7.12 | Log: `reports/rtx5090_kernel_fusion/bench_llm_default_topk_rollback.log`; JSON copied beside it. |
| 2026-05-03 | `8dec2ec` dirty | `build-luce-sm120`, CUDA arch `120a` | `Qwen3.6-27B-UD-Q5_K_XL.gguf` | `Qwen3.6-27B-DFlash model.safetensors` | prompt auto-fit, FP16/FP16 KV, FA window 4096, GPU DDTree top-K + rollback default | `bench_llm.py` GSM8K | 10 | 58.04 | 190.93 | 3.29x | 5.88 | Speed-only run; no answer accuracy scored. |
| 2026-05-03 | `8dec2ec` dirty | `build-luce-sm120`, CUDA arch `120a` | `Qwen3.6-27B-UD-Q5_K_XL.gguf` | `Qwen3.6-27B-DFlash model.safetensors` | prompt auto-fit, FP16/FP16 KV, FA window 4096, GPU DDTree top-K + rollback default | `bench_llm.py` Math500 | 10 | 58.55 | 237.01 | 4.05x | 7.31 | Speed-only run; no answer accuracy scored. |

## Proposed RTX 5090 Base Build

This is the first local 5090 base entry for `dflash/RESULTS.md`. It is
published as draft PR #86:
`https://github.com/Luce-Org/lucebox-hub/pull/86`.

Power and thermal notes are not captured yet; add them in a follow-up benchmark
rather than changing this base entry.

- Hardware: NVIDIA GeForce RTX 5090, 32 GB VRAM, driver 595.58.03.
- CUDA/toolchain: CUDA 13.0.88, CMake Release.
- Repo: `Luce-Org/lucebox-hub` commit `69ebfb9`; remote `origin/main` matched
  local `HEAD` at sweep time.
- Build: `dflash/build-luce-sm120`; configured with
  `-DCMAKE_CUDA_ARCHITECTURES=120`,
  `-DDFLASH27B_USER_CUDA_ARCHITECTURES=120`,
  `-DDFLASH27B_ENABLE_BSA=ON`; ggml emitted arch `120a`.
- Target: `Qwen3.6-27B-UD-Q5_K_XL.gguf` from
  `unsloth/Qwen3.6-27B-GGUF`, 19 GB local file.
- Draft: `Qwen3.6-27B-DFlash model.safetensors`, 3.3 GB local file.
- Runtime: concurrency 1, greedy decode, DDTree budget 22, FP16/FP16 KV,
  FA window 4096.
- Server profile: 64K max context, PFlash auto threshold 32K, keep ratio 0.05.

Headline candidate:

| Task | AR tok/s | DFlash tok/s | AL | Speedup |
|---|---:|---:|---:|---:|
| HumanEval | 57.76 | 230.95 | 7.12 | 4.00x |
| Math500 | 58.55 | 237.01 | 7.31 | 4.05x |
| GSM8K | 58.04 | 190.93 | 5.88 | 3.29x |

Suggested upstream wording:

```md
## RTX 5090 (Blackwell, sm_120/sm_120a, 32 GB)

Single RTX 5090 32 GB, CUDA 13.0.88, driver 595.58.03.
Target: `unsloth/Qwen3.6-27B-GGUF` (`Qwen3.6-27B-UD-Q5_K_XL.gguf`, ~19 GB).
Draft:  local Qwen3.6-27B DFlash safetensors (`model.safetensors`, ~3.3 GB).
Concurrency = 1, greedy decoding, `n_gen=128`.
Build: `cmake -B build-luce-sm120 -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DDFLASH27B_USER_CUDA_ARCHITECTURES=120 -DDFLASH27B_ENABLE_BSA=ON`
Runtime: FP16/FP16 KV, FA window 4096, DDTree budget 22, GPU DDTree top-K and
rollback enabled by default.

### RTX 5090 headline

| Task      | AR tok/s | DFlash tok/s | AL   | Speedup |
|-----------|:--------:|:------------:|:----:|:-------:|
| HumanEval | 57.76    | **230.95**   | 7.12 | **4.00x** |
| Math500   | 58.55    | **237.01**   | 7.31 | **4.05x** |
| GSM8K     | 58.04    | **190.93**   | 5.88 | **3.29x** |
```

## DDTree Budget Sweep

Fast HumanEval sweep, 10 prompts, `n_gen=128`, Qwen3.6 Q5_K_XL target,
Qwen3.6 DFlash draft, FP16/FP16 KV, FA window 4096. Raw logs:
`reports/rtx5090_budget_sweep/budget_*.log`.

| Budget | Mean AL | Accept % | Mean tok/s | Notes |
|---:|---:|---:|---:|---|
| 12 | n/a | n/a | n/a | Failed all prompts with `GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b))`. |
| 15 | 4.99 | 31.2 | 174.45 | Too low; weak acceptance. |
| 16 | 5.76 | 36.0 | 176.98 | Useful long-context fallback, not best short-context speed. |
| 18 | 6.93 | 43.3 | 206.62 | Near plateau. |
| 20 | 6.94 | 43.4 | 204.03 | Near plateau. |
| **22** | **7.25** | **45.3** | **211.20** | Best throughput in this sweep; keep as base default. |
| 24 | 7.19 | 44.9 | 203.08 | No improvement over 22. |
| 26 | 7.09 | 44.3 | 199.96 | Slower. |
| 30 | 7.44 | 46.5 | 206.19 | Highest AL, but still slower than 22. |
| 32 | 6.87 | 42.9 | 183.34 | Regression. |
| 40 | 6.97 | 43.6 | 174.52 | Too much verify overhead. |
| 48 | 7.07 | 44.2 | 165.24 | Too much verify overhead. |
| 64 | 7.14 | 44.6 | 148.12 | Too much verify overhead. |

Conclusion: budget 22 remains the 5090 short-context throughput default for
this Qwen3.6 Q5_K_XL build. Budget 30 can be kept as an experimental
quality-biased knob because it produced the highest mean AL, but it is not the
base speed setting.

## Build / Bench / Test Workflow

Use this loop for future 5090 work:

1. Build from the current Lucebox commit with explicit Blackwell arch flags.
2. Run a clean `bench_llm.py` reproduction and save the full stdout log.
3. Run focused sweeps for the parameter being changed, one axis at a time.
4. Record raw logs under `reports/`, summary rows in this file, and decisions
   in `RTX5090_TUNING_LEDGER.md`.
5. Promote only stable, upstream-worthy benchmark sections into
   `dflash/RESULTS.md`.
6. Keep local launchers, build dirs, raw logs, and scratch patch files out of
   upstream docs commits.

Next automation target: add a checked-in or local `bench_5090_profile.sh` that
captures GPU metadata, build flags, model paths/sizes, `bench_llm.py`, DDTree
sweep, and a machine-readable JSON/CSV manifest in one run.

## Accuracy / Intelligence Benchmarks

No scored accuracy run has been completed yet in this fresh Lucebox workspace.
The current `bench_llm.py` measures throughput and AL only. Accuracy should be
tracked separately so decode speed changes do not hide quality regressions.

Recommended first checks:

- HumanEval correctness: run generated completions through unit tests or an
  existing HumanEval evaluator.
- GSM8K exact match: extract final numeric answer and compare to dataset answer.
- Math500 exact match / boxed-answer match: extract final answer and compare to
  dataset target.
- Long-context retrieval: PFlash NIAH-style benchmark from `pflash/tests`.

## Command Log

Latest speed run:

```bash
DFLASH_TARGET=/home/jake/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q5_K_XL.gguf \
DFLASH_DRAFT=/home/jake/models/Qwen3.6-27B-DFlash-safetensors/model.safetensors \
DFLASH_BIN=/home/jake/lucebox-hub/dflash/build-luce-sm120/test_dflash \
DFLASH_BIN_AR=/home/jake/lucebox-hub/dflash/build-luce-sm120/test_generate \
DFLASH27B_KV_K=f16 \
DFLASH27B_KV_V=f16 \
DFLASH27B_FA_WINDOW=4096 \
/home/jake/lucebox-hub/dflash/.venv/bin/python scripts/bench_llm.py
```

## Kernel Fusion Prototypes

Date: 2026-05-03 22:41 EDT. Branch: `codex/kernel-fusion-prototypes`.
Target: `Qwen3.6-27B-UD-Q5_K_XL.gguf`. Draft:
`Qwen3.6-27B-DFlash-safetensors/model.safetensors`. Prompt:
`/tmp/dflash_bench/b_HumanEval_00.bin`, `n_gen=128`, `--max-ctx=512`,
`--ddtree --ddtree-budget=22`, FP16/FP16 KV, FA window 4096.

Focused DDTree runs, all compared byte-identical to the CPU/staged baseline:

| Flags | draft_logits ms | verify_set ms | sum ms | tok/s | AL | Notes |
|---|---:|---:|---:|---:|---:|---|
| baseline | 2.53 | 0.17 | 35.41 | 223.75 | 8.00 | CPU full-logits top-K and CPU staged verify inputs. |
| `DFLASH27B_GPU_DDTREE_TOPK=1` | 0.46 | 0.11 | 31.53 | 251.20 | 8.00 | Best isolated gain; avoids full draft logits host copy/scan. |
| `DFLASH27B_GPU_DDTREE_ROLLBACK=1` | 2.54 | 0.12 | 34.35 | 231.47 | 8.00 | Correct isolated rollback/compaction helper. |
| `DFLASH27B_GPU_DDTREE_PREP=1` | 2.51 | 0.12 | 34.93 | 226.81 | 8.00 | Correct, modest isolated gain on this small tree. |
| top-K + rollback | 0.47 | 0.10 | 31.61 | 251.44 | 8.00 | Best recommended DDTree prototype combination from this sample. |
| top-K + rollback + prep | 0.51 | 0.10 | 32.54 | 244.27 | 8.00 | Correct but slower than top-K + rollback here; keep opt-in. |

FlashPrefill fused score/select focused test:

| Flags | Result | e2e flash_prefill_forward_bf16 at S=8192 |
|---|---|---:|
| default | PASS | 5.9 ms / iter |
| `DFLASH_FP_FUSED_SELECT=1` | PASS | 5.0 ms / iter |

Current recommendation: default GPU DDTree top-K and rollback on, while keeping
`DFLASH27B_GPU_DDTREE_TOPK=0` and `DFLASH27B_GPU_DDTREE_ROLLBACK=0` as escape
hatches. Keep `DFLASH27B_GPU_DDTREE_PREP=1` experimental until it wins on larger
DDTree budgets or longer FA windows.
