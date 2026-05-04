# RTX 5090 DFlash/PFlash Tuning Ledger

Purpose: track local decisions for hosting Qwen-family models on an RTX 5090
32 GB system using Lucebox, llama.cpp/ggml kernels, DFlash decode, and PFlash
prefill. Keep entries concrete enough that a run can be reproduced.

## Operating Goals

- Prefer local model hosting through Lucebox/DFlash/PFlash, not an external
  agent runtime.
- Tune for RTX 5090 32 GB instead of the original RTX 3090 24 GB assumptions.
- Push context only as far as is reasonable with FP16 KV first; use quantized KV
  only when a target context cannot fit or speed/memory tradeoffs justify it.
- Track every benchmark with build commit, model files, runtime parameters, and
  quality/intelligence checks.
- Clarify with the user before destructive changes, long benchmark sweeps, or
  accuracy tasks that require choosing datasets/metrics.

## Current Local Baseline

Date: 2026-05-03

- Repo: `/home/jake/lucebox-hub`
- Commit: `69ebfb9`
- Build dir: `dflash/build-luce-sm120`
- CUDA: CUDA 13.0.88, ggml configured Blackwell arch as `120a`
- Target model: `/home/jake/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q5_K_XL.gguf`
- DFlash draft: `/home/jake/models/Qwen3.6-27B-DFlash-safetensors/model.safetensors`
- PFlash drafter: `/home/jake/models/Qwen3-0.6B-GGUF/Qwen3-0.6B-BF16.gguf`
- Default server profile: 64K context, FP16/FP16 KV, FA window 4096,
  DDTree budget 22, PFlash auto threshold 32K, keep ratio 0.05.
- Local launcher: `dflash/scripts/serve_dflash_luce.sh`
- Auto-update wrapper: `/home/jake/bin/lucebox`

## Change Log

| Date | Area | Change | Reason | Verification |
|---|---|---|---|---|
| 2026-05-03 | Workspace | Removed old Luce/Hermes folders and recloned `Luce-Org/lucebox-hub` into `/home/jake/lucebox-hub`. | Start fresh without Hermes-specific profile state. | `lucebox-update` succeeds; only `/home/jake/lucebox-hub` and Luce bin wrappers remain. |
| 2026-05-03 | Launcher | Added local `dflash/scripts/serve_dflash_luce.sh`. | Provide direct OpenAI-compatible Luce server launch using local models. | `bash -n` passes. |
| 2026-05-03 | Auto-update | Added `/home/jake/bin/lucebox-update` and `/home/jake/bin/lucebox`. | Pull latest repo/submodules before launch. | `/home/jake/bin/lucebox-update` succeeds at `69ebfb9`. |
| 2026-05-03 | Build | Built `test_dflash`, `test_flashprefill_kernels`, and `test_generate` in `build-luce-sm120`. | Prepare latest DFlash/PFlash and AR bench binaries for RTX 5090. | CMake build completed. |
| 2026-05-03 | Runtime baseline | Benchmarked Qwen3.6 Q5_K_XL with DFlash draft, FP16/FP16 KV, FA window 4096, DDTree budget 22. | Establish speed baseline on 5090 with the larger local target quant. | Full `bench_llm.py` summary recorded in `RTX5090_BENCHMARKS.md`. |
| 2026-05-03 | Results proposal | Added an RTX 5090 base-build section to `dflash/RESULTS.md` modeled after commit `f85bb3b`. | Prepare an upstream-style docs commit with headline, per-prompt tables, and DDTree sweep. | Clean `bench_llm.py` reproduction logged at `reports/rtx5090_base_build/bench_llm_qwen36_q5xl_budget22.log`; DDTree sweep logs in `reports/rtx5090_budget_sweep/`. |
| 2026-05-03 | GitHub | Opened draft PR #86 from `grearjake-star/lucebox-hub` to `Luce-Org/lucebox-hub`. | Publish the focused RTX 5090 `dflash/RESULTS.md` docs update without local launcher/build artifacts. | PR is open, draft, and mergeable: `https://github.com/Luce-Org/lucebox-hub/pull/86`. |
| 2026-05-03 | GitHub cleanup | Deleted old `grearjake-star` repos except the Lucebox fork. | Keep the GitHub account focused on the active Lucebox contribution. | Remaining repo list shows only `grearjake-star/lucebox-hub`. |

## Open Tuning Questions

- Maximum FP16/FP16 KV context that fits comfortably on 32 GB with target +
  DFlash draft + PFlash drafter resident.
- Best FA window for 64K and larger contexts on 5090.
- Best DDTree budget sweep for Qwen3.6 Q5_K_XL on 5090: budget 22 is the
  current short-context throughput default; budget 30 is a quality-biased
  experiment with higher AL but lower tok/s.
- Whether Q5_K_XL target quality gain is worth speed/memory cost versus Q4_K_M
  or other Unsloth Dynamic GGUF variants.
- Best PFlash threshold/keep ratio for long-context workloads at 64K, 96K,
  128K, and beyond.

## Repo Survey Notes

Completed by read-only helper agents on 2026-05-03.

Relevant existing references:

- `dflash/docs/SPEC_PREFILL.md`: strongest current DFlash/PFlash build and
  runtime note set.
- `pflash/README.md`: PFlash results and kernel/runtime tunables.
- `dflash/RESULTS.md`: best static DFlash result table.
- `dflash/scripts/bench_llm.py`: AR vs DFlash speed/AL harness, writes JSON to
  `/tmp/dflash_bench/bench_llm_results.json`.
- `dflash/scripts/bench_he.py`: fast HumanEval prompt speed/AL harness.
- `dflash/scripts/bench_daemon.py`: server/API timing harness.
- `pflash/tests/bench_niah_cpp.py`: long-context retrieval/PFlash benchmark.
- `dflash/scripts/phase_split_dual_gpu.py`: closest existing structured report
  writer, but focused on PFlash phase-split rather than full DFlash speed and
  quality.

Gaps:

- No persistent first-party benchmark ledger existed before this file and
  `RTX5090_BENCHMARKS.md`.
- Existing speed harnesses mostly print to stdout or `/tmp`, not a checked
  repo-local result source.
- Quality/intelligence scoring is not part of `bench_llm.py`; it should be
  added or handled by separate evaluators.
- Build metadata is scattered across CMake, docs, launcher env vars, and bench
  commands. Benchmark rows must normalize it.

Important tunables identified:

- Build: `CMAKE_CUDA_ARCHITECTURES`, `DFLASH27B_USER_CUDA_ARCHITECTURES`,
  `DFLASH27B_ENABLE_BSA`, `DFLASH27B_FA_ALL_QUANTS`.
- KV: `DFLASH27B_KV_K`, `DFLASH27B_KV_V`, legacy `DFLASH27B_KV_F16`,
  `DFLASH27B_KV_Q4`, `DFLASH27B_KV_TQ3`.
- Context: `--max-ctx`; direct `test_dflash` default is 4096, upstream servers
  default to 16384, local Luce launcher defaults to 65536.
- Attention: `DFLASH27B_FA_WINDOW` / `--fa-window`; `0` means full attention.
- DFlash: `--ddtree-budget`, `--ddtree-temp`, `--ddtree-no-chain-seed`.
- PFlash server: `--prefill-compression`, `--prefill-threshold`,
  `--prefill-keep-ratio`, `--prefill-drafter`, `DFLASH_FP_USE_BSA`,
  `DFLASH_FP_ALPHA`.
- PFlash kernel/scoring internals: `block_size=128`, `attention_sink=2`,
  `window=4`, `last_n_full=2`, `chunk_size=32`, `n_lookahead=8`,
  `pool_kernel=13`.

5090-specific caution:

- Upstream docs and many defaults are shaped by RTX 3090 24 GB.
- For RTX 5090, use explicit `-DCMAKE_CUDA_ARCHITECTURES=120` or a build may
  include legacy SMs that disable BSA.
- The local launcher passes explicit FP16/FP16 KV; direct upstream server runs
  may still auto-enable TQ3 at larger contexts unless given explicit KV flags.
- DDTree budget 22 is the current short-context throughput default for
  Qwen3.6 Q5_K_XL on this 5090 build. Re-sweep if target quant, FA window,
  context length, or draft model changes.
- PFlash park/unpark memory behavior may be unnecessary or suboptimal on 32 GB;
  revalidate target + drafter residency.

## Agent / Automation Notes

- Use helper agents for independent repo surveys, benchmark result parsing, and
  doc-format inspection when they can run in parallel with local verification.
- Keep agents read-only unless assigning a narrow file ownership scope.
- Before upstream PRs, stage only the reviewed upstream file set. Current PR #86
  intentionally includes only `dflash/RESULTS.md`.
- Next useful automation is a 5090 profile runner that emits a manifest with
  build metadata, model paths/sizes, GPU metadata, benchmark command lines,
  speed KPIs, and raw log paths.

## Required Benchmark Record Fields

Every benchmark entry should include:

- Date/time
- Git commit and dirty state
- Build directory and CUDA arch
- Target/draft/prefill model paths
- Max context, actual prompt length range, KV K/V type, FA window
- DDTree budget and mode
- PFlash settings if enabled
- Dataset/task and sample count
- Speed KPIs: AR tok/s, DFlash tok/s, speedup, AL, accept %, TTFT if server run
- Quality KPIs: exact match/pass@k/answer accuracy/judged quality, depending on
  task
- Notes on failures, OOMs, warnings, and thermal/power context when relevant

## 2026-05-03 Kernel Fusion Prototype Ledger

Goal: reduce CPU load and launch/copy overhead in the DFlash/PFlash hot paths
without changing decode output.

Implemented opt-in prototypes:

- `DFLASH27B_GPU_DDTREE_TOPK=1`: GPU draft top-K/logprob extraction for DDTree
  K=8. This removes the full `[q_len * vocab]` logits transfer and host top-K
  scan in the DDTree sibling path.
- `DFLASH27B_GPU_DDTREE_ROLLBACK=1`: GPU accepted-path compaction for
  `target_feat` and full-attention K/V cache slots, replacing many tiny host
  issued device-to-device copies.
- `DFLASH27B_GPU_DDTREE_PREP=1`: GPU fill for DDTree verify positions,
  `parent_ids`, and ancestor mask. Tree construction and token embedding remain
  CPU-side.
- `DFLASH_FP_FUSED_SELECT=1`: PFlash fused score/select prototype that avoids
  materializing the full block score/max scratch before sparse FlashPrefill.

Validation completed:

- `test_dflash` built successfully in `dflash/build-luce-sm120`.
- DDTree output files compared byte-identical against baseline for top-K,
  rollback, verify prep, and all flags together on one HumanEval prompt.
- `test_flashprefill_kernels` passed in default and fused-select modes.
- `git diff --check` passed.

Performance decision:

- Top-K is the clearest win: `draft_logits` dropped from 2.53 ms to 0.46 ms and
  throughput rose from 223.75 tok/s to 251.20 tok/s on the focused prompt.
- Rollback is correct and mildly positive in isolation; it should be retested on
  larger accept paths where copy-loop overhead is more visible.
- Verify prep is correct but not yet a default win on the small tree; keep it
  opt-in until larger tree/window testing proves it.
- PFlash fused select passed and improved the focused kernel test from 5.9 ms to
  5.0 ms per iteration at S=8192.

Next test pass:

- Run a 10-prompt HumanEval speed/AL sweep with top-K alone and top-K+rollback.
- Add a longer-context PFlash run with `DFLASH_FP_FUSED_SELECT=1` and profile
  memory scratch reduction.
- Add accuracy/intelligence checks before making any fusion flag default.

## 2026-05-03 Default GPU Top-K + Rollback Sweep

Decision: make `DFLASH27B_GPU_DDTREE_TOPK` and
`DFLASH27B_GPU_DDTREE_ROLLBACK` default-on in `test_dflash`; `=0` still disables
either path. Keep `DFLASH27B_GPU_DDTREE_PREP` default-off because the focused
combined run was correct but slower than top-K + rollback alone.

Full `bench_llm.py` speed/AL sweep completed after the default flip:

| Task | AR tok/s | DFlash tok/s | AL | Speedup |
|---|---:|---:|---:|---:|
| HumanEval | 57.76 | 230.95 | 7.12 | 4.00x |
| GSM8K | 58.04 | 190.93 | 5.88 | 3.29x |
| Math500 | 58.55 | 237.01 | 7.31 | 4.05x |

Compared with the previous clean base-build reproduction, DFlash throughput
improved by +12.72 tok/s on HumanEval, +11.86 tok/s on GSM8K, and +17.95 tok/s
on Math500, with unchanged AL on this sweep.

Raw artifacts:

- `reports/rtx5090_kernel_fusion/bench_llm_default_topk_rollback.log`
- `reports/rtx5090_kernel_fusion/bench_llm_default_topk_rollback.json`
