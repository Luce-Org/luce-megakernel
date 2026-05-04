# Luce DFlash

Bare runtime for Qwen3.5/3.6 27B GGUF target inference with DFlash speculative decoding and PFlash prefill support.

## 5090 Defaults

- Build arch: `sm_120`.
- Target: `/home/jake/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q5_K_XL.gguf`.
- DFlash draft: `/home/jake/models/Qwen3.6-27B-DFlash-safetensors/model.safetensors`.
- PFlash drafter: `/home/jake/models/Qwen3-0.6B-GGUF/Qwen3-0.6B-BF16.gguf`.
- KV cache: `DFLASH27B_KV_K=f16`, `DFLASH27B_KV_V=f16`.
- FA window: `DFLASH27B_FA_WINDOW=4096`.
- DDTree budget: `22`.
- GPU DDTree top-K and rollback are default-on.

Disable the new GPU paths only for debugging:

```bash
DFLASH27B_GPU_DDTREE_TOPK=0
DFLASH27B_GPU_DDTREE_ROLLBACK=0
```

`DFLASH27B_GPU_DDTREE_PREP=1` exists, but is not default because it did not win the combined 5090 run.

## Build

```bash
cmake -B build-luce-sm120 -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DDFLASH27B_USER_CUDA_ARCHITECTURES=120 \
  -DDFLASH27B_ENABLE_BSA=ON
cmake --build build-luce-sm120 --target test_dflash test_generate test_flashprefill_kernels -j 8
```

## Bench

```bash
DFLASH_TARGET=/home/jake/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q5_K_XL.gguf \
DFLASH_DRAFT=/home/jake/models/Qwen3.6-27B-DFlash-safetensors/model.safetensors \
DFLASH_BIN=/home/jake/lucebox-hub/dflash/build-luce-sm120/test_dflash \
DFLASH_BIN_AR=/home/jake/lucebox-hub/dflash/build-luce-sm120/test_generate \
DFLASH27B_KV_K=f16 \
DFLASH27B_KV_V=f16 \
DFLASH27B_FA_WINDOW=4096 \
python scripts/bench_llm.py
```

Current headline numbers are in `RESULTS.md`.
