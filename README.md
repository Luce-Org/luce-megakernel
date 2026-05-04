# Lucebox Hub

Minimal local-inference workspace for Jake's Lucebox builds.

## Contents

- `dflash/` - Qwen3.5/3.6 27B GGUF + DFlash/PFlash runtime.

## Current Default

The active 5090 path is in `dflash/`:

- Target: Qwen3.6 27B GGUF, `Q5_K_XL`.
- Draft: Qwen3.6 27B DFlash safetensors.
- KV cache: FP16/FP16.
- Context target: 64K for the server path.
- DDTree budget: 22.
- GPU DDTree top-K: default on.
- GPU DDTree rollback/compaction: default on.
- GPU DDTree verify prep: opt-in only.

## Build

```bash
cd dflash
cmake -B build-luce-sm120 -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DDFLASH27B_USER_CUDA_ARCHITECTURES=120 \
  -DDFLASH27B_ENABLE_BSA=ON
cmake --build build-luce-sm120 --target test_dflash test_generate test_flashprefill_kernels -j 8
```

## Bench

```bash
cd dflash
DFLASH_TARGET=/home/jake/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q5_K_XL.gguf \
DFLASH_DRAFT=/home/jake/models/Qwen3.6-27B-DFlash-safetensors/model.safetensors \
DFLASH_BIN=/home/jake/lucebox-hub/dflash/build-luce-sm120/test_dflash \
DFLASH_BIN_AR=/home/jake/lucebox-hub/dflash/build-luce-sm120/test_generate \
DFLASH27B_KV_K=f16 \
DFLASH27B_KV_V=f16 \
DFLASH27B_FA_WINDOW=4096 \
python scripts/bench_llm.py
```

## Notes

- Keep model weights outside this repo under `/home/jake/models`.
- `dflash/deps/llama.cpp` and `dflash/deps/Block-Sparse-Attention` are submodules.
- Generated build folders, virtualenvs, raw logs, and local launch scripts should stay untracked.
