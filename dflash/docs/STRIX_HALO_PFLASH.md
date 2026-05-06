# Strix Halo pflash / DFlash Runbook

This document records the Strix Halo ROCm setup used for the Qwen3.5-27B target
and small C++ compression drafters.

## Repository Layout

- Main repo: `/home/sam/projects/repos/lucebox-hub`
- DFlash daemon and C++ graph code: `dflash/`
- Python pflash benchmark/client code: `pflash/`
- Vendored llama.cpp fork: `dflash/deps/llama.cpp`
- Out-of-tree HIP build used during testing: `/tmp/dflash_build`

## Models

The current local model paths are:

- Target GGUF:
  `/home/sam/projects/models/llm/Qwen3.5-27B-Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf`
- DFlash draft weights:
  `/home/sam/projects/models/llm/Qwen3.5-27B-DFlash/model.safetensors`
- Qwen3.5 0.8B compression drafter:
  `/home/sam/projects/models/llm/qwen3.5-0.8b/Qwen3.5-0.8B-BF16.gguf`
- Qwen3 0.6B compression drafter:
  `/home/sam/projects/models/llm/qwen3-0.6b/Qwen3-0.6B-BF16.gguf`

Download commands, using the existing pflash venv:

```bash
cd /home/sam/projects/repos/lucebox-hub/pflash
source .venv/bin/activate

python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download

base = Path('/home/sam/projects/models/llm')
items = [
    ('unsloth/Qwen3.5-27B-GGUF', 'Qwen3.5-27B-Q4_K_M.gguf', base / 'Qwen3.5-27B-Q4_K_M'),
    ('z-lab/Qwen3.5-27B-DFlash', 'model.safetensors', base / 'Qwen3.5-27B-DFlash'),
    ('unsloth/Qwen3.5-0.8B-GGUF', 'Qwen3.5-0.8B-BF16.gguf', base / 'qwen3.5-0.8b'),
    ('unsloth/Qwen3-0.6B-GGUF', 'Qwen3-0.6B-BF16.gguf', base / 'qwen3-0.6b'),
]
for repo, filename, outdir in items:
    outdir.mkdir(parents=True, exist_ok=True)
    print(hf_hub_download(repo_id=repo, filename=filename, local_dir=str(outdir)))
PY
```

## Build

Use an out-of-tree build. CMake on this machine needs the HIP compiler and ROCm
root passed explicitly.

```bash
rm -rf /tmp/dflash_build
mkdir -p /tmp/dflash_build
cd /tmp/dflash_build

cmake /home/sam/projects/repos/lucebox-hub/dflash \
  -D DFLASH27B_USE_HIP=ON \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -D CMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm

cmake --build . --target test_dflash -j8
```

Expected success line:

```text
[100%] Built target test_dflash
```

The llama.cpp submodule must be on the luce-dflash branch/commit that contains
TQ3, Turbo WHT, and DeltaNet ops. The working commit used here is:

```text
ce3919b4a Merge pull request #5 from Luce-Org/feat/turbo-wht-parallel
```

## Python Environment

Use the existing pflash venv:

```bash
cd /home/sam/projects/repos/lucebox-hub/pflash
source .venv/bin/activate
python - <<'PY'
import torch, transformers
print(torch.__version__)
print(torch.cuda.is_available())
print(transformers.__version__)
PY
```

On this machine the venv has `torch 2.11.0+rocm7.2` and sees the Strix Halo iGPU
through `torch.cuda.is_available() == True`.

## llama-cli Wrapper

The current llama.cpp `llama-cli` rejects `-no-cnv` for Qwen3.5 chat mode and
prints banners/prompt text. The benchmark used this cleaning wrapper:

```bash
/tmp/opencode/llama_cli_clean.py
```

It invokes:

```text
/home/sam/projects/repos/lucebox-hub/dflash/deps/llama.cpp/build-hip/bin/llama-cli
```

and strips the interactive prompt/banner so benchmark scoring sees only the
model output.

## Generate A 16K RULER-Style Case

The selected diagnostic task is a recall-only common-words aggregation case:

```bash
cd /home/sam/projects/repos/lucebox-hub/pflash
source .venv/bin/activate

python tests/common_words_gen.py \
  --n 1 \
  --ctx 18000 \
  --lists 12 \
  --list-size 24 \
  --common 3 \
  --decoys 6 \
  --out /tmp/opencode/ruler_common_words_easy_16k_1.jsonl \
  --tokenizer Qwen/Qwen3.5-27B

python - <<'PY'
import json
src = '/tmp/opencode/ruler_common_words_easy_16k_1.jsonl'
dst = '/tmp/opencode/ruler_common_words_easy_16k_recall_only.jsonl'
row = json.loads(open(src).readline())
row.pop('forbidden', None)
open(dst, 'w').write(json.dumps(row) + '\n')
print(dst, row.get('n_tokens'), row['answer'])
PY
```

This specific case has expected answer:

```text
lichen, fennel, juniper
```

## Run 27B Null / Control

```bash
cd /home/sam/projects/repos/lucebox-hub/pflash
source .venv/bin/activate

python tests/bench_ruler64_compare.py \
  --cases /tmp/opencode/ruler_common_words_easy_16k_recall_only.jsonl \
  --out /tmp/opencode/common_words_easy_recall_null.jsonl \
  --modes null \
  --n 1 \
  --llama-cli /tmp/opencode/llama_cli_clean.py \
  --model /home/sam/projects/models/llm/Qwen3.5-27B-Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf \
  --exact-bin /tmp/dflash_build/test_dflash \
  --draft-spec /home/sam/projects/models/llm/Qwen3.5-27B-DFlash/model.safetensors \
  --drafter-gguf /home/sam/projects/models/llm/qwen3.5-0.8b/Qwen3.5-0.8B-BF16.gguf \
  --drafter-arch qwen35-0.8b \
  --target-tokenizer Qwen/Qwen3.5-27B \
  --drafter-tokenizer Qwen/Qwen3.5-0.8B \
  --keep-ratio 1.0 \
  --ctx-size 16384 \
  --exact-max-ctx 16384 \
  --n-gen 96
```

## Run Qwen3.5 0.8B Compression Drafter

10x nominal compression:

```bash
python tests/bench_ruler64_compare.py \
  --cases /tmp/opencode/ruler_common_words_easy_16k_recall_only.jsonl \
  --out /tmp/opencode/common_words_easy_recall_qwen35_0p8b_10x.jsonl \
  --modes exact \
  --n 1 \
  --llama-cli /tmp/opencode/llama_cli_clean.py \
  --model /home/sam/projects/models/llm/Qwen3.5-27B-Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf \
  --exact-bin /tmp/dflash_build/test_dflash \
  --draft-spec /home/sam/projects/models/llm/Qwen3.5-27B-DFlash/model.safetensors \
  --drafter-gguf /home/sam/projects/models/llm/qwen3.5-0.8b/Qwen3.5-0.8B-BF16.gguf \
  --drafter-arch qwen35-0.8b \
  --target-tokenizer Qwen/Qwen3.5-27B \
  --drafter-tokenizer Qwen/Qwen3.5-0.8B \
  --keep-ratio 0.10 \
  --ctx-size 16384 \
  --exact-max-ctx 16384 \
  --n-gen 96
```

3x nominal compression: change only `--keep-ratio` to `0.333333`.

## Run Qwen3 0.6B Compression Drafter

10x nominal compression:

```bash
python tests/bench_ruler64_compare.py \
  --cases /tmp/opencode/ruler_common_words_easy_16k_recall_only.jsonl \
  --out /tmp/opencode/common_words_easy_recall_qwen3_0p6b_10x.jsonl \
  --modes exact \
  --n 1 \
  --llama-cli /tmp/opencode/llama_cli_clean.py \
  --model /home/sam/projects/models/llm/Qwen3.5-27B-Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf \
  --exact-bin /tmp/dflash_build/test_dflash \
  --draft-spec /home/sam/projects/models/llm/Qwen3.5-27B-DFlash/model.safetensors \
  --drafter-gguf /home/sam/projects/models/llm/qwen3-0.6b/Qwen3-0.6B-BF16.gguf \
  --drafter-arch qwen3-0.6b \
  --target-tokenizer Qwen/Qwen3.5-27B \
  --drafter-tokenizer Qwen/Qwen3-0.6B \
  --keep-ratio 0.10 \
  --ctx-size 16384 \
  --exact-max-ctx 16384 \
  --n-gen 96
```

3x nominal compression: change only `--keep-ratio` to `0.333333`.

## What Changed

- The Qwen3.5 0.8B compression path now always applies head/tail retention,
  AvgPool score smoothing, final-query n-gram anchors, repeated-token anchors,
  and span merge. Repeated-token anchors are capped by
  `DFLASH_COMPRESS_REPEAT_CHUNKS` and default to the nominal selected chunk
  count, so they reprioritize selection without silently reducing compression.
- Qwen3 0.6B ROCm compression now defaults FlashPrefill to `alpha=0.95` on HIP,
  which keeps fewer K blocks during sparse forward and cuts compression time on
  the selected 16K task from roughly 58 seconds to roughly 13 seconds.
- The llama.cpp fork has HIP compatibility fixes needed by the luce-dflash
  branch: `cublasSgemmStridedBatched`, stream-capture aliases, and guarded
  `cuda_fp16.h` include for HIP.
- The dflash HIP runtime wrapper has additional CUDA-to-HIP aliases needed by
  `test_dflash.cpp`.

## Current Results On The Selected 16K Case

| run | score | compression | total |
|---|---:|---:|---:|
| 27B null/control | 100% | 1.00x | ~70s |
| Qwen3.5-0.8B 10x | 0% | ~10.06x | ~15s |
| Qwen3.5-0.8B 3x | 100% | ~3.01x | ~32s |
| Qwen3-0.6B 10x | 66.7% | ~10.01x | ~24s after HIP alpha tuning |
| Qwen3-0.6B 3x | 100% | ~3.01x | ~39s after HIP alpha tuning |

## KV-Cache / Conversation Options

Prompt compression invalidates the target model's normal KV-cache continuity:
the compressed prompt is not a prefix of the original conversation, so cached KV
for the uncompressed conversation cannot be directly reused after compression.

Practical options:

1. Cache compression outputs by prefix hash.
   - Store compressed token IDs plus metadata: source hash, compression ratio,
     drafter model, tokenizer, env knobs, and target model.
   - Best for repeated system prompts, tool schemas, or retrieved documents.
   - Safest and easiest; no model-internal state coupling.

2. Cache drafter scoring state for stable prefixes.
   - For Qwen3.5, cache completed layer activations/KV for a prefix inside the
     compression drafter, then score only appended conversation turns.
   - More complex because the selected chunks can change when the final query
     changes. Needs invalidation keyed on query tail and compression knobs.

3. Cache target KV only after compression.
   - Once a compressed prompt is generated, run target prefill once and cache KV
     for that compressed prompt.
   - Useful if multiple completions are sampled from the same compressed prompt.
   - Not useful when every user turn changes the compressed prompt.

4. Segment-stable compression.
   - Split conversation into immutable blocks: system, tools, memory, retrieved
     docs, conversation turns.
   - Compress/cache immutable blocks independently and append recent raw turns.
   - This is likely the best production direction for chat/agent use.

5. Prefix snapshot restoration in DFlash.
   - The repository has declarations and protocol hooks for prefix snapshots,
     but the current C++ snapshot functions are stubs in this branch. A complete
     implementation would snapshot both full-attention KV and DeltaNet recurrent
     state at compressed-prompt boundaries.

Recommended near-term path: implement output-token compression cache first, then
add segment-stable compression for system/tools/retrieval blocks. Avoid trying to
reuse uncompressed target KV across compression boundaries.
