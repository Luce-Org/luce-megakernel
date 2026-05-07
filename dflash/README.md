<p align="left">
  <a href="../README.md">← lucebox-hub</a>
</p>

<p align="center">
  <img src="hero.png" width="600" />
</p>

<h1 align="center">Luce DFlash</h1>

<p align="center">
  <strong>The first GGUF port of DFlash speculative decoding.</strong><br/>
  Qwen3.5-27B at up to 207 tok/s<sup>*</sup> on a single RTX 3090 (HumanEval 10-prompt bench mean 129.5 tok/s at DDTree budget=22). 128K context on 24 GB.<br/>
  3.43× faster than autoregressive (+15% over chain spec decoding), 2.8× faster than SGLang AWQ.<br/>
  <sub><sup>*</sup>Demo run: 207.6 tok/s DFlash vs 38.0 tok/s AR (5.46×).</sub><br/><br/>
  <a href="https://lucebox.com/blog/dflash27b">Blog post</a> · <a href="RESULTS.md">Benchmarks</a> · <a href="https://discord.gg/yHfswqZmJQ">Discord</a> · <a href="https://lucebox.com">lucebox.com</a>
</p>

<p align="center">
  <img src="demo.gif" width="600" />
</p>

---

```
                   AR (tok/s)   DFlash (tok/s)   Speedup
HumanEval             37.78        129.52          3.43x
Math500               37.71        110.51          2.93x
GSM8K                 37.65         96.15          2.55x
```

> Consumer GPUs can run 27B models at chat-grade speed without multi-GPU, without batching, without quantization compromises. The bottleneck was never hardware. It was the decoding algorithm.

## The gap we filled

On a 24 GB RTX 3090 with Q4_K_M weights, autoregressive decode of Qwen3.5-27B hits ~37.7 tok/s regardless of framework. Every token reads the full model from VRAM.

Speculative decoding breaks that ceiling: a tiny draft proposes multiple tokens per step, the target verifies them in one forward. [DFlash (z-lab, 2026)](https://arxiv.org/abs/2602.06036) takes this further with **block-diffusion drafting**: a 5-layer non-causal denoising draft conditioned on captured target hidden states. Accepts ~8 tokens/step vs ~3 for chain EAGLE. The official draft is [`z-lab/Qwen3.5-27B-DFlash`](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash). [DDTree (Ringel & Romano, 2026)](https://arxiv.org/abs/2604.12989) adds tree-structured verify on top, recovering the last 30% of the speedup.

**What was missing:** no public implementation ran either on consumer hardware. z-lab targets BF16 on B200 (60+ GB VRAM). No GGUF path. No DDTree port. AWQ INT4 of the target + BF16 draft doesn't leave room for the verify tree on 24 GB.

Q4_K_M GGUF (~16 GB) is the largest quantization that fits target + 3.46 GB draft + budget=22 tree state + KV cache on one RTX 3090. Picking it forced the port onto ggml, the only runtime with first-class Gated DeltaNet CUDA kernels and a GGUF Q4_K_M loader. This repo is that port:

- ~2000 lines of C++/CUDA on top of ggml (no libllama, no Python runtime)
- a pinned fork of llama.cpp at [`Luce-Org/llama.cpp@luce-dflash`](https://github.com/Luce-Org/llama.cpp/tree/luce-dflash) that adds three tree-mode ggml ops: `ggml_ssm_conv_tree`, `ggml_gated_delta_net_tree`, `ggml_gated_delta_net_tree_persist`
- hardcoded for the one model pair, decoding at 129.52 tok/s mean on HumanEval

## Results

Qwen3.5-27B Q4_K_M, concurrency=1, n_gen=256, 10 prompts/dataset:

| Task      | AR tok/s | DFlash+DDTree tok/s | AL   | Speedup |
|-----------|:--------:|:-------------------:|:----:|:-------:|
| HumanEval | 37.78    | **129.52**          | 8.31 | **3.43×** |
| Math500   | 37.71    | **110.51**          | 7.04 | **2.93×** |
| GSM8K     | 37.65    | **96.15**           | 6.14 | **2.55×** |

AR = autoregressive (`test_generate`). DFlash+DDTree = tree verify at budget=22 with fast rollback (`test_dflash`). AL = Acceptance Length, average committed tokens per draft/verify step. Reproduce via `python3 scripts/bench_llm.py`.

**Up to 256K context on 24 GB** via TQ3_0 KV cache (3.5 bpv, default; Q4_0 legacy path tops out near 128K) + sliding `target_feat` ring (4096 slots). TQ3 = ~9.7× memory saving vs F16; Q4_0 = 8×.

| Prompt length | KV     | Prefill time | Decode tok/s (FA window=2048) |
|:-------------:|:------:|:------------:|:------------:|
| 520 (HE)      | Q8_0   | 0.06 s       | ~104 (window inactive)        |
| 32K           | Q8_0   | 38 s         | ~95 (interp.)                 |
| 64K           | Q4_0   | 126 s        | **91** (PR #26)               |
| 128K          | TQ3_0  | ~10 min      | ~85–95 (sliding active)       |

Prefill numbers assume `--max-ctx` sized to the prompt (auto-fit in `run.py` / `bench_llm.py`). Oversizing — e.g. `--max-ctx=131072` on a 32K prompt — triggers FA stride over unused KV and slows prefill ~27× at that ratio.

HE 10-prompt bench mean in 128K mode (ctx=131072, ddtree-budget=16, FA window=2048): **134.78 tok/s** at AL 8.33.

Decode tok/s assume the default sliding-window flash attention (`--fa-window 2048`, lossless: 100% acceptance at all window sizes). Disable with `--fa-window 0` for full attention; expect ~25 tok/s at 60K+. Tune the window via `python3 scripts/run.py --fa-window N` or `--fa-window N` on `test_dflash`/`server.py` (sweet spot 1024–2048; bigger windows trade speed for marginally tighter attention).

Set `DFLASH27B_KV_TQ3=1` (TQ3_0, 3.5 bpv, default) or `DFLASH27B_KV_Q4=1` (Q4_0, 4.5 bpv, legacy) to enable. Full sweep in [RESULTS.md](RESULTS.md).

### Asymmetric K/V quantization

The cache now supports independent quantization types for keys and values, optimizing memory-asymmetric workloads. Set `DFLASH27B_KV_K=<type>` and `DFLASH27B_KV_V=<type>` via environment or CLI flags. Supported types (case-insensitive): `f16`, `bf16`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `tq3_0`.

**Supported (K, V) pairs:**
- K ∈ {F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0} × V ∈ {F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, TQ3_0}
- K = TQ3_0 × V ∈ {F16, BF16, Q4_0, Q8_0, TQ3_0}

Unsupported pairs abort at allocation with a printed list. Precedence (high→low): per-axis `_KV_K`/`_KV_V` override legacy shorthand (`--kv-tq3` / `--kv-q4` / `--kv-f16`); legacy shorthand last-wins among themselves. Default: `q8_0` for both.

**Environment variables:**
```bash
DFLASH27B_KV_K=q8_0 DFLASH27B_KV_V=q4_0 ./test_dflash …
```

**CLI flags on `test_dflash` / `test_generate`:**
```bash
./test_dflash … -ctk q8_0 -ctv q4_0
./test_dflash … --cache-type-k=q8_0 --cache-type-v=q4_0
```

**Target layer-split harness on `test_dflash`:**
```bash
./test_dflash target.gguf draft.safetensors prompt.bin 128 out.bin \
  --target-gpus=1,2,3,4 --target-layer-split=1,1,1,1

./test_dflash target.gguf draft.safetensors prompt.bin 128 out.bin \
  --draft-gpu=0 --target-gpus=0,1 --target-layer-split=1,3 \
  --target-split-load-draft

./test_dflash target.gguf draft.gguf prompt.bin 128 out.bin \
  --draft-gpu=0 --target-gpus=0,1 --target-layer-split=1,1 \
  --target-split-dflash
```

When more than one target GPU is listed, `test_dflash` runs a non-daemon
target-only layer-split harness. Each target GPU loads only its assigned
contiguous layer range; `output_norm` and `output.weight` stay on the last
target GPU. `--target-split-load-draft` additionally loads the DFlash draft
on `--draft-gpu` (`.safetensors` or quantized draft `.gguf`), captures target
features into a draft-side mirror, and runs a small draft forward smoke. This
allows capacity checks where the draft and a target layer range share one GPU
before serving integration. `--target-split-dflash` runs the same split target
placement through a chain DFlash decode loop and reports acceptance length.

**Python flags on `scripts/run.py`, `scripts/server.py`, `scripts/server_tools.py`:**
```bash
python3 scripts/run.py --ctk q8_0 --ctv q4_0 --prompt "hello"
python3 scripts/run.py --cache-type-k q8_0 --cache-type-v q4_0 --prompt "hello"
```

**TQ3 semantics under asymmetry:** TQ3_0 is K-side-driven. The FWHT rotation is applied to the query and inverse-applied to the attention output only when `K=TQ3_0`. V's type is independent of rotation. The 256-stride context alignment is triggered if *either* K or V is TQ3_0.

Legacy `--kv-tq3` / `--kv-q4` / `--kv-f16` flags continue to work as symmetric shorthand for backward compatibility.

## Qwen3.6-27B target (experimental)

Qwen3.6-27B ships the same `qwen35` architecture string and identical layer/head dims as 3.5, so `test_dflash` loads it with no code change:

```bash
# 1. target
huggingface-cli download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir models/

# 2. matched 3.6 draft (gated: accept terms + set HF_TOKEN first)
huggingface-cli download z-lab/Qwen3.6-27B-DFlash --local-dir models/draft/

# 3. bench
DFLASH_TARGET=models/Qwen3.6-27B-Q4_K_M.gguf python3 scripts/bench_he.py --n-gen 128
```

> The draft path is fixed at `models/draft/model.safetensors`; swapping targets requires also swapping the draft file (or pointing `DFLASH_DRAFT` at a different `model.safetensors`).

**Throughput is lower than on 3.5.** z-lab published a matched [Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) draft on 2026-04-26 (still under training). AL should climb as the draft matures. Measured on the same RTX 3090:

| Target | Draft | Bench | AL | Accept | Mean tok/s |
|---|---|---|---:|---:|---:|
| Qwen3.5-27B Q4_K_M | z-lab/Qwen3.5-27B-DFlash | HumanEval (README config) | 8.33 | ~65% | 134.78 |
| Qwen3.6-27B Q4_K_M | z-lab/Qwen3.5-27B-DFlash (mismatch) | HumanEval (10 prompts, n_gen=128) | 4.74 | 30.6% | 73.67 |
| Qwen3.6-27B Q4_K_M | z-lab/Qwen3.6-27B-DFlash (still training) | HumanEval (10 prompts, n_gen=128) | 5.05 | 32.3% | 77.77 |
| Qwen3.6-27B Q4_K_M | z-lab/Qwen3.5-27B-DFlash (mismatch) | Math (10 prompts, n_gen=128) | 3.63 | 23.7% | 57.00 |

Full `bench_llm.py` suite on **Qwen3.6-27B UD-Q4_K_XL** (unsloth Dynamic 2.0, 10 prompts, n_gen=256, RTX 3090 24 GB, auto-fit `--max-ctx`):

| Bench | AR tok/s | DFlash tok/s | AL | Speedup |
|---|---:|---:|---:|---:|
| HumanEval | 34.90 | 78.16 | 5.94 | **2.24×** |
| GSM8K | 34.89 | 59.65 | 4.43 | **1.71×** |
| Math500 | 35.13 | 69.77 | 5.15 | **1.99×** |
| **Mean** | 34.97 | 69.19 | 5.17 | **1.98×** |

Compare to Qwen3.5-27B on the same harness: 2.87× / 2.21× / 2.56× — cross-generation drop of ~22% uniformly from the draft mismatch, but still a clean ~2× with zero retraining.

Numbers will move once a Qwen3.6-matched DFlash draft lands; swap it in via `DFLASH_DRAFT=...` without rebuilding.

## Quick start

```bash
git clone --recurse-submodules https://github.com/Luce-Org/lucebox-hub
cd lucebox-hub/dflash

# Build (CUDA 12+, CMake 3.18+, sm_75+ GPU; CUDA 13+ required for Jetson AGX Thor sm_110)
# Pass -DCMAKE_CUDA_ARCHITECTURES matching your GPU. Common values:
#   75 = 2080 Ti (auto-converts BF16 draft → FP16 at load)
#   86 = RTX 3090 / A40
#   89 = RTX 4090
#   90 = H100
#   120 = Blackwell / DGX Spark
#   110 = Jetson AGX Thor (CUDA 13+)
# Omitting the flag falls back to the CMake-set default ("75;86"), which
# fails to compile pflash's BF16 WMMA kernels on the sm_75 pass — set the
# flag explicitly to your real arch to avoid this.
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target test_dflash -j

# Fetch models: ~16 GB target + 3.46 GB draft.
# Quickstart pins to Qwen3.6-27B (latest release). For Qwen3.5-27B swap in
# unsloth/Qwen3.5-27B-GGUF + z-lab/Qwen3.5-27B-DFlash; arch is identical so
# no rebuild is needed (see "Qwen3.6-27B target" section above for AL deltas).
huggingface-cli download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir models/
huggingface-cli download z-lab/Qwen3.6-27B-DFlash model.safetensors --local-dir models/draft/

# Streaming one-shot generate (run.py defaults to models/Qwen3.6-27B-Q4_K_M.gguf;
# override with --target or DFLASH_TARGET=... env var).
python3 scripts/run.py --prompt "def fibonacci(n):"

# Multi-turn chat REPL
python3 examples/chat.py

# OpenAI-compatible HTTP server (drop-in for Open WebUI / LM Studio / Cline)
python3 -m venv .venv
.venv/bin/pip install fastapi uvicorn transformers jinja2
.venv/bin/python scripts/server.py --port 8000 --daemon

# Reproduce paper numbers
python3 scripts/bench_llm.py                                 # HE + GSM8K + Math500
python3 scripts/bench_he.py --n-gen 256 --ddtree-budget 22   # minimal HE bench
```

**Long-context mode (up to 256K):**
```bash
DFLASH27B_KV_TQ3=1 DFLASH27B_PREFILL_UBATCH=16 \
  build/test_dflash models/Qwen3.6-27B-Q4_K_M.gguf \
  models/draft/model.safetensors /tmp/long_prompt.bin 64 /tmp/out.bin \
  --fast-rollback --ddtree --ddtree-budget=16 --max-ctx=4096   # align_up(prompt + n_gen + 64, 256); raise up to 262144 for long prompts
```

**Requirements:** NVIDIA sm_75+ GPU (2080 Ti, 3090, A10, A40, 4090) or Jetson AGX Thor sm_110, CUDA 12+ (CUDA 13+ required for Thor), 22+ GB VRAM, ~80 GB disk. On Turing (SM 7.5), BF16 draft weights are auto-converted to FP16 at load time for tensor core acceleration.

## How it works

**Block-diffusion draft.** Each step, the draft sees `[last_target_token, MASK×15]` plus the last 5 captured target hidden states. It denoises the masks in a single forward, producing 16 candidate tokens conditioned on real target features. Structurally stronger than chain EAGLE: every position conditions on the same captured context, not its own noisy predictions.

**DDTree tree verify.** Instead of one chain of 16 candidates, a best-first tree of up to 22 nodes spans the top-K branches at each position. One target forward verifies the whole tree via a causal mask derived from parent pointers. Budget=22 is the sweet spot where draft accuracy plateaus. Chain pre-seed matters: pure best-first construction with greedy verify on a quantized target can rescue an inferior suffix; the `chain_seed=true` flag in `build_ddtree` recovered AL from ~4 to ~9.

**Per-step rollback, kernel-free.** Before verify, the target's recurrent state (SSM intermediate, conv window, KV cache) is snapshotted; after accept, restored to the committed prefix. Three custom CUDA kernels keep rollback off the critical path:

| Kernel | Purpose |
|--------|---------|
| `ggml_gated_delta_net_tree_persist` | Direct-writes SSM intermediates into a persistent buffer, skipping a 9 ms `ggml_cpy` per step |
| `ggml_ssm_conv_tree` | Tree-aware conv state gather: each sibling reads its K-1 window along the DDTree parent chain, not DFS order |
| Sliding `target_feat` ring | 4096-slot ring via `(pos % cap)`, enables 128K without holding 6.6 GB of captured features |

Prefill and decode share one graph builder; chain mode is just DDTree with `budget=n_spec+1` and no branching.

## Architecture note

Qwen3.5-27B is **not** a dense transformer. llama.cpp calls the arch `qwen35`:

- 64 layers. Every 4th is full softmax attention, the rest are **Gated DeltaNet** (linear attention with learned recurrence)
- M-RoPE, dimension sections `[11, 11, 10, 0]`
- 24 Q heads, 4 KV heads, key/value length 256
- SSM state cache alongside the KV cache

The DeltaNet primitive is already a first-class ggml op (`ggml_gated_delta_net`). Our fork of llama.cpp adds three tree-mode variants (`ggml_ssm_conv_tree`, `ggml_gated_delta_net_tree`, `ggml_gated_delta_net_tree_persist`) so DDTree verify can roll back SSM state in place, without a replay forward. The full engine (graph builders + decode loop + rollback + kernels) is ~2000 lines.

## Why not llama.cpp / vLLM / z-lab?

- **llama.cpp**: runs Qwen3.5-27B via GGUF but has no DFlash integration. Chain EAGLE isn't enough; block diffusion + DDTree needs a custom decode loop that bypasses `llama_decode`.
- **vLLM / SGLang**: Qwen3.5-27B in BF16 is 54 GB, so a single 24 GB card forces a quantized path. GGUF for this arch is broken on SGLang as of 2026-04 and vLLM is dropping GGUF support. AWQ runs on SGLang as plain autoregressive at 46.6 tok/s but can't host the BF16 draft + DDTree tree state alongside it on 24 GB. Q4_K_M GGUF is the only format that fits the full spec-decode stack, this repo runs it at 129.5 tok/s mean on HumanEval, **2.8× faster** than SGLang AWQ autoregressive on the same hardware.
- **z-lab reference**: vLLM / SGLang integrations ship DFlash as a speculative-decoding method, but only on BF16 weights benchmarked on NVIDIA B200 (54+ GB VRAM). No GGUF path.

## Scope and limits

Research proof-of-concept, not production.

- **Batch size 1**, single-user local inference target (Ollama / LM Studio use case)
- **One model pair**: Qwen3.5-27B Q4_K_M target + z-lab DFlash BF16 draft. Does not generalize without rewriting the graph builders.
- **Optional sampling**: `temperature`, `top_p`, `top_k`, `seed`, and `frequency_penalty` are honored on the OpenAI endpoint. The DDTree verify skeleton stays argmax (preserves accept rate); only the *committed* token at each verify step is drawn from a small CPU sampler chain (rep-pen → top-k → top-p → temp → multinomial). `temperature=0` (default) keeps the path bit-exact greedy. Full Leviathan-style rejection sampling on the tree is still a future addition.
- **CUDA sm_86+ / sm_110 Thor** only. No Metal, ROCm, multi-GPU.
- **Q4_K_M target** costs ~30 points of per-position accept vs the paper's BF16. Q5_K_M / Q6_K would recover most of it, if they fit.

Correctness: `test_vs_oracle` validates the draft graph at cos sim 0.999812 vs the PyTorch reference. The target graph matches llama.cpp's `models/qwen35.cpp` semantically and produces bit-identical output to `test_generate` in autoregressive mode.

## Contributing

Open an issue or PR against `Luce-Org/lucebox-hub`. Good first picks:

- **Leviathan-style rejection sampling** on each DDTree branch (the current implementation samples only the committed token; full prob-matching across the tree is the next step)
- **Full llama.cpp integration**: new arch, `llama-speculative-dflash.cpp`, `llama-cli` / `llama-server` wiring

## Citation

```bibtex
@software{luce_dflash_2026,
  title  = {Luce DFlash: GGUF port of block-diffusion speculative decoding for Qwen3.5-27B on consumer GPUs},
  author = {Lucebox},
  url    = {https://github.com/Luce-Org/lucebox-hub/tree/main/dflash},
  year   = {2026}
}

@article{dflash2026,
  title   = {DFlash: Block-Diffusion Speculative Decoding},
  author  = {z-lab},
  journal = {arXiv:2602.06036},
  year    = {2026}
}

@article{ddtree2026,
  title   = {Accelerating Speculative Decoding with Block Diffusion Draft Trees},
  author  = {Ringel, Liran and Romano, Yaniv},
  journal = {arXiv:2604.12989},
  year    = {2026}
}
```

---

Apache 2.0 · [Lucebox](https://lucebox.com) · [Discord](https://discord.gg/yHfswqZmJQ)

Inspired by [z-lab/DFlash](https://arxiv.org/abs/2602.06036), [liranringel/ddtree](https://github.com/liranringel/ddtree), [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
l-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
