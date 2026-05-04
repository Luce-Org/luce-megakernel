<p align="center">
  <img src="hero.png" width="600" />
</p>

<h1 align="center">Luce Megakernel</h1>

<p align="center">
  <strong>The first megakernel for hybrid DeltaNet/Attention LLMs.</strong><br/>
  All 24 layers of Qwen 3.5-0.8B in a single CUDA dispatch.<br/>
  1.87 tok/J on a 2020 GPU, matching Apple's latest silicon at 2x the throughput.<br/><br/>
  <a href="https://lucebox.com/blog/megakernel">Blog post</a> · <a href="RESULTS.md">Benchmarks</a> · <a href="https://discord.gg/yHfswqZmJQ">Discord</a> · <a href="https://lucebox.com">lucebox.com</a>
</p>

---

```
                        Prefill      Decode      tok/J
Megakernel (RTX 3090)   21,347       413         1.87  @220W
Megakernel (RTX 2080 Ti) 13,919      250          —
llama.cpp  (RTX 3090)   11,247       267         0.76
Apple M5 Max               -         229         1.76
```

> The efficiency gap between NVIDIA and Apple isn't inherent to the silicon. It's an artifact of running generic software on capable hardware.

## Why this exists

Conventional wisdom says NVIDIA GPUs are fast but power hungry, and Apple Silicon is slower but efficient. On paper, that checks out: llama.cpp on an RTX 3090 gets 267 tok/s at 350W (0.76 tok/J), while an M5 Max gets 229 tok/s at ~130W (1.76 tok/J). NVIDIA is faster, but 2.3x worse on efficiency.
Qwen 3.5-0.8B uses a hybrid DeltaNet + Attention architecture (linear attention interleaved with standard attention). No fused kernel existed for this pattern. This is the first.

Inspired by [Hazy Research's megakernel work on Llama-1B](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles), we asked: can the same idea work for hybrid DeltaNet/Attention models on consumer GPUs?

We thought the problem was never the hardware. The RTX 3090 has 936 GB/s memory bandwidth and 142 TFLOPS FP16 compute. Extracting only 267 tok/s from that is a software problem.

**The culprit: ~100 kernel launches per token.** Each layer boundary returns control to the CPU, dispatches the next kernel, re-fetches weights from global memory, and synchronizes threads. For 24 layers, those microseconds add up, and each one burns power doing nothing useful.

So we fused everything into one kernel.

## Results

### Decode and Prefill (Qwen 3.5-0.8B)

| Method | Prefill pp520 (tok/s) | Decode tg128 (tok/s) |
|--------|:---------------------:|:--------------------:|
| **Megakernel (RTX 3090)** | **21,347** | **413** |
| **Megakernel (RTX 2080 Ti)** | **13,919** | **250** |
| llama.cpp BF16 (RTX 3090) | 11,247 | 267 |
| PyTorch HuggingFace (RTX 3090) | 7,578 | 108 |
| PyTorch HuggingFace (RTX 2080 Ti) | 1,050 | 12 |

RTX 3090: 1.9x faster prefill, 1.55x faster decode vs llama.cpp. RTX 2080 Ti: 13.3x faster prefill, 20x faster decode vs PyTorch HF on the same GPU.

### Energy Efficiency (DVFS Power Sweep)

| Power Limit | Clock | Draw | tok/s | tok/J | vs Stock |
|:-----------:|:-----:|:----:|:-----:|:-----:|:--------:|
| 420W (stock) | 1980 MHz | 314W | 433 | 1.38 | baseline |
| 300W | 1935 MHz | 299W | 432 | 1.44 | 99.8% speed, 5% less power |
| **220W** | **1635 MHz** | **220W** | **411** | **1.87** | **95% speed, 30% less power** |
| 150W | 405 MHz | 150W | 194 | 1.29 | too aggressive |

Sweet spot at 220W: 95% of the speed, 30% less power. The curve is nonlinear, tight execution converts directly into saved watts until you starve the GPU too aggressively.

### The Comparison That Shouldn't Exist

| Metric | RTX 3090 (llama.cpp) | M5 Max | RTX 3090 (Megakernel @220W) |
|--------|:--------------------:|:------:|:---------------------------:|
| tok/s | 267 | 229 | **411** |
| Power | 350W | ~130W | 220W |
| tok/J | 0.76 | 1.76 | **1.87** |
| GPU price | ~$900 | $2,499+ (system) | ~$900 |

A $900 GPU from 2020, power-limited to 220W, matches Apple's latest chip on efficiency while delivering 1.8x the throughput.

## How it works

A single persistent CUDA kernel processes the entire Qwen 3.5-0.8B forward pass in one dispatch. No CPU round-trips between layers.

**Architecture:** Qwen 3.5-0.8B is a hybrid model, 18 DeltaNet layers (linear attention with learned recurrence) and 6 full attention layers, in a 3:1 ratio. DeltaNet scales linearly with context length vs. quadratic for standard attention. It's an emerging pattern in next-gen models (Qwen3-Next, Kimi Linear), but no framework had optimized kernels for it.

**Kernel specs:**
- 82 blocks, 512 threads, all SMs on the RTX 3090 kept occupied; launch count is clamped to the resident-block ceiling on smaller GPUs to avoid grid-sync deadlock
- BF16 weights and activations on Ampere+ (sm_80+), FP16 on Turing (sm_75); FP32 accumulation where it matters
- DeltaNet recurrence via warp-cooperative state updates in F32 registers
- Full attention with online softmax (fused QKV, RoPE, causal mask, output projection)
- Cooperative grid sync between layers instead of kernel launches (zero inter-layer overhead)
- KV cache updates in-kernel
- Weights loaded directly from HuggingFace

**What traditional frameworks do:** launch ~100 separate kernels per token, each one paying the cost of CPU dispatch, weight re-fetch, and thread synchronization. The megakernel eliminates all of that.

## Why DeltaNet matters

Standard transformers have years of kernel optimization: FlashAttention, PagedAttention, continuous batching. Hybrid DeltaNet/Attention architectures are newer, and the kernel ecosystem is immature:

- **MLX:** no native DeltaNet kernels
- **llama.cpp:** generic DeltaNet support, no fusion
- **vLLM/SGLang:** Triton kernels via flash-linear-attention, but no megakernel fusion

As more models go hybrid (and they will, because linear attention scales better), what you run them on matters less than *how* you run them. When you write a kernel that actually uses what the GPU offers, tensor cores, shared memory, cooperative grid launches, register-resident state, a five-year-old GPU matches Apple's latest chip.

## Lessons from building this

**`grid.sync()` inside loops will deadlock silently.** We tried synchronizing all blocks within the per-token DeltaNet recurrence loop. No error message, just a hang. The fix: synchronize between layers, not within them.

**Register pressure kills performance quietly.** We attempted `S_TILE=16` for more instruction-level parallelism. Silent crash, no CUDA error, registers spilled to local memory, performance collapsed. `S_TILE=8` was the sweet spot.

**The power curve is nonlinear.** 420W to 300W lost almost nothing (99.8% speed). 300W to 220W lost marginally (95%). 220W to 150W collapsed to 45%. The megakernel's tight execution means the GPU hits its compute ceiling before its power ceiling, until you starve it too aggressively.

## Quick start

```bash
git clone https://github.com/Luce-Org/luce-megakernel
cd luce-megakernel
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch                          # install BEFORE the next step; setup.py imports torch at build time
pip install -e . --no-build-isolation      # --no-build-isolation lets the build see the torch you just installed
python final_bench.py    # runs pp520 tg128 (properly warmed), prints tok/s
```

**Requirements:**
- **RTX 3090** (sm_86, BF16) — primary target, all benchmarks above
- **RTX 2080 Ti** (sm_75, FP16) — supported via compile-time `TARGET_SM` flag; auto-detected at build time
- Portable to other Turing+ (sm_75+) NVIDIA GPUs
- CUDA 12+
- PyTorch 2.0+
- ~1.5 GB VRAM for weights

**Blackwell (sm_120 / sm_121a):** runs on Blackwell consumer GPUs (RTX 5090)
and the NVIDIA DGX Spark (GB10) via an NVFP4 decode path. The build
auto-detects your GPU and `final_bench.py` dispatches to the right backend;
use `--backend nvfp4` to force it. First DGX Spark numbers are in
[RESULTS.md](RESULTS.md#nvidia-dgx-spark-gb10-sm_121a).

**Optional:** Set a power limit to find your GPU's sweet spot:
```bash
sudo nvidia-smi -pl 220    # or whatever your target wattage
```

## Files

| File | Description |
|------|-------------|
| `kernel.cu` | Decode megakernel, all 24 layers in one dispatch |
| `prefill.cu` | Prefill (cuBLAS + standalone kernels) |
| `half_type.h` | Portable half-precision type alias (BF16 on sm_80+, FP16 on sm_75) |
| `torch_bindings.cpp` | PyTorch C++ bindings |
| `model.py` | Weight loading + decoder |
| `setup.py` | Build configuration (arch auto-detect, Blackwell gating) |
| `final_bench.py` | Benchmark (prefill + decode, 10 warmup + 20 timed runs averaged — use this for published numbers) |
| `bench_pp_tg.py` | Quick single-run benchmark with correctness check (not warmed, under-reports prefill) |
| `RESULTS.md` | Full benchmark results and DVFS sweep |
| `kernel_gb10_nvfp4.cu` | Blackwell-only: NVFP4 persistent decode megakernel |
| `prefill_megakernel.cu` | Blackwell-only: single-dispatch prefill megakernel |
| `model_nvfp4.py` | Blackwell-only: NVFP4 decoder driving the ops above |
| `final_bench_nvfp4.py` / `bench_pp_tg_nvfp4.py` | Blackwell bench siblings dispatched by `--backend nvfp4` |

## Scope and limitations

This is a **research proof-of-concept**, not a production inference server.

- **Batch size 1 only.** This targets single-user local inference (the llama.cpp/Ollama use case), not multi-tenant serving. If you need batched throughput, use vLLM or SGLang.
- **Single model, single architecture.** The kernel is hand-written for Qwen 3.5-0.8B's specific layer pattern (18 DeltaNet + 6 Attention). It does not generalize to other models without rewriting.
- **BF16 (Ampere+) / FP16 (Turing).** Uses BF16 on sm_80+ and FP16 on sm_75 via compile-time `TARGET_SM` flag. No quantization support (GGUF/GPTQ/AWQ). We benchmark at half-precision to isolate kernel-level efficiency from quantization tradeoffs.
- **0.8B parameters.** This is a small model. Megakernel fusion benefits shrink as model size grows and compute begins to dominate over launch overhead. We chose 0.8B because it's the first hybrid DeltaNet model available, not because it's representative of all workloads.
- **Power methodology.** Efficiency numbers measure accelerator power only (NVML for NVIDIA, `powermetrics` for Apple), following [Hazy Research's Intelligence Per Watt](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) methodology. Total system draw is higher for both platforms.
- **Correctness.** `bench_pp_tg.py` includes an end-to-end correctness check comparing megakernel output against a reference decode path. Use `final_bench.py` for performance numbers (properly warmed, 10 warmup + 20 timed runs averaged).

The goal is to demonstrate that architecture-specific kernel fusion eliminates a real efficiency gap on consumer hardware, and to do it in the open so others can reproduce, critique, and extend the work.

## Files

Questions, ideas, or want to see what others are building? Join the [Luce Discord](https://discord.gg/yHfswqZmJQ).

## Citation

If you use this work in your research:

```bibtex
@software{luce_megakernel_2026,
  title  = {Luce Megakernel: Fused Forward Pass for Hybrid DeltaNet/Attention LLMs},
  author = {Luce},
  url    = {https://github.com/Luce-Org/luce-megakernel},
  year   = {2026}
}
```

## Why llama.cpp as a baseline?

llama.cpp is the most widely used local inference engine. It's what most people actually run on consumer GPUs. We also include PyTorch HuggingFace numbers (3.8x slower) for a second reference point. This is not a critique of llama.cpp, it's an excellent project. The comparison shows what architecture-specific optimization can unlock on top of a generic framework.

## Why an RTX 3090?

Deliberately chosen as the "worst case" for NVIDIA: a 2020 GPU, widely dismissed as power-hungry, available for ~$900 used. If the software gap is real on old hardware, it's even larger on newer cards.

## RTX 2080 Ti support

The megakernel also runs on the RTX 2080 Ti (Turing, sm_75, 2018) using FP16 instead of BF16. The build system auto-detects the GPU and selects the right code path via a compile-time `TARGET_SM` flag. On the 2080 Ti, the megakernel delivers 250 tok/s decode and 13,919 tok/s prefill — 20x faster than PyTorch HuggingFace on the same GPU.
## Community

Questions, ideas, or want to see what others are building? Join the [Luce Discord](https://discord.gg/yHfswqZmJQ).

---

MIT · [Lucebox](https://lucebox.com)

Built with [Claude](https://claude.ai)

Inspired by [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel), [Infatoshi/MegaQwen](https://github.com/Infatoshi/MegaQwen)
