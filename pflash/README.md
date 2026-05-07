<p align="left">
  <a href="../README.md">← lucebox-hub</a>
</p>

<p align="center">
  <img src="hero.png" width="600" />
</p>

<h1 align="center">Luce PFlash</h1>

<p align="center">
  <strong>Speculative prefill in front of dflash. C++/CUDA only.</strong><br/>
  A drafter loaded in-process scores token importance; the heavy target only prefills the spans that matter.<br/>
  Qwen3.6-27B Q4_K_M at 128K on a single RTX 3090: <strong>24.8 s TTFT vs ~257 s llama.cpp</strong> = <strong>~10.4×</strong>, NIAH retrieval preserved.<br/><br/>
  <a href="https://lucebox.com/blog/pflash">Blog post</a> · <a href="https://discord.gg/yHfswqZmJQ">Discord</a> · <a href="https://lucebox.com">lucebox.com</a>
</p>

<p align="center">
  <img src="demo.gif" width="600" />
</p>

---

```
                       Cold TTFT (s)   Speedup   NIAH
llama.cpp pp131072         ~257           1.0x     ✓
dflash daemon @ 128K        24.8         10.4x     ✓
dflash daemon @  64K        13.5         10.0x     ✓
```

> Long context turns prefill into the dominant latency on quantized 27B targets. Speculative prefill scores token importance with a small drafter, then the heavy target only prefills the spans that matter. Quality preserved on NIAH at every measured context. The whole thing runs as a single C++/CUDA binary: no Python, no Triton, no PyTorch at runtime.

## The gap we filled

Long-context prefill is O(S²): vanilla llama.cpp on a single RTX 3090 takes **~257 s** to prefill 131,072 tokens of Qwen3.6-27B Q4_K_M (FA on, Q4_0 KV). Decode after that is fast (dflash spec decode runs at ~74 tok/s) but the user is staring at a blank screen for 4 minutes before the first token.

[Cross-Family Speculative Prefill (SambaNova ICLR 2026, Liu et al.)](https://arxiv.org/abs/2603.02631) showed a small drafter can score per-token importance over a long prompt and select a tiny fraction without losing the needle. The reference impl ([Jingyu6/speculative_prefill](https://github.com/Jingyu6/speculative_prefill)) wires this on top of vLLM with full BF16 targets on big GPUs.

**What was missing:** no implementation that sits in front of a quantized GGUF target on a 24 GB card without dragging Python+Triton into the runtime path. PFlash is that:

- C++/CUDA daemon-resident drafter + scoring + target generation, all in one process, one ggml allocator.
- Custom Qwen3-0.6B BF16 forward (`qwen3_0p6b_loader.cpp` + `qwen3_0p6b_graph.cpp`) — no libllama.
- 4 CUDA kernels for the FlashPrefill `mean_K → score → select → sparse_fwd` algorithm (`flashprefill_kernels.cu`).
- BSA ([mit-han-lab/Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention), FA-2 derived, sm_80+) for the long-context drafter forward, wired without `libtorch` via 3 ATen/c10 header stubs (`dflash/deps/bsa_stubs/`).
- 128K → 2.6K span selection at `keep_ratio=0.05`, NIAH retrieved at every measured context, decode ~74 tok/s downstream.

## Results

NIAH single-needle, RTX 3090 24 GB, Qwen3.6-27B Q4_K_M target, Qwen3-0.6B drafter, `DFLASH_FP_USE_BSA=1`, `DFLASH_FP_ALPHA=0.85`, `keep_ratio=0.05`.

| Source S | dflash TTFT | llama.cpp baseline | Speedup | NIAH |
|---|:---:|:---:|:---:|:---:|
| 64K  | **13.5 s** | 134.95 s (FA off, dense) | **10.0×** | ✅ |
| 128K | **24.8 s** | ~257 s (FA on, Q4_0 KV)  | **~10.4×** | ✅ |

Decode after prefill: ~74 tok/s (dflash spec decode + DDTree). The pipeline is the dflash binary on its own — no Python in the inference loop.

## Quick start

PFlash is the algorithm. The implementation lives in [`../dflash/`](../dflash/) as part of the dflash daemon. The `pflash/` directory in this repo only contains the Python tooling for **benchmarking** (NIAH case generation, bench harness around the daemon stdin protocol). Production deploys hit the dflash daemon directly.

```bash
# 1. build dflash with the BSA kernel (sm_80+; ~10 min cold compile pulls cutlass)
cd lucebox-hub/dflash
git submodule update --init --recursive
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
                    -DCMAKE_CUDA_ARCHITECTURES=86 \
                    -DDFLASH27B_ENABLE_BSA=ON
cmake --build build --target test_dflash test_flashprefill_kernels -j

# 2. fetch weights (target + spec-decode draft + drafter scorer)
huggingface-cli download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir models/
huggingface-cli download Qwen/Qwen3-0.6B model.safetensors tokenizer.json --local-dir models/drafter/
huggingface-cli download z-lab/Qwen3.6-27B-DFlash model.safetensors --local-dir models/draft/

# 2b. convert the drafter (Qwen3-0.6B HF) to a BF16 GGUF for the C++ scorer.
#     The submodule already vendors llama.cpp at deps/llama.cpp.
python deps/llama.cpp/convert_hf_to_gguf.py models/drafter \
       --outtype bf16 --outfile models/Qwen3-0.6B-BF16.gguf

# 3. install pflash bench harness (Python only used for benchmarking)
cd ../pflash
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 4. generate NIAH cases + run head-to-head bench against the C++ daemon
python tests/niah_gen.py --n 1 --ctx 131072 --out /tmp/niah_128k.jsonl
python tests/bench_niah_cpp.py \
  --bin    ../dflash/build/test_dflash \
  --target ../dflash/models/Qwen3.6-27B-Q4_K_M.gguf \
  --draft-spec ../dflash/models/draft/model.safetensors \
  --drafter-gguf ../dflash/models/Qwen3-0.6B-BF16.gguf \
  --cases  /tmp/niah_128k.jsonl --keep-ratio 0.05 --n-gen 256
```

## OpenAI server flags

For an OpenAI-compatible server with transparent compression on long prompts, run [`dflash/scripts/server.py`](../dflash/scripts/server.py) (or `server_tools.py` for tool-calling) with these flags:

| Flag | Choices / type | Default | Effect |
|---|---|:---:|---|
| `--prefill-compression` | `off` / `auto` / `always` | `off` | When to run pflash. `auto` compresses when total prompt ≥ threshold; `always` compresses every request. |
| `--prefill-threshold` | int (tokens) | `32000` | Token threshold for `auto` mode. |
| `--prefill-keep-ratio` | float `(0, 1]` | `0.05` | Fraction of source tokens to keep after compression. `0.02` for 128K, `0.10` for 32K. |
| `--prefill-drafter` | path to `.gguf` | required when not `off` | Drafter weights (Qwen3-0.6B BF16 GGUF). |
| `--prefill-drafter-tokenizer` | HF repo id | `Qwen/Qwen3-0.6B` | HF tokenizer for the drafter vocab. |

When `--prefill-compression != off`, the server auto-sets `DFLASH27B_LM_HEAD_FIX=0` and `DFLASH27B_FA_WINDOW=0` (matching the bench harness — needed so the post-compress draft graph fits on a 24 GB card without OOM).

```bash
python dflash/scripts/server.py \
  --target dflash/models/Qwen3.6-27B-Q4_K_M.gguf \
  --draft  dflash/models/draft/model.safetensors \
  --max-ctx 8192 --budget 16 --fa-window 0 \
  --prefill-compression auto \
  --prefill-threshold 4096 \
  --prefill-keep-ratio 0.02 \
  --prefill-drafter dflash/models/Qwen3-0.6B-BF16.gguf
```

Below the threshold the server runs the standard target generate (no compression). Above it, the server transparently runs `compress` on the daemon, swaps the prompt for the compressed text, and continues the normal `/v1/chat/completions` flow. Tool-calling requests (`req.tools` non-empty) skip compression so JSON tool definitions stay intact.

Validated end-to-end at 64K and 128K source on RTX 3090 (Qwen3.6-27B Q4_K_M target + Qwen3.5-DFlash draft + Qwen3-0.6B BF16 drafter).

## Daemon stdin protocol

The dflash daemon runs persistently and accepts these commands on stdin (newline-delimited):

| Command | Effect |
|---|---|
| `compress <ids.bin> <keep_x1000> <drafter.gguf>` | Drafter scores the prompt and emits the compressed token-id stream (terminated by `-1`). |
| `generate <prompt_ids.bin> <n_gen> <out_ids.bin>` | Target spec-decode on the (already compressed) prompt. Streams committed token ids on stdout. |
| `park draft` / `park target` / `park` | Free draft / target / both weights from VRAM. |
| `unpark draft` / `unpark target` / `unpark` | Restore weights from disk to VRAM. |
| `free drafter` | Release the spec-prefill drafter context (drafter weights + KV + BSA scratch). |

Typical flow at 128K on a 24 GB card: `park target` → `compress` → `free drafter` → `unpark target` → `unpark draft` → `generate` → `park draft`.

`pflash.dflash_client.DflashClient` is the Python wrapper around this protocol used by `tests/bench_niah_cpp.py`.

## Runtime tunables

Everything is configured via env vars on the daemon process. Full list in [`../dflash/src/flashprefill.h`](../dflash/src/flashprefill.h).

| Env var | Default | Purpose |
|---|:---:|---|
| `DFLASH_FP_USE_BSA` | `0` | Set to `1` to dispatch the sparse FA forward through the BSA cutlass kernel (sm_80+). Required for the headline 10.4× number; without it the WMMA fallback is used (slower at long ctx). |
| `DFLASH_FP_ALPHA` | `0.12` | Block-selection threshold. Higher = stricter = fewer K-blocks per Q-row. `0.85` is the bench setting; `0.99` cuts another second at 128K with a small NIAH-margin loss. |
| `DFLASH_FP_PROFILE` | `0` | Set to `1` to log per-stage timings (`mean_K / score / select / forward`). |
| `DFLASH_FP_DUMP_COUNTS` | `0` | Set to `1` to dump per-row K-block counts for debugging keep-ratio tuning. |
| `DFLASH27B_FA_WINDOW` | (auto) | Set to `0` to force full attention on the compressed prompt (recommended). |
| `DFLASH27B_KV_K` / `DFLASH27B_KV_V` | (auto) | KV-cache quant types. `q4_0` / `q4_0` is the bench setting. `tq3_0` saves another ~4 GB at 128K. |

## How it works

```
prompt (≤ 128K tokens)
   │
   ▼
┌──────────────────────────────────────────────┐
│  drafter (in-process)                        │
│   custom Qwen3-0.6B BF16 forward in ggml     │
│   FlashPrefill block-sparse via BSA (≥ 32K)  │
│   tail-attention scoring → score [S]         │
│   chunk(128) + alpha-threshold → top blocks  │
└──────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────┐
│  compressor (in-process)                     │
│   keep top keep_ratio of source tokens       │
│   re-emit compressed token-id stream         │
└──────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────┐
│  dflash spec decode (in-process)             │
│   target prefill of compressed prompt        │
│   DDTree spec decode + rollback              │
│   → answer tokens                            │
└──────────────────────────────────────────────┘
```

**Drafter forward.** Custom Qwen3-0.6B graph (`qwen3_0p6b_graph.cpp`) per-layer A/FP/B blocks: dense attention up to ~32K source, FlashPrefill sparse attention at and above. The 4 FP kernels live in `flashprefill_kernels.cu`; BSA dispatch is in `bsa_launcher.cu` + `bsa_fwd_inst.cu`.

**Scoring + selection.** Tail attention `Q[-N:] @ K^T / sqrt(d)` per layer/head, max over (L, H), mean over the tail window. Block-level threshold by `alpha * mean(scores)` selects which K-blocks each Q-block attends to. Configurable via `DFLASH_FP_ALPHA`.

**Memory budget on 24 GB.** Drafter scoring at 128K needs ~7-10 GB (drafter + KV + BSA scratch). Target + draft idle is ~18 GB. They can't coexist. The daemon's `park` / `unpark` / `free drafter` commands sequence VRAM occupancy across the request:

```
1. park draft + target          # daemon idles at ~3 GB
2. drafter loaded + scored      # ~10 GB peak
3. free drafter                 # release drafter weights + KV + BSA scratch
4. unpark target                # ~16 GB
5. unpark draft                 # +draft weights for spec decode
6. generate                     # spec decode the compressed prompt
7. park draft (idle)            # back to target only
```

## What's ours, what isn't

The algorithms are not ours:

- [**Cross-Family Speculative Prefill**](https://arxiv.org/abs/2603.02631) (SambaNova ICLR 2026): max-mean attention aggregation over a small drafter, lookahead-only attention.
- [**Speculative Prefill**](https://arxiv.org/abs/2502.02789) (Liu et al, 2025): the original Q-hook construction. Reference impl: [Jingyu6/speculative_prefill](https://github.com/Jingyu6/speculative_prefill).
- [**FlashPrefill**](https://arxiv.org/abs/2603.06199) (Fan et al, 2026): block-sparse attention with sink + window + dynamic top-K blocks. Original kernel: [qhfan/FlashPrefill](https://github.com/qhfan/FlashPrefill) (Triton).

What we built:

- C++/CUDA port of the FlashPrefill algorithm: 4 kernels (`mean_K / score / select / sparse_fwd`), no Triton dependency.
- BSA ([mit-han-lab/Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)) wired without `libtorch` via 3 ATen/c10 header stubs (`dflash/deps/bsa_stubs/`).
- Custom Qwen3-0.6B BF16 forward so the drafter runs through the same ggml allocator as the 27B target.
- Daemon stdin protocol (`compress` / `generate` / `park` / `unpark` / `free drafter`) so target + drafter coexist on a 24 GB card.
- NIAH harness against `llama-bench` for end-to-end validation.

## Scope and limits

- **Single 24 GB GPU** target (RTX 3090 reference). On 32+ GB cards, drafter + target can coexist and the park/unpark dance disappears.
- **Qwen3.6-27B Q4_K_M target + Qwen3-0.6B drafter** is the validated pair. Other targets/drafters need keep_ratio + alpha re-calibration.
- **NIAH single-needle** is the only retrieval task validated end-to-end. Multi-doc QA, long-form code retrieval, etc. still TBD.
- **sm_80+** required for BSA (RTX 3090 sm_86 is the reference). On sm_75 (Turing) the build auto-disables BSA and falls back to the WMMA path; expect a slower drafter forward at long ctx.

## Citation

```bibtex
@software{luce_pflash_2026,
  title  = {Luce PFlash: speculative prefill compression for long-context spec decode on consumer GPUs},
  author = {Lucebox},
  url    = {https://github.com/Luce-Org/lucebox-hub/tree/main/pflash},
  year   = {2026}
}

@article{spec_prefill_xfamily_2026,
  title   = {Cross-Family Speculative Prefill},
  author  = {Liu and others},
  journal = {arXiv:2603.02631},
  year    = {2026}
}

@article{flashprefill_2026,
  title   = {FlashPrefill: Block-Sparse Attention for Long-Context Prefill},
  author  = {Fan and others},
  journal = {arXiv:2603.06199},
  year    = {2026}
}
```

---

Apache 2.0 · [Lucebox](https://lucebox.com) · [Discord](https://discord.gg/yHfswqZmJQ)

Inspired by [Jingyu6/speculative_prefill](https://github.com/Jingyu6/speculative_prefill), [qhfan/FlashPrefill](https://github.com/qhfan/FlashPrefill), [z-lab/DFlash](https://arxiv.org/abs/2602.06036).
