# Laguna-XS.2 integration into dflash + PFlash (Path A: hand-rolled CUDA)

Status: scaffolding. PR #115 in lucebox-hub bumps llama.cpp submodule to `luce-dflash@706cd1f6b` (LAGUNA arch lives in libllama, used by quantize+inspect tools, NOT in dflash runtime). dflash runtime is and stays **ggml-only** (no libllama dependency). All Laguna forward code is hand-rolled CUDA mirroring the existing qwen35 path.

## Context

- `pflash_daemon` (test/pflash_daemon.cpp): drafter-only stdin compressor, loads Qwen3-0.6B via dflash's own loader, emits compressed token IDs in DRAFTER vocab. Already model-agnostic on the target side. **No change needed.**
- `test_dflash` (test/test_dflash.cpp 190 KB): main target runner. Hand-rolled CUDA forward graph for qwen35 hybrid. Loads via `load_target_gguf` which hardcodes `arch == "qwen35"`. **Hard-blocked on Laguna.**
- `qwen35_target_graph.cpp` (60 KB): hand-rolled CUDA forward, builds full-attn + delta-net + FFN. Uses `flash_prefill_forward_bf16` for sparse prefill.
- `flashprefill.{h,cpp}` + `flashprefill_kernels.cu`: model-agnostic block-sparse FA. Takes Q/K/V tensors, returns O. Already works for any GQA arch with head_dim 128. **Reusable as-is.**
- `server.py`: HTTP wrapper, subprocess-spawns `test_dflash`. Needs no change once a Laguna-capable target binary exists.

## Constraint

No libllama dependency in dflash runtime. Keep ggml-only stack. (libllama+LAGUNA arch from PR #7 is used by quantize/inspect tools at /workspace/lucebox-hub/dflash/deps/llama.cpp/build-standalone/ and at HF upload time, not by the daemon.)

## Implementation outline

### Files to add

1. `dflash/src/laguna_internal.h` (NEW, ~200 LOC) — structs:
   - `LagunaTargetLayer` — per-layer tensors (attn_norm, wq/wk/wv/wo, q_norm/k_norm, attn_gate, ffn_norm, dense MLP for layer 0, MoE: ffn_gate_inp + ffn_exp_probs_b + ffn_gate_exps + ffn_up_exps + ffn_down_exps + ffn_gate_shexp + ffn_up_shexp + ffn_down_shexp)
   - `LagunaTargetWeights` — collection of layers + tok_embd + output_norm + output, plus metadata (n_layer=40, n_head_per_layer[40] = [48,64,64,64]*10, n_head_kv=8, head_dim=128, n_embd=2048, n_ff=8192, n_ff_exp=512, n_ff_shexp=512, n_expert=256, n_expert_used=8, expert_weights_scale=2.5, sliding_window=512, rope_freq_base_full=500000, rope_freq_base_swa=10000, n_rot_full=64, n_rot_swa=128, eos_id=2, eot_id=24)
   - `LagunaTargetCache` — KV cache (Q8_0, per layer, max_ctx tokens), no SSM/conv state
   - `LagunaGraphInputs` / `LagunaGraphOutputs`

2. `dflash/src/laguna_target_loader.cpp` (NEW, ~500 LOC):
   - `load_target_gguf_laguna(path, backend, LagunaTargetWeights & out)`
   - Validates `arch == "laguna"`, reads all hparams, mmaps GGUF, copies tensors to ggml_backend buffer
   - Per-layer head count: reads `laguna.attention.head_count` as ARRAY (length 40) into `n_head_arr`
   - Tensor naming: matches gguf-py's MODEL_ARCH.LAGUNA list (token_embd, output_norm, output, blk.<i>.{attn_norm, attn_q, attn_k, attn_v, attn_output, attn_q_norm, attn_k_norm, attn_gate, ffn_norm, ffn_gate, ffn_down, ffn_up, ffn_gate_inp, ffn_gate_exps, ffn_down_exps, ffn_up_exps, ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp, exp_probs_b})
   - Layer 0: dense MLP (ffn_gate/down/up). Layers 1-39: sparse MoE (ffn_*_exps + shexp + gate_inp + exp_probs_b)

3. `dflash/src/laguna_target_graph.cpp` (NEW, ~1500 LOC — multi-session):
   - `build_laguna_full_attn_block` — full attention layer with YaRN RoPE (theta=500K, factor=32, partial_rotary=0.5, n_rot=64), per-head softplus gate, head_count from per-layer arr (48 on full)
   - `build_laguna_swa_block` — sliding-window attention layer (window=512, theta=10K, partial_rotary=1.0, n_rot=128), per-head softplus gate, head_count=64
   - `build_laguna_dense_mlp` — SwiGLU dense MLP (layer 0)
   - `build_laguna_moe_block` — sigmoid router + score-correction bias + sum-normalize + top-8, scale=2.5, plus always-on shared expert SwiGLU
   - `build_laguna_layer` — dispatches per-layer based on layer_types[il] (full vs swa) + mlp_layer_types[il] (dense vs sparse)
   - `build_laguna_graph` — token_embd → 40 layers → final RMSNorm → lm_head
   - `build_laguna_layer_for_prefill` — layer-segmented variant for the long-context prefill loop
   - Reuses `flash_prefill_forward_bf16` for sparse prefill on full-attention layers (for sliding-window layers, use dense FA since window=512 is small)
   - Cache mgmt: `create_laguna_target_cache`, `free_laguna_target_cache`, `reset_laguna_target_cache`, `snapshot_laguna_target_cache`, `restore_laguna_target_cache`

4. Modify `dflash/src/gguf_target_loader.cpp` (~30 LOC added):
   - Pre-detect arch string from GGUF header
   - Dispatch: arch == "qwen35" → existing path, arch == "laguna" → new path

5. Modify `dflash/src/internal.h` (~50 LOC added):
   - `enum class TargetArch { Qwen35, Laguna }` to tag the loaded weights
   - Forward decls for Laguna structs / functions (or include `laguna_internal.h`)

6. Modify `dflash/CMakeLists.txt`:
   - Add `src/laguna_target_loader.cpp` and `src/laguna_target_graph.cpp` to `dflash27b` library sources

7. Modify `dflash/test/test_dflash.cpp` (substantial changes — multi-session):
   - Detect arch from loaded weights
   - For Laguna arch, use `LagunaTargetCache` + `build_laguna_graph` instead of qwen35 equivalents
   - Adjust per-layer-head-count in attention buffer sizing
   - PFlash drafter call unchanged (drafter is Qwen3-0.6B regardless of target)
   - Cross-tokenizer mapping (Qwen3 IDs → Laguna IDs): byte-level round-trip via existing pflash/ Python module OR port to C++ helper

## Phasing

- **Phase 1 (this session)**: laguna_internal.h structs + load_target_gguf_laguna full impl + arch dispatch + CMakeLists. Deliverable: `smoke_load_target_laguna` (or modified smoke_load_target) successfully loads our Laguna Q4_K_M GGUF, prints arch info, exits cleanly. NO forward yet.
- **Phase 2 (next session)**: build_laguna_graph implementation. Verify forward parity vs HF reference (token-level prefix match like our llama-simple test). Hook into test_dflash.
- **Phase 3**: PFlash sparse-prefill on Laguna. Verify NIAH retention. TTFT bench at 16/32/64/128K.
- **Phase 4**: server.py support, HTTP API end-to-end, article.

## Constants for Laguna-XS.2 Q4_K_M

```
n_layer                 = 40
n_head (default)        = 48           # plus per-layer override array
n_head_kv               = 8
head_dim                = 128
n_embd                  = 2048
n_ff (dense layer 0)    = 8192
n_ff_exp                = 512
n_ff_shexp              = 512
n_expert                = 256
n_expert_used           = 8
expert_weights_scale    = 2.5
expert_weights_norm     = true (sum-normalize selected weights)
expert_gating_func      = SIGMOID
rope_freq_base_full     = 500000
rope_freq_base_swa      = 10000
rope_yarn_factor        = 32
rope_yarn_beta_fast     = 64
rope_yarn_beta_slow     = 1
rope_yarn_orig_ctx      = 4096
n_rot_full              = 64    # partial_rotary_factor=0.5 on full layers
n_rot_swa               = 128   # partial_rotary_factor=1.0 on sliding
sliding_window          = 512
full_attention_pattern  = (full, sw, sw, sw) x 10  // every 4th layer is full
layer 0 mlp             = dense SwiGLU
layers 1..39 mlp        = sparse MoE + always-on shared expert
vocab_size              = 100352
bos_id                  = 2
eos_ids                 = [2, 24]
pad_id                  = 9
```

GGUF KV keys: prefix `laguna.*` (e.g. `laguna.attention.head_count`, `laguna.rope.freq_base_swa`, `laguna.expert_weights_scale`, etc.).

GGUF tensor names: `token_embd.weight`, `output_norm.weight`, `output.weight`, `blk.<i>.attn_norm.weight`, `blk.<i>.attn_q.weight`, `blk.<i>.attn_k.weight`, `blk.<i>.attn_v.weight`, `blk.<i>.attn_output.weight`, `blk.<i>.attn_q_norm.weight`, `blk.<i>.attn_k_norm.weight`, `blk.<i>.attn_gate.weight`, `blk.<i>.ffn_norm.weight`, `blk.<i>.ffn_gate.weight` (layer 0), `blk.<i>.ffn_down.weight` (layer 0), `blk.<i>.ffn_up.weight` (layer 0), `blk.<i>.ffn_gate_inp.weight` (sparse), `blk.<i>.exp_probs_b.bias` (sparse), `blk.<i>.ffn_gate_exps.weight` (sparse, [n_expert, n_ff_exp, n_embd]), `blk.<i>.ffn_down_exps.weight` (sparse, [n_expert, n_embd, n_ff_exp]), `blk.<i>.ffn_up_exps.weight` (sparse, [n_expert, n_ff_exp, n_embd]), `blk.<i>.ffn_gate_shexp.weight` (sparse), `blk.<i>.ffn_up_shexp.weight` (sparse), `blk.<i>.ffn_down_shexp.weight` (sparse).
