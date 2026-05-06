// Laguna-XS.2 (Poolside) target structs for dflash daemon. Mirrors the qwen35
// abstractions in internal.h but for an iSWA + MoE arch with no SSM/delta-net
// and per-layer-varying head count.
//
// Architecture summary (from poolside/Laguna-XS.2 config):
//   - 40 layers. Pattern (FULL, SWA, SWA, SWA) x 10. Sliding window = 512.
//   - Per-layer head count: 48 on FULL layers, 64 on SWA layers. n_head_kv = 8 always.
//   - head_dim = 128. n_embd = 2048.
//   - Q-norm + K-norm (RMSNorm at head_dim level).
//   - Per-head softplus attention gate: g_proj : hidden -> n_head, applied as
//     attn_out.view(*, n_head, head_dim) * softplus(g_proj(x)).unsqueeze(-1).
//   - Per-layer-type RoPE:
//       FULL: YaRN (theta=500000, factor=32, partial_rotary=0.5, original=4096,
//             beta_slow=1, beta_fast=64). n_rot = 64.
//       SWA:  default (theta=10000, partial_rotary=1.0). n_rot = 128.
//   - Layer 0: dense SwiGLU (n_ff = 8192). Layers 1..39: sparse MoE
//     (256 experts, top-8, sigmoid router with score-correction bias,
//     sum-normalize selected weights, scale = 2.5) + always-on shared expert
//     SwiGLU (intermediate = 512).
//   - Vocab = 100352. BOS = 2. EOS = {2, 24}. Pad = 9.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

#include "internal.h"  // for CpuEmbedder

namespace dflash27b {

struct LagunaTargetLayer {
    // Pre-attn + pre-ffn norms (Laguna has only these two; no post norms).
    ggml_tensor * attn_norm   = nullptr;  // [n_embd]
    ggml_tensor * ffn_norm    = nullptr;  // [n_embd]

    // Attention projections. Q/O sizing is per-layer (n_head[il] varies).
    ggml_tensor * wq          = nullptr;  // [n_embd, n_head[il] * head_dim]
    ggml_tensor * wk          = nullptr;  // [n_embd, n_head_kv * head_dim]
    ggml_tensor * wv          = nullptr;  // [n_embd, n_head_kv * head_dim]
    ggml_tensor * wo          = nullptr;  // [n_head[il] * head_dim, n_embd]

    // Q/K head-dim RMSNorm (Qwen3-style).
    ggml_tensor * q_norm      = nullptr;  // [head_dim]
    ggml_tensor * k_norm      = nullptr;  // [head_dim]

    // Per-head softplus gate. wqkv_gate shape: [n_embd, n_head[il]] (NOT n_head*head_dim).
    // Forward: gate = softplus(x @ wqkv_gate.float()), broadcast over head_dim onto attn_out.
    ggml_tensor * wqkv_gate   = nullptr;  // [n_embd, n_head[il]]

    // Dense MLP (layer 0 only — nullptr for sparse layers).
    ggml_tensor * w_gate      = nullptr;  // [n_embd, n_ff]
    ggml_tensor * w_up        = nullptr;  // [n_embd, n_ff]
    ggml_tensor * w_down      = nullptr;  // [n_ff, n_embd]

    // Sparse MoE (layers 1..n-1 only — nullptr for dense layer 0).
    ggml_tensor * ffn_gate_inp     = nullptr;  // [n_embd, n_expert]      router weights
    ggml_tensor * ffn_exp_probs_b  = nullptr;  // [n_expert]              score-correction bias
    ggml_tensor * ffn_gate_exps    = nullptr;  // [n_embd, n_ff_exp, n_expert]  stacked routed gate
    ggml_tensor * ffn_up_exps      = nullptr;  // [n_embd, n_ff_exp, n_expert]  stacked routed up
    ggml_tensor * ffn_down_exps    = nullptr;  // [n_ff_exp, n_embd, n_expert]  stacked routed down
    ggml_tensor * ffn_gate_shexp   = nullptr;  // [n_embd, n_ff_shexp]    shared expert gate
    ggml_tensor * ffn_up_shexp     = nullptr;  // [n_embd, n_ff_shexp]    shared expert up
    ggml_tensor * ffn_down_shexp   = nullptr;  // [n_ff_shexp, n_embd]    shared expert down
};

struct LagunaTargetWeights {
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf     = nullptr;

    // CPU-side embedding table (matches qwen35 strategy: tok_embd stays on host
    // because CUDA's get_rows doesn't support k-quants).
    CpuEmbedder           embedder;

    ggml_tensor * tok_embd = nullptr;        // metadata only; data on host
    std::vector<LagunaTargetLayer> layers;   // size = n_layer
    ggml_tensor * out_norm = nullptr;        // [n_embd]
    ggml_tensor * output   = nullptr;        // [n_embd, n_vocab]  (lm_head)

    // Architecture metadata (validated at load time).
    int  n_layer              = 40;
    int  n_head_kv            = 8;
    int  head_dim             = 128;
    int  n_embd               = 2048;
    int  n_ff                 = 8192;
    int  n_ff_exp             = 512;
    int  n_ff_shexp           = 512;
    int  n_expert             = 256;
    int  n_expert_used        = 8;
    int  n_layer_dense_lead   = 1;     // layer 0 dense, rest sparse
    int  sliding_window       = 512;
    int  swa_pattern          = 4;     // (full, sw, sw, sw) repeating

    // Per-layer head count: 48 (full) / 64 (sliding). Always n_layer entries.
    int  n_head_arr[40]       = { 48,64,64,64, 48,64,64,64, 48,64,64,64, 48,64,64,64, 48,64,64,64,
                                  48,64,64,64, 48,64,64,64, 48,64,64,64, 48,64,64,64, 48,64,64,64 };

    // RoPE per layer-type.
    float rope_freq_base_full = 500000.0f;
    float rope_freq_base_swa  = 10000.0f;
    int   n_rot_full          = 64;     // partial_rotary_factor=0.5 on full layers
    int   n_rot_swa           = 128;    // partial_rotary_factor=1.0 on sliding
    // YaRN params for full layers:
    float yarn_factor         = 32.0f;
    float yarn_beta_fast      = 64.0f;
    float yarn_beta_slow      = 1.0f;
    int   yarn_orig_ctx       = 4096;

    // MoE gating + scaling.
    bool  expert_gating_sigmoid = true;   // sigmoid (vs softmax) router
    bool  expert_weights_norm   = true;   // sum-normalize selected weights
    float expert_weights_scale  = 2.5f;   // routed combine scale

    // Tokenizer special tokens.
    int32_t bos_id      = 2;
    int32_t eos_id      = 2;
    int32_t eos_chat_id = 24;             // second eos id in HF config
    int32_t pad_id      = 9;
};

// True if layer `il` uses full attention (the FIRST layer in each group of
// `swa_pattern` is full; the remaining are sliding-window).
inline bool laguna_is_full_attn_layer(const LagunaTargetWeights & w, int il) {
    return (il % w.swa_pattern) == 0;
}

// Loader. Validates arch == "laguna", reads all hparams, mmaps GGUF, copies
// tensors to backend buffer. Returns false + sets dflash27b_last_error on failure.
bool load_target_gguf_laguna(const std::string & path,
                              ggml_backend_t       backend,
                              LagunaTargetWeights & out);

void free_laguna_target_weights(LagunaTargetWeights & w);

// ---- Forward graph (Phase 2; signatures only for now) -------------------

struct LagunaTargetCache {
    ggml_context *        base_ctx = nullptr;
    ggml_backend_buffer_t base_buf = nullptr;
    ggml_backend_t        backend  = nullptr;

    int max_ctx  = 0;
    int cur_pos  = 0;
    int last_tok = -1;

    ggml_type kv_k_type = GGML_TYPE_Q8_0;
    ggml_type kv_v_type = GGML_TYPE_Q8_0;

    // Per-layer KV cache. ALL 40 layers have KV (both full + swa).
    // Layout: [head_dim, max_ctx, n_head_kv] f16/q8_0, contiguous per layer.
    std::vector<ggml_tensor *> attn_k;   // size = n_layer
    std::vector<ggml_tensor *> attn_v;
};

bool create_laguna_target_cache(const LagunaTargetWeights & w,
                                 int max_ctx,
                                 ggml_backend_t backend,
                                 LagunaTargetCache & out);
void free_laguna_target_cache(LagunaTargetCache & c);
void reset_laguna_target_cache(LagunaTargetCache & c);

struct LagunaGraphInputs {
    ggml_tensor * inp_embed;      // [n_embd, n_tokens, 1] f32 (CPU-embedded by caller)
    ggml_tensor * positions;      // [n_tokens] i32 (NeoX rope; not M-RoPE)
    ggml_tensor * attn_mask;      // optional [kv_len, n_tokens] f32 (causal)
    int           n_tokens;
    int           kv_start;
    bool          output_logits = true;
    bool          output_hidden_states = false;
    // If true, lm_head only runs on the LAST token (saves ~6 GB of logit memory
    // at n_tokens=16K, vocab=100352). Standard TTFT optimization.
    bool          output_last_only = false;
};

struct LagunaGraphOutputs {
    ggml_tensor * logits        = nullptr;  // [vocab, n_tokens] f32
    ggml_tensor * hidden_states = nullptr;  // [n_embd, n_tokens] f32
};

// Phase 2: implement.
LagunaGraphOutputs build_laguna_graph(
    ggml_context *               ctx,
    ggml_cgraph *                gf,
    const LagunaTargetWeights &  w,
    LagunaTargetCache &          cache,
    const LagunaGraphInputs &    in);

} // namespace dflash27b
