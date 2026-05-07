// Shared building blocks for qwen35 (dense) and qwen35moe (MoE) forward passes.
//
// These functions are defined in qwen35_target_graph.cpp and used by both the
// dense graph builder and the MoE graph builder (qwen35moe_target_graph.cpp).

#pragma once

#include "internal.h"

namespace dflash27b {

struct DeltaNetCapture;

// RMSNorm followed by element-wise multiplication with a weight tensor.
ggml_tensor * rms_norm_mul(ggml_context * ctx, ggml_tensor * x,
                           ggml_tensor * weight, float eps);

// SwiGLU FFN: gate * silu, element-wise multiply with up, then project down.
// Uses L.w_gate, L.w_up, L.w_down.
ggml_tensor * build_swiglu_ffn(ggml_context * ctx, ggml_tensor * cur,
                               const TargetLayer & L);

// Full multi-head attention block with M-RoPE, GQA, and KV cache update.
ggml_tensor * build_full_attn_block(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const TargetWeights & w,
    const TargetLayer & L,
    ggml_tensor * cur,
    ggml_tensor * positions,
    ggml_tensor * cache_k,
    ggml_tensor * cache_v,
    ggml_tensor * attn_mask,
    int kv_start,
    int n_tokens,
    ggml_type kv_k_type,
    ggml_type kv_v_type,
    int fa_window = 0);

// Gated DeltaNet block (causal conv + recurrent state update).
ggml_tensor * build_delta_net_block(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const TargetWeights & w,
    const TargetLayer & L,
    ggml_tensor * cur,
    ggml_tensor * conv_state,
    ggml_tensor * ssm_state,
    int n_tokens,
    DeltaNetCapture * cap,
    ggml_tensor * parent_ids);

} // namespace dflash27b
