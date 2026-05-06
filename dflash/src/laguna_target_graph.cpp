// Hand-rolled CUDA forward graph for Poolside Laguna-XS.2 in dflash.
//
// Mirrors qwen35_target_graph.cpp's structure but for the laguna arch:
//   - 40 layers, hybrid iSWA: every 4th layer is FULL attention, rest are SWA(512)
//   - Per-layer head count: 48 (full) / 64 (SWA), 8 KV heads always, head_dim=128
//   - Q-norm + K-norm RMSNorm at head_dim level (Qwen3-style)
//   - Per-head SOFTPLUS attention gate: g_proj : hidden -> n_head, broadcast over head_dim
//   - Per-layer-type partial RoPE:
//       FULL: YaRN (theta=500K, factor=32, partial=0.5  -> n_rot=64)
//       SWA:  default (theta=10K, partial=1.0  -> n_rot=128)
//   - Layer 0: dense SwiGLU MLP (n_ff=8192). Layers 1..39: sparse MoE (256 top-8,
//     sigmoid router with score-correction bias, sum-normalize, scale=2.5) +
//     always-on shared expert SwiGLU (intermediate=512).
//
// Phase 2 status: cache lifecycle + attention blocks (full + SWA) + dense MLP
// + MoE block + layer dispatcher + full graph. Forward parity vs HF reference
// is tested against our llama.cpp build_laguna (already verified to match HF
// for 30+ tokens on B-tree prompt; see Lucebox/Laguna-XS.2-GGUF README).

#include "laguna_internal.h"
#include "internal.h"
#include "dflash27b.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cuda.h"

namespace dflash27b {

static constexpr float LAGUNA_EPS = 1e-6f;

// ---- Cache lifecycle ----------------------------------------------------

bool create_laguna_target_cache(const LagunaTargetWeights & w,
                                 int max_ctx,
                                 ggml_backend_t backend,
                                 LagunaTargetCache & out) {
    out.backend  = backend;
    out.max_ctx  = max_ctx;
    out.cur_pos  = 0;
    out.last_tok = -1;
    // KV cache: per-layer, ALL 40 layers (full + SWA). Layout matches qwen35:
    //   [head_dim, max_ctx, n_head_kv]
    // dtype Q8_0 to halve VRAM vs F16.
    const ggml_type k_type = out.kv_k_type;
    const ggml_type v_type = out.kv_v_type;

    const size_t n_tensors_per_layer = 2;
    const size_t need_tensors = (size_t)w.n_layer * n_tensors_per_layer;

    ggml_init_params ip{};
    // Each tensor descriptor + overhead. Be generous.
    ip.mem_size = ggml_tensor_overhead() * (need_tensors + 16) + 4096;
    ip.no_alloc = true;
    out.base_ctx = ggml_init(ip);
    if (!out.base_ctx) { set_last_error("laguna cache: ggml_init failed"); return false; }

    out.attn_k.resize(w.n_layer, nullptr);
    out.attn_v.resize(w.n_layer, nullptr);
    for (int il = 0; il < w.n_layer; ++il) {
        char nm[32];
        std::snprintf(nm, sizeof(nm), "k_l%d", il);
        ggml_tensor * k = ggml_new_tensor_3d(out.base_ctx, k_type, w.head_dim, max_ctx, w.n_head_kv);
        ggml_set_name(k, nm);
        std::snprintf(nm, sizeof(nm), "v_l%d", il);
        ggml_tensor * v = ggml_new_tensor_3d(out.base_ctx, v_type, w.head_dim, max_ctx, w.n_head_kv);
        ggml_set_name(v, nm);
        out.attn_k[il] = k;
        out.attn_v[il] = v;
    }

    out.base_buf = ggml_backend_alloc_ctx_tensors(out.base_ctx, backend);
    if (!out.base_buf) {
        set_last_error("laguna cache: ggml_backend_alloc_ctx_tensors failed");
        ggml_free(out.base_ctx); out.base_ctx = nullptr;
        return false;
    }

    // Zero-init KV (so reads before any write don't see garbage).
    const size_t buf_sz = ggml_backend_buffer_get_size(out.base_buf);
    std::vector<uint8_t> zeros(std::min<size_t>(buf_sz, 64 * 1024 * 1024), 0);
    for (int il = 0; il < w.n_layer; ++il) {
        for (auto * t : { out.attn_k[il], out.attn_v[il] }) {
            const size_t sz = ggml_nbytes(t);
            for (size_t off = 0; off < sz; off += zeros.size()) {
                const size_t chunk = std::min(zeros.size(), sz - off);
                ggml_backend_tensor_set(t, zeros.data(), off, chunk);
            }
        }
    }
    return true;
}

void free_laguna_target_cache(LagunaTargetCache & c) {
    if (c.base_buf) { ggml_backend_buffer_free(c.base_buf); c.base_buf = nullptr; }
    if (c.base_ctx) { ggml_free(c.base_ctx);                c.base_ctx = nullptr; }
    c.attn_k.clear();
    c.attn_v.clear();
}

void reset_laguna_target_cache(LagunaTargetCache & c) {
    c.cur_pos  = 0;
    c.last_tok = -1;
    if (!c.base_ctx) return;
    std::vector<uint8_t> zeros(64 * 1024 * 1024, 0);
    for (auto * t : c.attn_k) {
        if (!t) continue;
        const size_t sz = ggml_nbytes(t);
        for (size_t off = 0; off < sz; off += zeros.size()) {
            const size_t chunk = std::min(zeros.size(), sz - off);
            ggml_backend_tensor_set(t, zeros.data(), off, chunk);
        }
    }
    for (auto * t : c.attn_v) {
        if (!t) continue;
        const size_t sz = ggml_nbytes(t);
        for (size_t off = 0; off < sz; off += zeros.size()) {
            const size_t chunk = std::min(zeros.size(), sz - off);
            ggml_backend_tensor_set(t, zeros.data(), off, chunk);
        }
    }
}

// ---- Helpers ------------------------------------------------------------

static ggml_tensor * laguna_rms_norm_mul(ggml_context * ctx, ggml_tensor * x,
                                          ggml_tensor * weight, float eps = LAGUNA_EPS) {
    ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, n, weight);
}

static ggml_tensor * build_laguna_dense_ffn(ggml_context * ctx, ggml_tensor * cur,
                                              const LagunaTargetLayer & L) {
    // SwiGLU: down( silu(gate(x)) * up(x) )
    ggml_tensor * gate = ggml_mul_mat(ctx, L.w_gate, cur);   // [n_ff, n_tokens]
    ggml_tensor * up   = ggml_mul_mat(ctx, L.w_up,   cur);   // [n_ff, n_tokens]
    ggml_tensor * gu   = ggml_swiglu_split(ctx, gate, up);
    return ggml_mul_mat(ctx, L.w_down, gu);                  // [n_embd, n_tokens]
}

// Forward decl for the full MoE block (defined further down).
static ggml_tensor * build_laguna_moe_block_full(ggml_context * ctx, ggml_tensor * cur,
                                                  const LagunaTargetWeights & w,
                                                  const LagunaTargetLayer & L);

// MoE block: sigmoid router with score-correction bias, sum-normalize selected
// weights, scale routed combine by expert_weights_scale (=2.5 for Laguna),
// add always-on shared expert SwiGLU. Matches modeling_laguna.LagunaSparseMoeBlock.
//
// `cur` shape [n_embd, n_tokens]. Returns [n_embd, n_tokens].
// PHASE 2.0 STUB: returns ONLY the shared expert (no routed dispatch). Lets us
// validate attention path + dense MLP path on layer 0 + final norm + lm_head.
// Numerically wrong (drops routed MoE contribution = ~80% of MLP signal) but
// graph builds + executes. Routed dispatch (sigmoid router + score-correction
// bias + sum-norm + ggml_mul_mat_id) is Phase 2.1.
// DEBUG SWITCH: env DFLASH_LAGUNA_MOE_STUB=1 routes to shared-only stub.
// Default: full MoE.
static ggml_tensor * build_laguna_moe_block(ggml_context * ctx, ggml_tensor * cur,
                                             const LagunaTargetWeights & w,
                                             const LagunaTargetLayer & L) {
    static const bool stub = (std::getenv("DFLASH_LAGUNA_MOE_STUB") != nullptr);
    if (stub) {
        ggml_tensor * sh_gate = ggml_mul_mat(ctx, L.ffn_gate_shexp, cur);
        ggml_tensor * sh_up   = ggml_mul_mat(ctx, L.ffn_up_shexp,   cur);
        ggml_tensor * sh_gu   = ggml_swiglu_split(ctx, sh_gate, sh_up);
        return ggml_mul_mat(ctx, L.ffn_down_shexp, sh_gu);
    }
    return build_laguna_moe_block_full(ctx, cur, w, L);
}

// Phase 2.1: full MoE dispatch (sigmoid + score-correction bias + sum-norm +
// scale 2.5 + always-on shared expert). Mirrors llama.cpp's build_moe_ffn for
// the SIGMOID + WEIGHTS_NORM + EXP_PROBS_B configuration that Laguna uses.
static ggml_tensor * build_laguna_moe_block_full(ggml_context * ctx, ggml_tensor * cur,
                                                  const LagunaTargetWeights & w,
                                                  const LagunaTargetLayer & L) {
    const int n_tokens = (int)cur->ne[1];
    const int n_expert = w.n_expert;
    const int n_used   = w.n_expert_used;
    const int n_embd   = w.n_embd;

    // Router logits + sigmoid
    ggml_tensor * logits = ggml_mul_mat(ctx, L.ffn_gate_inp, cur);  // [n_expert, n_tokens]
    ggml_tensor * probs  = ggml_sigmoid(ctx, logits);

    // Add score-correction bias for SELECTION (not for combine weights).
    ggml_tensor * scores_sel = ggml_add(ctx, probs, L.ffn_exp_probs_b);

    // Top-k selection: indices [n_used, n_tokens] i32.
    ggml_tensor * selected = ggml_top_k(ctx, scores_sel, n_used);

    // Gather ORIGINAL probs (no bias) at the selected indices for combine weights.
    // Trick: reshape probs to [1, n_expert, n_tokens] so ggml_get_rows treats
    // the expert axis as the row axis. Output shape [1, n_used, n_tokens].
    ggml_tensor * probs_3d = ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens);
    ggml_tensor * weights  = ggml_get_rows(ctx, probs_3d, selected);
    weights = ggml_reshape_2d(ctx, weights, n_used, n_tokens);

    // Sum-normalize selected weights (Laguna sets expert_weights_norm=true).
    ggml_tensor * w_sum = ggml_sum_rows(ctx, weights);  // [1, n_tokens]
    weights = ggml_div(ctx, weights, w_sum);

    // Scale routed combine.
    if (w.expert_weights_scale != 1.0f) {
        weights = ggml_scale(ctx, weights, w.expert_weights_scale);
    }

    // Per-expert SwiGLU via mul_mat_id.
    //   ffn_gate_exps: [n_embd, n_ff_exp, n_expert]
    //   ffn_up_exps:   [n_embd, n_ff_exp, n_expert]
    //   selected:      [n_used, n_tokens] i32  (ids->ne = {n_used, n_tokens, 1, 1})
    // ggml_mul_mat_id requires b->ne[2] == ids->ne[1]. cur is [n_embd, n_tokens]
    // (ne[2]=1) so reshape to [n_embd, 1, n_tokens] (ne[2]=n_tokens). Same trick
    // llama.cpp's build_moe_ffn uses.
    ggml_tensor * cur_3d = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);
    ggml_tensor * gate_e = ggml_mul_mat_id(ctx, L.ffn_gate_exps, cur_3d, selected);
    ggml_tensor * up_e   = ggml_mul_mat_id(ctx, L.ffn_up_exps,   cur_3d, selected);
    ggml_tensor * gu     = ggml_swiglu_split(ctx, gate_e, up_e);
    //   ffn_down_exps: [n_ff_exp, n_embd, n_expert]
    //   gu:            [n_ff_exp, n_used, n_tokens]   (ne[2] = n_tokens)
    //   experts out:   [n_embd, n_used, n_tokens]
    ggml_tensor * experts = ggml_mul_mat_id(ctx, L.ffn_down_exps, gu, selected);

    // Multiply per-expert outputs by their routing weights.
    //   experts: [n_embd, n_used, n_tokens]
    //   weights: [n_used, n_tokens] -> view as [1, n_used, n_tokens] for broadcast
    ggml_tensor * w_view = ggml_reshape_3d(ctx, weights, 1, n_used, n_tokens);
    experts = ggml_mul(ctx, experts, w_view);

    // Sum across the n_used axis: explicit slice + add loop (matches llama.cpp
    // pattern; ggml_sum_rows would sum over dim 0 which is n_embd, wrong).
    ggml_tensor * routed = nullptr;
    for (int i = 0; i < n_used; ++i) {
        ggml_tensor * slice = ggml_view_2d(ctx, experts,
            n_embd, n_tokens,
            experts->nb[2],
            (size_t)i * experts->nb[1]);
        routed = (i == 0) ? slice : ggml_add(ctx, routed, slice);
    }

    // Always-on shared expert (SwiGLU).
    ggml_tensor * sh_gate = ggml_mul_mat(ctx, L.ffn_gate_shexp, cur);
    ggml_tensor * sh_up   = ggml_mul_mat(ctx, L.ffn_up_shexp,   cur);
    ggml_tensor * sh_gu   = ggml_swiglu_split(ctx, sh_gate, sh_up);
    ggml_tensor * shared  = ggml_mul_mat(ctx, L.ffn_down_shexp, sh_gu);

    return ggml_add(ctx, routed, shared);
}

static ggml_tensor * build_laguna_moe_block_legacy(ggml_context * ctx, ggml_tensor * cur,
                                                    const LagunaTargetWeights & w,
                                                    const LagunaTargetLayer & L) {
    // ggml_build_moe is the standard helper used by deepseek/qwen3 MoE in
    // llama.cpp. We need an equivalent. ggml's ggml_mul_mat_id implements the
    // grouped per-expert matmul; the router/topk logic must be done via op
    // composition. Match llama.cpp's build_moe_ffn semantics.
    //
    //   logits = ffn_gate_inp @ x                    # [n_expert, n_tokens]
    //   probs  = sigmoid(logits)                     # SIGMOID gating
    //   sel_scores = probs + exp_probs_b             # bias-corrected for SELECTION
    //   topk = argtopk(sel_scores, n_expert_used)    # indices [topk, n_tokens]
    //   weights = gather(probs, topk)                # ORIGINAL probs (no bias) for combine
    //   weights = weights / sum(weights, axis=-2, keepdim=True)
    //   y_routed = sum_e weights[e] * down_e(silu(gate_e(x)) * up_e(x))
    //   y_routed = y_routed * scale
    //   y_shared = down_sh(silu(gate_sh(x)) * up_sh(x))
    //   return y_routed + y_shared
    //
    // ggml provides this composition via ggml_top_k + ggml_mul_mat_id. To keep
    // the file self-contained and avoid extending ggml here, route through
    // existing ops:

    // Router logits: [n_expert, n_tokens]
    ggml_tensor * router_logits = ggml_mul_mat(ctx, L.ffn_gate_inp, cur);
    // Sigmoid (Laguna router uses sigmoid, not softmax)
    ggml_tensor * probs = ggml_sigmoid(ctx, router_logits);

    // For selection only, add the score-correction bias. Don't use bias-added
    // values for the combine weights.
    //   bias is [n_expert]; broadcast over n_tokens.
    ggml_tensor * scores_for_sel = ggml_add(ctx, probs, L.ffn_exp_probs_b);

    // Top-k indices: [n_expert_used, n_tokens] i32.
    // ggml_top_k returns argmax indices into the n_expert axis.
    ggml_tensor * selected = ggml_top_k(ctx, scores_for_sel, w.n_expert_used);

    // Gather the ORIGINAL probs at the selected indices for combine weights.
    // ggml_get_rows would treat probs's first dim as rows; here we want to
    // index into the n_expert axis per-token, which matches the standard MoE
    // pattern in llama.cpp's build_moe_ffn (see deepseek path).
    ggml_tensor * weights = ggml_get_rows(ctx, probs, selected);
    // Reshape to [n_expert_used, n_tokens] (ggml_get_rows yields [..., 1]; squeeze).
    weights = ggml_reshape_2d(ctx, weights, w.n_expert_used, ggml_nelements(weights) / w.n_expert_used);

    // Sum-normalize selected weights along expert axis.
    ggml_tensor * w_sum = ggml_sum_rows(ctx, weights);  // [1, n_tokens]
    weights = ggml_div(ctx, weights, w_sum);

    // Apply the routed scaling factor.
    if (w.expert_weights_scale != 1.0f) {
        weights = ggml_scale(ctx, weights, w.expert_weights_scale);
    }

    // Routed expert gate+up (fused SwiGLU): use mul_mat_id to dispatch.
    //   ffn_gate_exps: [n_embd, n_ff_exp, n_expert]
    //   ffn_up_exps:   [n_embd, n_ff_exp, n_expert]
    //   ffn_down_exps: [n_ff_exp, n_embd, n_expert]
    // ggml_mul_mat_id expects ids tensor [n_expert_used, n_tokens] (we have it).
    ggml_tensor * gate_e = ggml_mul_mat_id(ctx, L.ffn_gate_exps, cur, selected);
    ggml_tensor * up_e   = ggml_mul_mat_id(ctx, L.ffn_up_exps,   cur, selected);
    ggml_tensor * gu     = ggml_swiglu_split(ctx, gate_e, up_e);
    ggml_tensor * routed = ggml_mul_mat_id(ctx, L.ffn_down_exps, gu, selected);

    // Multiply per-expert outputs by their routing weights and combine.
    // mul_mat_id gives [n_embd, n_expert_used, n_tokens]. Broadcast multiply
    // by weights [n_expert_used, n_tokens] -> need shape match. Reshape weights
    // to [1, n_expert_used, n_tokens] via view.
    ggml_tensor * w_view = ggml_reshape_3d(ctx, weights, 1, w.n_expert_used,
                                            ggml_nelements(weights) / w.n_expert_used);
    routed = ggml_mul(ctx, routed, w_view);
    routed = ggml_sum_rows(ctx, ggml_cont(ctx, ggml_permute(ctx, routed, 0, 2, 1, 3)));
    // Now [n_embd, n_tokens, 1]. Reshape to [n_embd, n_tokens].
    routed = ggml_reshape_2d(ctx, routed, w.n_embd, ggml_nelements(routed) / w.n_embd);

    // Shared expert (always on).
    ggml_tensor * sh_gate = ggml_mul_mat(ctx, L.ffn_gate_shexp, cur);
    ggml_tensor * sh_up   = ggml_mul_mat(ctx, L.ffn_up_shexp,   cur);
    ggml_tensor * sh_gu   = ggml_swiglu_split(ctx, sh_gate, sh_up);
    ggml_tensor * shared  = ggml_mul_mat(ctx, L.ffn_down_shexp, sh_gu);

    return ggml_add(ctx, routed, shared);
}

// Attention block. Handles BOTH full and SWA layers; the only differences are
//   - n_head[il] (per-layer)
//   - rope_freq_base + n_rot (per layer-type)
//   - YaRN params (full only) vs default rope (swa)
//   - sliding window mask (swa only; passed in via attn_mask)
//
// Per-head softplus gate is applied AFTER the FA output (broadcast over head_dim)
// and BEFORE the o_proj.
static ggml_tensor * build_laguna_attn_block(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const LagunaTargetWeights & w,
    const LagunaTargetLayer & L,
    int il,
    ggml_tensor * cur,
    ggml_tensor * positions,
    ggml_tensor * cache_k,
    ggml_tensor * cache_v,
    ggml_tensor * attn_mask,
    int kv_start,
    int n_tokens,
    bool is_full)
{
    const int head_dim   = w.head_dim;
    const int n_head     = w.n_head_arr[il];
    const int n_head_kv  = w.n_head_kv;
    const int q_dim      = n_head * head_dim;

    // ---- Q/K/V projections ---
    ggml_tensor * Qcur = ggml_mul_mat(ctx, L.wq, cur);  // [q_dim, n_tokens]
    ggml_tensor * Kcur = ggml_mul_mat(ctx, L.wk, cur);  // [n_head_kv*head_dim, n_tokens]
    ggml_tensor * Vcur = ggml_mul_mat(ctx, L.wv, cur);

    Qcur = ggml_reshape_3d(ctx, Qcur, head_dim, n_head,    n_tokens);
    Kcur = ggml_reshape_3d(ctx, Kcur, head_dim, n_head_kv, n_tokens);
    Vcur = ggml_reshape_3d(ctx, Vcur, head_dim, n_head_kv, n_tokens);

    // ---- Per-head Q/K RMSNorm (norm over head_dim) ---
    Qcur = laguna_rms_norm_mul(ctx, Qcur, L.q_norm);
    Kcur = laguna_rms_norm_mul(ctx, Kcur, L.k_norm);

    // ---- Per-head softplus attention gate ---
    // wqkv_gate : [n_embd, n_head]; gate_proj output [n_head, n_tokens] f32.
    ggml_tensor * gate_logits = ggml_mul_mat(ctx, L.wqkv_gate, cur); // [n_head, n_tokens]
    // Cast to f32 to match HF reference (computed in fp32, then cast back).
    gate_logits = ggml_cast(ctx, gate_logits, GGML_TYPE_F32);
    ggml_tensor * gate = ggml_softplus(ctx, gate_logits);            // [n_head, n_tokens] f32

    // ---- Partial RoPE (NeoX layout: rotate first n_rot dims, leave the rest) ---
    const int n_rot     = is_full ? w.n_rot_full : w.n_rot_swa;
    const float rope_th = is_full ? w.rope_freq_base_full : w.rope_freq_base_swa;
    // YaRN params: only on full layers (sliding uses default rope, attn_factor=0
    // makes ggml_rope_ext fall back to plain math).
    const float ext_factor  = is_full ? 1.0f : 0.0f;
    const float attn_factor = is_full ? 1.0f : 1.0f;
    const float beta_fast   = is_full ? w.yarn_beta_fast : 32.0f;
    const float beta_slow   = is_full ? w.yarn_beta_slow :  1.0f;
    const int   n_ctx_orig  = is_full ? w.yarn_orig_ctx  : 0;
    const float freq_scale  = is_full ? (1.0f / w.yarn_factor) : 1.0f;

    Qcur = ggml_rope_ext(ctx, Qcur, positions, /*freq_factors=*/nullptr,
                          n_rot, /*mode=*/GGML_ROPE_TYPE_NEOX,
                          n_ctx_orig, rope_th, freq_scale,
                          ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_ext(ctx, Kcur, positions, nullptr,
                          n_rot, GGML_ROPE_TYPE_NEOX,
                          n_ctx_orig, rope_th, freq_scale,
                          ext_factor, attn_factor, beta_fast, beta_slow);

    // ---- Write K/V to cache slot [kv_start, kv_start+n_tokens) ---
    // cache_k layout: [head_dim, max_ctx, n_head_kv]. Permute Kcur from
    // [head_dim, n_head_kv, n_tokens] to [head_dim, n_tokens, n_head_kv]
    // so axes line up.
    ggml_tensor * Kcur_T = ggml_permute(ctx, Kcur, 0, 2, 1, 3);
    ggml_tensor * Vcur_T = ggml_permute(ctx, Vcur, 0, 2, 1, 3);

    ggml_tensor * k_slot = ggml_view_3d(ctx, cache_k,
        head_dim, n_tokens, n_head_kv,
        cache_k->nb[1], cache_k->nb[2],
        cache_k->nb[1] * (size_t)kv_start);
    ggml_tensor * v_slot = ggml_view_3d(ctx, cache_v,
        head_dim, n_tokens, n_head_kv,
        cache_v->nb[1], cache_v->nb[2],
        cache_v->nb[1] * (size_t)kv_start);

    ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur_T, k_slot));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur_T, v_slot));

    // ---- Flash attention over [0..kv_len) (full) or sliding window ---
    const int kv_len = kv_start + n_tokens;
    int win_start = 0;
    if (!is_full) {
        const int sw = w.sliding_window;
        if (kv_len > sw) win_start = kv_len - sw;
    }
    const int win_len = kv_len - win_start;

    ggml_tensor * Qfa = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
    Qfa = ggml_cont(ctx, Qfa);

    ggml_tensor * Kfa = ggml_view_3d(ctx, cache_k,
        head_dim, win_len, n_head_kv,
        cache_k->nb[1], cache_k->nb[2], cache_k->nb[1] * (size_t)win_start);
    ggml_tensor * Vfa = ggml_view_3d(ctx, cache_v,
        head_dim, win_len, n_head_kv,
        cache_v->nb[1], cache_v->nb[2], cache_v->nb[1] * (size_t)win_start);

    const float kq_scale = 1.0f / std::sqrt((float)head_dim);
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, Qfa, Kfa, Vfa, attn_mask,
                                              kq_scale, 0.0f, 0.0f);
    // attn: [head_dim, n_head, n_tokens]

    // Per-head softplus gate broadcast over head_dim:
    //   attn[d, h, t] *= gate[h, t]
    // Reshape gate to [1, n_head, n_tokens] so broadcast works.
    ggml_tensor * gate_b = ggml_reshape_3d(ctx, gate, 1, n_head, n_tokens);
    // Cast gate back to attn dtype to keep mul homogeneous.
    gate_b = ggml_cast(ctx, gate_b, attn->type);
    attn = ggml_mul(ctx, attn, gate_b);

    attn = ggml_reshape_2d(ctx, attn, q_dim, n_tokens);

    // ---- Output projection ---
    return ggml_mul_mat(ctx, L.wo, attn);  // [n_embd, n_tokens]
}

// ---- Layer dispatch ----------------------------------------------------

static ggml_tensor * build_laguna_layer(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const LagunaTargetWeights & w,
    LagunaTargetCache & cache,
    int il,
    ggml_tensor * inp,
    ggml_tensor * positions,
    ggml_tensor * attn_mask,
    int kv_start,
    int n_tokens)
{
    const LagunaTargetLayer & L = w.layers[il];

    // Pre-attn norm
    ggml_tensor * cur = laguna_rms_norm_mul(ctx, inp, L.attn_norm);

    // Attention
    const bool is_full = laguna_is_full_attn_layer(w, il);
    cur = build_laguna_attn_block(ctx, gf, w, L, il, cur,
                                    positions, cache.attn_k[il], cache.attn_v[il],
                                    attn_mask, kv_start, n_tokens, is_full);

    // Residual
    ggml_tensor * ffn_inp = ggml_add(ctx, cur, inp);

    // Pre-FFN norm
    cur = laguna_rms_norm_mul(ctx, ffn_inp, L.ffn_norm);

    // Dense MLP (layer 0) or sparse MoE+shared (layers 1..n)
    const bool is_dense = (il < w.n_layer_dense_lead);
    cur = is_dense ? build_laguna_dense_ffn(ctx, cur, L)
                   : build_laguna_moe_block(ctx, cur, w, L);

    return ggml_add(ctx, cur, ffn_inp);
}

LagunaGraphOutputs build_laguna_graph(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const LagunaTargetWeights & w,
    LagunaTargetCache & cache,
    const LagunaGraphInputs & in)
{
    LagunaGraphOutputs out{};

    ggml_tensor * cur = in.inp_embed;  // [n_embd, n_tokens, 1] f32 (CPU-embedded)
    if (cur->ne[2] == 1) {
        cur = ggml_reshape_2d(ctx, cur, w.n_embd, in.n_tokens);
    }

    for (int il = 0; il < w.n_layer; ++il) {
        cur = build_laguna_layer(ctx, gf, w, cache, il, cur,
                                  in.positions, in.attn_mask, in.kv_start, in.n_tokens);
    }

    // Final norm + lm_head
    cur = laguna_rms_norm_mul(ctx, cur, w.out_norm);

    if (in.output_hidden_states) {
        out.hidden_states = cur;
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);
    }

    if (in.output_logits) {
        ggml_tensor * head_in = cur;  // [n_embd, n_tokens]
        if (in.output_last_only && in.n_tokens > 1) {
            head_in = ggml_view_2d(ctx, cur, w.n_embd, 1,
                                    cur->nb[1],
                                    (size_t)(in.n_tokens - 1) * cur->nb[1]);
        }
        out.logits = ggml_mul_mat(ctx, w.output, head_in);  // [vocab, 1] or [vocab, n_tokens]
        ggml_set_output(out.logits);
        ggml_build_forward_expand(gf, out.logits);
    }

    return out;
}

} // namespace dflash27b
