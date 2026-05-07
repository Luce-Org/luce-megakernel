// MoE forward pass for Qwen3.5-MoE (qwen35moe) in pure ggml.
//
// Two-graph-per-layer execution pattern:
//   Graph A: attention (full-attn or deltanet) + post-norm + router + top-k + weights
//   Graph B: expert FFN via ggml_mul_mat_id + shared expert + residual
//
// Between the two graphs, the CPU reads router outputs, prepares the
// expert cache (loading misses from mmap), and computes slot_ids for
// the mul_mat_id indexing.
//
// This avoids destabilizing the dense code path while reusing ~90% of the
// shared building blocks (rms_norm_mul, build_full_attn_block, build_delta_net_block,
// build_swiglu_ffn).

#include "internal.h"
#include "expert_cache.h"
#include "qwen35_blocks.h"

#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>

// Temporary debug: set to true to add sync checks per-layer.
bool g_moe_debug_sync = false;
int  g_moe_debug_call = 0;

// Timing accumulators (reset per forward call)
static double g_time_graph_a_ms = 0;
static double g_time_graph_b_ms = 0;
static double g_time_cache_ms   = 0;
static double g_time_alloc_ms   = 0;

namespace dflash27b {

static constexpr float MOE_EPS = 1e-6f;

// ─── Graph A: attention + router ─────────────────────────────────────

// Builds Graph A for one MoE layer. Saves intermediates (post, ffn_residual,
// weights) to MoeState persistent tensors via ggml_cpy. Returns the
// selected_experts tensor (marked as output for D2H after graph_compute).
static ggml_tensor * build_moe_graph_a(
    ggml_context *        ctx,
    ggml_cgraph *         gf,
    const TargetWeights & w,
    TargetCache &         cache,
    MoeState &            moe,
    int                   layer_idx,
    ggml_tensor *         inp,         // [hidden, n_tokens]
    ggml_tensor *         positions,   // [4*n_tokens] i32 or nullptr
    ggml_tensor *         attn_mask,   // optional
    int                   kv_start,
    int                   n_tokens,
    int                   fa_window = 0)
{
    const int hidden  = w.n_embd;
    const int n_used  = w.n_experts_active;
    const int n_total = w.n_experts;
    const TargetLayer & L = w.layers[layer_idx];
    const bool is_attn = (((layer_idx + 1) % w.full_attention_interval) == 0);

    // ── Attention ──
    ggml_tensor * inpSA = inp;
    ggml_tensor * cur = rms_norm_mul(ctx, inp, L.attn_norm, MOE_EPS);

    if (is_attn) {
        int fa_idx = 0;
        for (int il = 0; il < layer_idx; il++)
            if (((il + 1) % w.full_attention_interval) == 0) fa_idx++;
        cur = build_full_attn_block(ctx, gf, w, L, cur, positions,
                                     cache.attn_k[fa_idx], cache.attn_v[fa_idx],
                                     attn_mask, kv_start, n_tokens,
                                     cache.kv_k_type, cache.kv_v_type, fa_window);
    } else {
        int dn_idx = 0;
        for (int il = 0; il < layer_idx; il++)
            if (((il + 1) % w.full_attention_interval) != 0) dn_idx++;
        cur = build_delta_net_block(ctx, gf, w, L, cur,
                                     cache.conv_state[dn_idx], cache.ssm_state[dn_idx],
                                     n_tokens, nullptr, nullptr);
    }

    // Residual after attention
    cur = ggml_add(ctx, cur, inpSA);

    // Post-attention norm → input to MoE FFN
    ggml_tensor * post = rms_norm_mul(ctx, cur, L.attn_post_norm, MOE_EPS);

    // ── Save intermediates to MoeState (F32→F32 GPU copies) ──
    ggml_tensor * post_dst = ggml_view_2d(ctx, moe.post,
        hidden, n_tokens, moe.post->nb[1], 0);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, post, post_dst));

    ggml_tensor * res_dst = ggml_view_2d(ctx, moe.ffn_residual,
        hidden, n_tokens, moe.ffn_residual->nb[1], 0);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, cur, res_dst));

    // ── Router ──
    ggml_tensor * logits = ggml_mul_mat(ctx, L.ffn_gate_inp, post); // [n_expert, n_tokens]
    ggml_tensor * probs  = ggml_soft_max(ctx, logits);               // [n_expert, n_tokens]

    // Top-k selection: argsort_top_k returns a non-contiguous view (stride=n_total).
    // Use ggml_cont to make it contiguous for linear D2H read.
    ggml_tensor * selected = ggml_cont(ctx, ggml_argsort_top_k(ctx, probs, n_used));
    ggml_set_name(selected, "selected_experts");
    ggml_set_output(selected);  // persist for D2H after graph_compute

    // ── Weight extraction + normalization ──
    ggml_tensor * probs_3d = ggml_reshape_3d(ctx, probs, 1, n_total, n_tokens);
    ggml_tensor * weights  = ggml_get_rows(ctx, probs_3d, selected); // [1, n_used, n_tokens]

    // Normalize: weights /= sum(weights) per token
    ggml_tensor * weights_2d  = ggml_reshape_2d(ctx, weights, n_used, n_tokens);
    ggml_tensor * weights_sum = ggml_sum_rows(ctx, weights_2d); // [1, n_tokens]
    weights_sum = ggml_clamp(ctx, weights_sum, 6.103515625e-5f, INFINITY);
    weights_2d  = ggml_div(ctx, weights_2d, weights_sum);        // [n_used, n_tokens]
    weights     = ggml_reshape_3d(ctx, weights_2d, 1, n_used, n_tokens);

    // Save weights to MoeState
    ggml_tensor * wt_dst = ggml_view_3d(ctx, moe.weights,
        1, n_used, n_tokens,
        moe.weights->nb[1], moe.weights->nb[2], 0);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, weights, wt_dst));

    return selected;
}

// ─── Graph B: MoE FFN ─────────────────────────────────────────────────

// Builds Graph B: expert FFN via mul_mat_id + shared expert FFN + residual.
// Reads from MoeState persistent tensors (post, ffn_residual, weights, slot_ids).
// Returns the final layer output [hidden, n_tokens].
static ggml_tensor * build_moe_graph_b(
    ggml_context *        ctx,
    ggml_cgraph *         gf,
    const TargetWeights & w,
    ExpertCache &         ecache,
    MoeState &            moe,
    int                   layer_idx,
    int                   n_tokens)
{
    const int hidden = w.n_embd;
    const int n_used = w.n_experts_active;
    const TargetLayer & L = w.layers[layer_idx];

    // Read persistent tensors (views with current n_tokens)
    ggml_tensor * post = ggml_view_2d(ctx, moe.post,
        hidden, n_tokens, moe.post->nb[1], 0);
    ggml_set_input(post);

    ggml_tensor * residual = ggml_view_2d(ctx, moe.ffn_residual,
        hidden, n_tokens, moe.ffn_residual->nb[1], 0);
    ggml_set_input(residual);

    ggml_tensor * weights = ggml_view_3d(ctx, moe.weights,
        1, n_used, n_tokens,
        moe.weights->nb[1], moe.weights->nb[2], 0);
    ggml_set_input(weights);

    ggml_tensor * slot_ids = ggml_view_2d(ctx, moe.slot_ids,
        n_used, n_tokens, moe.slot_ids->nb[1], 0);
    ggml_set_input(slot_ids);

    ggml_tensor * down_slot_ids = ggml_view_2d(ctx, moe.down_slot_ids,
        n_used, n_tokens, moe.down_slot_ids->nb[1], 0);
    ggml_set_input(down_slot_ids);

    // ── Expert FFN via mul_mat_id ──
    // Input: [hidden, 1, n_tokens] (broadcast across n_used experts)
    ggml_tensor * cur_3d = ggml_reshape_3d(ctx, post, hidden, 1, n_tokens);

    // Gate projection: [hidden, expert_ffn, n_slots] × [hidden, 1, n_tokens] → [expert_ffn, n_used, n_tokens]
    ggml_tensor * gate_out = ggml_mul_mat_id(ctx, ecache.gate_3d(), cur_3d, slot_ids);

    // Up projection: same shape
    ggml_tensor * up_out = ggml_mul_mat_id(ctx, ecache.up_3d(), cur_3d, slot_ids);

    // SwiGLU activation
    ggml_tensor * swiglu = ggml_swiglu_split(ctx, gate_out, up_out); // [expert_ffn, n_used, n_tokens]

    // Down projection: [expert_ffn, hidden, n_slots] × [expert_ffn, n_used, n_tokens] → [hidden, n_used, n_tokens]
    ggml_tensor * experts = ggml_mul_mat_id(ctx, ecache.down_3d_for_layer(layer_idx), swiglu, down_slot_ids);

    // Apply gate weights: [hidden, n_used, n_tokens] * [1, n_used, n_tokens]
    experts = ggml_mul(ctx, experts, weights);

    // ── Sum over experts (chain of view_2d + add) ──
    ggml_tensor * moe_out = ggml_view_2d(ctx, experts,
        hidden, n_tokens, experts->nb[2], 0);
    ggml_build_forward_expand(gf, moe_out);

    for (int i = 1; i < n_used; i++) {
        ggml_tensor * slice = ggml_view_2d(ctx, experts,
            hidden, n_tokens, experts->nb[2], (size_t)i * experts->nb[1]);
        ggml_build_forward_expand(gf, slice);
        moe_out = ggml_add(ctx, moe_out, slice);
        ggml_build_forward_expand(gf, moe_out);
    }

    // ── Shared expert FFN ──
    // Reuse build_swiglu_ffn with a temp layer referencing shared expert weights.
    TargetLayer shared_L{};
    shared_L.w_gate = L.shared_w_gate;
    shared_L.w_up   = L.shared_w_up;
    shared_L.w_down = L.shared_w_down;
    ggml_tensor * shared_ffn = build_swiglu_ffn(ctx, post, shared_L);

    // Shared expert gate: sigmoid(matmul(gate_inp_shexp, post)) → [1, n_tokens]
    if (L.ffn_gate_inp_shexp) {
        ggml_tensor * shared_gate_logit = ggml_mul_mat(ctx, L.ffn_gate_inp_shexp, post); // [1, n_tokens]
        ggml_tensor * shared_gate = ggml_sigmoid(ctx, shared_gate_logit);
        shared_ffn = ggml_mul(ctx, shared_ffn, shared_gate); // broadcast [hidden, n_tokens] * [1, n_tokens]
    }

    // Combine routed experts + shared expert
    ggml_tensor * ffn_out = ggml_add(ctx, moe_out, shared_ffn);

    // Add pre-FFN residual
    return ggml_add(ctx, ffn_out, residual);
}

// ─── Per-layer execution ──────────────────────────────────────────────

bool run_qwen35moe_layer(
    ggml_backend_t         backend,
    const TargetWeights &  w,
    TargetCache &          cache,
    ExpertCache &          ecache,
    MoeState &             moe,
    const MoeExpertSource & source,
    int                    layer_idx,
    ggml_tensor *          act_in,
    ggml_tensor *          act_out,
    int                    chunk_start,
    int                    n_tokens,
    ggml_tensor *          positions,
    ggml_tensor *          attn_mask,
    int                    kv_start,
    bool                   capture,
    int                    fa_window,
    ggml_gallocr_t         gallocr_a,
    ggml_gallocr_t         gallocr_b)
{
    const int hidden = w.n_embd;
    const int n_used = w.n_experts_active;

    // ════════════════════════════════════════════════════════════════════
    // Graph A: attention + router
    // ════════════════════════════════════════════════════════════════════

    ggml_init_params ip{};
    ip.mem_size   = 16 * 1024 * 1024;  // 16 MB metadata (plenty for ~200 tensors)
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;

    ggml_context * ctx_a = ggml_init(ip);
    if (!ctx_a) return false;

    ggml_cgraph * gf_a = ggml_new_graph_custom(ctx_a, 4096, false);

    // Input view into activation buffer
    ggml_tensor * inp = ggml_view_2d(ctx_a, act_in,
        hidden, n_tokens, act_in->nb[1], (size_t)chunk_start * act_in->nb[1]);
    ggml_set_input(inp);

    ggml_tensor * selected = build_moe_graph_a(
        ctx_a, gf_a, w, cache, moe, layer_idx,
        inp, positions, attn_mask, kv_start, n_tokens, fa_window);

    // Allocate graph work buffer + execute (reuses pre-allocated gallocr)
    auto t0 = std::chrono::steady_clock::now();
    if (!ggml_gallocr_alloc_graph(gallocr_a, gf_a)) {
        std::fprintf(stderr, "[MoE] Graph A alloc failed layer=%d\n", layer_idx);
        ggml_free(ctx_a);
        return false;
    }
    auto t1 = std::chrono::steady_clock::now();

    auto status = ggml_backend_graph_compute(backend, gf_a);
    auto t2 = std::chrono::steady_clock::now();
    g_time_alloc_ms += std::chrono::duration<double,std::milli>(t1-t0).count();
    g_time_graph_a_ms += std::chrono::duration<double,std::milli>(t2-t1).count();
    if (g_moe_debug_sync) {
        cudaError_t cerr = cudaDeviceSynchronize();
        if (cerr != cudaSuccess) {
            std::fprintf(stderr, "[MoE-DBG] CUDA error after Graph A layer=%d call=%d: %s\n",
                layer_idx, g_moe_debug_call, cudaGetErrorString(cerr));
            return false;  // bail before ggml_abort
        }
        std::fprintf(stderr, "[MoE-DBG] L%d-A OK\n", layer_idx);
    }
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[MoE] Graph A compute failed layer=%d status=%d\n",
                     layer_idx, (int)status);
        ggml_free(ctx_a);
        return false;
    }

    // D2H: read selected experts (gallocr buffer still alive — owned by caller)
    std::vector<int32_t> expert_ids((size_t)n_used * n_tokens);
    ggml_backend_tensor_get(selected, expert_ids.data(), 0,
        sizeof(int32_t) * expert_ids.size());

    ggml_free(ctx_a);

    // ════════════════════════════════════════════════════════════════════
    // Cache preparation (CPU)
    // ════════════════════════════════════════════════════════════════════
    auto tc0 = std::chrono::steady_clock::now();

    // Validate expert IDs are in range
    for (size_t i = 0; i < expert_ids.size(); i++) {
        if (expert_ids[i] < 0 || expert_ids[i] >= w.n_experts) {
            std::fprintf(stderr, "[MoE] invalid expert_id[%zu]=%d at layer=%d (range [0,%d))\n",
                         i, expert_ids[i], layer_idx, w.n_experts);
            return false;
        }
    }

    // Collect unique experts for this layer, then batch-load all misses.
    std::unordered_set<int> unique_set(expert_ids.begin(), expert_ids.end());
    std::vector<int> unique_experts(unique_set.begin(), unique_set.end());
    ecache.batch_ensure_cached(layer_idx, unique_experts.data(),
                               (int)unique_experts.size(), source);

    // Compute slot_ids (gate/up) and down_slot_ids (down).
    std::vector<int32_t> slot_ids((size_t)n_used * n_tokens);
    std::vector<int32_t> down_slot_ids((size_t)n_used * n_tokens);
    for (size_t i = 0; i < expert_ids.size(); i++) {
        slot_ids[i]      = ecache.get_slot(layer_idx, expert_ids[i]);
        down_slot_ids[i] = ecache.get_down_slot(layer_idx, expert_ids[i]);
    }

    // Validate slot_ids are in range for the target tensors.
    if (g_moe_debug_sync) {
        int max_gu = ecache.n_slots();
        int max_dn = (int)ecache.down_3d_for_layer(layer_idx)->ne[2];
        for (size_t i = 0; i < slot_ids.size(); i++) {
            if (slot_ids[i] < 0 || slot_ids[i] >= max_gu) {
                std::fprintf(stderr, "[MoE-DBG] OOB gate/up slot[%zu]=%d (max=%d) layer=%d call=%d\n",
                    i, slot_ids[i], max_gu, layer_idx, g_moe_debug_call);
            }
            if (down_slot_ids[i] < 0 || down_slot_ids[i] >= max_dn) {
                std::fprintf(stderr, "[MoE-DBG] OOB down slot[%zu]=%d (max=%d) layer=%d call=%d\n",
                    i, down_slot_ids[i], max_dn, layer_idx, g_moe_debug_call);
            }
        }
    }

    ggml_backend_tensor_set(moe.slot_ids, slot_ids.data(), 0,
        sizeof(int32_t) * slot_ids.size());
    ggml_backend_tensor_set(moe.down_slot_ids, down_slot_ids.data(), 0,
        sizeof(int32_t) * down_slot_ids.size());

    auto tc1 = std::chrono::steady_clock::now();
    g_time_cache_ms += std::chrono::duration<double,std::milli>(tc1-tc0).count();

    // ════════════════════════════════════════════════════════════════════
    // Graph B: MoE FFN
    // ════════════════════════════════════════════════════════════════════

    ggml_context * ctx_b = ggml_init(ip);
    if (!ctx_b) return false;

    ggml_cgraph * gf_b = ggml_new_graph_custom(ctx_b, 4096, false);

    ggml_tensor * layer_out = build_moe_graph_b(
        ctx_b, gf_b, w, ecache, moe, layer_idx, n_tokens);

    // Copy output to activation buffer
    ggml_tensor * out_view = ggml_view_2d(ctx_b, act_out,
        hidden, n_tokens, act_out->nb[1], (size_t)chunk_start * act_out->nb[1]);
    ggml_build_forward_expand(gf_b, ggml_cpy(ctx_b, layer_out, out_view));

    // ── DFlash feature capture (after MoE FFN, same pattern as dense) ──
    if (capture && cache.target_feat) {
        int capture_idx = -1;
        for (int k = 0; k < DFLASH27B_DRAFT_N_TARGET_LAYERS; k++) {
            if (w.capture_layer_ids[k] == layer_idx) { capture_idx = k; break; }
        }
        if (capture_idx >= 0) {
            const size_t elt        = ggml_element_size(cache.target_feat);
            const size_t col_stride = cache.target_feat->nb[1];
            const int    cap        = cache.target_feat_cap;
            const int    slot_start = kv_start % cap;
            const int    pre_n      = std::min(n_tokens, cap - slot_start);
            const int    post_n     = n_tokens - pre_n;

            ggml_tensor * cur_2d = ggml_reshape_2d(ctx_b, layer_out, hidden, n_tokens);

            {
                const size_t offset =
                    (size_t)slot_start * col_stride +
                    (size_t)capture_idx * hidden * elt;
                ggml_tensor * slot = ggml_view_2d(ctx_b, cache.target_feat,
                    hidden, pre_n, col_stride, offset);
                ggml_tensor * src = ggml_view_2d(ctx_b, cur_2d,
                    hidden, pre_n, cur_2d->nb[1], 0);
                ggml_build_forward_expand(gf_b, ggml_cpy(ctx_b, src, slot));
            }
            if (post_n > 0) {
                const size_t offset =
                    (size_t)capture_idx * hidden * elt;
                ggml_tensor * slot = ggml_view_2d(ctx_b, cache.target_feat,
                    hidden, post_n, col_stride, offset);
                ggml_tensor * src = ggml_view_2d(ctx_b, cur_2d,
                    hidden, post_n, cur_2d->nb[1],
                    (size_t)pre_n * cur_2d->nb[1]);
                ggml_build_forward_expand(gf_b, ggml_cpy(ctx_b, src, slot));
            }
        }
    }

    // Allocate + execute Graph B (reuses pre-allocated gallocr)
    auto tb0 = std::chrono::steady_clock::now();
    if (!ggml_gallocr_alloc_graph(gallocr_b, gf_b)) {
        std::fprintf(stderr, "[MoE] Graph B alloc failed layer=%d\n", layer_idx);
        ggml_free(ctx_b);
        return false;
    }
    auto tb1 = std::chrono::steady_clock::now();

    status = ggml_backend_graph_compute(backend, gf_b);
    auto tb2 = std::chrono::steady_clock::now();
    g_time_alloc_ms += std::chrono::duration<double,std::milli>(tb1-tb0).count();
    g_time_graph_b_ms += std::chrono::duration<double,std::milli>(tb2-tb1).count();
    if (g_moe_debug_sync) {
        cudaError_t cerr = cudaDeviceSynchronize();
        if (cerr != cudaSuccess) {
            std::fprintf(stderr, "[MoE-DBG] CUDA error after Graph B layer=%d call=%d: %s\n",
                layer_idx, g_moe_debug_call, cudaGetErrorString(cerr));
            ggml_free(ctx_b);
            return false;
        }
        std::fprintf(stderr, "[MoE-DBG] L%d-B OK\n", layer_idx);
    }

    ggml_free(ctx_b);

    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[MoE] Graph B compute failed layer=%d status=%d\n",
                     layer_idx, (int)status);
        return false;
    }

    return true;
}

// ─── Full forward pass (all layers + lm_head) ────────────────────────

bool run_qwen35moe_forward(
    ggml_backend_t         backend,
    const TargetWeights &  w,
    TargetCache &          cache,
    ExpertCache &          ecache,
    MoeState &             moe,
    const MoeExpertSource & source,
    ggml_tensor *          act_a,        // [hidden, max_tokens] persistent buffer A
    ggml_tensor *          act_b,        // [hidden, max_tokens] persistent buffer B
    int                    n_tokens,
    ggml_tensor *          positions,    // [4*n_tokens] i32 (pre-filled)
    ggml_tensor *          attn_mask,    // optional [kv_pad, q_pad] f16
    int                    kv_start,
    bool                   capture,
    int                    fa_window,
    ggml_tensor *          logits_out,   // [vocab, n_tokens] f32 (pre-allocated)
    ggml_tensor *          argmax_out)   // [n_tokens] i32 (pre-allocated, or nullptr)
{
    const int hidden  = w.n_embd;
    const int n_layer = w.n_layer;

    g_moe_debug_call++;
    if (g_moe_debug_sync) {
        std::fprintf(stderr, "[MoE-DBG] forward call=%d n_tokens=%d kv_start=%d\n",
            g_moe_debug_call, n_tokens, kv_start);
    }

    // Reset timing accumulators
    g_time_graph_a_ms = 0;
    g_time_graph_b_ms = 0;
    g_time_cache_ms   = 0;
    g_time_alloc_ms   = 0;

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Pre-allocate graph allocators (reused across all layers to avoid per-layer CUDA alloc)
    ggml_gallocr_t gallocr_a = ggml_gallocr_new(buft);
    ggml_gallocr_t gallocr_b = ggml_gallocr_new(buft);

    // Double-buffer: layers read from act_in, write to act_out, then swap.
    ggml_tensor * act_in  = act_a;
    ggml_tensor * act_out = act_b;

    for (int il = 0; il < n_layer; il++) {
        if (g_moe_debug_sync) {
            std::fprintf(stderr, "[MoE-DBG] layer %d/%d (call=%d)\n", il, n_layer, g_moe_debug_call);
        }
        bool ok = run_qwen35moe_layer(
            backend, w, cache, ecache, moe, source,
            il, act_in, act_out,
            /*chunk_start=*/0, n_tokens,
            positions, attn_mask, kv_start, capture, fa_window,
            gallocr_a, gallocr_b);
        if (!ok) {
            std::fprintf(stderr, "[MoE] forward failed at layer %d\n", il);
            ggml_gallocr_free(gallocr_a);
            ggml_gallocr_free(gallocr_b);
            return false;
        }
        std::swap(act_in, act_out);
    }

    ggml_gallocr_free(gallocr_a);
    ggml_gallocr_free(gallocr_b);

    // Print timing breakdown (when debug enabled)
    if (g_moe_debug_sync) {
        std::fprintf(stderr, "[MoE-PERF] call=%d n_tok=%d: graph_a=%.1f graph_b=%.1f cache=%.1f alloc=%.1f ms\n",
            g_moe_debug_call, n_tokens,
            g_time_graph_a_ms, g_time_graph_b_ms, g_time_cache_ms, g_time_alloc_ms);
    }

    // act_in now holds the output of the last layer (after final swap).
    // Build a small graph: RMSNorm → lm_head → (argmax)
    ggml_init_params ip{};
    ip.mem_size   = 64 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;

    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 256, false);

    ggml_tensor * inp_view = ggml_view_2d(ctx, act_in,
        hidden, n_tokens, act_in->nb[1], 0);
    ggml_set_input(inp_view);

    // Final RMSNorm
    ggml_tensor * cur = ggml_rms_norm(ctx, inp_view, 1e-6f);
    cur = ggml_mul(ctx, cur, w.out_norm);

    // lm_head
    ggml_tensor * logits = ggml_mul_mat(ctx, w.output, cur);
    ggml_set_name(logits, "logits");

    // Copy logits to persistent output tensor
    ggml_tensor * logits_dst = ggml_view_2d(ctx, logits_out,
        logits_out->ne[0], n_tokens, logits_out->nb[1], 0);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, logits, logits_dst));

    // Optional argmax
    if (argmax_out) {
        ggml_tensor * argmax = ggml_argmax(ctx, logits);
        ggml_tensor * argmax_dst = ggml_view_1d(ctx, argmax_out, n_tokens, 0);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, argmax, argmax_dst));
    }

    ggml_gallocr_t gallocr = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(gallocr, gf)) {
        std::fprintf(stderr, "[MoE] lm_head graph alloc failed\n");
        ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        return false;
    }

    auto status = ggml_backend_graph_compute(backend, gf);
    if (g_moe_debug_sync) {
        cudaError_t cerr = cudaDeviceSynchronize();
        if (cerr != cudaSuccess) {
            std::fprintf(stderr, "[MoE-DBG] CUDA error in lm_head graph call=%d: %s\n",
                g_moe_debug_call, cudaGetErrorString(cerr));
            ggml_gallocr_free(gallocr);
            ggml_free(ctx);
            return false;
        }
        std::fprintf(stderr, "[MoE-DBG] lm_head OK (call=%d)\n", g_moe_debug_call);
    }
    ggml_gallocr_free(gallocr);
    ggml_free(ctx);

    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[MoE] lm_head graph compute failed status=%d\n", (int)status);
        return false;
    }

    return true;
}

} // namespace dflash27b
