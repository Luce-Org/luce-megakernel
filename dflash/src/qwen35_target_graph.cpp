// Forward pass of Qwen3.5-27B (qwen35 hybrid) in pure ggml.
//
// Translates llama.cpp's `src/models/qwen35.cpp` + `delta-net-base.cpp` into
// our standalone library, hardcoded for Qwen3.5-27B dimensions. No
// llama.cpp runtime is linked — only ggml ops.
//
// Architecture highlights:
//   - 64 layers; every 4th (il % 4 == 3) is full attention, rest are Gated DeltaNet
//   - Full-attention Q projection is PACKED with a gate (attn_q has width 2*q_dim)
//   - Full attention uses M-RoPE with sections [11,11,10,0]
//   - Flash attention is GQA 24/4, causal
//   - Delta-net uses ggml_ssm_conv for the 1D conv + ggml_gated_delta_net for the recurrence
//   - FFN is SwiGLU (w_gate * silu, element-wise multiply with w_up, then w_down)
//
// State (persisted in TargetCache across calls):
//   - attn_k[16], attn_v[16]     : KV cache for full-attn layers, f16
//   - conv_state[48]             : 1D conv recurrence state, f32
//   - ssm_state[48]              : delta-net recurrent state (head_v^2 × H_v), f32
//
// Key dimensions (all hardcoded via DFLASH27B_* macros):
//   n_embd           = 5120
//   n_head           = 24    head_dim = 256   q_dim = n_head * head_dim = 6144
//   n_head_kv        = 4     kv_dim = 4 * 256 = 1024
//   n_ff             = 17408
//   d_inner (ssm)    = 6144
//   d_state (ssm)    = 128
//   dt_rank (ssm)    = 48    (num_v_heads)
//   n_group (ssm)    = 16    (num_k_heads)
//   head_v_dim       = d_inner / dt_rank = 128
//   head_k_dim       = d_state           = 128
//   conv_kernel      = 4

#include "internal.h"
#include "delta_net_chunked.h"
#include "kv_quant.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace dflash27b {

// ─── Local qwen35 constants (from the GGUF, hardcoded for this model) ─
// These complement the DFLASH27B_* macros in dflash27b.h with qwen35-specific
// hparams that differ from the draft (which uses plain Qwen3 dims).
namespace q35 {
constexpr int N_HEAD        = 24;
constexpr int N_HEAD_KV     = 4;
constexpr int HEAD_DIM      = 256;   // key_length == value_length
constexpr int Q_DIM         = N_HEAD * HEAD_DIM;    // 6144
constexpr int KV_DIM        = N_HEAD_KV * HEAD_DIM; // 1024
constexpr int FFN_DIM       = 17408;

constexpr int SSM_D_INNER   = 6144;
constexpr int SSM_D_STATE   = 128;
constexpr int SSM_DT_RANK   = 48;
constexpr int SSM_N_GROUP   = 16;
constexpr int SSM_CONV_KERN = 4;

// Derived
constexpr int HEAD_V_DIM    = SSM_D_INNER / SSM_DT_RANK;  // 128
constexpr int HEAD_K_DIM    = SSM_D_STATE;                // 128
constexpr int CONV_CHANNELS = SSM_D_INNER + 2 * SSM_N_GROUP * SSM_D_STATE; // 6144 + 2*16*128 = 10240

constexpr float EPS         = 1e-6f;
constexpr float ROPE_THETA  = 10000000.0f;
}  // namespace q35

// ─── TargetCache allocation ─────────────────────────────────────────

bool create_target_cache(const TargetWeights & w,
                         int max_ctx,
                         int max_verify_tokens,
                         ggml_backend_t backend,
                         TargetCache & out,
                         bool prefill_only) {
    return create_target_cache_partial(w, max_ctx, max_verify_tokens, backend,
                                       out, prefill_only,
                                       0, w.n_layer, true);
}

bool create_target_cache_partial(const TargetWeights & w,
                                 int max_ctx,
                                 int max_verify_tokens,
                                 ggml_backend_t backend,
                                 TargetCache & out,
                                 bool prefill_only,
                                 int layer_begin,
                                 int layer_end,
                                 bool allocate_target_feat) {
    if (layer_begin < 0) layer_begin = 0;
    if (layer_end < 0 || layer_end > w.n_layer) layer_end = w.n_layer;
    if (layer_begin > layer_end) {
        set_last_error("invalid target cache layer range");
        return false;
    }
    out.backend = backend;
    out.max_ctx = max_ctx;
    out.cur_pos = 0;
    if (max_verify_tokens <= 0) {
        max_verify_tokens = DFLASH27B_DRAFT_BLOCK_SIZE;
    }

    const int n_full_attn = w.n_layer / w.full_attention_interval; // 16
    const int n_delta     = w.n_layer - n_full_attn;               // 48
    const int head_dim    = w.n_embd_head_k;
    const int head_v_dim  = w.ssm_d_inner / w.ssm_dt_rank;
    const int conv_ch     = w.ssm_d_inner + 2 * w.ssm_n_group * w.ssm_d_state;

    out.attn_k.assign(n_full_attn, nullptr);
    out.attn_v.assign(n_full_attn, nullptr);
    out.ssm_state.assign(n_delta, nullptr);
    out.conv_state.assign(n_delta, nullptr);
    out.ssm_state_snap.assign(n_delta, nullptr);
    out.conv_state_snap.assign(n_delta, nullptr);
    out.ssm_intermediate.assign(n_delta, nullptr);
    out.conv_input_cache.assign(n_delta, nullptr);

    // KV cache element types (resolved from env; aborts on unsupported pair).
    ggml_type kv_k_type = GGML_TYPE_Q8_0;
    ggml_type kv_v_type = GGML_TYPE_Q8_0;
    dflash::resolve_kv_types(kv_k_type, kv_v_type);
    out.kv_k_type = kv_k_type;
    out.kv_v_type = kv_v_type;
    const int max_ctx_alloc = (kv_k_type == GGML_TYPE_TQ3_0 || kv_v_type == GGML_TYPE_TQ3_0)
        ? ((max_ctx + 255) / 256) * 256
        : max_ctx;

    // ── Base context: KV cache + SSM/conv state + target_feat ────────
    {
        const int base_tensors = 2 * n_full_attn + 2 * n_delta + 1;
        ggml_init_params ip{};
        ip.mem_size   = (size_t)(base_tensors + 16) * ggml_tensor_overhead();
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        out.base_ctx = ggml_init(ip);
        if (!out.base_ctx) { set_last_error("base cache ggml_init failed"); return false; }

        int fa_idx = 0, dn_idx = 0;
        for (int il = 0; il < w.n_layer; il++) {
            const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);
            const bool owns_layer = il >= layer_begin && il < layer_end;
            if (is_attn) {
                if (!owns_layer) { fa_idx++; continue; }
                // [head_dim, max_ctx_alloc, n_head_kv]
                ggml_tensor * K = ggml_new_tensor_3d(out.base_ctx, kv_k_type,
                                                     head_dim, max_ctx_alloc, w.n_head_kv);
                ggml_tensor * V = ggml_new_tensor_3d(out.base_ctx, kv_v_type,
                                                     head_dim, max_ctx_alloc, w.n_head_kv);
                char name[64];
                std::snprintf(name, sizeof(name), "cache_k_%d", il);
                ggml_set_name(K, name);
                std::snprintf(name, sizeof(name), "cache_v_%d", il);
                ggml_set_name(V, name);
                out.attn_k[fa_idx] = K;
                out.attn_v[fa_idx] = V;
                fa_idx++;
            } else {
                if (!owns_layer) { dn_idx++; continue; }
                // ssm_state: [head_v_dim, head_v_dim, num_v_heads]
                ggml_tensor * S = ggml_new_tensor_3d(out.base_ctx, GGML_TYPE_F32,
                                                     head_v_dim, head_v_dim, w.ssm_dt_rank);
                // conv_state: [kernel-1, conv_channels]
                ggml_tensor * C = ggml_new_tensor_2d(out.base_ctx, GGML_TYPE_F32,
                                                     w.ssm_d_conv - 1, conv_ch);
                char name[64];
                std::snprintf(name, sizeof(name), "ssm_state_%d", il);  ggml_set_name(S, name);
                std::snprintf(name, sizeof(name), "conv_state_%d", il); ggml_set_name(C, name);
                out.ssm_state[dn_idx]  = S;
                out.conv_state[dn_idx] = C;
                dn_idx++;
            }
        }

        constexpr int TARGET_FEAT_CAP_DEFAULT = 4096;
        out.target_feat_cap = std::min(max_ctx, TARGET_FEAT_CAP_DEFAULT);
        if (allocate_target_feat) {
            const int fc_in = DFLASH27B_DRAFT_N_TARGET_LAYERS * w.n_embd;  // 25600
            out.target_feat = ggml_new_tensor_2d(out.base_ctx, GGML_TYPE_BF16, fc_in, out.target_feat_cap);
            ggml_set_name(out.target_feat, "target_feat");
        } else {
            out.target_feat = nullptr;
        }

        out.base_buf = ggml_backend_alloc_ctx_tensors(out.base_ctx, backend);
        if (!out.base_buf) {
            set_last_error("ggml_backend_alloc_ctx_tensors failed for base cache");
            ggml_free(out.base_ctx);
            out.base_ctx = nullptr;
            return false;
        }
    }

    // ── Rollback context: snapshots + intermediates ───────────────────
    if (!prefill_only) {
        const int rb_tensors = 4 * n_delta;
        ggml_init_params ip{};
        ip.mem_size   = (size_t)(rb_tensors + 16) * ggml_tensor_overhead();
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        out.rollback_ctx = ggml_init(ip);
        if (!out.rollback_ctx) { set_last_error("rollback cache ggml_init failed"); return false; }

        int dn_idx = 0;
        for (int il = 0; il < w.n_layer; il++) {
            if (((il + 1) % w.full_attention_interval) != 0) {
                const bool owns_layer = il >= layer_begin && il < layer_end;
                if (!owns_layer) { dn_idx++; continue; }
                ggml_tensor * Sn = ggml_new_tensor_3d(out.rollback_ctx, GGML_TYPE_F32,
                                                       head_v_dim, head_v_dim, w.ssm_dt_rank);
                ggml_tensor * Cn = ggml_new_tensor_2d(out.rollback_ctx, GGML_TYPE_F32,
                                                       w.ssm_d_conv - 1, conv_ch);
                ggml_tensor * Si = ggml_new_tensor_4d(out.rollback_ctx, GGML_TYPE_Q8_0,
                                                       head_v_dim, head_v_dim,
                                                       w.ssm_dt_rank, max_verify_tokens);
                ggml_tensor * Ci = ggml_new_tensor_3d(out.rollback_ctx, GGML_TYPE_F32,
                                                       (w.ssm_d_conv - 1) + max_verify_tokens,
                                                       conv_ch, 1);
                char name[64];
                std::snprintf(name, sizeof(name), "ssm_state_snap_%d", il);  ggml_set_name(Sn, name);
                std::snprintf(name, sizeof(name), "conv_state_snap_%d", il); ggml_set_name(Cn, name);
                std::snprintf(name, sizeof(name), "ssm_intermediate_%d", il); ggml_set_name(Si, name);
                std::snprintf(name, sizeof(name), "conv_input_cache_%d", il); ggml_set_name(Ci, name);
                out.ssm_state_snap[dn_idx]  = Sn;
                out.conv_state_snap[dn_idx] = Cn;
                out.ssm_intermediate[dn_idx] = Si;
                out.conv_input_cache[dn_idx] = Ci;
                dn_idx++;
            }
        }

        out.rollback_buf = ggml_backend_alloc_ctx_tensors(out.rollback_ctx, backend);
        if (!out.rollback_buf) {
            set_last_error("ggml_backend_alloc_ctx_tensors failed for rollback cache");
            ggml_free(out.rollback_ctx);
            out.rollback_ctx = nullptr;
            return false;
        }
    }

    // ── Zero-initialize all state tensors ─────────────────────────────
    std::vector<uint8_t> zeros(1 * 1024 * 1024, 0);
    ggml_context * ctx_list[] = { out.base_ctx, out.rollback_ctx };
    for (int ci = 0; ci < 2; ci++) {
        ggml_context * c = ctx_list[ci];
        if (!c) continue;
        for (ggml_tensor * t = ggml_get_first_tensor(c); t != nullptr;
             t = ggml_get_next_tensor(c, t)) {
            size_t nb = ggml_nbytes(t);
            size_t off = 0;
            while (off < nb) {
                size_t chunk = std::min(nb - off, zeros.size());
                ggml_backend_tensor_set(t, zeros.data(), off, chunk);
                off += chunk;
            }
        }
    }

    return true;
}

void free_target_cache(TargetCache & c) {
    if (c.base_buf)     { ggml_backend_buffer_free(c.base_buf);     c.base_buf     = nullptr; }
    if (c.base_ctx)     { ggml_free(c.base_ctx);                   c.base_ctx     = nullptr; }
    if (c.rollback_buf) { ggml_backend_buffer_free(c.rollback_buf); c.rollback_buf = nullptr; }
    if (c.rollback_ctx) { ggml_free(c.rollback_ctx);               c.rollback_ctx = nullptr; }
    c.attn_k.clear();
    c.attn_v.clear();
    c.ssm_state.clear();
    c.conv_state.clear();
    c.ssm_state_snap.clear();
    c.conv_state_snap.clear();
    c.ssm_intermediate.clear();
    c.conv_input_cache.clear();
    c.target_feat = nullptr;
    c.cur_pos = 0;
}

void reset_target_cache(TargetCache & c) {
    c.cur_pos = 0;
    std::vector<uint8_t> zeros(1 * 1024 * 1024, 0);
    ggml_context * ctx_list[] = { c.base_ctx, c.rollback_ctx };
    for (int ci = 0; ci < 2; ci++) {
        ggml_context * ctx = ctx_list[ci];
        if (!ctx) continue;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr;
             t = ggml_get_next_tensor(ctx, t)) {
            size_t nb = ggml_nbytes(t);
            size_t off = 0;
            while (off < nb) {
                size_t chunk = std::min(nb - off, zeros.size());
                ggml_backend_tensor_set(t, zeros.data(), off, chunk);
                off += chunk;
            }
        }
    }
}

// Attach rollback tensors to an existing prefill cache without touching the
// base tensors (KV, SSM, conv, target_feat) that prefill already populated.
// No D2D copies — the base tensors stay right where the graph wrote them.
// If rollback tensors are already present (e.g. daemon mode second request),
// this is a no-op.
bool migrate_prefill_cache(const TargetWeights & w,
                           int max_ctx,
                           int max_verify_tokens,
                           ggml_backend_t backend,
                           TargetCache & cache) {
    // Already migrated (e.g. daemon mode second+ request after reset_target_cache).
    if (cache.rollback_ctx) return true;

    const int n_delta = (int)cache.ssm_state.size(); // 48
    const int head_v_dim = w.ssm_d_inner / w.ssm_dt_rank;
    const int conv_ch = w.ssm_d_inner + 2 * w.ssm_n_group * w.ssm_d_state;
    if (max_verify_tokens <= 0) {
        max_verify_tokens = DFLASH27B_DRAFT_BLOCK_SIZE;
    }

    cache.ssm_state_snap.assign(n_delta, nullptr);
    cache.conv_state_snap.assign(n_delta, nullptr);
    cache.ssm_intermediate.assign(n_delta, nullptr);
    cache.conv_input_cache.assign(n_delta, nullptr);

    const int rb_tensors = 4 * n_delta;
    ggml_init_params ip{};
    ip.mem_size   = (size_t)(rb_tensors + 16) * ggml_tensor_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    cache.rollback_ctx = ggml_init(ip);
    if (!cache.rollback_ctx) { set_last_error("rollback cache ggml_init failed"); return false; }

    int dn_idx = 0;
    for (int il = 0; il < w.n_layer; il++) {
        if (((il + 1) % w.full_attention_interval) != 0) {
            ggml_tensor * Sn = ggml_new_tensor_3d(cache.rollback_ctx, GGML_TYPE_F32,
                                                   head_v_dim, head_v_dim, w.ssm_dt_rank);
            ggml_tensor * Cn = ggml_new_tensor_2d(cache.rollback_ctx, GGML_TYPE_F32,
                                                   w.ssm_d_conv - 1, conv_ch);
            ggml_tensor * Si = ggml_new_tensor_4d(cache.rollback_ctx, GGML_TYPE_F16,
                                                   head_v_dim, head_v_dim,
                                                   w.ssm_dt_rank, max_verify_tokens);
            ggml_tensor * Ci = ggml_new_tensor_3d(cache.rollback_ctx, GGML_TYPE_F32,
                                                   (w.ssm_d_conv - 1) + max_verify_tokens,
                                                   conv_ch, 1);
            char name[64];
            std::snprintf(name, sizeof(name), "ssm_state_snap_%d", il);  ggml_set_name(Sn, name);
            std::snprintf(name, sizeof(name), "conv_state_snap_%d", il); ggml_set_name(Cn, name);
            std::snprintf(name, sizeof(name), "ssm_intermediate_%d", il); ggml_set_name(Si, name);
            std::snprintf(name, sizeof(name), "conv_input_cache_%d", il); ggml_set_name(Ci, name);
            cache.ssm_state_snap[dn_idx]  = Sn;
            cache.conv_state_snap[dn_idx] = Cn;
            cache.ssm_intermediate[dn_idx] = Si;
            cache.conv_input_cache[dn_idx] = Ci;
            dn_idx++;
        }
    }

    cache.rollback_buf = ggml_backend_alloc_ctx_tensors(cache.rollback_ctx, backend);
    if (!cache.rollback_buf) {
        set_last_error("ggml_backend_alloc_ctx_tensors failed for rollback cache");
        ggml_free(cache.rollback_ctx);
        cache.rollback_ctx = nullptr;
        return false;
    }

    // Zero-initialize rollback tensors
    std::vector<uint8_t> zeros(1 * 1024 * 1024, 0);
    for (ggml_tensor * t = ggml_get_first_tensor(cache.rollback_ctx); t != nullptr;
         t = ggml_get_next_tensor(cache.rollback_ctx, t)) {
        size_t nb = ggml_nbytes(t);
        size_t off = 0;
        while (off < nb) {
            size_t chunk = std::min(nb - off, zeros.size());
            ggml_backend_tensor_set(t, zeros.data(), off, chunk);
            off += chunk;
        }
    }

    return true;
}

// Snapshot/restore SSM+conv state for speculative rollback. Uses device-side
// tensor copy (ggml_backend_tensor_copy). Called outside of any compute graph.
void snapshot_ssm_state(TargetCache & c) {
    for (size_t i = 0; i < c.ssm_state.size(); i++) {
        if (!c.ssm_state[i] || !c.ssm_state_snap[i]) continue;
        ggml_backend_tensor_copy(c.ssm_state[i], c.ssm_state_snap[i]);
        if (!c.conv_state[i] || !c.conv_state_snap[i]) continue;
        ggml_backend_tensor_copy(c.conv_state[i], c.conv_state_snap[i]);
    }
}

void restore_ssm_state(TargetCache & c) {
    for (size_t i = 0; i < c.ssm_state.size(); i++) {
        if (!c.ssm_state_snap[i] || !c.ssm_state[i]) continue;
        ggml_backend_tensor_copy(c.ssm_state_snap[i], c.ssm_state[i]);
        if (!c.conv_state_snap[i] || !c.conv_state[i]) continue;
        ggml_backend_tensor_copy(c.conv_state_snap[i], c.conv_state[i]);
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────

static ggml_tensor * rms_norm_mul(ggml_context * ctx, ggml_tensor * x,
                                  ggml_tensor * weight, float eps) {
    ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, n, weight);
}

static ggml_tensor * build_swiglu_ffn(ggml_context * ctx, ggml_tensor * cur,
                                      const TargetLayer & L) {
    ggml_tensor * gate = ggml_mul_mat(ctx, L.w_gate, cur);   // [inter, n_tokens]
    gate = ggml_silu(ctx, gate);
    ggml_tensor * up = ggml_mul_mat(ctx, L.w_up, cur);
    ggml_tensor * gu = ggml_mul(ctx, gate, up);
    return ggml_mul_mat(ctx, L.w_down, gu);                  // [hidden, n_tokens]
}

// Full-attention block (matches llama.cpp's build_layer_attn for qwen35)
//
// `cache_k` / `cache_v` are the persistent KV buffers for this layer
// (shape [head_dim, max_ctx, n_head_kv] f16). We write the new K/V for
// `n_tokens` new positions starting at `kv_start`, then run causal attention
// over [0..kv_start + n_tokens).
static ggml_tensor * build_full_attn_block(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const TargetWeights & w,
    const TargetLayer & L,
    ggml_tensor * cur,
    ggml_tensor * positions,
    const int * rope_sections,
    ggml_tensor * cache_k,
    ggml_tensor * cache_v,
    ggml_tensor * attn_mask,
    int kv_start,
    int n_tokens,
    ggml_type kv_k_type,
    ggml_type kv_v_type,
    int fa_window = 0,
    ggml_tensor * q_tail_capture = nullptr,
    int q_tail_start = 0
) {
    const int head_dim = w.n_embd_head_k;
    const int n_head = w.n_head;
    const int n_head_kv = w.n_head_kv;
    const int q_dim = head_dim * n_head;
    // ── Q projection (packed Q || gate), shape [2*q_dim, n_tokens]
    ggml_tensor * QG = ggml_mul_mat(ctx, L.wq, cur);
    // Reshape to [head_dim*2, n_head, n_tokens] so we can view the Q and gate halves
    QG = ggml_reshape_3d(ctx, QG, head_dim * 2, n_head, n_tokens);

    // Q half: view at offset 0, stride head_dim*2
    // Layout: [head_dim, n_head, n_tokens]
    ggml_tensor * Q = ggml_view_3d(ctx, QG,
        head_dim, n_head, n_tokens,
        ggml_element_size(QG) * head_dim * 2,                 // nb1: stride over n_head
        ggml_element_size(QG) * head_dim * 2 * n_head,   // nb2: stride over n_tokens
        /*offset*/ 0);
    Q = rms_norm_mul(ctx, Q, L.q_norm, w.rms_eps);

    // Gate half: view at offset head_dim
    ggml_tensor * gate = ggml_view_3d(ctx, QG,
        head_dim, n_head, n_tokens,
        ggml_element_size(QG) * head_dim * 2,
        ggml_element_size(QG) * head_dim * 2 * n_head,
        ggml_element_size(QG) * head_dim);
    gate = ggml_cont_2d(ctx, gate, q_dim, n_tokens);  // [q_dim, n_tokens]

    // ── K and V projections
    ggml_tensor * Kcur = ggml_mul_mat(ctx, L.wk, cur);   // [kv_dim, n_tokens]
    ggml_tensor * Vcur = ggml_mul_mat(ctx, L.wv, cur);   // [kv_dim, n_tokens]

    Kcur = ggml_reshape_3d(ctx, Kcur, head_dim, n_head_kv, n_tokens);
    Kcur = rms_norm_mul(ctx, Kcur, L.k_norm, w.rms_eps);
    Vcur = ggml_reshape_3d(ctx, Vcur, head_dim, n_head_kv, n_tokens);

    // ── M-RoPE (multi-axis rotary). n_rot = HEAD_DIM/4 * 4 ? Actually
    //    ggml_rope_multi takes n_dims = the number of dims to rotate; for
    //    qwen35 that's rope.dimension_count=64 (out of head_dim=256).
    int n_rot = w.rope_dimension_count;
    int sections[4];
    for (int i = 0; i < 4; i++) sections[i] = rope_sections[i];

    Q = ggml_rope_multi(ctx, Q, positions, /*freq_factors=*/nullptr,
                        n_rot, sections, GGML_ROPE_TYPE_MROPE,
                        /*n_ctx_orig=*/0, w.rope_theta, 1.0f,
                        0.0f, 1.0f, 0.0f, 0.0f);
    Kcur = ggml_rope_multi(ctx, Kcur, positions, nullptr,
                           n_rot, sections, GGML_ROPE_TYPE_MROPE,
                           0, w.rope_theta, 1.0f,
                           0.0f, 1.0f, 0.0f, 0.0f);

    if (q_tail_capture) {
        const int chunk_lo = kv_start;
        const int chunk_hi = kv_start + n_tokens;
        const int cap_n = (int) q_tail_capture->ne[2];
        const int tail_lo = q_tail_start;
        const int tail_hi = q_tail_start + cap_n;
        const int ov_lo = std::max(chunk_lo, tail_lo);
        const int ov_hi = std::min(chunk_hi, tail_hi);
        if (ov_lo < ov_hi) {
            const int local_lo = ov_lo - chunk_lo;
            const int cap_lo = ov_lo - tail_lo;
            const int n_cap = ov_hi - ov_lo;
            ggml_tensor * q_src = ggml_view_3d(ctx, Q,
                head_dim, n_head, n_cap,
                Q->nb[1], Q->nb[2], (size_t)local_lo * Q->nb[2]);
            q_src = ggml_cont(ctx, q_src);
            q_src = ggml_reshape_1d(ctx, q_src, head_dim * n_head * n_cap);
            ggml_tensor * q_dst = ggml_view_1d(ctx, q_tail_capture,
                head_dim * n_head * n_cap,
                (size_t)cap_lo * q_tail_capture->nb[2]);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, q_src, q_dst));
        }
    }

    // ── Write K/V into the persistent cache at slot [kv_start..kv_start+n_tokens)
    //
    // cache_k is [head_dim, max_ctx, n_head_kv]. We want to copy Kcur
    // [head_dim, n_head_kv, n_tokens] into cache_k[:, kv_start:kv_start+n_tokens, :].
    //
    // Easiest: transpose Kcur to [head_dim, n_tokens, n_head_kv] so its axes
    // line up with cache_k's [head_dim, max_ctx, n_head_kv], then view a slice
    // of cache_k and copy.
    ggml_tensor * Kcur_T = ggml_permute(ctx, Kcur, 0, 2, 1, 3);  // [head_dim, n_tokens, n_head_kv]
    ggml_tensor * Vcur_T = ggml_permute(ctx, Vcur, 0, 2, 1, 3);  // [head_dim, n_tokens, n_head_kv]

    ggml_tensor * k_slot = ggml_view_3d(ctx, cache_k,
        head_dim, n_tokens, n_head_kv,
        cache_k->nb[1], cache_k->nb[2],
        /*offset*/ cache_k->nb[1] * kv_start);
    ggml_tensor * v_slot = ggml_view_3d(ctx, cache_v,
        head_dim, n_tokens, n_head_kv,
        cache_v->nb[1], cache_v->nb[2],
        cache_v->nb[1] * kv_start);

    ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur_T, k_slot));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur_T, v_slot));

    // ── Flash attention over the valid slice
    // When fa_window > 0 and kv_start >= fa_window, only attend to the last
    // fa_window positions. This dramatically reduces FA cost during speculative
    // decode verify/replay at long contexts (60K+ kv entries).
    const int win_start = (fa_window > 0 && kv_start > fa_window)
                              ? (kv_start - fa_window) : 0;
    const int kv_len = kv_start + n_tokens;
    const int win_len = kv_len - win_start;

    const int fattn_stride  = (kv_k_type == GGML_TYPE_TQ3_0 || kv_v_type == GGML_TYPE_TQ3_0) ? 256 : 1;
    const int win_len_padded = ((win_len + fattn_stride - 1) / fattn_stride) * fattn_stride;

    ggml_tensor * Qfa = ggml_permute(ctx, Q, 0, 2, 1, 3);
    Qfa = ggml_cont(ctx, Qfa);

    // For TQ3_0 KV cache, K/V are stored in FWHT-rotated space (the f32->TQ3_0
    // quantize kernel applies tq3_rotate_forward before the centroid search,
    // see ggml-cuda/cpy-utils.cuh quantize_f32_tq3_0_group).
    // Rotation gates are independent for K and V:
    //   * K=TQ3 needs Q rotated forward so softmax(Qfa . Kfa^T) = softmax(QK^T)
    //   * V=TQ3 needs attn_out inverse-rotated to recover plain V space
    const bool q_rotate   = (kv_k_type == GGML_TYPE_TQ3_0);
    const bool out_rotate = (kv_v_type == GGML_TYPE_TQ3_0);
    if (q_rotate) {
        Qfa = ggml_turbo_wht(ctx, Qfa, 0);
    }

    // K and V from cache: a windowed view starting at win_start.
    ggml_tensor * Kfa = ggml_view_3d(ctx, cache_k,
        head_dim, win_len_padded, n_head_kv,
        cache_k->nb[1], cache_k->nb[2], cache_k->nb[1] * win_start);
    ggml_tensor * Vfa = ggml_view_3d(ctx, cache_v,
        head_dim, win_len_padded, n_head_kv,
        cache_v->nb[1], cache_v->nb[2], cache_v->nb[1] * win_start);

    // Causal mask: for n_tokens==1 we don't need one (a single query attending
    // to all keys is trivially causal). For n_tokens>1 the caller must provide
    // a mask shaped [kv_len, n_tokens] with 0 for attendable positions and
    // -inf for positions beyond the causal boundary.
    const float kq_scale = 1.0f / std::sqrt((float)head_dim);
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, Qfa, Kfa, Vfa, attn_mask,
                                             kq_scale, 0.0f, 0.0f);
    // attn: [head_dim, n_head, n_tokens] (permuted)

    // Un-rotate the FA output from FWHT-rotated V space (only when V is TQ3).
    if (out_rotate) {
        attn = ggml_cont(ctx, attn);
        attn = ggml_turbo_wht(ctx, attn, 1);
    }

    attn = ggml_reshape_2d(ctx, attn, q_dim, n_tokens);

    // ── Apply the sigmoid gate from the packed Q
    ggml_tensor * gate_sig = ggml_sigmoid(ctx, gate);
    attn = ggml_mul(ctx, attn, gate_sig);

    // ── Output projection
    attn = ggml_mul_mat(ctx, L.wo, attn);  // [hidden, n_tokens]
    return attn;
}

// Gated DeltaNet block using the fused ggml_gated_delta_net primitive.
//
// Matches the semantics of llama.cpp's build_layer_attn_linear + build_delta_net_fused.
// Updates cache->conv_state and cache->ssm_state in place.
//
// When `cap` is non-null, the function populates `cap->ssm_intermediate_states`
// with a view into the gated_delta_net result's per-step recurrent states and
// `cap->conv_input` with the concatenated conv input (old state + new tokens),
// both of which are marked as graph outputs so the caller can rollback SSM and
// conv state to any intermediate step commit_n-1 without a replay forward pass.
static ggml_tensor * build_delta_net_block(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const TargetWeights & w,
    const TargetLayer & L,
    ggml_tensor * cur,            // [hidden, n_tokens]
    ggml_tensor * conv_state,     // [kernel-1, conv_channels] persistent
    ggml_tensor * ssm_state,      // [head_v_dim, head_v_dim, num_v_heads] persistent
    int n_tokens,
    DeltaNetCapture * cap,        // optional: populated on capture_delta_intermediate
    ggml_tensor * parent_ids      // optional [n_tokens] i32; tree mode when non-null
) {
    const int head_k_dim   = w.ssm_d_state;
    const int num_k_heads  = w.ssm_n_group;
    const int num_v_heads  = w.ssm_dt_rank;
    const int head_v_dim   = w.ssm_d_inner / w.ssm_dt_rank;
    const int conv_channels = w.ssm_d_inner + 2 * w.ssm_n_group * w.ssm_d_state;
    const int n_seqs       = 1;
    const int n_seq_tokens = n_tokens;

    // ── qkv_mixed = wqkv @ cur         [10240, n_tokens]
    ggml_tensor * qkv_mixed = ggml_mul_mat(ctx, L.wqkv, cur);
    qkv_mixed = ggml_reshape_3d(ctx, qkv_mixed, conv_channels, n_seq_tokens, n_seqs);

    // ── z = wqkv_gate @ cur            [inner, n_tokens]
    ggml_tensor * z = ggml_mul_mat(ctx, L.wqkv_gate, cur);

    // ── beta = ssm_beta @ cur          [dt_rank, n_tokens]
    ggml_tensor * beta = ggml_mul_mat(ctx, L.ssm_beta, cur);
    beta = ggml_reshape_4d(ctx, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx, beta);

    // ── alpha = ssm_alpha @ cur        [dt_rank, n_tokens]
    //    alpha = alpha + ssm_dt_bias          (per-head bias)
    //    alpha = softplus(alpha)
    //    g     = alpha * ssm_a                (-A_log.exp() * softplus)
    ggml_tensor * alpha = ggml_mul_mat(ctx, L.ssm_alpha, cur);
    alpha = ggml_reshape_3d(ctx, alpha, num_v_heads, n_seq_tokens, n_seqs);
    alpha = ggml_add(ctx, alpha, L.ssm_dt_bias);
    alpha = ggml_softplus(ctx, alpha);
    ggml_tensor * g_tensor = ggml_mul(ctx, alpha, L.ssm_a);
    g_tensor = ggml_reshape_4d(ctx, g_tensor, 1, num_v_heads, n_seq_tokens, n_seqs);

    // ── Fetch conv state [kernel-1, conv_channels] and prepend to qkv_mixed
    //    along the token axis to form the convolution input.
    ggml_tensor * conv_states_r = ggml_reshape_3d(ctx, conv_state,
        w.ssm_d_conv - 1, conv_channels, n_seqs);

    // qkv_mixed currently is [conv_channels, n_tokens, n_seqs]; we need
    // [n_tokens, conv_channels, n_seqs] to concat on dim 0.
    ggml_tensor * qkv_T = ggml_transpose(ctx, qkv_mixed);

    ggml_tensor * conv_input = ggml_concat(ctx, conv_states_r, qkv_T, 0);
    // conv_input: [kernel-1 + n_tokens, conv_channels, n_seqs]

    // For spec-decode rollback: copy the full conv_input into the persistent
    // cache buffer via an in-graph ggml_cpy. This avoids marking conv_input as
    // a graph output (which would force the gallocr to preserve its memory
    // past graph_compute). After graph_compute, the cache buffer's data is
    // always valid; the rollback code slices it at commit_n.
    if (cap && cap->conv_input) {
        ggml_build_forward_expand(gf, ggml_cpy(ctx, conv_input, cap->conv_input));
    }

    // ── Save the last (kernel-1) steps back to conv_state
    ggml_tensor * last_conv = ggml_view_3d(ctx, conv_input,
        w.ssm_d_conv - 1, conv_channels, n_seqs,
        conv_input->nb[1], conv_input->nb[2],
        (conv_input->ne[0] - (w.ssm_d_conv - 1)) * ggml_element_size(conv_input));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, last_conv, conv_state));

    // ── 1D conv + silu
    //    Tree mode: use the parent-chain-aware variant so sibling nodes gather
    //    their conv window from their actual tree parent instead of the DFS
    //    predecessor. Without this, siblings get garbage logits (the conv
    //    output would mix unrelated branches).
    ggml_tensor * conv_out = parent_ids
        ? ggml_ssm_conv_tree(ctx, conv_input, L.ssm_conv1d, parent_ids)
        : ggml_ssm_conv     (ctx, conv_input, L.ssm_conv1d);
    conv_out = ggml_silu(ctx, conv_out);

    // conv_out: [conv_channels, n_tokens, n_seqs]
    const int64_t q_offset = 0;
    const int64_t k_offset = num_k_heads * head_k_dim;
    const int64_t v_offset = 2 * num_k_heads * head_k_dim;

    const size_t elt = ggml_element_size(conv_out);
    const size_t row_size = conv_channels * elt;

    ggml_tensor * q_c = ggml_view_4d(ctx, conv_out,
        head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        head_k_dim * elt,
        row_size,
        row_size * n_seq_tokens,
        q_offset * elt);
    ggml_tensor * k_c = ggml_view_4d(ctx, conv_out,
        head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        head_k_dim * elt,
        row_size,
        row_size * n_seq_tokens,
        k_offset * elt);
    ggml_tensor * v_c = ggml_view_4d(ctx, conv_out,
        head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
        head_v_dim * elt,
        row_size,
        row_size * n_seq_tokens,
        v_offset * elt);

    // L2 norm on Q and K
    q_c = ggml_l2_norm(ctx, q_c, w.rms_eps);
    k_c = ggml_l2_norm(ctx, k_c, w.rms_eps);

    // Repeat Q and K from num_k_heads to num_v_heads so they match V's layout
    // (only needed if not using the fused op's broadcast support).
    if (num_k_heads != num_v_heads) {
        q_c = ggml_repeat_4d(ctx, q_c, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_c = ggml_repeat_4d(ctx, k_c, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    // ── SSM state (recurrent): reshape to [S_v, S_v, H_v, n_seqs]
    ggml_tensor * s = ggml_reshape_4d(ctx, ssm_state,
        head_v_dim, head_v_dim, num_v_heads, n_seqs);

    // ── Fused Gated DeltaNet op — returns packed (output | new_state [| intermediates]).
    //    In tree mode, the kernel uses parent_ids to reload state at DFS
    //    branch transitions (ported from sglang's retrieve_parent_token path).
    //    When `cap->ssm_intermediate_states` is present AND we are in tree
    //    mode, use the _tree_persist variant: the kernel writes per-token
    //    intermediate states DIRECTLY into the persistent cache buffer,
    //    eliminating the downstream ggml_cpy that would otherwise copy them.
    //    Saves ~5-10 ms per verify step (memory-bandwidth bound) on 27B.
    // tree_persist writes directly to the intermediate buffer. It only supports
    // F32/F16 output; for Q8_0 intermediates, fall back to the legacy ggml_cpy
    // path which handles F32→Q8_0 quantization automatically.
    ggml_tensor * persist_inter = (parent_ids && cap && cap->ssm_intermediate_states
                                   && (cap->ssm_intermediate_states->type == GGML_TYPE_F32
                                       || cap->ssm_intermediate_states->type == GGML_TYPE_F16))
        ? cap->ssm_intermediate_states
        : nullptr;

    // Chunked delta-net path: chain-only (no parent_ids), no per-token
    // capture (no cap). Ported from llama.cpp
    // src/models/delta-net-base.cpp::build_delta_net_chunking. At n_tokens=16
    // and 48 delta-net layers it eliminates the serial per-token loop that
    // dominates target-verify compute at long ctx. Currently OFF by
    // default — port produces correct shape but slightly wrong final state,
    // causing AL degradation and loopy output. Set DFLASH27B_CHUNKED=1 to
    // opt in for A/B testing while debugging.
    bool use_chunked = false;
    if (!parent_ids && !cap && n_seq_tokens > 1) {
        if (const char * s_env = std::getenv("DFLASH27B_CHUNKED")) {
            use_chunked = (std::atoi(s_env) != 0);
        }
    }

    ggml_tensor * output = nullptr;
    ggml_tensor * new_state = nullptr;

    if (use_chunked) {
        auto r = build_delta_net_chunked(ctx, q_c, k_c, v_c, g_tensor, beta, s);
        output    = r.output;
        new_state = r.new_state;
        goto after_delta_net;
    }

    ggml_tensor * result;
    result =
        persist_inter
            ? ggml_gated_delta_net_tree_persist(ctx, q_c, k_c, v_c, g_tensor, beta, s, parent_ids, persist_inter)
            : (parent_ids
                ? ggml_gated_delta_net_tree(ctx, q_c, k_c, v_c, g_tensor, beta, s, parent_ids)
                : ggml_gated_delta_net     (ctx, q_c, k_c, v_c, g_tensor, beta, s));

    // Slice output and new_state out of the packed result
    {
    const int64_t S_v = head_v_dim;
    const int64_t H_v = num_v_heads;
    const size_t r_elt = ggml_element_size(result);
    output = ggml_view_4d(ctx, result,
        S_v, H_v, n_seq_tokens, n_seqs,
        S_v * r_elt,
        S_v * H_v * r_elt,
        S_v * H_v * n_seq_tokens * r_elt,
        0);
    new_state = ggml_view_4d(ctx, result,
        S_v, S_v, H_v, n_seqs,
        S_v * r_elt,
        S_v * S_v * r_elt,
        S_v * S_v * H_v * r_elt,
        S_v * H_v * n_seq_tokens * n_seqs * r_elt);

    // Persist new_state back to cache
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state, ssm_state));

    // Expose per-step intermediate states for spec-decode rollback. The patched
    // ggml_gated_delta_net kernel appends an intermediate-states region to the
    // result tensor after the final-state slot. Layout in result->data:
    //   [ attn_out: S_v*H_v*n_seq_tokens*n_seqs floats
    //   | final_state: S_v*S_v*H_v*n_seqs floats
    //   | intermediate_states: S_v*S_v*H_v*n_seq_tokens*n_seqs floats ]
    //
    // Instead of marking the whole `result` tensor as a graph output (which
    // forces gallocr to preserve ~50 MB per layer × 48 layers of otherwise
    // transient memory and inflates graph_build by ~35 ms), we create a VIEW
    // into the intermediate region and ggml_cpy it into the persistent cache
    // buffer cap->ssm_intermediate_states. The gallocr is unaware of the
    // persistent cache, so verify_build stays cheap. Matches SGLang's
    // mamba_caches.intermediate_ssm pattern.
    if (cap && cap->ssm_intermediate_states && !persist_inter) {
        // Legacy cpy path: only used when the kernel wrote intermediates into
        // its own result region (i.e. when we did NOT use _tree_persist).
        // The _tree_persist variant writes directly to the cache buffer and
        // this cpy becomes redundant, saving ~5-10 ms per verify step.
        const size_t inter_offset =
            S_v * H_v * n_seq_tokens * n_seqs * r_elt        // attn output region
          + S_v * S_v * H_v * n_seqs * r_elt;                // final-state region
        ggml_tensor * inter_view = ggml_view_4d(ctx, result,
            S_v, S_v, H_v, n_seq_tokens,
            S_v * r_elt,
            S_v * S_v * r_elt,
            S_v * S_v * H_v * r_elt,
            inter_offset);
        ggml_build_forward_expand(gf,
            ggml_cpy(ctx, inter_view, cap->ssm_intermediate_states));
    }
    } // end of block started at `{` before `const int64_t S_v = head_v_dim;`

after_delta_net:
    // Chunked path writes directly into the same ssm_state slot via its 4D
    // view `s` (which is a live view over ssm_state), using the same cpy
    // pattern the sequential path uses for `new_state`. Sequential path's
    // cpy was already emitted above; guard this second cpy on use_chunked
    // so we don't double-write.
    if (use_chunked) {
        ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state, s));
    }

    // ── Gated output norm: rms_norm(output) * silu(z_4d)
    ggml_tensor * z_4d = ggml_reshape_4d(ctx, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor * output_n = ggml_rms_norm(ctx, output, w.rms_eps);
    output_n = ggml_mul(ctx, output_n, L.ssm_norm);
    ggml_tensor * z_silu  = ggml_silu(ctx, z_4d);
    output_n = ggml_mul(ctx, output_n, z_silu);

    // Reshape to [d_inner, n_tokens]
    ggml_tensor * flat = ggml_reshape_3d(ctx, output_n,
        head_v_dim * num_v_heads, n_seq_tokens, n_seqs);

    // Output projection
    ggml_tensor * out = ggml_mul_mat(ctx, L.ssm_out, flat);
    out = ggml_reshape_2d(ctx, out, w.n_embd, n_seq_tokens * n_seqs);
    return out;
}

// ─── Main graph builder ─────────────────────────────────────────────

// Build a single layer of the Qwen3.5-27B model.
// layer_idx: which of the 64 layers to build (0-based).
// inp:      input activation [hidden, n_tokens]
// Returns the output activation [hidden, n_tokens].
static ggml_tensor * build_single_layer(
    ggml_context *        ctx,
    ggml_cgraph *         gf,
    const TargetWeights & w,
    TargetCache &         cache,
    int                   layer_idx,
    ggml_tensor *         inp,         // [hidden, n_tokens]
    ggml_tensor *         positions,   // [4 * n_tokens] i32 (M-RoPE)
    ggml_tensor *         attn_mask,   // optional causal mask
    int                   kv_start,
    int                   n_tokens,
    bool                  capture,
    int                   fa_window = 0,
    ggml_tensor *         q_tail_capture = nullptr,
    int                   q_tail_start = 0)
{
    const int hidden = w.n_embd;
    const float eps   = w.rms_eps;
    const TargetLayer & L = w.layers[layer_idx];
    const bool is_attn = (((layer_idx + 1) % w.full_attention_interval) == 0);

    static const int CAPTURE_LAYERS[DFLASH27B_DRAFT_N_TARGET_LAYERS] =
        { 1, 16, 31, 46, 61 };

    ggml_tensor * inpSA = inp;
    ggml_tensor * cur   = rms_norm_mul(ctx, inp, L.attn_norm, eps);

    if (is_attn) {
        int fa_idx = 0;
        for (int il = 0; il < layer_idx; il++) {
            if (((il + 1) % w.full_attention_interval) == 0) fa_idx++;
        }
        cur = build_full_attn_block(ctx, gf, w, L, cur, positions, w.rope_sections,
                                    cache.attn_k[fa_idx], cache.attn_v[fa_idx],
                                    attn_mask, kv_start, n_tokens,
                                    cache.kv_k_type, cache.kv_v_type, fa_window,
                                    q_tail_capture, q_tail_start);
    } else {
        int dn_idx = 0;
        for (int il = 0; il < layer_idx; il++) {
            if (((il + 1) % w.full_attention_interval) != 0) dn_idx++;
        }
        cur = build_delta_net_block(ctx, gf, w, L, cur,
                                    cache.conv_state[dn_idx], cache.ssm_state[dn_idx],
                                    n_tokens, nullptr, nullptr);
    }

    cur = ggml_add(ctx, cur, inpSA);

    ggml_tensor * ffn_residual = cur;
    ggml_tensor * post = rms_norm_mul(ctx, cur, L.attn_post_norm, eps);
    ggml_tensor * ffn  = build_swiglu_ffn(ctx, post, L);
    cur = ggml_add(ctx, ffn, ffn_residual);

    if (capture && cache.target_feat) {
        int capture_idx = -1;
        for (int k = 0; k < DFLASH27B_DRAFT_N_TARGET_LAYERS; k++) {
            if (CAPTURE_LAYERS[k] == layer_idx) { capture_idx = k; break; }
        }
        if (capture_idx >= 0) {
            const size_t elt        = ggml_element_size(cache.target_feat);
            const size_t col_stride = cache.target_feat->nb[1];
            const int    cap        = cache.target_feat_cap;
            const int    slot_start = kv_start % cap;
            const int    pre_n      = std::min(n_tokens, cap - slot_start);
            const int    post_n     = n_tokens - pre_n;

            ggml_tensor * cur_2d = ggml_reshape_2d(ctx, cur, hidden, n_tokens);

            {
                const size_t offset =
                    (size_t)slot_start * col_stride +
                    (size_t)capture_idx * hidden * elt;
                ggml_tensor * slot = ggml_view_2d(ctx, cache.target_feat,
                    hidden, pre_n, col_stride, offset);
                ggml_tensor * src  = ggml_view_2d(ctx, cur_2d,
                    hidden, pre_n, cur_2d->nb[1], 0);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, src, slot));
            }
            if (post_n > 0) {
                const size_t offset =
                    (size_t)capture_idx * hidden * elt;
                ggml_tensor * slot = ggml_view_2d(ctx, cache.target_feat,
                    hidden, post_n, col_stride, offset);
                ggml_tensor * src  = ggml_view_2d(ctx, cur_2d,
                    hidden, post_n, cur_2d->nb[1],
                    (size_t)pre_n * cur_2d->nb[1]);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, src, slot));
            }
        }
    }

    return cur;
}

QwenGraphOutputs build_qwen35_graph(
    ggml_context *         ctx,
    ggml_cgraph *          gf,
    const TargetWeights &  w,
    TargetCache &          cache,
    const QwenGraphInputs & in) {

    const int n_tokens = in.n_tokens;

    // 1. Caller supplies pre-embedded inputs via in.inp_embed (CPU lookup done
    //    ahead of time, zero GPU cost for the embedding table).
    ggml_tensor * inpL = in.inp_embed;

    int fa_idx = 0, dn_idx = 0;

    // If the caller requested capture, size the output list to the total delta-
    // net layer count so we can index by dn_idx as we iterate the layers.
    QwenGraphOutputs og_early{};
    if (in.capture_delta_intermediate) {
        const int n_full_attn = w.n_layer / w.full_attention_interval;
        const int n_delta     = w.n_layer - n_full_attn;
        og_early.delta_captures.resize(n_delta);
    }

    // DFlash target layer IDs for feature capture: {1, 16, 31, 46, 61}
    // HF hidden_states[lid+1] convention — capture AFTER layer 'lid' runs.
    static const int CAPTURE_LAYERS[DFLASH27B_DRAFT_N_TARGET_LAYERS] =
        { 1, 16, 31, 46, 61 };

    const int hidden = w.n_embd;
    const float eps  = w.rms_eps;

    for (int il = 0; il < w.n_layer; il++) {
        const TargetLayer & L = w.layers[il];
        const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);

        ggml_tensor * inpSA = inpL;

        // Pre-attention norm
        ggml_tensor * cur = rms_norm_mul(ctx, inpL, L.attn_norm, eps);

        if (is_attn) {
            cur = build_full_attn_block(ctx, gf, w, L, cur, in.positions, w.rope_sections,
                                        cache.attn_k[fa_idx], cache.attn_v[fa_idx],
                                        in.attn_mask, in.kv_start, n_tokens,
                                        cache.kv_k_type, cache.kv_v_type, in.fa_window);
            fa_idx++;
        } else {
            DeltaNetCapture * cap_ptr = nullptr;
            if (in.capture_delta_intermediate) {
                cap_ptr = &og_early.delta_captures[dn_idx];
                // Point at the persistent per-layer cache buffers so
                // build_delta_net_block can ggml_cpy into them during graph
                // execution. The caller (test_dflash.cpp spec loop) reads from
                // these tensors post-compute; their ->data pointers are always
                // valid because they're cache-resident, not gallocr-managed.
                cap_ptr->ssm_intermediate_states = cache.ssm_intermediate[dn_idx];
                cap_ptr->conv_input              = cache.conv_input_cache[dn_idx];
            }
            cur = build_delta_net_block(ctx, gf, w, L, cur,
                                        cache.conv_state[dn_idx], cache.ssm_state[dn_idx],
                                        n_tokens, cap_ptr, in.parent_ids);
            dn_idx++;
        }

        // Residual
        cur = ggml_add(ctx, cur, inpSA);

        // Post-attention norm (before FFN)
        ggml_tensor * ffn_residual = cur;
        ggml_tensor * post = rms_norm_mul(ctx, cur, L.attn_post_norm, eps);

        // SwiGLU FFN
        ggml_tensor * ffn = build_swiglu_ffn(ctx, post, L);
        cur = ggml_add(ctx, ffn, ffn_residual);

        // ── DFlash layer feature capture ──
        // Write `cur` into the rolling target_feat buffer. The buffer is a
        // ring of `target_feat_cap` slots; position P maps to slot P%cap.
        // Within a single build call we may straddle the wrap boundary, so
        // we split the copy into up to two contiguous ggml_cpy ops.
        if (in.capture_layers && cache.target_feat) {
            int capture_idx = -1;
            for (int k = 0; k < DFLASH27B_DRAFT_N_TARGET_LAYERS; k++) {
                if (CAPTURE_LAYERS[k] == il) { capture_idx = k; break; }
            }
            if (capture_idx >= 0) {
                const size_t elt        = ggml_element_size(cache.target_feat);
                const size_t col_stride = cache.target_feat->nb[1];
                const int    cap        = cache.target_feat_cap;
                const int    slot_start = in.kv_start % cap;
                const int    pre_n      = std::min(n_tokens, cap - slot_start);
                const int    post_n    = n_tokens - pre_n;

                ggml_tensor * cur_2d = ggml_reshape_2d(ctx, cur, hidden, n_tokens);

                // First slice: [slot_start..slot_start+pre_n) in the ring.
                {
                    const size_t offset =
                        (size_t)slot_start * col_stride +
                        (size_t)capture_idx * hidden * elt;
                    ggml_tensor * slot = ggml_view_2d(ctx, cache.target_feat,
                        hidden, pre_n, col_stride, offset);
                    ggml_tensor * src  = ggml_view_2d(ctx, cur_2d,
                        hidden, pre_n, cur_2d->nb[1], 0);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx, src, slot));
                }

                // Second slice: wrap-around at [0..post_n) if needed.
                if (post_n > 0) {
                    const size_t offset =
                        (size_t)capture_idx * hidden * elt;
                    ggml_tensor * slot = ggml_view_2d(ctx, cache.target_feat,
                        hidden, post_n, col_stride, offset);
                    ggml_tensor * src  = ggml_view_2d(ctx, cur_2d,
                        hidden, post_n, cur_2d->nb[1],
                        (size_t)pre_n * cur_2d->nb[1]);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx, src, slot));
                }
            }
        }

        inpL = cur;
    }

    // 2. Final norm
    ggml_tensor * out = rms_norm_mul(ctx, inpL, w.out_norm, w.rms_eps);

    // 3. LM head
    ggml_tensor * logits = nullptr;
    if (w.output) {
        logits = ggml_mul_mat(ctx, w.output, out);
        ggml_set_name(logits, "logits");
        ggml_build_forward_expand(gf, logits);
    } else {
        ggml_set_name(out, "result_norm");
        ggml_build_forward_expand(gf, out);
    }

    QwenGraphOutputs og = std::move(og_early);
    og.logits = logits;
    return og;
}

ggml_tensor * build_qwen35_layer(
    ggml_context *        ctx,
    ggml_cgraph *         gf,
    const TargetWeights & w,
    TargetCache &         cache,
    int                   layer_idx,
    ggml_tensor *         inp,
    ggml_tensor *         positions,
    ggml_tensor *         attn_mask,
    int                   kv_start,
    int                   n_tokens,
    bool                  capture,
    int                   fa_window,
    ggml_tensor *         q_tail_capture,
    int                   q_tail_start)
{
    return build_single_layer(ctx, gf, w, cache, layer_idx, inp, positions,
                              attn_mask, kv_start, n_tokens, capture, fa_window,
                              q_tail_capture, q_tail_start);
}

} // namespace dflash27b
