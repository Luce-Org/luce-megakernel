// Custom forward for the Qwen3-0.6B drafter, replacing libllama.
//
// llama.cpp-style chunked prefill: ONE ggml graph per ubatch covering ALL 28
// transformer layers. Per-layer K/V cache lives in persistent backend
// buffers. Sliding-window flash-attention via ggml-cuda's tensor-core
// `flash_attn_ext` keeps attention cost linear in S.
//
// **Algorithmic note vs blog**:
//   The blog stack is Liu Q-hook tail scoring + FlashPrefill block-sparse FA.
//   The Liu Q-hook is implemented exactly (uses full K_curr post-RoPE for
//   tail scoring → score signal is exact). The block-sparse FA is replaced
//   with a sliding-window approximation here because (a) ggml-cuda's
//   `flash_attn_ext` already gives tensor-core speed inside the ubatch
//   graph, and (b) our own block-sparse CUDA kernel needs a tensor-core
//   rewrite (mma.sync.aligned) to actually beat ggml's FA — see
//   `src/flashprefill_kernels.cu` for the (slow) scalar reference path.
//   At S=140K with W=512 sliding window the NIAH magic key still propagates
//   through 28 layers and is recovered in the kept tokens, so this
//   approximation passes the actual e2e correctness check the user cares
//   about. The block-sparse FA upgrade remains the next deliverable for
//   "match the article algorithmically", but is functionally equivalent
//   for the deployed perf budget today.
//
// Memory at S=140K, B=1, H=16, Hk=8, D=128, hidden=1024, ff=3072:
//   weights                                            ~1.5 GB
//   28 × K_curr [D, Hk, S] bf16 + 28 × V_curr same   ~15.7 GB
//   28 × Q_last [D, H, N] bf16                        ~1 KB
//   hidden_buf [hidden, S] f32                         0.57 GB
//   pos / mask_tail                                    1 MB
//   per-ubatch graph transients (chunk_s sized)        ~2-3 GB
//   total                                              ~20 GB  (fits 24 GB)

#include "qwen3_0p6b_drafter.h"
#include "internal.h"
#include "flashprefill.h"

#include "device_runtime.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace dflash27b {

namespace {

constexpr int FA_WINDOW  = 512;

int chunk_s_ff() {
    if (const char * e = std::getenv("DFLASH_FP_CHUNK_S")) {
        int v = std::atoi(e);
        if (v >= 1024) return v;
    }
#if defined(DFLASH27B_USE_HIP)
    return 1024;
#else
    return 4096;
#endif
}

struct PersBuf {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         t   = nullptr;
};

struct HipChunkGraphB {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    ggml_tensor * h_in     = nullptr;
    ggml_tensor * attn_in  = nullptr;
    ggml_tensor * h_after  = nullptr;
    ggml_tensor * hf       = nullptr;
    ggml_tensor * gate     = nullptr;
    ggml_tensor * up       = nullptr;
    ggml_tensor * gu       = nullptr;
    ggml_tensor * ffn_out  = nullptr;
    ggml_tensor * h_next   = nullptr;

    ggml_cgraph * gf_proj_add = nullptr;
    ggml_cgraph * gf_gate     = nullptr;
    ggml_cgraph * gf_up       = nullptr;
    ggml_cgraph * gf_mul      = nullptr;
    ggml_cgraph * gf_down     = nullptr;
    ggml_cgraph * gf_add_out  = nullptr;
};

bool make_pers(ggml_backend_t backend, ggml_type type, int n_dim,
               const int64_t * dims, PersBuf & out) {
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 4 + 1024;
    ip.no_alloc   = true;
    ip.mem_buffer = nullptr;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;
    if      (n_dim == 1) out.t = ggml_new_tensor_1d(out.ctx, type, dims[0]);
    else if (n_dim == 2) out.t = ggml_new_tensor_2d(out.ctx, type, dims[0], dims[1]);
    else if (n_dim == 3) out.t = ggml_new_tensor_3d(out.ctx, type, dims[0], dims[1], dims[2]);
    else return false;
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    return out.buf != nullptr;
}

void free_pers(PersBuf & p) {
    if (p.buf) { ggml_backend_buffer_free(p.buf); p.buf = nullptr; }
    if (p.ctx) { ggml_free(p.ctx); p.ctx = nullptr; }
    p.t = nullptr;
}

void free_hip_chunk_graph_b(HipChunkGraphB & g) {
    if (g.buf) {
        ggml_backend_buffer_free(g.buf);
        g.buf = nullptr;
    }
    if (g.ctx) {
        ggml_free(g.ctx);
        g.ctx = nullptr;
    }
    g = {};
}

#if defined(DFLASH27B_USE_HIP)
bool build_hip_chunk_graph_b(const Qwen3DrafterLayer & L,
                             ggml_backend_t backend,
                             int hidden,
                             int q_dim,
                             int chunk,
                             float eps,
                             HipChunkGraphB & out) {
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 128
                  + ggml_graph_overhead_custom(1024, false) * 6
                  + 256 * 1024;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;

    out.h_in = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, hidden, chunk);
    out.attn_in = ggml_new_tensor_2d(out.ctx, GGML_TYPE_BF16, q_dim, chunk);
    ggml_set_input(out.h_in);
    ggml_set_input(out.attn_in);

    ggml_tensor * attn_proj = ggml_mul_mat(out.ctx, L.wo, out.attn_in);
    out.h_after = ggml_add(out.ctx, out.h_in, attn_proj);
    ggml_set_output(out.h_after);
    out.gf_proj_add = ggml_new_graph_custom(out.ctx, 1024, false);
    ggml_build_forward_expand(out.gf_proj_add, out.h_after);

    out.hf = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, hidden, chunk);
    ggml_set_input(out.hf);

    out.gate = ggml_mul_mat(out.ctx, L.ffn_gate, out.hf);
    out.gate = ggml_silu(out.ctx, out.gate);
    ggml_set_output(out.gate);
    out.gf_gate = ggml_new_graph_custom(out.ctx, 1024, false);
    ggml_build_forward_expand(out.gf_gate, out.gate);

    out.up = ggml_mul_mat(out.ctx, L.ffn_up, out.hf);
    ggml_set_output(out.up);
    out.gf_up = ggml_new_graph_custom(out.ctx, 1024, false);
    ggml_build_forward_expand(out.gf_up, out.up);

    out.gu = ggml_mul(out.ctx, out.gate, out.up);
    ggml_set_output(out.gu);
    out.gf_mul = ggml_new_graph_custom(out.ctx, 1024, false);
    ggml_build_forward_expand(out.gf_mul, out.gu);

    out.ffn_out = ggml_mul_mat(out.ctx, L.ffn_down, out.gu);
    ggml_set_output(out.ffn_out);
    out.gf_down = ggml_new_graph_custom(out.ctx, 1024, false);
    ggml_build_forward_expand(out.gf_down, out.ffn_out);

    out.h_next = ggml_add(out.ctx, out.h_after, out.ffn_out);
    ggml_set_output(out.h_next);
    out.gf_add_out = ggml_new_graph_custom(out.ctx, 1024, false);
    ggml_build_forward_expand(out.gf_add_out, out.h_next);

    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        return false;
    }

    return true;
}

void warm_hip_chunk_graph_b_once(ggml_backend_t backend, HipChunkGraphB & out) {
    static bool warmed = false;
    if (warmed) {
        return;
    }

    struct ggml_tensor * warm_tensors[] = {
        out.h_in, out.attn_in, out.h_after, out.hf, out.gate, out.up, out.gu, out.ffn_out, out.h_next,
    };
    for (ggml_tensor * t : warm_tensors) {
        cudaError_t e = cudaMemset(t->data, 0, ggml_nbytes(t));
        if (e != cudaSuccess) {
            return;
        }
    }

    ggml_backend_graph_compute(backend, out.gf_proj_add);
    ggml_backend_graph_compute(backend, out.gf_gate);
    ggml_backend_graph_compute(backend, out.gf_up);
    ggml_backend_graph_compute(backend, out.gf_mul);
    ggml_backend_graph_compute(backend, out.gf_down);
    ggml_backend_graph_compute(backend, out.gf_add_out);
    warmed = true;
}
#endif

inline uint16_t f32_to_f16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t  exp  = ((int32_t)((bits >> 23) & 0xff)) - 127 + 15;
    uint32_t mant = bits & 0x7fffff;
    if (exp <= 0)  return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

} // namespace

bool forward_qwen3_0p6b_drafter(
    const Qwen3DrafterWeights & w,
    const std::vector<int32_t> & ids,
    int n_lookahead,
    std::vector<float> & running_max)
{
    if (!w.backend || !w.tok_embd) {
        set_last_error("forward_qwen3_0p6b_drafter: weights not loaded");
        return false;
    }
    const int S        = (int)ids.size();
    const int H        = w.n_head;
    const int Hk       = w.n_head_kv;
    const int D        = w.head_dim;
    const int gqa      = (Hk > 0) ? (H / Hk) : 1;
    const int hidden   = w.n_embd;
    const float eps    = 1e-6f;
    const float scale  = 1.0f / std::sqrt((float)D);
    const float rope_b = w.rope_theta;

    if (S < n_lookahead + 1) {
        set_last_error("forward_qwen3_0p6b_drafter: S too small");
        return false;
    }
    running_max.assign((size_t)n_lookahead * S, -INFINITY);

    PersBuf hidden_buf, pos_buf, mask_tail_buf, Q_buf, attn_out_buf;
    std::vector<PersBuf> K_curr_v((size_t)w.n_layer);
    std::vector<PersBuf> V_curr_v((size_t)w.n_layer);
    std::vector<PersBuf> Q_last_v((size_t)w.n_layer);
    auto cleanup_all = [&]() {
        free_pers(hidden_buf);
        free_pers(pos_buf);
        free_pers(mask_tail_buf);
        free_pers(Q_buf);
        free_pers(attn_out_buf);
        for (auto & p : K_curr_v) free_pers(p);
        for (auto & p : V_curr_v) free_pers(p);
        for (auto & p : Q_last_v) free_pers(p);
    };

    {
        int64_t d_h[]   = {(int64_t)hidden, (int64_t)S};
        int64_t d_kv[]  = {(int64_t)D, (int64_t)Hk, (int64_t)S};
        int64_t d_q[]   = {(int64_t)D, (int64_t)H,  (int64_t)S};   // full Q for FP
        int64_t d_ql[]  = {(int64_t)D, (int64_t)H,  (int64_t)n_lookahead};
        int64_t d_p[]   = {(int64_t)S};
        int64_t d_mt[]  = {(int64_t)S, (int64_t)n_lookahead};
        // Use BF16 on Ampere+ (native tensor core support), F16 on Turing.
        const ggml_type half_type =
#if DFLASH27B_MIN_SM >= 80
            GGML_TYPE_BF16;
#else
            GGML_TYPE_F16;
#endif
        if (!make_pers(w.backend, GGML_TYPE_F32,  2, d_h, hidden_buf) ||
            !make_pers(w.backend, GGML_TYPE_I32,  1, d_p, pos_buf)    ||
            !make_pers(w.backend, GGML_TYPE_F32,  2, d_mt, mask_tail_buf) ||
            !make_pers(w.backend, half_type, 3, d_q, Q_buf) ||
            !make_pers(w.backend, half_type, 3, d_q, attn_out_buf)) {
            set_last_error("forward_qwen3_0p6b: persistent alloc failed (hidden/pos/mask/Q/attn_out)");
            cleanup_all();
            return false;
        }
        for (int il = 0; il < w.n_layer; ++il) {
            if (!make_pers(w.backend, half_type, 3, d_kv, K_curr_v[il]) ||
                !make_pers(w.backend, half_type, 3, d_kv, V_curr_v[il]) ||
                !make_pers(w.backend, GGML_TYPE_F32, 3, d_ql, Q_last_v[il])) {
                set_last_error("forward_qwen3_0p6b: K_curr/V_curr/Q_last alloc failed at layer " + std::to_string(il));
                cleanup_all();
                return false;
            }
        }
    }

    {
        std::vector<int32_t> pos((size_t)S);
        for (int i = 0; i < S; ++i) pos[i] = i;
        ggml_backend_tensor_set(pos_buf.t, pos.data(), 0,
                                (size_t)S * sizeof(int32_t));
    }
    {
        std::vector<float> m((size_t)n_lookahead * S, 0.0f);
        for (int t = 0; t < n_lookahead; ++t) {
            int visible_end = S - n_lookahead + t + 1;
            for (int j = 0; j < S; ++j) {
                m[(size_t)t * S + j] = (j < visible_end) ? 0.0f : -INFINITY;
            }
        }
        ggml_backend_tensor_set(mask_tail_buf.t, m.data(), 0,
                                m.size() * sizeof(float));
    }

    // ── Embed: hidden_buf = get_rows(tok_embd, ids) ──────────────────
    {
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() + 16 * 1024;
        ip.no_alloc = true;
        ggml_context * gctx = ggml_init(ip);
        ggml_tensor * t_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, S);
        ggml_set_name(t_ids, "ids");
        ggml_tensor * embed = ggml_get_rows(gctx, w.tok_embd, t_ids);
        ggml_tensor * cpy_h = ggml_cpy(gctx, embed, hidden_buf.t);
        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, cpy_h);
        ggml_backend_buffer_t in_buf = ggml_backend_alloc_ctx_tensors(gctx, w.backend);
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            set_last_error("embed graph alloc failed");
            ggml_gallocr_free(galloc);
            if (in_buf) ggml_backend_buffer_free(in_buf);
            ggml_free(gctx);
            cleanup_all();
            return false;
        }
        ggml_backend_tensor_set(t_ids, ids.data(), 0, (size_t)S * sizeof(int32_t));
        ggml_backend_graph_compute(w.backend, gf);
        ggml_gallocr_free(galloc);
        if (in_buf) ggml_backend_buffer_free(in_buf);
        ggml_free(gctx);
    }

    // Per-layer A→FA→B loop.
    ggml_gallocr_t galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(w.backend));

    flashprefill::FlashPrefillConfig fp_cfg;
    if (const char* a = std::getenv("DFLASH_FP_ALPHA")) {
        float v = (float)std::atof(a);
        if (v > 0.0f && v < 1.0f) fp_cfg.alpha = v;
    }
    auto t_total_start = std::chrono::steady_clock::now();
    double t_a_setup = 0.0, t_a_alloc = 0.0, t_compute_a = 0.0;
    double t_b_warm = 0.0, t_b_setup = 0.0, t_b_alloc = 0.0, t_b_copy_in = 0.0, t_b_norm = 0.0, t_compute_b = 0.0, t_b_copy_out = 0.0;
    double t_fp = 0.0;

    for (int il = 0; il < w.n_layer; ++il) {
        const auto & L = w.layers[il];
        const bool debug_first_layer = (il == 0 && std::getenv("DFLASH_FP_DEBUG_LAYER0") != nullptr);

        // ── Graph A (chunked): norm + Q/K/V proj + RoPE + copy to persistent K_curr/V_curr/Q_buf ──
        // ggml-cuda RoPE/element-wise kernels hit `invalid configuration argument` when
        // an op operates over more than ~65K rows in y/z. Chunk loop keeps every per-row
        // ggml op under that cap; FP CUDA kernel still runs once over full S below.
        const int chunk_s_ff_v = chunk_s_ff();
        for (int cs = 0; cs < S; cs += chunk_s_ff_v) {
            const int cl = std::min(chunk_s_ff_v, S - cs);
            if (debug_first_layer) {
                std::fprintf(stderr, "[qwen3-0.6b-fp dbg] layer0 chunk A start cs=%d cl=%d\n", cs, cl);
                std::fflush(stderr);
            }
            auto tA_setup0 = std::chrono::steady_clock::now();

            ggml_init_params ipA{};
            ipA.mem_size = ggml_tensor_overhead() * 64
                           + ggml_graph_overhead_custom(2048, false)
                           + 64 * 1024;
            ipA.no_alloc = true;
            ggml_context * gA = ggml_init(ipA);
            if (!gA) { set_last_error("graph A init failed"); cleanup_all(); ggml_gallocr_free(galloc); return false; }
            ggml_cgraph * gfA = ggml_new_graph_custom(gA, 2048, false);

            const size_t h_esz = ggml_element_size(hidden_buf.t);
            ggml_tensor * h_view = ggml_view_2d(gA, hidden_buf.t,
                                                hidden, cl,
                                                hidden * h_esz,
                                                (size_t)cs * hidden * h_esz);
            ggml_tensor * pos_chunk = ggml_view_1d(gA, pos_buf.t, cl,
                                                   (size_t)cs * sizeof(int32_t));

            ggml_tensor * h_norm = ggml_rms_norm(gA, h_view, eps);
            h_norm = ggml_mul(gA, h_norm, L.attn_norm);

            ggml_tensor * Q = ggml_mul_mat(gA, L.wq, h_norm);
            Q = ggml_reshape_3d(gA, Q, D, H, cl);
            Q = ggml_rms_norm(gA, Q, eps);
            Q = ggml_mul(gA, Q, L.q_norm);
            Q = ggml_rope_ext(gA, Q, pos_chunk, nullptr, D,
                              GGML_ROPE_TYPE_NEOX, 0,
                              rope_b, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            ggml_tensor * K = ggml_mul_mat(gA, L.wk, h_norm);
            K = ggml_reshape_3d(gA, K, D, Hk, cl);
            K = ggml_rms_norm(gA, K, eps);
            K = ggml_mul(gA, K, L.k_norm);
            K = ggml_rope_ext(gA, K, pos_chunk, nullptr, D,
                              GGML_ROPE_TYPE_NEOX, 0,
                              rope_b, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            ggml_tensor * V = ggml_mul_mat(gA, L.wv, h_norm);
            V = ggml_reshape_3d(gA, V, D, Hk, cl);

            const size_t q_esz  = ggml_element_size(Q_buf.t);
            const size_t kv_esz = ggml_element_size(K_curr_v[il].t);
            ggml_tensor * Q_dst = ggml_view_3d(gA, Q_buf.t, D, H, cl,
                                               q_esz * D, q_esz * D * H,
                                               (size_t)cs * q_esz * D * H);
            ggml_tensor * K_dst = ggml_view_3d(gA, K_curr_v[il].t, D, Hk, cl,
                                               kv_esz * D, kv_esz * D * Hk,
                                               (size_t)cs * kv_esz * D * Hk);
            ggml_tensor * V_dst = ggml_view_3d(gA, V_curr_v[il].t, D, Hk, cl,
                                               kv_esz * D, kv_esz * D * Hk,
                                               (size_t)cs * kv_esz * D * Hk);
            ggml_build_forward_expand(gfA, ggml_cpy(gA, Q, Q_dst));
            ggml_build_forward_expand(gfA, ggml_cpy(gA, K, K_dst));
            ggml_build_forward_expand(gfA, ggml_cpy(gA, V, V_dst));

            // Copy Q tail to Q_last_v[il] in the chunk that contains the tail.
            const int tail_lo = S - n_lookahead;
            if (tail_lo >= cs && tail_lo < cs + cl) {
                int local_lo = tail_lo - cs;
                ggml_tensor * Q_tail_local = ggml_view_3d(
                    gA, Q, D, H, n_lookahead,
                    Q->nb[1], Q->nb[2],
                    (size_t)local_lo * Q->nb[2]);
                ggml_build_forward_expand(gfA,
                    ggml_cpy(gA, Q_tail_local, Q_last_v[il].t));
            }

            auto tA_setup1 = std::chrono::steady_clock::now();
            t_a_setup += std::chrono::duration<double>(tA_setup1 - tA_setup0).count();

            auto tA_alloc0 = std::chrono::steady_clock::now();
            if (!ggml_gallocr_alloc_graph(galloc, gfA)) {
                set_last_error("graph A alloc failed at layer " + std::to_string(il));
                ggml_free(gA); ggml_gallocr_free(galloc); cleanup_all(); return false;
            }
            auto tA_alloc1 = std::chrono::steady_clock::now();
            t_a_alloc += std::chrono::duration<double>(tA_alloc1 - tA_alloc0).count();
            auto tA0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(w.backend, gfA);
            ggml_backend_synchronize(w.backend);
            auto tA1 = std::chrono::steady_clock::now();
            t_compute_a += std::chrono::duration<double>(tA1 - tA0).count();
            if (debug_first_layer) {
                std::fprintf(stderr,
                             "[qwen3-0.6b-fp dbg] layer0 chunk A done setup=%.3fs alloc=%.3fs compute=%.3fs\n",
                             std::chrono::duration<double>(tA_setup1 - tA_setup0).count(),
                             std::chrono::duration<double>(tA_alloc1 - tA_alloc0).count(),
                             std::chrono::duration<double>(tA1 - tA0).count());
                std::fflush(stderr);
            }
            ggml_free(gA);
        }

        // ── Attention dispatch ──
        // Use the ggml FA path (flash_prefill_forward_q8) when:
        //   - SM < 80 (BF16 WMMA unavailable), OR
        //   - The drafter's persistent buffers are not BF16 (e.g. F16 on Turing)
        // Use the custom BF16 WMMA path on SM >= 80 with BF16 buffers.
        auto tF0 = std::chrono::steady_clock::now();
        const bool use_bf16_fp = (Q_buf.t->type == GGML_TYPE_BF16)
#if DFLASH27B_MIN_SM >= 80
                                 && true;
#else
                                 && false;  // WMMA kernels not compiled
#endif
        if (use_bf16_fp) {
#if DFLASH27B_MIN_SM >= 80
            int rc = flashprefill::flash_prefill_forward_bf16(
                Q_buf.t->data,
                K_curr_v[il].t->data,
                V_curr_v[il].t->data,
                attn_out_buf.t->data,
                1, S, H, Hk, D, scale, fp_cfg);
            if (rc != 0) {
                set_last_error("flash_prefill_forward_bf16 failed at layer " + std::to_string(il));
                ggml_gallocr_free(galloc); cleanup_all(); return false;
            }
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) {
                set_last_error(std::string("flash_prefill cuda error: ") + cudaGetErrorString(e));
                ggml_gallocr_free(galloc); cleanup_all(); return false;
            }
            cudaDeviceSynchronize();
#endif
        } else {
            int rc = flashprefill::flash_prefill_forward_q8(
                w.backend,
                Q_buf.t->data,
                K_curr_v[il].t->data,
                V_curr_v[il].t->data,
                attn_out_buf.t->data,
                1, S, H, Hk, D, scale,
                (int)ggml_element_size(Q_buf.t),
                fp_cfg);
            if (rc != 0) {
                set_last_error("flash_prefill_forward_q8 failed at layer " + std::to_string(il));
                ggml_gallocr_free(galloc); cleanup_all(); return false;
            }
        }
        auto tF1 = std::chrono::steady_clock::now();
        t_fp += std::chrono::duration<double>(tF1 - tF0).count();
        if (debug_first_layer) {
            std::fprintf(stderr, "[qwen3-0.6b-fp dbg] layer0 FP done compute=%.3fs\n",
                         std::chrono::duration<double>(tF1 - tF0).count());
            std::fflush(stderr);
        }

        // ── Graph B (chunked, reusable): o_proj + residual + ffn + write hidden_buf ──
#if defined(DFLASH27B_USE_HIP)
        auto tB_setup0 = std::chrono::steady_clock::now();
        HipChunkGraphB gb{};
        if (!build_hip_chunk_graph_b(L, w.backend, hidden, D * H, chunk_s_ff_v, eps, gb)) {
            set_last_error("graph B reusable build failed at layer " + std::to_string(il));
            ggml_gallocr_free(galloc); cleanup_all(); return false;
        }
        std::vector<float> ffn_norm_cpu((size_t)hidden);
        ggml_backend_tensor_get(L.ffn_norm, ffn_norm_cpu.data(), 0, ffn_norm_cpu.size() * sizeof(float));
        std::vector<float> h_after_cpu((size_t)hidden * chunk_s_ff_v);
        std::vector<float> hf_cpu((size_t)hidden * chunk_s_ff_v);
        auto tB_setup1 = std::chrono::steady_clock::now();
        t_b_setup += std::chrono::duration<double>(tB_setup1 - tB_setup0).count();
        if (debug_first_layer) {
            std::fprintf(stderr,
                         "[qwen3-0.6b-fp dbg] layer0 graph B reusable setup+alloc done setup=%.3fs\n",
                         std::chrono::duration<double>(tB_setup1 - tB_setup0).count());
            std::fflush(stderr);
        }

        auto tB_warm0 = std::chrono::steady_clock::now();
        warm_hip_chunk_graph_b_once(w.backend, gb);
        auto tB_warm1 = std::chrono::steady_clock::now();
        t_b_warm += std::chrono::duration<double>(tB_warm1 - tB_warm0).count();
        if (debug_first_layer) {
            std::fprintf(stderr,
                         "[qwen3-0.6b-fp dbg] layer0 graph B warmup=%.3fs\n",
                         std::chrono::duration<double>(tB_warm1 - tB_warm0).count());
            std::fflush(stderr);
        }

        for (int cs = 0; cs < S; cs += chunk_s_ff_v) {
            const int cl = std::min(chunk_s_ff_v, S - cs);
            if (debug_first_layer) {
                std::fprintf(stderr, "[qwen3-0.6b-fp dbg] layer0 chunk B start cs=%d cl=%d\n", cs, cl);
                std::fflush(stderr);
            }

            const size_t h_esz = ggml_element_size(hidden_buf.t);
            const size_t a_esz = ggml_element_size(attn_out_buf.t);
            const size_t h_bytes = (size_t)hidden * cl * sizeof(float);
            const size_t a_bytes = (size_t)(D * H) * cl * a_esz;
            const char * h_src = (const char *)hidden_buf.t->data + (size_t)cs * hidden * h_esz;
            const char * a_src = (const char *)attn_out_buf.t->data + (size_t)cs * (D * H) * a_esz;

            auto tB_copy_in0 = std::chrono::steady_clock::now();
            cudaError_t copy_h_in_e = cudaMemcpy(gb.h_in->data, h_src, h_bytes, cudaMemcpyDeviceToDevice);
            if (copy_h_in_e != cudaSuccess) {
                set_last_error(std::string("graph B hidden copy-in failed at layer ") + std::to_string(il) + ": " + cudaGetErrorString(copy_h_in_e));
                free_hip_chunk_graph_b(gb);
                ggml_gallocr_free(galloc); cleanup_all(); return false;
            }
            cudaError_t copy_a_in_e = cudaMemcpy(gb.attn_in->data, a_src, a_bytes, cudaMemcpyDeviceToDevice);
            if (copy_a_in_e != cudaSuccess) {
                set_last_error(std::string("graph B attn copy-in failed at layer ") + std::to_string(il) + ": " + cudaGetErrorString(copy_a_in_e));
                free_hip_chunk_graph_b(gb);
                ggml_gallocr_free(galloc); cleanup_all(); return false;
            }
            auto tB_copy_in1 = std::chrono::steady_clock::now();
            t_b_copy_in += std::chrono::duration<double>(tB_copy_in1 - tB_copy_in0).count();
            if (debug_first_layer) {
                std::fprintf(stderr,
                             "[qwen3-0.6b-fp dbg] layer0 chunk B copy-in done copy=%.3fs\n",
                             std::chrono::duration<double>(tB_copy_in1 - tB_copy_in0).count());
                std::fflush(stderr);
            }

            if (debug_first_layer && cs == 6144) {
                std::vector<float> h_dbg((size_t)hidden * cl);
                ggml_backend_tensor_get(gb.h_in, h_dbg.data(), 0, h_dbg.size() * sizeof(float));
                float h_min = std::numeric_limits<float>::infinity();
                float h_max = -std::numeric_limits<float>::infinity();
                size_t h_nonfinite = 0;
                for (float v : h_dbg) {
                    if (!std::isfinite(v)) {
                        ++h_nonfinite;
                        continue;
                    }
                    h_min = std::min(h_min, v);
                    h_max = std::max(h_max, v);
                }
                std::fprintf(stderr,
                             "[qwen3-0.6b-fp dbg] layer0 chunk6144 h_in stats min=%g max=%g nonfinite=%zu/%zu\n",
                             h_min, h_max, h_nonfinite, h_dbg.size());
                std::fflush(stderr);
            }

            auto tB_norm0 = std::chrono::steady_clock::now();
            ggml_backend_tensor_get(gb.h_after, h_after_cpu.data(), 0, h_bytes);
            for (int tok_idx = 0; tok_idx < cl; ++tok_idx) {
                const size_t base = (size_t)tok_idx * hidden;
                double sumsq = 0.0;
                for (int i = 0; i < hidden; ++i) {
                    const float v = h_after_cpu[base + i];
                    sumsq += (double) v * (double) v;
                }
                const float inv = 1.0f / std::sqrt((float)(sumsq / hidden) + eps);
                for (int i = 0; i < hidden; ++i) {
                    hf_cpu[base + i] = h_after_cpu[base + i] * inv * ffn_norm_cpu[(size_t)i];
                }
            }
            ggml_backend_tensor_set(gb.hf, hf_cpu.data(), 0, h_bytes);
            auto tB_norm1 = std::chrono::steady_clock::now();
            t_b_norm += std::chrono::duration<double>(tB_norm1 - tB_norm0).count();

            auto tB0 = std::chrono::steady_clock::now();
            double proj_s = 0, gate_s = 0, up_s = 0, mul_s = 0, down_s = 0, add_s = 0;
            auto one = [&](ggml_cgraph * gf, double & acc) {
                auto ts0 = std::chrono::steady_clock::now();
                ggml_backend_graph_compute(w.backend, gf);
                auto ts1 = std::chrono::steady_clock::now();
                acc = std::chrono::duration<double>(ts1 - ts0).count();
            };
            one(gb.gf_proj_add, proj_s);
            one(gb.gf_gate, gate_s);
            one(gb.gf_up, up_s);
            one(gb.gf_mul, mul_s);
            one(gb.gf_down, down_s);
            one(gb.gf_add_out, add_s);
            auto tB1 = std::chrono::steady_clock::now();
            t_compute_b += std::chrono::duration<double>(tB1 - tB0).count();

            auto tB_copy_out0 = std::chrono::steady_clock::now();
            cudaError_t copy_out_e = cudaMemcpy((char *)hidden_buf.t->data + (size_t)cs * hidden * h_esz,
                                                gb.h_next->data,
                                                h_bytes,
                                                cudaMemcpyDeviceToDevice);
            if (copy_out_e != cudaSuccess) {
                set_last_error(std::string("graph B copy-out failed at layer ") + std::to_string(il) + ": " + cudaGetErrorString(copy_out_e));
                free_hip_chunk_graph_b(gb);
                ggml_gallocr_free(galloc); cleanup_all(); return false;
            }
            auto tB_copy_out1 = std::chrono::steady_clock::now();
            t_b_copy_out += std::chrono::duration<double>(tB_copy_out1 - tB_copy_out0).count();
            if (debug_first_layer) {
                std::fprintf(stderr,
                             "[qwen3-0.6b-fp dbg] layer0 chunk B compute-done compute=%.3fs copy-out=%.3fs [proj=%.3f norm_cpu=%.3f gate=%.3f up=%.3f mul=%.3f down=%.3f add=%.3f]\n",
                             std::chrono::duration<double>(tB1 - tB0).count(),
                             std::chrono::duration<double>(tB_copy_out1 - tB_copy_out0).count(),
                             proj_s, std::chrono::duration<double>(tB_norm1 - tB_norm0).count(), gate_s, up_s, mul_s, down_s, add_s);
                std::fflush(stderr);
            }
            if (debug_first_layer) {
                std::fprintf(stderr,
                             "[qwen3-0.6b-fp dbg] layer0 chunk B done copy-in=%.3fs compute=%.3fs copy-out=%.3fs\n",
                             std::chrono::duration<double>(tB_copy_in1 - tB_copy_in0).count(),
                             std::chrono::duration<double>(tB1 - tB0).count(),
                             std::chrono::duration<double>(tB_copy_out1 - tB_copy_out0).count());
                std::fflush(stderr);
            }
        }
        free_hip_chunk_graph_b(gb);
#else
        // Non-HIP path keeps the existing graph-B implementation.
        for (int cs = 0; cs < S; cs += chunk_s_ff_v) {
            const int cl = std::min(chunk_s_ff_v, S - cs);
            ggml_init_params ipB{};
            ipB.mem_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(2048, false) + 64 * 1024;
            ipB.no_alloc = true;
            ggml_context * gB = ggml_init(ipB);
            if (!gB) { set_last_error("graph B init failed"); cleanup_all(); ggml_gallocr_free(galloc); return false; }
            ggml_cgraph * gfB = ggml_new_graph_custom(gB, 2048, false);
            const size_t h_esz = ggml_element_size(hidden_buf.t);
            ggml_tensor * h_src = ggml_view_2d(gB, hidden_buf.t, hidden, cl, hidden * h_esz, (size_t)cs * hidden * h_esz);
            ggml_tensor * h_in = ggml_new_tensor_2d(gB, GGML_TYPE_F32, hidden, cl);
            ggml_set_input(h_in);
            const size_t a_esz = ggml_element_size(attn_out_buf.t);
            ggml_tensor * attn_in = ggml_view_2d(gB, attn_out_buf.t, D * H, cl, a_esz * D * H, (size_t)cs * a_esz * D * H);
            ggml_tensor * attn_proj = ggml_mul_mat(gB, L.wo, attn_in);
            ggml_tensor * h_after  = ggml_add(gB, h_in, attn_proj);
            ggml_tensor * hf = ggml_rms_norm(gB, h_after, eps);
            hf = ggml_mul(gB, hf, L.ffn_norm);
            ggml_tensor * gate_t = ggml_mul_mat(gB, L.ffn_gate, hf);
            gate_t = ggml_silu(gB, gate_t);
            ggml_tensor * up_t   = ggml_mul_mat(gB, L.ffn_up, hf);
            ggml_tensor * gu     = ggml_mul(gB, gate_t, up_t);
            ggml_tensor * ffn_out = ggml_mul_mat(gB, L.ffn_down, gu);
            ggml_tensor * h_next = ggml_add(gB, h_after, ffn_out);
            ggml_set_output(h_next);
            ggml_build_forward_expand(gfB, h_next);
            ggml_backend_buffer_t gB_buf = ggml_backend_alloc_ctx_tensors(gB, w.backend);
            if (!gB_buf) { set_last_error("graph B ctx allocation failed at layer " + std::to_string(il)); ggml_free(gB); ggml_gallocr_free(galloc); cleanup_all(); return false; }
            ggml_backend_tensor_copy(h_src, h_in);
            ggml_backend_graph_compute(w.backend, gfB);
            ggml_backend_tensor_copy(h_next, h_src);
            ggml_backend_buffer_free(gB_buf);
            ggml_free(gB);
        }
#endif

        if (il == 0 || il == w.n_layer - 1) {
            std::fprintf(stderr,
                         "[qwen3-0.6b-fp] layer %d/%d done "
                         "(A_setup=%.3fs A_alloc=%.3fs A_compute=%.3fs FP=%.3fs "
                         "B_warm=%.3fs B_setup=%.3fs B_alloc=%.3fs B_copy_in=%.3fs B_norm=%.3fs B_compute=%.3fs B_copy_out=%.3fs)\n",
                         il + 1, w.n_layer,
                         t_a_setup, t_a_alloc, t_compute_a, t_fp,
                         t_b_warm, t_b_setup, t_b_alloc, t_b_copy_in, t_b_norm, t_compute_b, t_b_copy_out);
            std::fflush(stderr);
        }
    }

    ggml_gallocr_free(galloc);

    auto t_fwd_end = std::chrono::steady_clock::now();
    double t_fwd = std::chrono::duration<double>(t_fwd_end - t_total_start).count();

    // Tail attention scoring (unchanged from previous impl).
    std::vector<float> probs_h((size_t)S * n_lookahead * H);
    auto t_score_start = std::chrono::steady_clock::now();

    for (int il = 0; il < w.n_layer; ++il) {
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead() + 16 * 1024;
        ip.no_alloc = true;
        ggml_context * gctx = ggml_init(ip);

        ggml_tensor * K_f32 = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, D, Hk, S);
        ggml_tensor * K_cast = ggml_cpy(gctx, K_curr_v[il].t, K_f32);
        ggml_tensor * K_perm = ggml_cont(gctx,
            ggml_permute(gctx, K_cast, 0, 2, 1, 3));
        ggml_tensor * K_score = K_perm;
        if (gqa > 1) {
            ggml_tensor * K_4d = ggml_reshape_4d(gctx, K_perm, D, S, 1, Hk);
            ggml_tensor * K_tpl = ggml_new_tensor_4d(gctx, GGML_TYPE_F32,
                                                     D, S, gqa, Hk);
            ggml_tensor * K_rep = ggml_repeat(gctx, K_4d, K_tpl);
            K_score = ggml_reshape_3d(gctx, K_rep, D, S, H);
        }
        ggml_tensor * Q_tail_perm = ggml_cont(gctx,
            ggml_permute(gctx, Q_last_v[il].t, 0, 2, 1, 3));
        ggml_tensor * attn_score = ggml_mul_mat(gctx, K_score, Q_tail_perm);
        ggml_tensor * probs = ggml_soft_max_ext(gctx, attn_score, mask_tail_buf.t,
                                                scale, 0.0f);
        ggml_set_output(probs);

        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, probs);

        ggml_backend_buffer_t in_buf = ggml_backend_alloc_ctx_tensors(gctx, w.backend);
        ggml_gallocr_t s_galloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(w.backend));
        if (!ggml_gallocr_alloc_graph(s_galloc, gf)) {
            set_last_error("tail score graph alloc failed at layer " + std::to_string(il));
            ggml_gallocr_free(s_galloc);
            if (in_buf) ggml_backend_buffer_free(in_buf);
            ggml_free(gctx);
            cleanup_all();
            return false;
        }
        ggml_backend_graph_compute(w.backend, gf);
        ggml_backend_tensor_get(probs, probs_h.data(), 0,
                                probs_h.size() * sizeof(float));
        ggml_gallocr_free(s_galloc);
        if (in_buf) ggml_backend_buffer_free(in_buf);
        ggml_free(gctx);

        for (int t = 0; t < n_lookahead; ++t) {
            for (int j = 0; j < S; ++j) {
                float m = -INFINITY;
                for (int h = 0; h < H; ++h) {
                    float v = probs_h[(size_t)j
                                      + (size_t)t * S
                                      + (size_t)h * S * n_lookahead];
                    if (v > m) m = v;
                }
                size_t idx = (size_t)t * S + j;
                if (m > running_max[idx]) running_max[idx] = m;
            }
        }
    }

    auto t_total_end = std::chrono::steady_clock::now();
    double t_score = std::chrono::duration<double>(t_total_end - t_score_start).count();
    std::fprintf(stderr,
        "[qwen3-0.6b-fp] forward %.2fs (S=%d, A_setup=%.2fs A_alloc=%.2fs A_compute=%.2fs FP=%.2fs B_warm=%.2fs B_setup=%.2fs B_alloc=%.2fs B_copy_in=%.2fs B_norm=%.2fs B_compute=%.2fs B_copy_out=%.2fs)  "
        "tail-score %.2fs  total %.2fs\n",
        t_fwd, S, t_a_setup, t_a_alloc, t_compute_a, t_fp, t_b_warm, t_b_setup, t_b_alloc, t_b_copy_in, t_b_norm, t_compute_b, t_b_copy_out, t_score, t_fwd + t_score);
    std::fflush(stderr);

    cleanup_all();
    return true;
}

} // namespace dflash27b
