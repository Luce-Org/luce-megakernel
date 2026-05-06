// Qwen3-0.6B drafter for pflash speculative prefill, hosted in-process.
//
// Wires three pieces:
//   - qwen3_0p6b_loader.cpp : mmap GGUF + populate ggml tensors on backend
//   - qwen3_0p6b_graph.cpp  : custom forward (per-layer ggml + FP CUDA kernel)
//   - chunk-top-K + span merge (this file)
//
// Single-pass forward at full S using a custom Qwen3-0.6B graph with the
// FlashPrefill block-sparse attention kernel (or BSA when enabled). Tail
// attention scoring runs in a separate post-forward graph using saved Q_last
// and K_curr per layer.
//
// Result running_max [n_lookahead, S] f32 is reduced to per-token scores via
// mean-over-lookahead, smoothed with AvgPool, scored per chunk, top-K kept.

#include "qwen3_drafter.h"
#include "qwen3_0p6b_drafter.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unordered_map>
#include <vector>

namespace dflash27b {

namespace {

static constexpr uint16_t F16_ZERO = 0x0000;
static constexpr uint16_t F16_NEG_INF = 0xFC00;

static int align_up_i(int x, int a) { return ((x + a - 1) / a) * a; }

static void build_causal_mask_f16(std::vector<uint16_t> & out, int kv_len, int n_tokens, int kv_start) {
    const int kv_pad = align_up_i(kv_len, 32);
    const int q_pad = align_up_i(n_tokens, 32);
    out.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    for (int q = 0; q < n_tokens; ++q) {
        const int abs_q = kv_start + q;
        for (int k = 0; k <= abs_q && k < kv_len; ++k) {
            out[(size_t)q * kv_pad + k] = F16_ZERO;
        }
    }
}

struct Qwen35DrafterState {
    TargetWeights weights;
};

static int env_int(const char * name, int fallback) {
    if (const char * v = std::getenv(name)) {
        int x = std::atoi(v);
        if (x >= 0) return x;
    }
    return fallback;
}

static void force_chunk_neighborhood(std::vector<uint8_t> & forced, int n_chunks,
                                     int chunk, int radius) {
    int lo = std::max(0, chunk - radius);
    int hi = std::min(n_chunks - 1, chunk + radius);
    for (int c = lo; c <= hi; ++c) forced[(size_t)c] = 1;
}

#if defined(DFLASH27B_USE_HIP)
bool prewarm_drafter_once(const Qwen3DrafterWeights & w) {
    static bool warmed = false;
    if (warmed || std::getenv("DFLASH_FP_SKIP_PREWARM")) {
        return true;
    }

    const int warm_tokens = 1024;
    const int n_lookahead = 8;
    std::vector<int32_t> ids((size_t)warm_tokens, 0);
    std::vector<float> running_max;

    auto t0 = std::chrono::steady_clock::now();
    bool ok = forward_qwen3_0p6b_drafter(w, ids, n_lookahead, running_max);
    auto t1 = std::chrono::steady_clock::now();
    if (!ok) {
        return false;
    }

    std::fprintf(stderr, "[drafter] HIP prewarm %.2fs (%d tokens)\n",
                 std::chrono::duration<double>(t1 - t0).count(), warm_tokens);
    std::fflush(stderr);
    warmed = true;
    return true;
}
#endif

} // namespace

bool parse_drafter_arch(const std::string & name, DrafterArch & out) {
    if (name == "qwen3-0.6b" || name == "qwen3_0p6b" || name == "qwen3") {
        out = DrafterArch::Qwen3_0p6b;
        return true;
    }
    if (name == "qwen35-0.8b" || name == "qwen3.5-0.8b" || name == "qwen35_0p8b" || name == "qwen35") {
        out = DrafterArch::Qwen35_0p8b;
        return true;
    }
    return false;
}

const char * drafter_arch_name(DrafterArch arch) {
    switch (arch) {
        case DrafterArch::Qwen3_0p6b: return "qwen3-0.6b";
        case DrafterArch::Qwen35_0p8b: return "qwen35-0.8b";
    }
    return "unknown";
}

bool load_drafter(const std::string & gguf_path, int /*gpu_layers*/,
                  DrafterContext & out) {
    return load_drafter(gguf_path, /*gpu_layers=*/999, DrafterArch::Qwen3_0p6b, out);
}

bool load_drafter(const std::string & gguf_path, int /*gpu_layers*/,
                  DrafterArch arch, DrafterContext & out) {
    if (out.loaded) {
        set_last_error("drafter already loaded");
        return false;
    }

    // If caller didn't supply a backend, spin up our own CUDA one. Sharing
    // would be ideal but we don't have a handle to the daemon's backend
    // through this API. Same-process CUDA pools coexist fine — fragmentation
    // is the only cost, and we free everything in free_drafter.
    if (!out.backend) {
        size_t n_dev = ggml_backend_dev_count();
        for (size_t i = 0; i < n_dev; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                out.backend = ggml_backend_dev_init(dev, nullptr);
                break;
            }
        }
        if (!out.backend) {
            set_last_error("load_drafter: no GPU backend available");
            return false;
        }
    }

    if (arch == DrafterArch::Qwen35_0p8b) {
        auto * st = new Qwen35DrafterState();
        if (!load_target_gguf(gguf_path, out.backend, st->weights)) {
            delete st;
            return false;
        }
        out.arch_state = st;
        out.loaded = true;
        out.arch = arch;
        std::fprintf(stderr,
            "[drafter] loaded %s qwen35: n_layer=%d n_head=%d n_head_kv=%d "
            "n_embd=%d n_ff=%d head_dim=%d vocab=%d\n",
            drafter_arch_name(arch),
            st->weights.n_layer, st->weights.n_head, st->weights.n_head_kv,
            st->weights.n_embd, st->weights.n_ff, st->weights.n_embd_head_k,
            st->weights.n_vocab);
        std::fflush(stderr);
        return true;
    }

    if (!load_qwen3_0p6b_drafter(gguf_path, out.backend, out.weights)) {
        return false;
    }

    out.loaded = true;
    out.arch = arch;
    std::fprintf(stderr,
        "[drafter] loaded %s BF16: n_layer=%d n_head=%d n_kv=%d "
        "n_embd=%d n_ff=%d head_dim=%d vocab=%d\n",
        drafter_arch_name(arch),
        out.weights.n_layer, out.weights.n_head, out.weights.n_head_kv,
        out.weights.n_embd, out.weights.n_ff, out.weights.head_dim,
        out.weights.n_vocab);
    std::fflush(stderr);

#if defined(DFLASH27B_USE_HIP)
    if (!prewarm_drafter_once(out.weights)) {
        return false;
    }
#endif

    return true;
}

void free_drafter(DrafterContext & ctx) {
    if (ctx.arch == DrafterArch::Qwen35_0p8b && ctx.arch_state) {
        auto * st = static_cast<Qwen35DrafterState *>(ctx.arch_state);
        free_target_weights(st->weights);
        delete st;
        ctx.arch_state = nullptr;
    }
    if (ctx.loaded) {
        if (ctx.arch == DrafterArch::Qwen3_0p6b) {
            free_qwen3_0p6b_drafter(ctx.weights);
        }
    }
    if (ctx.backend) {
        ggml_backend_free(ctx.backend);
        ctx.backend = nullptr;
    }
    ctx.loaded = false;
}

static std::vector<int32_t> qwen35_score_and_compress(
    TargetWeights & w,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int chunk_size,
    int n_lookahead) {

    const int S = (int)ids.size();
    const int hidden = w.n_embd;
    if (S < n_lookahead + 1) return ids;

    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> running_max((size_t)n_lookahead * S, -INFINITY);

    TargetCache cache;
#if defined(_WIN32)
    _putenv_s("DFLASH27B_KV_TQ3", "0");
#else
    const char * old_tq3 = std::getenv("DFLASH27B_KV_TQ3");
    std::string old_tq3_s = old_tq3 ? old_tq3 : "";
    setenv("DFLASH27B_KV_TQ3", "0", 1);
#endif
    if (!create_target_cache(w, S, 0, w.backend, cache, true)) {
#if !defined(_WIN32)
        if (old_tq3) setenv("DFLASH27B_KV_TQ3", old_tq3_s.c_str(), 1);
        else unsetenv("DFLASH27B_KV_TQ3");
#endif
        return {};
    }
#if !defined(_WIN32)
    if (old_tq3) setenv("DFLASH27B_KV_TQ3", old_tq3_s.c_str(), 1);
    else unsetenv("DFLASH27B_KV_TQ3");
#endif

    ggml_init_params act_ip{};
    act_ip.mem_size = (size_t)8 * ggml_tensor_overhead() + 4096;
    act_ip.no_alloc = true;
    ggml_context * act_ctx = ggml_init(act_ip);
    if (!act_ctx) {
        free_target_cache(cache);
        set_last_error("qwen35 drafter activation ctx init failed");
        return {};
    }
    ggml_tensor * act_in = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, S);
    ggml_tensor * act_out = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, S);
    ggml_backend_buffer_t act_buf = ggml_backend_alloc_ctx_tensors(act_ctx, w.backend);
    if (!act_buf) {
        ggml_free(act_ctx);
        free_target_cache(cache);
        set_last_error("qwen35 drafter activation allocation failed");
        return {};
    }

    {
        const int batch = 2048;
        std::vector<float> emb((size_t)hidden * batch);
        for (int i = 0; i < S; i += batch) {
            const int n = std::min(batch, S - i);
            if (!w.embedder.embed(ids.data() + i, n, emb.data())) {
                ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 drafter embedding failed");
                return {};
            }
            ggml_backend_tensor_set(act_in, emb.data(), (size_t)i * act_in->nb[1], (size_t)hidden * n * sizeof(float));
        }
    }

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
    const int ubatch = 1024;
    for (int il = 0; il < w.n_layer; ++il) {
        const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);
        int fa_idx = 0;
        if (is_attn) {
            for (int k = 0; k < il; ++k) if (((k + 1) % w.full_attention_interval) == 0) ++fa_idx;
        }
        for (int start = 0; start < S; start += ubatch) {
            const int n = std::min(ubatch, S - start);
            const int kv_len = start + n;

            ggml_init_params ip{};
            ip.mem_size = 512 * 1024 * 1024;
            ip.no_alloc = true;
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) {
                ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 drafter layer graph ctx init failed");
                return {};
            }
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);
            ggml_tensor * inp = ggml_view_2d(ctx, act_in, hidden, n, act_in->nb[1], (size_t)start * act_in->nb[1]);
            ggml_tensor * pos = nullptr;
            ggml_tensor * mask = nullptr;
            if (is_attn) {
                pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n);
                ggml_set_input(pos);
                mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, align_up_i(kv_len, 32), align_up_i(n, 32));
                ggml_set_input(mask);
            }
            ggml_tensor * out = build_qwen35_layer(ctx, gf, w, cache, il, inp, pos, mask, start, n, false, 0);
            ggml_tensor * dst = ggml_view_2d(ctx, act_out, hidden, n, act_out->nb[1], (size_t)start * act_out->nb[1]);
            if (ggml_nelements(out) != ggml_nelements(dst)) {
                std::fprintf(stderr,
                    "[qwen35-drafter] layer output shape mismatch il=%d start=%d out=[%lld,%lld,%lld,%lld] dst=[%lld,%lld,%lld,%lld]\n",
                    il, start,
                    (long long)out->ne[0], (long long)out->ne[1], (long long)out->ne[2], (long long)out->ne[3],
                    (long long)dst->ne[0], (long long)dst->ne[1], (long long)dst->ne[2], (long long)dst->ne[3]);
                ggml_free(ctx); ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 layer output shape mismatch");
                return {};
            }
            ggml_build_forward_expand(gf, ggml_cpy(ctx, out, dst));
            if (!ggml_gallocr_alloc_graph(alloc, gf)) {
                ggml_free(ctx); ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 drafter graph allocation failed");
                return {};
            }
            if (is_attn) {
                std::vector<int32_t> p4((size_t)4 * n, 0);
                for (int i = 0; i < n; ++i) {
                    int p = start + i;
                    p4[(size_t)0 * n + i] = p;
                    p4[(size_t)1 * n + i] = p;
                    p4[(size_t)2 * n + i] = p;
                }
                ggml_backend_tensor_set(pos, p4.data(), 0, p4.size() * sizeof(int32_t));
                std::vector<uint16_t> m;
                build_causal_mask_f16(m, kv_len, n, start);
                ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(uint16_t));
            }
            auto st = ggml_backend_graph_compute(w.backend, gf);
            ggml_free(ctx);
            if (st != GGML_STATUS_SUCCESS) {
                ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 drafter graph compute failed");
                return {};
            }
        }

        if (is_attn) {
            ggml_init_params sip{};
            sip.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(1024, false) + 64 * 1024;
            sip.no_alloc = true;
            ggml_context * sctx = ggml_init(sip);
            if (!sctx) {
                ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 score graph ctx allocation failed");
                return {};
            }
            ggml_cgraph * sgf = ggml_new_graph_custom(sctx, 1024, false);
            const int K_len = (int) cache.attn_k[(size_t)fa_idx]->ne[1];
            ggml_tensor * mask_tail = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, K_len, n_lookahead);
            ggml_tensor * K_f32 = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, w.n_embd_head_k, K_len, w.n_head_kv);
            ggml_tensor * K_cast = ggml_cpy(sctx, cache.attn_k[(size_t)fa_idx], K_f32);
            ggml_tensor * K_score = nullptr;
            if (w.n_head != w.n_head_kv) {
                const int gqa = w.n_head / w.n_head_kv;
                ggml_tensor * K_4d = ggml_reshape_4d(sctx, K_cast, w.n_embd_head_k, K_len, 1, w.n_head_kv);
                ggml_tensor * K_tpl = ggml_new_tensor_4d(sctx, GGML_TYPE_F32, w.n_embd_head_k, K_len, gqa, w.n_head_kv);
                ggml_tensor * K_rep = ggml_repeat(sctx, K_4d, K_tpl);
                K_score = ggml_reshape_3d(sctx, K_rep, w.n_embd_head_k, K_len, w.n_head);
            } else {
                K_score = K_cast;
            }
            const TargetLayer & L = w.layers[il];
            ggml_tensor * inp_tail = ggml_view_2d(sctx, act_in, hidden, n_lookahead,
                act_in->nb[1], (size_t)(S - n_lookahead) * act_in->nb[1]);
            ggml_tensor * q_cur = ggml_rms_norm(sctx, inp_tail, w.rms_eps);
            q_cur = ggml_mul(sctx, q_cur, L.attn_norm);
            ggml_tensor * QG = ggml_mul_mat(sctx, L.wq, q_cur);
            QG = ggml_reshape_3d(sctx, QG, w.n_embd_head_k * 2, w.n_head, n_lookahead);
            ggml_tensor * Q = ggml_view_3d(sctx, QG,
                w.n_embd_head_k, w.n_head, n_lookahead,
                ggml_element_size(QG) * w.n_embd_head_k * 2,
                ggml_element_size(QG) * w.n_embd_head_k * 2 * w.n_head,
                0);
            Q = ggml_rms_norm(sctx, Q, w.rms_eps);
            Q = ggml_mul(sctx, Q, L.q_norm);
            ggml_tensor * pos_tail = ggml_new_tensor_1d(sctx, GGML_TYPE_I32, 4 * n_lookahead);
            int sections[4];
            for (int k = 0; k < 4; ++k) sections[k] = w.rope_sections[k];
            Q = ggml_rope_multi(sctx, Q, pos_tail, nullptr,
                                w.rope_dimension_count, sections, GGML_ROPE_TYPE_MROPE,
                                0, w.rope_theta, 1.0f,
                                0.0f, 1.0f, 0.0f, 0.0f);
            ggml_tensor * Q_tail_perm = ggml_cont(sctx, ggml_permute(sctx, Q, 0, 2, 1, 3));
            ggml_tensor * attn_score = ggml_mul_mat(sctx, K_score, Q_tail_perm);
            ggml_tensor * probs = ggml_soft_max_ext(sctx, attn_score, mask_tail, 1.0f / std::sqrt((float)w.n_embd_head_k), 0.0f);
            ggml_set_output(probs);
            ggml_build_forward_expand(sgf, probs);
            ggml_gallocr_t salloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
            if (!ggml_gallocr_alloc_graph(salloc, sgf)) {
                ggml_gallocr_free(salloc); ggml_free(sctx); ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 score graph allocation failed");
                return {};
            }
            std::vector<int32_t> pos4((size_t)4 * n_lookahead, 0);
            for (int i = 0; i < n_lookahead; ++i) {
                int p = S - n_lookahead + i;
                pos4[(size_t)0 * n_lookahead + i] = p;
                pos4[(size_t)1 * n_lookahead + i] = p;
                pos4[(size_t)2 * n_lookahead + i] = p;
            }
            ggml_backend_tensor_set(pos_tail, pos4.data(), 0, pos4.size() * sizeof(int32_t));
            std::vector<float> mask((size_t)n_lookahead * K_len, 0.0f);
            for (int t = 0; t < n_lookahead; ++t) {
                const int visible_end = S - n_lookahead + t + 1;
                for (int j = 0; j < K_len; ++j) {
                    mask[(size_t)t * K_len + j] = (j < visible_end) ? 0.0f : -INFINITY;
                }
            }
            ggml_backend_tensor_set(mask_tail, mask.data(), 0, mask.size() * sizeof(float));
            auto st = ggml_backend_graph_compute(w.backend, sgf);
            if (st != GGML_STATUS_SUCCESS) {
                ggml_gallocr_free(salloc); ggml_free(sctx); ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 score graph compute failed");
                return {};
            }
            std::vector<float> tmp((size_t)K_len * n_lookahead * w.n_head);
            ggml_backend_tensor_get(probs, tmp.data(), 0, tmp.size() * sizeof(float));
            for (int h = 0; h < w.n_head; ++h) {
                for (int t = 0; t < n_lookahead; ++t) {
                    for (int j = 0; j < S; ++j) {
                        const size_t src = (size_t)h * K_len * n_lookahead + (size_t)t * K_len + j;
                        const size_t dst = (size_t)t * S + j;
                        running_max[dst] = std::max(running_max[dst], tmp[src]);
                    }
                }
            }
            ggml_gallocr_free(salloc);
            ggml_free(sctx);
        }
        std::swap(act_in, act_out);
    }
    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(act_buf);
    ggml_free(act_ctx);
    free_target_cache(cache);

    std::vector<float> score((size_t)S, 0.0f);
    for (int j = 0; j < S; ++j) {
        float s = 0.0f;
        for (int t = 0; t < n_lookahead; ++t) s += running_max[(size_t)t * S + j];
        score[(size_t)j] = s / (float)n_lookahead;
    }

    const int n_chunks = (S + chunk_size - 1) / chunk_size;
    const int n_keep = std::max(1, (int)((float)n_chunks * keep_ratio));
    
    std::vector<float> smooth_score = score;
    int pool_kernel = std::max(3, env_int("DFLASH_COMPRESS_POOL_KERNEL", 5));
    std::vector<float> smoothed((size_t)S, 0.0f);
    int half = pool_kernel / 2;
    for (int j = 0; j < S; ++j) {
        int lo = std::max(0, j - half);
        int hi = std::min(S - 1, j + half);
        float s = 0.0f;
        int n = 0;
        for (int k = lo; k <= hi; ++k) { s += score[(size_t)k]; ++n; }
        smoothed[(size_t)j] = (n > 0) ? (s / (float)n) : 0.0f;
    }
    smooth_score.swap(smoothed);
    
    std::vector<std::pair<float, int>> chunk_means;
    for (int c = 0; c < n_chunks; ++c) {
        int lo = c * chunk_size, hi = std::min(S, lo + chunk_size);
        float s = 0.0f;
        for (int j = lo; j < hi; ++j) s += smooth_score[(size_t)j];
        chunk_means.push_back({s / std::max(1, hi - lo), c});
    }
    std::sort(chunk_means.begin(), chunk_means.end(), [](auto a, auto b) { return a.first > b.first; });
    
    std::vector<uint8_t> selected((size_t)n_chunks, 0);
    int count = 0;
    for (int c = 0; c < std::min(n_chunks, env_int("DFLASH_COMPRESS_HEAD_CHUNKS", 8)); ++c) { selected[(size_t)c] = 1; ++count; }
    for (int c = std::max(0, n_chunks - env_int("DFLASH_COMPRESS_TAIL_CHUNKS", 24)); c < n_chunks; ++c) if (!selected[(size_t)c]) { selected[(size_t)c] = 1; ++count; }
    
    const int query_tokens = env_int("DFLASH_COMPRESS_QUERY_TOKENS", 96);
    const int anchor_radius = env_int("DFLASH_COMPRESS_ANCHOR_RADIUS", 2);
    const int max_anchor_hits = env_int("DFLASH_COMPRESS_MAX_ANCHOR_HITS", 8);
    std::vector<uint8_t> forced((size_t)n_chunks, 0);

    const int q0 = std::max(0, S - query_tokens);
    constexpr int NGRAM = 4;
    for (int q = q0; q + NGRAM <= S; ++q) {
        int hits = 0;
        int hit_pos[8];
        const int search_end = std::max(0, q0 - NGRAM);
        for (int p = 0; p <= search_end && hits <= max_anchor_hits; ++p) {
            bool same = true;
            for (int k = 0; k < NGRAM; ++k) {
                if (ids[(size_t)p + k] != ids[(size_t)q + k]) { same = false; break; }
            }
            if (same) {
                if (hits < 8) hit_pos[hits] = p;
                ++hits;
            }
        }
        if (hits > 0 && hits <= max_anchor_hits) {
            for (int i = 0; i < hits && i < 8; ++i) {
                force_chunk_neighborhood(forced, n_chunks, hit_pos[i] / chunk_size, anchor_radius);
            }
        }
    }
    for (int c = 0; c < n_chunks; ++c) {
        if (forced[(size_t)c] && !selected[(size_t)c]) {
            selected[(size_t)c] = 1;
            ++count;
        }
    }

    // Global aggregation tasks often depend on repeated rare tokens that do
    // not appear in the final query. Preserve high-frequency-but-not-filler
    // token chunks before filling with model-score top-K.
    const int repeat_min = env_int("DFLASH_COMPRESS_REPEAT_MIN", 4);
    const int repeat_max = env_int("DFLASH_COMPRESS_REPEAT_MAX", 32);
    const int repeat_limit = env_int("DFLASH_COMPRESS_REPEAT_CHUNKS", n_keep);
    if (repeat_min > 1 && count < repeat_limit) {
        std::unordered_map<int32_t, int> freq;
        freq.reserve((size_t)S);
        const int repeat_scan_end = std::max(0, S - query_tokens);
        for (int j = 0; j < repeat_scan_end; ++j) {
            ++freq[ids[(size_t)j]];
        }
        std::vector<std::pair<int, int32_t>> repeated;
        repeated.reserve(freq.size());
        for (const auto & kv : freq) {
            if (kv.second >= repeat_min && kv.second <= repeat_max) {
                repeated.push_back({kv.second, kv.first});
            }
        }
        std::sort(repeated.begin(), repeated.end(), [](const auto & a, const auto & b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        });
        for (const auto & rp : repeated) {
            if (count >= repeat_limit) break;
            const int32_t tok = rp.second;
            for (int j = 0; j < repeat_scan_end && count < repeat_limit; ++j) {
                if (ids[(size_t)j] != tok) continue;
                const int c = j / chunk_size;
                if (!selected[(size_t)c]) {
                    selected[(size_t)c] = 1;
                    ++count;
                }
            }
        }
    }
    
    for (auto [_, c] : chunk_means) {
        if (count >= n_keep) break;
        if (!selected[(size_t)c]) { selected[(size_t)c] = 1; ++count; }
    }
    
    std::vector<int32_t> out_ids;
    std::vector<int> selected_chunks;
    for (int c = 0; c < n_chunks; ++c) {
        if (selected[(size_t)c]) selected_chunks.push_back(c);
    }
    int span_start = -1, span_end = -1;
    for (int c : selected_chunks) {
        int s_ = c * chunk_size;
        int e_ = std::min(S, (c + 1) * chunk_size);
        if (span_start < 0) {
            span_start = s_; span_end = e_;
        } else if (s_ == span_end) {
            span_end = e_;
        } else {
            for (int j = span_start; j < span_end; ++j) out_ids.push_back(ids[j]);
            span_start = s_; span_end = e_;
        }
    }
    if (span_start >= 0) {
        for (int j = span_start; j < span_end; ++j) out_ids.push_back(ids[j]);
    }

    auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[qwen35-drafter] forward+compress %.2fs S=%d kept=%zu (%d/%d chunks)\n",
                 std::chrono::duration<double>(t1 - t0).count(), S, out_ids.size(), count, n_chunks);
    std::fflush(stderr);
    return out_ids;
}

std::vector<int32_t> drafter_score_and_compress(
    DrafterContext & ctx,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int chunk_size,
    int n_lookahead,
    int pool_kernel) {
    if (!ctx.loaded) {
        set_last_error("drafter not loaded");
        return {};
    }
    if (ctx.arch == DrafterArch::Qwen35_0p8b) {
        if (!ctx.arch_state) {
            set_last_error("qwen35 drafter state missing");
            return {};
        }
        auto * st = static_cast<Qwen35DrafterState *>(ctx.arch_state);
        return qwen35_score_and_compress(st->weights, ids, keep_ratio, chunk_size, n_lookahead);
    }
    const int S = (int)ids.size();
    if (S < n_lookahead + 1) {
        // Too short to score — return as-is.
        return ids;
    }

    // ── 1. Custom forward + GPU tail-attention scoring ────────────────
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> running_max;
    if (!forward_qwen3_0p6b_drafter(ctx.weights, ids, n_lookahead, running_max)) {
        return {};
    }
    auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[drafter] forward+score in %.2fs S=%d\n",
        std::chrono::duration<double>(t1 - t0).count(), S);
    std::fflush(stderr);

    // ── 2. Mean over lookahead → per-token score [S] ──────────────────
    std::vector<float> score((size_t)S, 0.0f);
    for (int j = 0; j < S; ++j) {
        float s = 0.0f;
        for (int t = 0; t < n_lookahead; ++t) {
            s += running_max[(size_t)t * S + j];
        }
        score[j] = s / (float)n_lookahead;
    }

    // ── 3. AvgPool 1D smoothing ───────────────────────────────────────
    std::vector<float> smooth((size_t)S, 0.0f);
    int half = pool_kernel / 2;
    for (int j = 0; j < S; ++j) {
        int lo = std::max(0, j - half);
        int hi = std::min(S - 1, j + half);
        float s = 0.0f;
        int n = 0;
        for (int k = lo; k <= hi; ++k) { s += score[k]; ++n; }
        smooth[j] = (n > 0) ? (s / (float)n) : 0.0f;
    }

    // ── 4. Chunk-top-K + span merge ───────────────────────────────────
    int n_chunks = (S + chunk_size - 1) / chunk_size;
    int n_keep   = std::max(1, (int)((float)n_chunks * keep_ratio));
    std::vector<std::pair<float, int>> chunk_means;
    chunk_means.reserve((size_t)n_chunks);
    for (int c = 0; c < n_chunks; ++c) {
        int s_ = c * chunk_size;
        int e_ = std::min(S, (c + 1) * chunk_size);
        float m = 0.0f;
        for (int j = s_; j < e_; ++j) m += smooth[j];
        m /= std::max(1, e_ - s_);
        chunk_means.push_back({m, c});
    }
    std::sort(chunk_means.begin(), chunk_means.end(),
                      [](auto a, auto b) { return a.first > b.first; });

    // Retrieval tasks often repeat a rare key in the final query and in the
    // needle span. Exact scores alone can keep the query while dropping the
    // neighboring answer chunk, so force a small token-only anchor neighborhood.
    const int head_chunks = env_int("DFLASH_COMPRESS_HEAD_CHUNKS", 8);
    const int tail_chunks = env_int("DFLASH_COMPRESS_TAIL_CHUNKS", 24);
    const int query_tokens = env_int("DFLASH_COMPRESS_QUERY_TOKENS", 96);
    const int anchor_radius = env_int("DFLASH_COMPRESS_ANCHOR_RADIUS", 2);
    const int max_anchor_hits = env_int("DFLASH_COMPRESS_MAX_ANCHOR_HITS", 8);
    std::vector<uint8_t> selected_mask((size_t)n_chunks, 0);
    std::vector<uint8_t> forced((size_t)n_chunks, 0);
    for (int c = 0; c < std::min(n_chunks, head_chunks); ++c) forced[(size_t)c] = 1;
    for (int c = std::max(0, n_chunks - tail_chunks); c < n_chunks; ++c) forced[(size_t)c] = 1;

    const int q0 = std::max(0, S - query_tokens);
    constexpr int NGRAM = 4;
    for (int q = q0; q + NGRAM <= S; ++q) {
        int hits = 0;
        int hit_pos[8];
        const int search_end = std::max(0, q0 - NGRAM);
        for (int p = 0; p <= search_end && hits <= max_anchor_hits; ++p) {
            bool same = true;
            for (int k = 0; k < NGRAM; ++k) {
                if (ids[(size_t)p + k] != ids[(size_t)q + k]) { same = false; break; }
            }
            if (same) {
                if (hits < 8) hit_pos[hits] = p;
                ++hits;
            }
        }
        if (hits > 0 && hits <= max_anchor_hits) {
            for (int i = 0; i < hits && i < 8; ++i) {
                force_chunk_neighborhood(forced, n_chunks, hit_pos[i] / chunk_size, anchor_radius);
            }
        }
    }

    int selected_count = 0;
    int forced_count = 0;
    for (int c = 0; c < n_chunks; ++c) {
        if (forced[(size_t)c]) {
            selected_mask[(size_t)c] = 1;
            ++selected_count;
            ++forced_count;
        }
    }
    for (const auto & cm : chunk_means) {
        if (selected_count >= n_keep) break;
        int c = cm.second;
        if (!selected_mask[(size_t)c]) {
            selected_mask[(size_t)c] = 1;
            ++selected_count;
        }
    }

    std::vector<int> selected;
    selected.reserve((size_t)selected_count);
    for (int c = 0; c < n_chunks; ++c) {
        if (selected_mask[(size_t)c]) selected.push_back(c);
    }

    std::vector<int32_t> out;
    out.reserve((size_t)n_keep * chunk_size + 16);
    int span_start = -1, span_end = -1;
    for (int c : selected) {
        int s_ = c * chunk_size;
        int e_ = std::min(S, (c + 1) * chunk_size);
        if (span_start < 0) {
            span_start = s_; span_end = e_;
        } else if (s_ == span_end) {
            span_end = e_;
        } else {
            for (int j = span_start; j < span_end; ++j) out.push_back(ids[j]);
            span_start = s_; span_end = e_;
        }
    }
    if (span_start >= 0) {
        for (int j = span_start; j < span_end; ++j) out.push_back(ids[j]);
    }

    auto t2 = std::chrono::steady_clock::now();
    std::fprintf(stderr,
        "[drafter] score_and_compress total %.2fs S=%d kept=%zu (%d/%d chunks, forced=%d)\n",
        std::chrono::duration<double>(t2 - t0).count(),
        S, out.size(), (int)selected.size(), n_chunks, forced_count);
    std::fflush(stderr);

    return out;
}

} // namespace dflash27b
