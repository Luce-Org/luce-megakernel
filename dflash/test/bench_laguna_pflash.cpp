// End-to-end PFlash + Laguna TTFT bench. Mirrors the qwen3.6-27B PFlash flow:
//
//   1. Tokenize input (synthetic in DRAFTER vocab for the bench)
//   2. Drafter (Qwen3-0.6B BF16) score_and_compress  -> surviving Qwen3 IDs
//   3. Cross-tokenizer mapping Qwen3 IDs -> Laguna IDs (NOT plumbed yet; we
//      use a fake target token for compute-time-only measurement)
//   4. Laguna build_laguna_graph dense prefill on the COMPRESSED sequence
//   5. Report drafter time + target time + total TTFT and the compression
//      ratio achieved.
//
// Usage:
//   bench_laguna_pflash <laguna.gguf> <drafter.gguf> <N> [keep_ratio=0.10] [chunk=2048]

#include "laguna_internal.h"
#include "internal.h"
#include "qwen3_drafter.h"
#include "dflash27b.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"

using namespace dflash27b;

static ggml_status laguna_chunked_prefill(
    ggml_backend_t backend,
    const LagunaTargetWeights & w,
    LagunaTargetCache & cache,
    const std::vector<float> & embed_full,   // [n_tokens, n_embd]
    int n_tokens,
    int chunk,
    bool no_mask,
    double * out_pf_s,
    int * out_argmax,
    float * out_logit)
{
    *out_pf_s = 0.0;
    *out_argmax = -1;
    *out_logit  = 0.0f;
    const int n_chunks = (n_tokens + chunk - 1) / chunk;
    ggml_gallocr_t galloc = nullptr;

    for (int c = 0; c < n_chunks; ++c) {
        const int kv_start = c * chunk;
        const int n_tok    = std::min(chunk, n_tokens - c * chunk);
        const bool last    = (c == n_chunks - 1);

        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead() + 16 * 1024 * 1024;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);

        ggml_tensor * ie = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w.n_embd, n_tok, 1);
        ggml_set_input(ie);
        ggml_tensor * pp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tok);
        ggml_set_input(pp);
        ggml_tensor * mk = nullptr, * mkc = nullptr;
        if (!no_mask && n_tok > 1) {
            const int kv_len = kv_start + n_tok;
            mk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kv_len, n_tok, 1, 1);
            ggml_set_input(mk);
            mkc = ggml_cast(ctx, mk, GGML_TYPE_F16);
        }

        LagunaGraphInputs gi{};
        gi.inp_embed       = ie;
        gi.positions       = pp;
        gi.attn_mask       = mkc;
        gi.n_tokens        = n_tok;
        gi.kv_start        = kv_start;
        gi.output_logits   = last;
        gi.output_last_only= last;

        LagunaGraphOutputs go = build_laguna_graph(ctx, gf, w, cache, gi);

        if (!galloc) galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            std::fprintf(stderr, "gallocr_alloc chunk=%d failed\n", c);
            ggml_free(ctx); if (galloc) ggml_gallocr_free(galloc);
            return GGML_STATUS_FAILED;
        }

        ggml_backend_tensor_set(ie, embed_full.data() + (size_t)kv_start * w.n_embd,
                                 0, (size_t)n_tok * w.n_embd * sizeof(float));
        std::vector<int32_t> ppos(n_tok);
        for (int i = 0; i < n_tok; ++i) ppos[i] = kv_start + i;
        ggml_backend_tensor_set(pp, ppos.data(), 0, ppos.size() * sizeof(int32_t));
        if (mk) {
            const int kv_len = kv_start + n_tok;
            std::vector<float> mb((size_t)kv_len * n_tok, 0.0f);
            for (int t = 0; t < n_tok; ++t) {
                const int abs_q = kv_start + t;
                for (int j = 0; j < kv_len; ++j) {
                    if (j > abs_q) mb[(size_t)t * kv_len + j] = -INFINITY;
                }
            }
            ggml_backend_tensor_set(mk, mb.data(), 0, mb.size() * sizeof(float));
        }

        if (c == 0) {
            ggml_backend_graph_compute(backend, gf);  // warm
            ggml_backend_synchronize(backend);
        }
        auto tA = std::chrono::steady_clock::now();
        ggml_status st = ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        auto tB = std::chrono::steady_clock::now();
        if (st != GGML_STATUS_SUCCESS) {
            ggml_free(ctx); if (galloc) ggml_gallocr_free(galloc);
            return st;
        }
        cache.cur_pos = kv_start + n_tok;
        *out_pf_s += std::chrono::duration<double>(tB - tA).count();

        if (last && go.logits) {
            const int64_t vocab = go.logits->ne[0];
            std::vector<float> ll((size_t)vocab);
            ggml_backend_tensor_get(go.logits, ll.data(), 0, ll.size() * sizeof(float));
            int best = 0; float bv = ll[0];
            for (int i = 0; i < (int)vocab; ++i) if (ll[i] > bv) { bv = ll[i]; best = i; }
            *out_argmax = best;
            *out_logit  = bv;
        }
        ggml_free(ctx);
    }
    if (galloc) ggml_gallocr_free(galloc);
    return GGML_STATUS_SUCCESS;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <laguna.gguf> <drafter.gguf> <N> [keep_ratio=0.10] [chunk=2048]\n", argv[0]);
        return 2;
    }
    const std::string laguna_path  = argv[1];
    const std::string drafter_path = argv[2];
    const int N           = std::atoi(argv[3]);
    const float keep_r    = (argc >= 5) ? std::atof(argv[4]) : 0.10f;
    const int chunk_arg   = (argc >= 6) ? std::atoi(argv[5]) : 2048;
    const int32_t fake_q  = 1972;  // any non-special drafter id
    const int32_t fake_l  = 1972;  // dummy laguna id (cross-tokenizer skipped)
    const bool no_mask    = (std::getenv("DFLASH_NO_MASK") != nullptr);

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    // ---- Phase 1: drafter only. Loads ~1.2 GB BF16 weights, runs compress
    //               with full VRAM available, then frees so target gets max
    //               headroom for activations + cache.
    DrafterContext drafter;
    auto td0 = std::chrono::steady_clock::now();
    if (!load_drafter(drafter_path, /*gpu_layers=*/-1, drafter)) {
        std::fprintf(stderr, "load_drafter failed: %s\n", dflash27b_last_error());
        return 1;
    }
    auto td1 = std::chrono::steady_clock::now();
    std::printf("[pflash] drafter loaded in %.2fs vocab=%d\n",
                std::chrono::duration<double>(td1 - td0).count(), drafter.weights.n_vocab);

    std::vector<int32_t> input(N, fake_q);
    auto tc0 = std::chrono::steady_clock::now();
    std::vector<int32_t> compressed = drafter_score_and_compress(
        drafter, input, keep_r, /*chunk_size=*/32, /*n_lookahead=*/8, /*pool_kernel=*/13);
    auto tc1 = std::chrono::steady_clock::now();
    if (compressed.empty()) {
        std::fprintf(stderr, "drafter compress failed: %s\n", dflash27b_last_error());
        free_drafter(drafter); return 1;
    }
    const int M = (int)compressed.size();
    const double drafter_s = std::chrono::duration<double>(tc1 - tc0).count();
    std::printf("[pflash] drafter compress N=%d -> M=%d ratio=%.4f in %.3fs\n",
                N, M, (double)M / N, drafter_s);

    // Free drafter so its 1.2 GB BF16 weights + scratch buffers don't compete
    // with target cache + activations.
    free_drafter(drafter);

    // ---- Phase 2: load Laguna target now that drafter VRAM is free ----
    LagunaTargetWeights w;
    if (!load_target_gguf_laguna(laguna_path, backend, w)) {
        std::fprintf(stderr, "load_laguna failed: %s\n", dflash27b_last_error());
        return 1;
    }

    LagunaTargetCache cache;
    if (const char * kv_t = std::getenv("DFLASH_KV_TYPE")) {
        const std::string s = kv_t;
        if      (s == "q4_0") { cache.kv_k_type = GGML_TYPE_Q4_0; cache.kv_v_type = GGML_TYPE_Q4_0; }
        else if (s == "q5_0") { cache.kv_k_type = GGML_TYPE_Q5_0; cache.kv_v_type = GGML_TYPE_Q5_0; }
        else if (s == "q8_0") { cache.kv_k_type = GGML_TYPE_Q8_0; cache.kv_v_type = GGML_TYPE_Q8_0; }
        else if (s == "f16")  { cache.kv_k_type = GGML_TYPE_F16;  cache.kv_v_type = GGML_TYPE_F16;  }
    }
    // Cache sized for COMPRESSED length M (much smaller than raw N).
    if (!create_laguna_target_cache(w, M, backend, cache)) {
        std::fprintf(stderr, "create_laguna_target_cache: %s\n", dflash27b_last_error());
        free_laguna_target_weights(w); return 1;
    }
    std::printf("[pflash] laguna cache max_ctx=%d KV=%s/%s\n", M,
                ggml_type_name(cache.kv_k_type), ggml_type_name(cache.kv_v_type));

    // ---- Embed compressed token (single fake) for Laguna --------------
    std::vector<int32_t> laguna_ids(M, fake_l);
    std::vector<float> embed_full((size_t)M * w.n_embd);
    if (!w.embedder.embed(laguna_ids.data(), M, embed_full.data())) {
        std::fprintf(stderr, "laguna embed failed at M=%d\n", M);
        free_laguna_target_cache(cache); free_laguna_target_weights(w); free_drafter(drafter);
        return 1;
    }

    // ---- Laguna chunked prefill on COMPRESSED sequence -----------------
    const int chunk = std::min(M, chunk_arg);
    double tgt_s = 0.0; int argmax = 0; float logit = 0.0f;
    auto tt0 = std::chrono::steady_clock::now();
    ggml_status st = laguna_chunked_prefill(backend, w, cache, embed_full,
                                              M, chunk, no_mask,
                                              &tgt_s, &argmax, &logit);
    auto tt1 = std::chrono::steady_clock::now();
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "laguna prefill failed status=%d\n", (int)st);
        free_laguna_target_cache(cache); free_laguna_target_weights(w); free_drafter(drafter);
        return 1;
    }
    const double tgt_total_s = std::chrono::duration<double>(tt1 - tt0).count();
    const double total_ttft  = drafter_s + tgt_s;

    std::printf("[pflash] laguna prefill M=%d in %.3fs (graph %.3fs, %.1f tok/s effective on N=%d)\n",
                M, tgt_total_s, tgt_s, N / std::max(1e-9, total_ttft), N);
    std::printf("[pflash] === SUMMARY ===\n");
    std::printf("[pflash] N=%d  M=%d  compress=%.4f  drafter=%.3fs  target=%.3fs  TTFT=%.3fs  effective=%.1f tok/s\n",
                N, M, (double)M / N, drafter_s, tgt_s, total_ttft,
                N / std::max(1e-9, total_ttft));
    std::printf("[pflash] argmax=%d logit=%.3f\n", argmax, logit);

    free_laguna_target_cache(cache);
    free_laguna_target_weights(w);
    ggml_backend_free(backend);
    return 0;
}
