// Smoke test for MoE forward pass: loads Qwen3.6-35B-A3B, runs one token
// through all 40 layers using run_qwen35moe_layer, then applies lm_head.
//
// Usage: smoke_moe_forward <path/to/qwen35moe.gguf> [n_cache_slots]
//
// If n_cache_slots is not specified, uses 200 (small for functional test).

#include "dflash27b.h"
#include "internal.h"
#include "expert_cache.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

using namespace dflash27b;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <qwen35moe.gguf> [n_cache_slots]\n", argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const int n_cache_slots = (argc >= 3) ? std::atoi(argv[2]) : 200;

    // ── 1. CUDA backend ──
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    // ── 2. Load model ──
    TargetWeights w;
    if (!load_target_gguf(model_path, backend, w)) {
        std::fprintf(stderr, "load_target_gguf failed: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("%s\n", dflash27b_last_error());

    if (!w.is_moe) {
        std::fprintf(stderr, "error: model is not MoE (arch != qwen35moe)\n");
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }

    const int hidden  = w.n_embd;
    const int n_layer = w.n_layer;
    const int n_tokens = 1;  // single-token smoke test
    const int kv_start = 0;

    std::printf("MoE model: hidden=%d n_layer=%d n_experts=%d n_active=%d expert_ffn=%d\n",
        hidden, n_layer, w.n_experts, w.n_experts_active, w.expert_ffn_dim);

    // ── 3. ExpertCache ──
    // Determine alt slot count based on how many layers use non-primary down type.
    int n_alt_layers = 0;
    const auto & es = w.expert_source;
    for (int l = 0; l < es.n_layers; l++) {
        if (!es.layer_down_types.empty() && es.layer_down_types[l] != es.down_type)
            n_alt_layers++;
    }
    int n_alt_slots = n_alt_layers > 0 ? (n_cache_slots * n_alt_layers / es.n_layers) + 1 : 0;
    // Ensure alt slots have at least 8*n_alt_layers + 1 for NULL_SLOT.
    if (n_alt_slots > 0 && n_alt_slots < n_alt_layers * 9 + 1)
        n_alt_slots = n_alt_layers * 9 + 1;

    ExpertCache ecache;
    if (!ecache.init(backend, w.expert_source, n_cache_slots, n_alt_slots)) {
        std::fprintf(stderr, "ExpertCache init failed\n");
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }

    // Warmup: distribute slots across layers
    ecache.warmup(w.expert_source);

    // ── 4. MoeState ──
    MoeState moe{};
    if (!create_moe_state(backend, hidden, w.n_experts_active, n_tokens, moe)) {
        std::fprintf(stderr, "create_moe_state failed\n");
        ecache.destroy();
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }

    // ── 5. TargetCache (prefill_only, minimal context) ──
    TargetCache cache;
    if (!create_target_cache(w, /*max_ctx=*/32, /*max_verify_tokens=*/0, backend, cache,
                             /*prefill_only=*/true)) {
        std::fprintf(stderr, "create_target_cache failed: %s\n", dflash27b_last_error());
        free_moe_state(moe);
        ecache.destroy();
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }

    // ── 6. Activation buffers ──
    ggml_init_params act_ip{};
    act_ip.mem_size   = 8 * ggml_tensor_overhead() + 4096;
    act_ip.mem_buffer = nullptr;
    act_ip.no_alloc   = true;
    ggml_context * act_ctx = ggml_init(act_ip);
    ggml_tensor * act_in  = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, n_tokens);
    ggml_tensor * act_out = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, n_tokens);
    ggml_set_name(act_in, "act_in");
    ggml_set_name(act_out, "act_out");
    ggml_backend_buffer_t act_buf = ggml_backend_alloc_ctx_tensors(act_ctx, backend);
    if (!act_buf) {
        std::fprintf(stderr, "activation alloc failed\n");
        free_target_cache(cache);
        free_moe_state(moe);
        ecache.destroy();
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }

    // ── 7. Embed token 0 (BOS-like) ──
    std::vector<float> emb(hidden);
    int32_t tok_id = 0;
    if (!w.embedder.embed(&tok_id, 1, emb.data())) {
        std::fprintf(stderr, "embed failed\n");
        return 1;
    }
    ggml_backend_tensor_set(act_in, emb.data(), 0, sizeof(float) * hidden);

    // ── 8. Positions tensor (exact 4*n_tokens for M-RoPE) ──
    ggml_init_params pos_ip{};
    pos_ip.mem_size   = 4 * ggml_tensor_overhead() + 4096;
    pos_ip.mem_buffer = nullptr;
    pos_ip.no_alloc   = true;
    ggml_context * pos_ctx = ggml_init(pos_ip);
    ggml_tensor * positions = ggml_new_tensor_1d(pos_ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);
    ggml_backend_buffer_t pos_buf = ggml_backend_alloc_ctx_tensors(pos_ctx, backend);
    if (!pos_buf) {
        std::fprintf(stderr, "positions alloc failed\n");
        return 1;
    }
    // Fill positions: [pos, pos, pos, 0] for M-RoPE (3 spatial + 1 temporal)
    int32_t pos4[4] = { kv_start, kv_start, kv_start, 0 };
    ggml_backend_tensor_set(positions, pos4, 0, sizeof(pos4));

    // ── 9. Forward pass: 40 layers ──
    std::printf("\n─── Forward pass (%d layers, %d token) ───\n", n_layer, n_tokens);
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int il = 0; il < n_layer; il++) {
        bool ok = run_qwen35moe_layer(
            backend, w, cache, ecache, moe, w.expert_source,
            il, act_in, act_out,
            /*chunk_start=*/0, n_tokens,
            positions,
            /*attn_mask=*/nullptr,  // single token, no mask needed
            kv_start,
            /*capture=*/false);

        if (!ok) {
            std::fprintf(stderr, "run_qwen35moe_layer failed at layer %d\n", il);
            return 1;
        }

        // Diagnostic: check for NaN after each layer
        {
            std::vector<float> dbg(hidden);
            ggml_backend_tensor_get(act_out, dbg.data(), 0, sizeof(float) * hidden);
            bool has_nan = false;
            float sum = 0;
            for (int i = 0; i < hidden; i++) {
                if (std::isnan(dbg[i]) || std::isinf(dbg[i])) { has_nan = true; break; }
                sum += dbg[i] * dbg[i];
            }
            std::printf("  layer %2d: %s  L2=%.4f  first=[%.4f, %.4f, %.4f, %.4f]\n",
                il, has_nan ? "NaN!" : "OK  ",
                std::sqrt(sum), dbg[0], dbg[1], dbg[2], dbg[3]);
            if (has_nan) {
                std::printf("  *** NaN detected at layer %d, stopping.\n", il);
                break;
            }
        }

        // Swap: output of this layer is input to next
        std::swap(act_in, act_out);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double fwd_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("Forward pass: %.1f ms\n", fwd_ms);
    std::printf("Expert cache: %" PRId64 " hits, %" PRId64 " misses (%.1f%% hit rate)\n",
        ecache.hits(), ecache.misses(),
        ecache.hits() + ecache.misses() > 0
            ? 100.0 * ecache.hits() / (ecache.hits() + ecache.misses()) : 0.0);

    // ── 10. Output: RMSNorm + lm_head → argmax ──
    // act_in now has the final layer output (after last swap)
    {
        ggml_init_params lm_ip{};
        lm_ip.mem_size   = 256 * 1024 * 1024;
        lm_ip.mem_buffer = nullptr;
        lm_ip.no_alloc   = true;
        ggml_context * lm_ctx = ggml_init(lm_ip);

        ggml_tensor * inp_view = ggml_view_2d(lm_ctx, act_in,
            hidden, n_tokens, act_in->nb[1], 0);
        ggml_set_input(inp_view);

        // RMSNorm
        ggml_tensor * cur = ggml_rms_norm(lm_ctx, inp_view, 1e-6f);
        cur = ggml_mul(lm_ctx, cur, w.out_norm);

        // lm_head projection → logits
        ggml_tensor * logits = ggml_mul_mat(lm_ctx, w.output, cur); // [vocab, n_tokens]
        ggml_set_name(logits, "logits");
        ggml_set_output(logits);

        ggml_cgraph * gf = ggml_new_graph_custom(lm_ctx, 4096, false);
        ggml_build_forward_expand(gf, logits);

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(alloc, gf)) {
            std::fprintf(stderr, "lm_head alloc failed\n");
            return 1;
        }
        auto st = ggml_backend_graph_compute(backend, gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "lm_head compute failed: %d\n", (int)st);
            return 1;
        }

        // D2H logits and argmax
        const int64_t vocab = logits->ne[0];
        std::vector<float> logit_buf(vocab);
        ggml_backend_tensor_get(logits, logit_buf.data(), 0, sizeof(float) * vocab);

        int argmax_id = 0;
        float max_val = logit_buf[0];
        for (int64_t i = 1; i < vocab; i++) {
            if (logit_buf[i] > max_val) {
                max_val = logit_buf[i];
                argmax_id = (int)i;
            }
        }

        std::printf("\nPredicted next token: id=%d  logit=%.4f\n", argmax_id, max_val);
        std::printf("Top-5 logits: ");
        // Simple top-5 via repeated scan
        std::vector<bool> used(vocab, false);
        for (int k = 0; k < 5 && k < (int)vocab; k++) {
            int best = -1;
            float best_v = -1e30f;
            for (int64_t i = 0; i < vocab; i++) {
                if (!used[i] && logit_buf[i] > best_v) {
                    best_v = logit_buf[i];
                    best = (int)i;
                }
            }
            if (best >= 0) {
                used[best] = true;
                std::printf("[%d]=%.3f ", best, best_v);
            }
        }
        std::printf("\n");

        ggml_gallocr_free(alloc);
        ggml_free(lm_ctx);
    }

    // ── Cleanup ──
    ggml_backend_buffer_free(pos_buf);
    ggml_free(pos_ctx);
    ggml_backend_buffer_free(act_buf);
    ggml_free(act_ctx);
    free_target_cache(cache);
    free_moe_state(moe);
    ecache.destroy();
    free_target_weights(w);
    ggml_backend_free(backend);

    std::printf("\nOK\n");
    return 0;
}
