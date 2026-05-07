// Test multi-token batch MoE forward (simulating DFlash verify phase).
//
// Runs a 4-token sequence through all 40 layers with a causal attention mask.
// This validates that the two-graph-per-layer architecture handles batched
// tokens correctly (critical for the N=16 verify step in DFlash).
//
// Usage: test_moe_batch <path/to/qwen35moe.gguf> [n_cache_slots]

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

// F16 constants for attention mask
static const uint16_t F16_NEG_INF = 0xFC00;  // -inf in f16
static const uint16_t F16_ZERO    = 0x0000;

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
        std::fprintf(stderr, "error: model is not MoE\n");
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }

    const int hidden    = w.n_embd;
    const int n_layer   = w.n_layer;
    const int n_tokens  = 4;  // batch of 4 tokens
    const int kv_start  = 0;

    std::printf("Batch MoE test: hidden=%d n_layer=%d n_tokens=%d\n",
        hidden, n_layer, n_tokens);

    // ── 3. ExpertCache ──
    int n_alt_layers = 0;
    const auto & es = w.expert_source;
    for (int l = 0; l < es.n_layers; l++) {
        if (!es.layer_down_types.empty() && es.layer_down_types[l] != es.down_type)
            n_alt_layers++;
    }
    int n_alt_slots = n_alt_layers > 0 ? (n_cache_slots * n_alt_layers / es.n_layers) + 1 : 0;
    if (n_alt_slots > 0 && n_alt_slots < n_alt_layers * 9 + 1)
        n_alt_slots = n_alt_layers * 9 + 1;

    ExpertCache ecache;
    if (!ecache.init(backend, w.expert_source, n_cache_slots, n_alt_slots)) {
        std::fprintf(stderr, "ExpertCache init failed\n");
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }
    ecache.warmup(w.expert_source);

    // ── 4. MoeState (sized for batch) ──
    MoeState moe{};
    if (!create_moe_state(backend, hidden, w.n_experts_active, n_tokens, moe)) {
        std::fprintf(stderr, "create_moe_state failed\n");
        ecache.destroy();
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }

    // ── 5. TargetCache ──
    TargetCache cache;
    if (!create_target_cache(w, /*max_ctx=*/64, /*max_verify_tokens=*/n_tokens, backend, cache,
                             /*prefill_only=*/false)) {
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
    ggml_tensor * act_a = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, n_tokens);
    ggml_tensor * act_b = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, n_tokens);
    ggml_set_name(act_a, "act_a");
    ggml_set_name(act_b, "act_b");
    ggml_backend_buffer_t act_buf = ggml_backend_alloc_ctx_tensors(act_ctx, backend);
    if (!act_buf) {
        std::fprintf(stderr, "activation alloc failed\n");
        return 1;
    }

    // ── 7. Embed 4 tokens: [0, 1, 2, 3] ──
    std::vector<float> emb(hidden * n_tokens);
    int32_t tok_ids[4] = {0, 1, 2, 3};
    for (int i = 0; i < n_tokens; i++) {
        if (!w.embedder.embed(&tok_ids[i], 1, emb.data() + i * hidden)) {
            std::fprintf(stderr, "embed failed for token %d\n", i);
            return 1;
        }
    }
    ggml_backend_tensor_set(act_a, emb.data(), 0, sizeof(float) * hidden * n_tokens);

    // ── 8. Positions: M-RoPE format [4*n_tokens] = [pos0,pos1,..., pos0,pos1,..., pos0,pos1,..., 0,0,...] ──
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
    // M-RoPE: 3 spatial dims each get the position, 4th (temporal) is 0
    std::vector<int32_t> pos4(4 * n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        pos4[0 * n_tokens + i] = kv_start + i;
        pos4[1 * n_tokens + i] = kv_start + i;
        pos4[2 * n_tokens + i] = kv_start + i;
        pos4[3 * n_tokens + i] = 0;
    }
    ggml_backend_tensor_set(positions, pos4.data(), 0, sizeof(int32_t) * 4 * n_tokens);

    // ── 9. Attention mask (causal) ──
    // For full-attention layers: kv_pad × q_pad causal mask
    // kv_len = kv_start + n_tokens = 4, padded to 32 (KQ_MASK_PAD)
    const int kv_len = kv_start + n_tokens;
    const int kv_pad = ((kv_len + 31) / 32) * 32;  // align to 32
    const int q_pad  = ((n_tokens + 31) / 32) * 32;

    ggml_init_params mask_ip{};
    mask_ip.mem_size   = 4 * ggml_tensor_overhead() + 4096;
    mask_ip.mem_buffer = nullptr;
    mask_ip.no_alloc   = true;
    ggml_context * mask_ctx = ggml_init(mask_ip);
    ggml_tensor * attn_mask = ggml_new_tensor_2d(mask_ctx, GGML_TYPE_F16, kv_pad, q_pad);
    ggml_set_name(attn_mask, "attn_mask");
    ggml_set_input(attn_mask);
    ggml_backend_buffer_t mask_buf = ggml_backend_alloc_ctx_tensors(mask_ctx, backend);
    if (!mask_buf) {
        std::fprintf(stderr, "mask alloc failed\n");
        return 1;
    }
    // Fill causal mask: position q can attend to positions [0, kv_start+q]
    std::vector<uint16_t> mask_data((size_t)kv_pad * q_pad, F16_NEG_INF);
    for (int q = 0; q < n_tokens; q++) {
        for (int k = 0; k <= kv_start + q; k++) {
            mask_data[(size_t)q * kv_pad + k] = F16_ZERO;
        }
    }
    ggml_backend_tensor_set(attn_mask, mask_data.data(), 0, sizeof(uint16_t) * mask_data.size());

    // ── 10. Logits + argmax output tensors ──
    const int64_t vocab = w.embedder.n_vocab;
    ggml_init_params out_ip{};
    out_ip.mem_size   = 8 * ggml_tensor_overhead() + 4096;
    out_ip.mem_buffer = nullptr;
    out_ip.no_alloc   = true;
    ggml_context * out_ctx = ggml_init(out_ip);
    ggml_tensor * logits_out = ggml_new_tensor_2d(out_ctx, GGML_TYPE_F32, vocab, n_tokens);
    ggml_tensor * argmax_out = ggml_new_tensor_1d(out_ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(logits_out, "logits_out");
    ggml_set_name(argmax_out, "argmax_out");
    ggml_backend_buffer_t out_buf = ggml_backend_alloc_ctx_tensors(out_ctx, backend);
    if (!out_buf) {
        std::fprintf(stderr, "output alloc failed\n");
        return 1;
    }

    // ── 11. Run full forward via run_qwen35moe_forward ──
    std::printf("\n─── Batch forward (%d tokens) ───\n", n_tokens);
    auto t0 = std::chrono::high_resolution_clock::now();

    bool ok = run_qwen35moe_forward(
        backend, w, cache, ecache, moe, w.expert_source,
        act_a, act_b, n_tokens, positions, attn_mask,
        kv_start, /*capture=*/false, /*fa_window=*/0,
        logits_out, argmax_out);

    auto t1 = std::chrono::high_resolution_clock::now();
    double fwd_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!ok) {
        std::fprintf(stderr, "run_qwen35moe_forward FAILED\n");
        return 1;
    }

    std::printf("Forward pass: %.1f ms (%.1f ms/token)\n", fwd_ms, fwd_ms / n_tokens);
    std::printf("Expert cache: %" PRId64 " hits, %" PRId64 " misses (%.1f%% hit rate)\n",
        ecache.hits(), ecache.misses(),
        ecache.hits() + ecache.misses() > 0
            ? 100.0 * ecache.hits() / (ecache.hits() + ecache.misses()) : 0.0);

    // ── 12. Read argmax results ──
    std::vector<int32_t> argmax_ids(n_tokens);
    ggml_backend_tensor_get(argmax_out, argmax_ids.data(), 0, sizeof(int32_t) * n_tokens);

    std::printf("\nPredicted tokens: ");
    for (int i = 0; i < n_tokens; i++) {
        std::printf("[pos %d → id=%d] ", i, argmax_ids[i]);
    }
    std::printf("\n");

    // Read top logits for last token
    std::vector<float> logit_buf(vocab);
    ggml_backend_tensor_get(logits_out, logit_buf.data(),
        (size_t)(n_tokens - 1) * vocab * sizeof(float), sizeof(float) * vocab);

    std::printf("Last token top-5: ");
    std::vector<bool> used(vocab, false);
    for (int k = 0; k < 5; k++) {
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

    // Sanity check: argmax should be valid token IDs
    bool all_valid = true;
    for (int i = 0; i < n_tokens; i++) {
        if (argmax_ids[i] < 0 || argmax_ids[i] >= (int)vocab) {
            std::fprintf(stderr, "ERROR: invalid argmax[%d]=%d\n", i, argmax_ids[i]);
            all_valid = false;
        }
    }

    // ── Cleanup ──
    ggml_backend_buffer_free(out_buf);
    ggml_free(out_ctx);
    ggml_backend_buffer_free(mask_buf);
    ggml_free(mask_ctx);
    ggml_backend_buffer_free(pos_buf);
    ggml_free(pos_ctx);
    ggml_backend_buffer_free(act_buf);
    ggml_free(act_ctx);
    free_target_cache(cache);
    free_moe_state(moe);
    ecache.destroy();
    free_target_weights(w);
    ggml_backend_free(backend);

    if (!all_valid) {
        std::printf("\nFAIL: invalid argmax values\n");
        return 1;
    }

    std::printf("\nPASS\n");
    return 0;
}
