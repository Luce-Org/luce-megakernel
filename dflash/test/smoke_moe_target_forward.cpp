// Smoke test: full MoE target forward pass.
// Loads Qwen3.6-35B-A3B, creates cache, runs 1-token decode through all 40 layers,
// validates logits are sane.
//
// Usage: smoke_moe_target_forward <path/to/qwen35moe.gguf>

#include "dflash27b.h"
#include "internal.h"

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

using namespace dflash27b;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <qwen35moe.gguf>\n", argv[0]);
        return 2;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    TargetWeights w;
    if (!load_target_gguf(argv[1], backend, w)) {
        std::fprintf(stderr, "load_target_gguf: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("[target] %s\n", dflash27b_last_error());

    if (w.n_expert == 0) {
        std::fprintf(stderr, "FAIL: expected MoE model\n");
        return 1;
    }

    // Create cache
    TargetCache cache;
    const int max_ctx = 512;
    if (!create_target_cache(w, max_ctx, 0, backend, cache)) {
        std::fprintf(stderr, "FAIL: create_target_cache: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("[cache] attn_k=%zu attn_v=%zu ssm=%zu conv=%zu\n",
        cache.attn_k.size(), cache.attn_v.size(),
        cache.ssm_state.size(), cache.conv_state.size());

    // Validate cache dimensions
    bool ok = true;
    // 10 full-attn layers (indices 3,7,11,15,19,23,27,31,35,39)
    if (cache.attn_k.size() != 10) {
        std::fprintf(stderr, "FAIL: expected 10 attn_k, got %zu\n", cache.attn_k.size());
        ok = false;
    }
    // 30 delta-net layers
    if (cache.ssm_state.size() != 30) {
        std::fprintf(stderr, "FAIL: expected 30 ssm_state, got %zu\n", cache.ssm_state.size());
        ok = false;
    }

    // Embed a single token
    const int n_tokens = 1;
    const int token_id = 1; // arbitrary
    std::vector<float> embed_data(w.n_embd * n_tokens);
    if (!w.embedder.embed(&token_id, n_tokens, embed_data.data())) {
        std::fprintf(stderr, "FAIL: embedder.embed failed\n");
        return 1;
    }

    // Build graph
    ggml_context * ctx = nullptr;
    {
        struct ggml_init_params params = {};
        params.mem_size = 512 * 1024 * 1024;
        params.no_alloc = true;
        ctx = ggml_init(params);
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    ggml_tensor * inp_embed = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w.n_embd, n_tokens, 1);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_name(inp_embed, "inp_embed");
    ggml_set_name(positions, "positions");
    ggml_set_input(inp_embed);
    ggml_set_input(positions);

    QwenGraphInputs in;
    in.inp_embed  = inp_embed;
    in.positions  = positions;
    in.n_tokens   = n_tokens;
    in.kv_start   = 0;
    in.attn_mask  = nullptr; // single token, no mask needed
    in.capture_layers = false;
    in.capture_delta_intermediate = false;

    QwenGraphOutputs og = build_qwen35_graph(ctx, gf, w, cache, in);
    ggml_set_output(og.logits);

    ggml_build_forward_expand(gf, og.logits);
    std::printf("[graph] nodes=%d\n", ggml_graph_n_nodes(gf));

    // Allocate + set input
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        std::fprintf(stderr, "FAIL: graph alloc failed\n");
        return 1;
    }
    ggml_backend_tensor_set(inp_embed, embed_data.data(), 0, sizeof(float) * embed_data.size());
    int32_t pos4[4] = { 0, 0, 0, 0 };
    ggml_backend_tensor_set(positions, pos4, 0, sizeof(int32_t) * 4);

    // Compute
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: graph compute failed\n");
        return 1;
    }

    // Read logits
    std::printf("[logits_tensor] ne=[%lld,%lld,%lld]\n",
        (long long)og.logits->ne[0], (long long)og.logits->ne[1], (long long)og.logits->ne[2]);
    const int64_t vocab = og.logits->ne[0];
    const int64_t logits_count = vocab * og.logits->ne[1];
    std::vector<float> logits(logits_count);
    ggml_backend_tensor_get(og.logits, logits.data(), 0, sizeof(float) * logits.size());

    int n_nan = 0, n_inf = 0;
    float min_l = INFINITY, max_l = -INFINITY;
    for (int64_t i = 0; i < logits_count; i++) {
        auto v = logits[i];
        if (std::isnan(v)) n_nan++;
        if (std::isinf(v)) n_inf++;
        if (v < min_l) min_l = v;
        if (v > max_l) max_l = v;
    }
    std::printf("[logits] vocab=%lld nan=%d inf=%d min=%.2f max=%.2f\n",
        (long long)vocab, n_nan, n_inf, min_l, max_l);

    if (n_nan > 0) { std::fprintf(stderr, "FAIL: logits contain NaN\n"); ok = false; }
    if (n_inf > 0) { std::fprintf(stderr, "FAIL: logits contain Inf\n"); ok = false; }
    if (max_l - min_l < 1.0f) { std::fprintf(stderr, "FAIL: logits range too small, possible constant output\n"); ok = false; }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    free_target_cache(cache);
    free_target_weights(w);
    ggml_backend_free(backend);

    if (ok) {
        std::printf("OK\n");
        return 0;
    }
    return 1;
}
