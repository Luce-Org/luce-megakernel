// Smoke: load Laguna GGUF, build a 1-token forward graph, run on CUDA, print
// argmax token id. Validates that build_laguna_graph composes cleanly through
// ggml's CUDA backend without crashes / shape mismatches / NaNs.
//
// NOTE: actual logits parity vs HF will be validated against our libllama-based
// llama-simple binary (already verified to match HF for 30+ tokens). This smoke
// just proves the graph executes.
//
// Usage: smoke_laguna_forward <laguna-xs2-Q4_K_M.gguf> [bos_token_id]

#include "laguna_internal.h"
#include "internal.h"
#include "dflash27b.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"

using namespace dflash27b;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <laguna.gguf> [bos_id]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int bos = (argc >= 3) ? std::atoi(argv[2]) : 2;  // Laguna BOS = 2

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    LagunaTargetWeights w;
    if (!load_target_gguf_laguna(path, backend, w)) {
        std::fprintf(stderr, "load failed: %s\n", dflash27b_last_error());
        ggml_backend_free(backend); return 1;
    }

    LagunaTargetCache cache;
    if (!create_laguna_target_cache(w, /*max_ctx=*/4096, backend, cache)) {
        std::fprintf(stderr, "cache failed: %s\n", dflash27b_last_error());
        free_laguna_target_weights(w); ggml_backend_free(backend); return 1;
    }

    // Embed BOS via CpuEmbedder.
    std::vector<float> embed_f32((size_t)w.n_embd);
    int32_t id = bos;
    if (!w.embedder.embed(&id, 1, embed_f32.data())) {
        std::fprintf(stderr, "embed failed for id=%d\n", id);
        free_laguna_target_cache(cache); free_laguna_target_weights(w);
        ggml_backend_free(backend); return 1;
    }

    // Build a per-call ggml_context for graph tensors. Sized generously.
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 4096 + ggml_graph_overhead() + 4 * 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { std::fprintf(stderr, "ggml_init failed\n"); return 1; }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    // Inputs (allocated on backend via gallocr below).
    ggml_tensor * inp_embed = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w.n_embd, 1, 1);
    ggml_set_name(inp_embed, "inp_embed");
    ggml_set_input(inp_embed);

    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    LagunaGraphInputs gi{};
    gi.inp_embed     = inp_embed;
    gi.positions     = positions;
    gi.attn_mask     = nullptr;
    gi.n_tokens      = 1;
    gi.kv_start      = 0;
    gi.output_logits = true;

    LagunaGraphOutputs go = build_laguna_graph(ctx, gf, w, cache, gi);
    if (!go.logits) { std::fprintf(stderr, "no logits\n"); return 1; }

    std::printf("[smoke-fwd] graph nodes=%d leafs=%d\n", ggml_graph_n_nodes(gf), 0);

    // Allocate the graph on the backend.
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "gallocr_alloc_graph failed\n"); return 1;
    }

    // Upload inputs.
    ggml_backend_tensor_set(inp_embed, embed_f32.data(), 0, embed_f32.size() * sizeof(float));
    int32_t pos = 0;
    ggml_backend_tensor_set(positions, &pos, 0, sizeof(pos));

    auto status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "compute failed status=%d\n", (int)status);
        return 1;
    }

    // Pull logits to host + argmax.
    const int64_t vocab = go.logits->ne[0];
    std::vector<float> logits((size_t)vocab);
    ggml_backend_tensor_get(go.logits, logits.data(), 0, logits.size() * sizeof(float));

    int best = 0;
    float bv = logits[0];
    int n_inf = 0, n_nan = 0;
    for (int i = 0; i < (int)vocab; ++i) {
        float v = logits[i];
        if (std::isnan(v)) ++n_nan;
        if (std::isinf(v)) ++n_inf;
        if (v > bv) { bv = v; best = i; }
    }
    std::printf("[smoke-fwd] vocab=%lld  argmax=%d  logit=%.4f  nan=%d inf=%d\n",
                (long long)vocab, best, bv, n_nan, n_inf);

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    free_laguna_target_cache(cache);
    free_laguna_target_weights(w);
    ggml_backend_free(backend);
    std::printf("[smoke-fwd] OK\n");
    return 0;
}
