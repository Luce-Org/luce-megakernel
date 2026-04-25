// Smoke test: single MoE FFN layer forward pass.
// Loads the 35B-A3B MoE GGUF, picks layer 0, feeds a random vector through
// just the MoE FFN (post-attention-norm → expert routing → shared expert → output),
// and prints the output shape + a spot value.
//
// Usage: smoke_moe_ffn <path/to/qwen35moe.gguf>

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

    if (w.n_expert == 0) {
        std::fprintf(stderr, "FAIL: expected MoE model, got dense (n_expert=0)\n");
        return 1;
    }

    // Build a simple graph: MoE FFN on a single token
    const int n_tokens = 1;
    const int n_embd   = w.n_embd;

    ggml_context * ctx = nullptr;
    {
        struct ggml_init_params params = {};
        params.mem_size = 256 * 1024 * 1024;
        params.no_alloc = true;
        ctx = ggml_init(params);
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);

    // Input: random-ish values (just use 1.0 everywhere)
    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(inp, "inp");

    // Apply post-attention norm (RMS norm)
    const float eps = 1e-6f;
    ggml_tensor * normed = ggml_rms_norm(ctx, inp, eps);
    normed = ggml_mul(ctx, normed, w.layers[0].attn_post_norm);

    // MoE FFN
    ggml_tensor * moe_out = build_moe_ffn(ctx, gf, normed, w.layers[0], w);

    // Residual add
    ggml_tensor * out = ggml_add(ctx, moe_out, inp);
    ggml_set_name(out, "out");
    ggml_set_output(out);

    ggml_build_forward_expand(gf, out);

    // Allocate
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        std::fprintf(stderr, "FAIL: graph alloc failed\n");
        return 1;
    }

    // Set input data
    std::vector<float> inp_data(n_embd * n_tokens, 1.0f);
    ggml_backend_tensor_set(inp, inp_data.data(), 0, sizeof(float) * inp_data.size());

    // Compute
    ggml_backend_graph_compute(backend, gf);

    // Read output
    std::vector<float> out_data(n_embd * n_tokens);
    ggml_backend_tensor_get(out, out_data.data(), 0, sizeof(float) * out_data.size());

    std::printf("MoE FFN output shape: [%lld, %lld]\n",
        (long long)out->ne[0], (long long)out->ne[1]);
    std::printf("First 8 values: ");
    for (int i = 0; i < 8 && i < (int)out_data.size(); i++) {
        std::printf("%.6f ", out_data[i]);
    }
    std::printf("\n");

    // Sanity: output should not be all zeros or NaN/Inf
    bool ok = true;
    bool all_zero = true;
    bool has_nan = false;
    bool has_inf = false;
    for (auto v : out_data) {
        if (v != 0.0f) all_zero = false;
        if (std::isnan(v)) has_nan = true;
        if (std::isinf(v)) has_inf = true;
    }
    if (all_zero) { std::fprintf(stderr, "FAIL: output is all zeros\n"); ok = false; }
    if (has_nan)  { std::fprintf(stderr, "FAIL: output contains NaN\n"); ok = false; }
    if (has_inf)  { std::fprintf(stderr, "FAIL: output contains Inf\n"); ok = false; }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    free_target_weights(w);
    ggml_backend_free(backend);

    if (ok) {
        std::printf("OK\n");
        return 0;
    }
    return 1;
}
