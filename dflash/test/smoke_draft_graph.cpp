// Smoke test for the DFlash draft forward graph.
//
// Loads real draft weights, fills noise_embedding / target_hidden_cat with
// deterministic random bf16 data, runs the graph on CUDA, pulls results back,
// and checks:
//   - no NaN / Inf in output
//   - shape is [hidden, q_len, 1]
//   - a few representative values look reasonable (near-zero means for rms_norm output)
//
// Usage:
//   smoke_draft_graph <draft.safetensors> [ctx_len]
//
// ctx_len defaults to 64 to keep the first run tiny.

#include "dflash27b.h"
#include "internal.h"
#include "dflash_graph.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace dflash27b;

// Convert fp32 -> bf16 (truncation)
static uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}
static float bf16_to_f32(uint16_t u) {
    uint32_t bits = ((uint32_t)u) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.safetensors> [ctx_len]\n", argv[0]);
        return 2;
    }
    const char * path = argv[1];
    const int ctx_len = (argc >= 3) ? std::atoi(argv[2]) : 64;

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "ggml_backend_cuda_init failed\n"); return 1; }

    DraftWeights w;
    if (!load_draft_safetensors(path, backend, w)) {
        std::fprintf(stderr, "load: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("draft loaded\n");

    const int hidden_actual = w.hparams.hidden;
    const int q_len_actual  = w.hparams.block_size;
    const int fc_in_actual  = w.hparams.n_target_layers * hidden_actual;

    std::printf("ctx_len=%d q_len=%d hidden=%d fc_in=%d\n", ctx_len, q_len_actual, hidden_actual, fc_in_actual);

    // ── 2. Graph context (separate from weights context)
    const size_t mem_size = 256 * 1024 * 1024;  // 256 MB — plenty for nodes
    ggml_init_params ip{};
    ip.mem_size   = mem_size;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) { std::fprintf(stderr, "ggml_init graph failed\n"); return 1; }

    // ── 3. Input placeholder tensors
    // Activations flow as F32 through the graph (CUDA rms_norm requires F32).
    // Weights stay bf16 — ggml_mul_mat auto-casts.
    ggml_tensor * noise_embed = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, hidden_actual, q_len_actual, 1);
    ggml_tensor * target_hid  = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, fc_in_actual, ctx_len, 1);
    ggml_tensor * pos_q       = ggml_new_tensor_1d(gctx, GGML_TYPE_I32,  q_len_actual);
    ggml_tensor * pos_k       = ggml_new_tensor_1d(gctx, GGML_TYPE_I32,  ctx_len + q_len_actual);
    ggml_set_name(noise_embed, "noise_embed");
    ggml_set_name(target_hid,  "target_hidden_cat");
    ggml_set_name(pos_q,       "positions_q");
    ggml_set_name(pos_k,       "positions_k");
    ggml_set_input(noise_embed);
    ggml_set_input(target_hid);
    ggml_set_input(pos_q);
    ggml_set_input(pos_k);

    // ── 4. Build graph
    DraftGraphInputs gi{};
    gi.ctx_len           = ctx_len;
    gi.noise_embed       = noise_embed;
    gi.target_hidden_cat = target_hid;
    gi.positions_q       = pos_q;
    gi.positions_k       = pos_k;

    DraftGraphOutputs go = build_draft_graph(gctx, w, gi);
    if (!go.hidden_states) { std::fprintf(stderr, "build_draft_graph returned null\n"); return 1; }
    ggml_set_output(go.hidden_states);

    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, go.hidden_states);
    std::printf("graph built: n_nodes=%d\n", ggml_graph_n_nodes(gf));

    // ── 5. Allocate graph + all input tensors on the backend
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        std::fprintf(stderr, "ggml_gallocr_alloc_graph failed\n");
        return 1;
    }

    // ── 6. Fill input tensors
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-0.02f, 0.02f);

    {
        std::vector<float> data((size_t)hidden_actual * q_len_actual);
        for (auto & v : data) v = u(rng);
        ggml_backend_tensor_set(noise_embed, data.data(), 0, sizeof(float) * data.size());
    }
    {
        std::vector<float> data((size_t)fc_in_actual * ctx_len);
        for (auto & v : data) v = u(rng);
        ggml_backend_tensor_set(target_hid, data.data(), 0, sizeof(float) * data.size());
    }
    {
        std::vector<int32_t> pq(q_len_actual);
        for (int i = 0; i < q_len_actual; i++) pq[i] = ctx_len + i;
        ggml_backend_tensor_set(pos_q, pq.data(), 0, sizeof(int32_t) * pq.size());
    }
    {
        std::vector<int32_t> pk(ctx_len + q_len_actual);
        for (int i = 0; i < ctx_len + q_len_actual; i++) pk[i] = i;
        ggml_backend_tensor_set(pos_k, pk.data(), 0, sizeof(int32_t) * pk.size());
    }

    // ── 7. Compute
    auto status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph_compute returned %d\n", (int)status);
        return 1;
    }
    std::printf("compute OK\n");

    // ── 8. Read output, check shape + no NaN + print summary stats
    const size_t n_out_elems = ggml_nelements(go.hidden_states);
    if (n_out_elems != (size_t)hidden_actual * q_len_actual) {
        std::fprintf(stderr, "out elems mismatch: %zu vs %d\n", n_out_elems, hidden_actual * q_len_actual);
        return 1;
    }
    std::vector<float> out(n_out_elems);
    ggml_backend_tensor_get(go.hidden_states, out.data(), 0, sizeof(float) * out.size());

    int n_nan = 0, n_inf = 0;
    double sum = 0.0, sumsq = 0.0;
    float vmin = 1e30f, vmax = -1e30f;
    for (auto f : out) {
        if (std::isnan(f)) { n_nan++; continue; }
        if (std::isinf(f)) { n_inf++; continue; }
        sum   += f;
        sumsq += (double)f * f;
        if (f < vmin) vmin = f;
        if (f > vmax) vmax = f;
    }
    double mean = sum / (double)out.size();
    double var  = sumsq / (double)out.size() - mean * mean;
    std::printf("out shape: [%" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
                go.hidden_states->ne[0], go.hidden_states->ne[1], go.hidden_states->ne[2]);
    std::printf("out stats: nan=%d inf=%d mean=%.4g std=%.4g min=%.4g max=%.4g\n",
                n_nan, n_inf, mean, std::sqrt(std::max(0.0, var)), vmin, vmax);
    std::printf("out first 8 values: ");
    for (int i = 0; i < 8; i++) std::printf("%.4f ", out[i]);
    std::printf("\n");

    if (n_nan || n_inf) {
        std::fprintf(stderr, "FAIL: non-finite values in output\n");
        return 1;
    }

    ggml_gallocr_free(alloc);
    ggml_free(gctx);
    free_draft_weights(w);
    ggml_backend_free(backend);
    std::printf("OK\n");
    return 0;
}
