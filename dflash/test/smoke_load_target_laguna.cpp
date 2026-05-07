// Smoke test: load Laguna-XS.2 GGUF via dflash's hand-rolled loader
// (load_target_gguf with arch dispatch to Laguna), validate hparams + tensor
// counts, exit. NO forward yet — forward graph is Phase 2.
//
// Usage: smoke_load_target_laguna <laguna-xs2-Q4_K_M.gguf>

#include "dflash27b.h"
#include "internal.h"
#include "laguna_internal.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#include "ggml-backend.h"
#include "ggml-cuda.h"

using namespace dflash27b;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <laguna-xs2-Q4_K_M.gguf>\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "[smoke-laguna] cuda init failed\n"); return 1; }

    LagunaTargetWeights w;
    if (!load_target_gguf_laguna(path, backend, w)) {
        std::fprintf(stderr, "[smoke-laguna] load_target_gguf_laguna failed: %s\n", dflash27b_last_error());
        ggml_backend_free(backend);
        return 1;
    }

    std::printf("[smoke-laguna] loaded %s\n", path.c_str());
    std::printf("  n_layer=%d  n_embd=%d  n_head_kv=%d  head_dim=%d\n", w.n_layer, w.n_embd, w.n_head_kv, w.head_dim);
    std::printf("  n_ff=%d  n_ff_exp=%d  n_ff_shexp=%d  n_expert=%d  n_expert_used=%d\n",
                w.n_ff, w.n_ff_exp, w.n_ff_shexp, w.n_expert, w.n_expert_used);
    std::printf("  rope_full=%g rope_swa=%g  n_rot_full=%d n_rot_swa=%d  sliding_window=%d\n",
                w.rope_freq_base_full, w.rope_freq_base_swa, w.n_rot_full, w.n_rot_swa, w.sliding_window);
    std::printf("  per-layer heads: [");
    for (int i = 0; i < w.n_layer; ++i) std::printf("%d%s", w.n_head_arr[i], i+1<w.n_layer?",":"");
    std::printf("]\n");
    std::printf("  expert_weights_scale=%g  sigmoid_router=%d  n_layer_dense_lead=%d\n",
                w.expert_weights_scale, (int)w.expert_gating_sigmoid, w.n_layer_dense_lead);
    std::printf("  eos_ids=[%d, %d]  pad=%d\n", w.eos_id, w.eos_chat_id, w.pad_id);

    free_laguna_target_weights(w);
    ggml_backend_free(backend);
    std::printf("[smoke-laguna] OK\n");
    return 0;
}
