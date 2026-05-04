// In-process Qwen3-0.6B drafter for pflash speculative prefill.
//
// Hosted in the SAME process / SAME ggml allocator as the dflash target, so
// we never pay the cross-process VRAM contention that broke the Python
// subprocess integration. Drafter uses our custom Qwen3-0.6B forward
// (qwen3_0p6b_graph.cpp + qwen3_0p6b_loader.cpp) which calls our FlashPrefill
// CUDA kernels for the attention compute, replacing libllama. This removes
// the dense O(S²) FA cost that made libllama 3+ minutes at 140K.
//
// Public entry point: drafter_score_and_compress() takes raw input token IDs,
// runs the full pflash compression pipeline in C++, returns the surviving
// token IDs (drafter vocab).

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "qwen3_0p6b_drafter.h"

struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;

namespace dflash27b {

enum class DrafterArch {
    Qwen3_0p6b,
    Qwen35_0p8b,
};

bool parse_drafter_arch(const std::string & name, DrafterArch & out);
const char * drafter_arch_name(DrafterArch arch);

struct DrafterContext {
    ggml_backend_t        backend = nullptr;   // owned (created in load_drafter)
    Qwen3DrafterWeights   weights;             // BF16 weights on the backend
    DrafterArch           arch    = DrafterArch::Qwen3_0p6b;
    bool                  loaded  = false;
};

// Load the drafter GGUF (e.g. /opt/lucebox/models/drafter/Qwen3-0.6B-BF16.gguf).
// Creates a fresh CUDA backend if `backend` is null. Otherwise uses the
// caller-provided backend (so the drafter shares the daemon's allocator).
//
// `gpu_layers` is accepted for API compat but ignored — every layer goes on
// the GPU since the drafter weights are only ~1.5 GB.
bool load_drafter(const std::string & gguf_path, int gpu_layers,
                  DrafterContext & out);
bool load_drafter(const std::string & gguf_path, int gpu_layers,
                  DrafterArch arch, DrafterContext & out);

void free_drafter(DrafterContext & ctx);

// Score importance per token via Liu Q-hook tail attention, then chunk-top-K
// span merge. Returns surviving token IDs (drafter vocab).
//
//   ids          input token IDs of length S
//   keep_ratio   fraction of `chunk_size`-token chunks to keep
//   chunk_size   span granularity (default 32)
//   n_lookahead  trailing Q tokens used for tail attention (default 8)
//   pool_kernel  AvgPool kernel for score smoothing (default 13)
//
// On failure returns empty vector + sets last_error.
std::vector<int32_t> drafter_score_and_compress(
    DrafterContext & ctx,
    const std::vector<int32_t> & ids,
    float  keep_ratio,
    int    chunk_size  = 32,
    int    n_lookahead = 8,
    int    pool_kernel = 13);

} // namespace dflash27b
