// Custom Qwen3-0.6B drafter forward, in dflash, replacing libllama.
//
// Uses our FlashPrefill CUDA kernels for the attention compute. Single
// process, single CUDA context, single ggml allocator — no Python, no
// Triton, no subprocess.
//
// Public API:
//   bool load_qwen3_0p6b_drafter(path, backend, out)  → load GGUF weights
//   bool forward_qwen3_0p6b_drafter(weights, ids, out_q_capture, out_k_capture)
//   void free_qwen3_0p6b_drafter(weights)
//
// Status (2026-04-29 session): scaffolding written; full graph + integration
// is multi-hour work, in progress.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

namespace dflash27b {

struct Qwen3DrafterLayer {
    ggml_tensor * attn_norm   = nullptr;  // [hidden]
    ggml_tensor * wq          = nullptr;  // [hidden, q_dim] = [1024, 2048]
    ggml_tensor * wk          = nullptr;  // [hidden, kv_dim] = [1024, 1024]
    ggml_tensor * wv          = nullptr;  // [hidden, kv_dim]
    ggml_tensor * wo          = nullptr;  // [q_dim, hidden] = [2048, 1024]
    ggml_tensor * q_norm      = nullptr;  // [head_dim] = [128]
    ggml_tensor * k_norm      = nullptr;  // [head_dim]
    ggml_tensor * ffn_norm    = nullptr;  // [hidden]
    ggml_tensor * ffn_gate    = nullptr;  // [hidden, ffn]
    ggml_tensor * ffn_up      = nullptr;  // [hidden, ffn]
    ggml_tensor * ffn_down    = nullptr;  // [ffn, hidden]
};

struct Qwen3DrafterWeights {
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf     = nullptr;

    ggml_tensor * tok_embd    = nullptr;  // [hidden, vocab]
    ggml_tensor * out_norm    = nullptr;  // [hidden]
    ggml_tensor * output      = nullptr;  // [hidden, vocab] (lm_head)

    std::vector<Qwen3DrafterLayer> layers;  // size = n_layer = 28

    // Projection/embedding compute dtype. SM75 should use F16; BF16 is only
    // kept for native-BF16 GPUs.
    ggml_type compute_type = GGML_TYPE_BF16;

    // Architecture metadata.
    int n_layer    = 28;
    int n_head     = 16;
    int n_head_kv  = 8;
    int n_embd     = 1024;
    int n_ff       = 3072;
    int head_dim   = 128;
    int n_vocab    = 151936;
    int n_ctx_max  = 40960;
    float rope_theta = 1000000.0f;
};

bool load_qwen3_0p6b_drafter(const std::string & gguf_path,
                              ggml_backend_t backend,
                              Qwen3DrafterWeights & out);

void free_qwen3_0p6b_drafter(Qwen3DrafterWeights & w);

// Custom Qwen3-0.6B forward, fused with Liu Q-hook tail attention scoring.
//
// Inputs:
//   w           — loaded weights (must be on a CUDA backend)
//   ids         — input token IDs of length S (drafter vocab)
//   n_lookahead — number of trailing query tokens for tail attention (=8)
//
// Outputs:
//   running_max — flat [n_lookahead, S] f32, max-over-heads-and-layers of
//                 softmax(Q_tail @ K^T / sqrt(D)) per (lookahead, key) pair.
//                 Caller does AvgPool + chunk-top-K + span merge.
//
// Returns true on success. On failure sets last_error and returns false.
bool forward_qwen3_0p6b_drafter(
    const Qwen3DrafterWeights & w,
    const std::vector<int32_t> & ids,
    int n_lookahead,
    std::vector<float> & running_max);

} // namespace dflash27b
