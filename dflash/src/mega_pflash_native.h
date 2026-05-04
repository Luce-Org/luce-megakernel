// Native Qwen3.5-0.8B Mega PFlash daemon boundary.
//
// This header intentionally contains no Torch/Python types. The implementation
// is expected to own a native C++/CUDA Qwen3.5-0.8B loader and scorer.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dflash27b {

struct MegaPFlashContext {
    std::string model_path;
    int max_seq_len = 0;
    std::vector<void *> owned_device_ptrs;
    void * layer_weights_dev = nullptr;
    void * embed_weight = nullptr;
    void * final_norm_weight = nullptr;
    void * lm_head_weight = nullptr;
    void * fused_fa_qkv = nullptr;
    void * fused_gate_up = nullptr;
    void * fa_k_cache = nullptr;
    void * fa_v_cache = nullptr;
    void * fa_q_tail = nullptr;
    void * dn_states = nullptr;
    void * conv_bufs = nullptr;
    void * hidden = nullptr;
    void * residual = nullptr;
    void * normalized = nullptr;
    void * proj_buf = nullptr;
    void * proj_buf2 = nullptr;
    void * attn_buf = nullptr;
    void * mlp_buf = nullptr;
    void * dn_out_buf = nullptr;
    void * beta_buf = nullptr;
    void * alpha_buf = nullptr;
    void * dn_pre_qkv = nullptr;
    void * dn_u_scratch = nullptr;
    void * dn_w_scratch = nullptr;
    void * dn_cs_scratch = nullptr;
    void * final_normed = nullptr;
    void * hidden_bf16_out = nullptr;
    void * lm_bmv = nullptr;
    void * lm_bmi = nullptr;
    void * output_token = nullptr;
    bool loaded = false;
};

bool load_mega_pflash(const std::string & model_path,
                      int max_seq_len,
                      MegaPFlashContext & out);

void free_mega_pflash(MegaPFlashContext & ctx);

std::vector<int32_t> mega_pflash_score_and_compress(
    MegaPFlashContext & ctx,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int chunk_size = 32,
    int n_lookahead = 8,
    int pool_kernel = 13);

} // namespace dflash27b
