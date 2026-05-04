/**
 * PyTorch bindings for Qwen3.5-0.8B bf16 megakernel — decode.
 *
 * Blackwell-only bindings (NVFP4 decode, bf16 prefill megakernel, prefill
 * megakernel NVFP4) are gated behind MEGAKERNEL_HAS_NVFP4, which is only
 * defined by setup.py for sm_12+ builds. On sm_86 the symbol set is
 * identical to the original upstream build.
 */

#include <Python.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/library.h>

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)
#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

#define REGISTER_EXTENSION(NAME)                                               \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                     \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                 \
                                        STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                           \
  }

struct LayerWeights {
    int layer_type;
    int _pad[3];
    void *ptrs[14];  // max(11 FA, 14 DN) pointers — all bf16, no scales
};

#ifdef MEGAKERNEL_HAS_NVFP4
struct LayerWeightsNVFP4 {
    int layer_type;
    int group_size;
    int _pad[2];
    void *ptrs[24];  // hot decode weights become packed fp4 + per-group scales
};

// Layout-compatible with PFFusedLayerWeights in prefill_bw.cu — used by the
// hybrid bf16 prefill + NVFP4 LM head path (launch_prefill_bf16_nvfp4_lm).
struct PrefillFusedLayerWeights {
    void *proj_weight;
    void *gate_up_weight;
    void *proj_weight_packed;
    void *proj_weight_scales;
    void *gate_up_weight_packed;
    void *gate_up_weight_scales;
};
#endif

extern "C" void launch_decode(
    int input_token_id, int *output_token_id,
    const void *embed_weight, const LayerWeights *layer_weights,
    const void *final_norm_weight, const void *lm_head_weight,
    void *fa_k_cache, void *fa_v_cache,
    void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_activations, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out,
    void *g_mlp_inter, void *g_z_scratch, void *g_beta_scratch,
    void *g_alpha_scratch, void *g_normalized,
    unsigned int *barrier_counter, unsigned int *barrier_generation,
    float *block_max_vals, int *block_max_idxs,
    unsigned int *lm_sync_counter,
    float *seen_token_mask,
    float repetition_penalty,
    int position, int max_seq_len, cudaStream_t stream);

#ifdef MEGAKERNEL_HAS_NVFP4
extern "C" void launch_decode_nvfp4(
    const int *input_token_ptr, int *output_token_id,
    const void *embed_weight, const LayerWeightsNVFP4 *layer_weights,
    const void *final_norm_weight,
    const void *lm_head_weight_packed, const void *lm_head_scales,
    void *lm_hidden_bf16, void *lm_hidden_packed, void *lm_hidden_scales, void *lm_logits_f16,
    void *fa_k_cache, void *fa_v_cache,
    void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_activations, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out,
    void *g_mlp_inter, void *g_z_scratch, void *g_beta_scratch,
    void *g_alpha_scratch, void *g_normalized,
    unsigned int *barrier_counter, unsigned int *barrier_generation,
    float *block_max_vals, int *block_max_idxs,
    unsigned int *lm_sync_counter,
    int position, int max_seq_len, int group_size, cudaStream_t stream);

extern "C" void launch_decode_many_nvfp4(
    int *token_buffer, int *output_tokens, int steps,
    const void *embed_weight, const LayerWeightsNVFP4 *layer_weights,
    const void *final_norm_weight,
    const void *lm_head_weight_packed, const void *lm_head_scales,
    void *lm_hidden_bf16, void *lm_hidden_packed, void *lm_hidden_scales, void *lm_logits_f16,
    void *fa_k_cache, void *fa_v_cache,
    void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_activations, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out,
    void *g_mlp_inter, void *g_z_scratch, void *g_beta_scratch,
    void *g_alpha_scratch, void *g_normalized,
    unsigned int *barrier_counter, unsigned int *barrier_generation,
    float *block_max_vals, int *block_max_idxs,
    unsigned int *lm_sync_counter,
    int position, int max_seq_len, int group_size, cudaStream_t stream);

extern "C" void launch_quantize_nvfp4_out(
    const void *weight, int rows, int cols, int group_size,
    void *packed_out, void *scales_out, cudaStream_t stream);

extern "C" void launch_quantize_nvfp4_lm_out(
    const void *weight, int rows, int cols,
    void *packed_out, void *scales_out, cudaStream_t stream);

extern "C" void launch_prefill_megakernel_nvfp4(
    const int *token_ids, int seq_len, int *output_token_id,
    const void *embed_weight, const LayerWeightsNVFP4 *layer_weights,
    const void *final_norm_weight,
    const void *lm_head_weight_packed, const void *lm_head_scales,
    void *lm_hidden_bf16, void *lm_hidden_packed, void *lm_hidden_scales, void *lm_logits_f16,
    void *fa_k_cache, void *fa_v_cache,
    void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_activations, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out,
    void *g_mlp_inter, void *g_z_scratch, void *g_beta_scratch,
    void *g_alpha_scratch, void *g_normalized,
    unsigned int *barrier_counter, unsigned int *barrier_generation,
    float *block_max_vals, int *block_max_idxs,
    unsigned int *lm_sync_counter,
    int max_seq_len, int group_size, cudaStream_t stream);

static void seed_token_buffer(torch::Tensor token_buffer, int token_id) {
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = cudaMemcpyAsync(
        token_buffer.data_ptr(),
        &token_id,
        sizeof(token_id),
        cudaMemcpyHostToDevice,
        stream);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync(token_buffer) failed: ", cudaGetErrorString(err));
}
#endif  // MEGAKERNEL_HAS_NVFP4
extern "C" void set_decode_blocks_override(int blocks);
extern "C" int query_max_safe_decode_blocks();

void decode(
    torch::Tensor output_token, int64_t input_token_id,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden_buffer, torch::Tensor activations, torch::Tensor residual,
    torch::Tensor qkv_scratch, torch::Tensor kv_scratch, torch::Tensor attn_out,
    torch::Tensor mlp_inter, torch::Tensor z_scratch, torch::Tensor beta_scratch,
    torch::Tensor alpha_scratch, torch::Tensor normalized,
    torch::Tensor barrier_counter, torch::Tensor barrier_generation,
    torch::Tensor block_max_vals, torch::Tensor block_max_idxs,
    torch::Tensor lm_sync_counter, torch::Tensor seen_token_mask,
    double repetition_penalty, int64_t position, int64_t max_seq_len)
{
    launch_decode(
        (int)input_token_id, (int*)output_token.data_ptr(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeights*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden_buffer.data_ptr(), activations.data_ptr(), residual.data_ptr(),
        qkv_scratch.data_ptr(), kv_scratch.data_ptr(), attn_out.data_ptr(),
        mlp_inter.data_ptr(), z_scratch.data_ptr(), beta_scratch.data_ptr(),
        alpha_scratch.data_ptr(), normalized.data_ptr(),
        (unsigned int*)barrier_counter.data_ptr(), (unsigned int*)barrier_generation.data_ptr(),
        (float*)block_max_vals.data_ptr(), (int*)block_max_idxs.data_ptr(),
        (unsigned int*)lm_sync_counter.data_ptr(),
        (float*)seen_token_mask.data_ptr(), (float)repetition_penalty,
        (int)position, (int)max_seq_len,
        c10::cuda::getCurrentCUDAStream().stream());
}

#ifdef MEGAKERNEL_HAS_NVFP4
void decode_nvfp4(
    torch::Tensor output_token, int64_t input_token_id,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight,
    torch::Tensor lm_head_weight_packed, torch::Tensor lm_head_scales,
    torch::Tensor lm_hidden_bf16, torch::Tensor lm_hidden_packed, torch::Tensor lm_hidden_scales, torch::Tensor lm_logits_f16,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden_buffer, torch::Tensor activations, torch::Tensor residual,
    torch::Tensor qkv_scratch, torch::Tensor kv_scratch, torch::Tensor attn_out,
    torch::Tensor mlp_inter, torch::Tensor z_scratch, torch::Tensor beta_scratch,
    torch::Tensor alpha_scratch, torch::Tensor normalized,
    torch::Tensor barrier_counter, torch::Tensor barrier_generation,
    torch::Tensor block_max_vals, torch::Tensor block_max_idxs,
    torch::Tensor lm_sync_counter, int64_t position, int64_t max_seq_len,
    int64_t group_size)
{
    seed_token_buffer(output_token, (int)input_token_id);
    launch_decode_nvfp4(
        (const int*)output_token.data_ptr(),
        (int*)output_token.data_ptr(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeightsNVFP4*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(),
        lm_head_weight_packed.data_ptr(), lm_head_scales.data_ptr(),
        lm_hidden_bf16.data_ptr(), lm_hidden_packed.data_ptr(), lm_hidden_scales.data_ptr(), lm_logits_f16.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden_buffer.data_ptr(), activations.data_ptr(), residual.data_ptr(),
        qkv_scratch.data_ptr(), kv_scratch.data_ptr(), attn_out.data_ptr(),
        mlp_inter.data_ptr(), z_scratch.data_ptr(), beta_scratch.data_ptr(),
        alpha_scratch.data_ptr(), normalized.data_ptr(),
        (unsigned int*)barrier_counter.data_ptr(), (unsigned int*)barrier_generation.data_ptr(),
        (float*)block_max_vals.data_ptr(), (int*)block_max_idxs.data_ptr(),
        (unsigned int*)lm_sync_counter.data_ptr(),
        (int)position, (int)max_seq_len, (int)group_size,
        c10::cuda::getCurrentCUDAStream().stream());
}

void decode_many_nvfp4(
    torch::Tensor output_tokens,
    torch::Tensor token_buffer,
    int64_t input_token_id,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight,
    torch::Tensor lm_head_weight_packed, torch::Tensor lm_head_scales,
    torch::Tensor lm_hidden_bf16, torch::Tensor lm_hidden_packed, torch::Tensor lm_hidden_scales, torch::Tensor lm_logits_f16,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden_buffer, torch::Tensor activations, torch::Tensor residual,
    torch::Tensor qkv_scratch, torch::Tensor kv_scratch, torch::Tensor attn_out,
    torch::Tensor mlp_inter, torch::Tensor z_scratch, torch::Tensor beta_scratch,
    torch::Tensor alpha_scratch, torch::Tensor normalized,
    torch::Tensor barrier_counter, torch::Tensor barrier_generation,
    torch::Tensor block_max_vals, torch::Tensor block_max_idxs,
    torch::Tensor lm_sync_counter, int64_t position, int64_t max_seq_len,
    int64_t group_size)
{
    TORCH_CHECK(output_tokens.is_cuda(), "output_tokens must be CUDA");
    TORCH_CHECK(output_tokens.is_contiguous(), "output_tokens must be contiguous");
    TORCH_CHECK(output_tokens.scalar_type() == torch::kInt32, "output_tokens must be int32");
    TORCH_CHECK(output_tokens.dim() == 1, "output_tokens must be 1D");
    TORCH_CHECK(token_buffer.is_cuda(), "token_buffer must be CUDA");
    TORCH_CHECK(token_buffer.is_contiguous(), "token_buffer must be contiguous");
    TORCH_CHECK(token_buffer.scalar_type() == torch::kInt32, "token_buffer must be int32");
    TORCH_CHECK(token_buffer.numel() == 1, "token_buffer must contain exactly one int32 token");

    seed_token_buffer(token_buffer, (int)input_token_id);
    launch_decode_many_nvfp4(
        (int*)token_buffer.data_ptr(),
        (int*)output_tokens.data_ptr(),
        (int)output_tokens.numel(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeightsNVFP4*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(),
        lm_head_weight_packed.data_ptr(), lm_head_scales.data_ptr(),
        lm_hidden_bf16.data_ptr(), lm_hidden_packed.data_ptr(), lm_hidden_scales.data_ptr(), lm_logits_f16.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden_buffer.data_ptr(), activations.data_ptr(), residual.data_ptr(),
        qkv_scratch.data_ptr(), kv_scratch.data_ptr(), attn_out.data_ptr(),
        mlp_inter.data_ptr(), z_scratch.data_ptr(), beta_scratch.data_ptr(),
        alpha_scratch.data_ptr(), normalized.data_ptr(),
        (unsigned int*)barrier_counter.data_ptr(), (unsigned int*)barrier_generation.data_ptr(),
        (float*)block_max_vals.data_ptr(), (int*)block_max_idxs.data_ptr(),
        (unsigned int*)lm_sync_counter.data_ptr(),
        (int)position, (int)max_seq_len, (int)group_size,
        c10::cuda::getCurrentCUDAStream().stream());
}

void quantize_nvfp4_out(
    torch::Tensor packed_out,
    torch::Tensor scales_out,
    torch::Tensor weight,
    int64_t group_size)
{
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D [out_dim, in_dim] tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(group_size > 0 && (group_size % 2) == 0, "group_size must be a positive even integer");

    auto rows = static_cast<int>(weight.size(0));
    auto cols = static_cast<int>(weight.size(1));
    TORCH_CHECK((cols % 2) == 0, "in_dim must be divisible by 2 for packed fp4 output");
    TORCH_CHECK((cols % group_size) == 0, "in_dim must be divisible by group_size");
    TORCH_CHECK(packed_out.is_cuda() && packed_out.is_contiguous(), "packed_out must be contiguous CUDA");
    TORCH_CHECK(scales_out.is_cuda() && scales_out.is_contiguous(), "scales_out must be contiguous CUDA");
    TORCH_CHECK(packed_out.scalar_type() == torch::kUInt8, "packed_out must be uint8");
    TORCH_CHECK(scales_out.scalar_type() == torch::kFloat16, "scales_out must be float16");
    TORCH_CHECK(
        packed_out.numel() == (int64_t)rows * (cols / 2),
        "packed_out has the wrong size");
    TORCH_CHECK(
        scales_out.numel() == (int64_t)rows * (cols / group_size),
        "scales_out has the wrong size");

    launch_quantize_nvfp4_out(
        weight.data_ptr(), rows, cols, (int)group_size,
        packed_out.data_ptr(), scales_out.data_ptr(),
        c10::cuda::getCurrentCUDAStream().stream());
}

void quantize_nvfp4_lm_out(
    torch::Tensor packed_out,
    torch::Tensor scales_out,
    torch::Tensor weight)
{
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D [out_dim, in_dim] tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "weight must be bfloat16");

    auto rows = static_cast<int>(weight.size(0));
    auto cols = static_cast<int>(weight.size(1));
    TORCH_CHECK((rows % 128) == 0, "out_dim must be divisible by 128");
    TORCH_CHECK((cols % 64) == 0, "in_dim must be divisible by 64");
    TORCH_CHECK(packed_out.is_cuda() && packed_out.is_contiguous(), "packed_out must be contiguous CUDA");
    TORCH_CHECK(scales_out.is_cuda() && scales_out.is_contiguous(), "scales_out must be contiguous CUDA");
    TORCH_CHECK(packed_out.scalar_type() == torch::kUInt8, "packed_out must be uint8");
    TORCH_CHECK(scales_out.scalar_type() == torch::kUInt8, "scales_out must be uint8");
    TORCH_CHECK(
        packed_out.numel() == (int64_t)rows * (cols / 2),
        "packed_out has the wrong size");

    int scale_tiles = cols / 64;
    int expected_scales = (rows / 128) * scale_tiles * 512;
    TORCH_CHECK(
        scales_out.numel() == expected_scales,
        "scales_out has the wrong size");

    launch_quantize_nvfp4_lm_out(
        weight.data_ptr(), rows, cols,
        packed_out.data_ptr(), scales_out.data_ptr(),
        c10::cuda::getCurrentCUDAStream().stream());
}
#endif  // MEGAKERNEL_HAS_NVFP4

int64_t max_safe_decode_blocks()
{
    return query_max_safe_decode_blocks();
}

void set_decode_blocks(int64_t blocks)
{
    set_decode_blocks_override((int)blocks);
}

// ===== Prefill BF16 =====

// chunk-parallel DeltaNet prefill (was previously v2; promoted to canonical)
// adds 4 fp32 scratch buffers + 2 fused weight bases (FA QKV, MLP gate+up)
extern "C" void launch_prefill_bf16(
    const int *token_ids, int seq_len, int *output_token,
    const void *embed_weight, const LayerWeights *layers,
    const void *final_norm_w, const void *lm_head_w,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden, void *residual, void *normalized,
    void *proj_buf, void *proj_buf2, void *attn_buf, void *mlp_buf,
    void *dn_out_buf,
    void *beta_buf, void *alpha_buf, void *dn_pre_qkv,
    void *dn_u_scratch, void *dn_w_scratch, void *dn_cs_scratch,
    const void *fused_fa_qkv_base, const void *fused_gate_up_base,
    void *final_normed, void *hidden_bf16_out,
    void *lm_bmv, void *lm_bmi,
    void *fa_q_tail,
    int max_seq_len, int q_tail_len,
    cudaStream_t stream);

extern "C" void launch_mega_pflash_score(
    const void *q_tail,
    const void *k_cache,
    float *logit_scratch,
    float *token_scores,
    float *chunk_scores,
    int seq_len,
    int max_seq_len,
    int tail_len,
    int chunk_size,
    int smooth_kernel,
    int reduction_mode,
    cudaStream_t stream);

void prefill_bf16(
    torch::Tensor output_token, torch::Tensor token_ids,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden, torch::Tensor residual, torch::Tensor normalized,
    torch::Tensor proj_buf, torch::Tensor proj_buf2,
    torch::Tensor attn_buf, torch::Tensor mlp_buf,
    torch::Tensor dn_out_buf, torch::Tensor beta_buf, torch::Tensor alpha_buf,
    torch::Tensor dn_pre_qkv,
    torch::Tensor dn_u_scratch, torch::Tensor dn_w_scratch, torch::Tensor dn_cs_scratch,
    torch::Tensor fused_fa_qkv, torch::Tensor fused_gate_up,
    torch::Tensor final_normed, torch::Tensor hidden_bf16_out,
    torch::Tensor lm_bmv, torch::Tensor lm_bmi,
    int64_t max_seq_len)
{
    launch_prefill_bf16(
        (const int*)token_ids.data_ptr(), token_ids.size(0),
        (int*)output_token.data_ptr(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeights*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden.data_ptr(), residual.data_ptr(), normalized.data_ptr(),
        proj_buf.data_ptr(), proj_buf2.data_ptr(),
        attn_buf.data_ptr(), mlp_buf.data_ptr(),
        dn_out_buf.data_ptr(),
        beta_buf.data_ptr(), alpha_buf.data_ptr(), dn_pre_qkv.data_ptr(),
        dn_u_scratch.data_ptr(), dn_w_scratch.data_ptr(), dn_cs_scratch.data_ptr(),
        fused_fa_qkv.data_ptr(), fused_gate_up.data_ptr(),
        final_normed.data_ptr(), hidden_bf16_out.data_ptr(),
        lm_bmv.data_ptr(), lm_bmi.data_ptr(),
        nullptr,
        (int)max_seq_len, 0,
        c10::cuda::getCurrentCUDAStream().stream());
}

void prefill_bf16_qtail(
    torch::Tensor output_token, torch::Tensor token_ids,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden, torch::Tensor residual, torch::Tensor normalized,
    torch::Tensor proj_buf, torch::Tensor proj_buf2,
    torch::Tensor attn_buf, torch::Tensor mlp_buf,
    torch::Tensor dn_out_buf, torch::Tensor beta_buf, torch::Tensor alpha_buf,
    torch::Tensor dn_pre_qkv,
    torch::Tensor dn_u_scratch, torch::Tensor dn_w_scratch, torch::Tensor dn_cs_scratch,
    torch::Tensor fused_fa_qkv, torch::Tensor fused_gate_up,
    torch::Tensor final_normed, torch::Tensor hidden_bf16_out,
    torch::Tensor lm_bmv, torch::Tensor lm_bmi,
    torch::Tensor fa_q_tail,
    int64_t max_seq_len, int64_t q_tail_len)
{
    launch_prefill_bf16(
        (const int*)token_ids.data_ptr(), token_ids.size(0),
        (int*)output_token.data_ptr(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeights*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden.data_ptr(), residual.data_ptr(), normalized.data_ptr(),
        proj_buf.data_ptr(), proj_buf2.data_ptr(),
        attn_buf.data_ptr(), mlp_buf.data_ptr(),
        dn_out_buf.data_ptr(),
        beta_buf.data_ptr(), alpha_buf.data_ptr(), dn_pre_qkv.data_ptr(),
        dn_u_scratch.data_ptr(), dn_w_scratch.data_ptr(), dn_cs_scratch.data_ptr(),
        fused_fa_qkv.data_ptr(), fused_gate_up.data_ptr(),
        final_normed.data_ptr(), hidden_bf16_out.data_ptr(),
        lm_bmv.data_ptr(), lm_bmi.data_ptr(),
        fa_q_tail.data_ptr(),
        (int)max_seq_len, (int)q_tail_len,
        c10::cuda::getCurrentCUDAStream().stream());
}

void mega_pflash_score(
    torch::Tensor q_tail,
    torch::Tensor k_cache,
    torch::Tensor logit_scratch,
    torch::Tensor token_scores,
    torch::Tensor chunk_scores,
    int64_t seq_len,
    int64_t max_seq_len,
    int64_t tail_len,
    int64_t chunk_size,
    int64_t smooth_kernel,
    int64_t reduction_mode)
{
    launch_mega_pflash_score(
        q_tail.data_ptr(),
        k_cache.data_ptr(),
        static_cast<float*>(logit_scratch.data_ptr()),
        static_cast<float*>(token_scores.data_ptr()),
        static_cast<float*>(chunk_scores.data_ptr()),
        (int)seq_len,
        (int)max_seq_len,
        (int)tail_len,
        (int)chunk_size,
        (int)smooth_kernel,
        (int)reduction_mode,
        c10::cuda::getCurrentCUDAStream().stream());
}

#ifdef MEGAKERNEL_HAS_NVFP4
extern "C" void launch_prefill_bf16_mega(
    const int *token_ids, int seq_len, int *output_token,
    const void *embed_weight, const LayerWeights *layers,
    const void *final_norm_w, const void *lm_head_w,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden, void *residual, void *normalized,
    void *proj_buf, void *proj_buf2, void *attn_buf, void *mlp_buf,
    void *dn_out_buf, void *beta_buf, void *alpha_buf,
    void *final_normed, void *hidden_bf16_out,
    void *lm_bmv, void *lm_bmi,
    cudaStream_t stream);

void prefill_bf16_mega(
    torch::Tensor output_token, torch::Tensor token_ids,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden, torch::Tensor residual, torch::Tensor normalized,
    torch::Tensor proj_buf, torch::Tensor proj_buf2,
    torch::Tensor attn_buf, torch::Tensor mlp_buf,
    torch::Tensor dn_out_buf, torch::Tensor beta_buf, torch::Tensor alpha_buf,
    torch::Tensor final_normed, torch::Tensor hidden_bf16_out,
    torch::Tensor lm_bmv, torch::Tensor lm_bmi)
{
    launch_prefill_bf16_mega(
        (const int*)token_ids.data_ptr(), token_ids.size(0),
        (int*)output_token.data_ptr(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeights*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden.data_ptr(), residual.data_ptr(), normalized.data_ptr(),
        proj_buf.data_ptr(), proj_buf2.data_ptr(),
        attn_buf.data_ptr(), mlp_buf.data_ptr(),
        dn_out_buf.data_ptr(), beta_buf.data_ptr(), alpha_buf.data_ptr(),
        final_normed.data_ptr(), hidden_bf16_out.data_ptr(),
        lm_bmv.data_ptr(), lm_bmi.data_ptr(),
        c10::cuda::getCurrentCUDAStream().stream());
}

extern "C" void launch_prefill_bf16_nvfp4_lm(
    const int *token_ids, int seq_len, int *output_token,
    const void *embed_weight, const LayerWeights *layers,
    const PrefillFusedLayerWeights *fused_layers,
    const void *final_norm_w, const void *lm_head_w,
    const void *lm_head_weight_packed, const void *lm_head_scales,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden, void *residual, void *normalized,
    void *proj_buf, void *proj_buf2, void *proj_buf_half, void *proj_act_packed, void *proj_act_scales,
    void *attn_buf, void *mlp_buf,
    void *dn_out_buf, void *beta_buf, void *alpha_buf,
    void *final_normed, void *hidden_bf16_out,
    void *lm_bmv, void *lm_bmi,
    void *lm_hidden_bf16, void *lm_hidden_packed,
    void *lm_hidden_scales, void *lm_logits_f16,
    cudaStream_t stream);

void prefill_bf16_nvfp4_lm(
    torch::Tensor output_token, torch::Tensor token_ids,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor prefill_fused_weights_packed,
    torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
    torch::Tensor lm_head_weight_packed, torch::Tensor lm_head_scales,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden, torch::Tensor residual, torch::Tensor normalized,
    torch::Tensor proj_buf, torch::Tensor proj_buf2, torch::Tensor proj_buf_half,
    torch::Tensor proj_act_packed, torch::Tensor proj_act_scales,
    torch::Tensor attn_buf, torch::Tensor mlp_buf,
    torch::Tensor dn_out_buf, torch::Tensor beta_buf, torch::Tensor alpha_buf,
    torch::Tensor final_normed, torch::Tensor hidden_bf16_out,
    torch::Tensor lm_bmv, torch::Tensor lm_bmi,
    torch::Tensor lm_hidden_bf16, torch::Tensor lm_hidden_packed,
    torch::Tensor lm_hidden_scales, torch::Tensor lm_logits_f16)
{
    launch_prefill_bf16_nvfp4_lm(
        (const int*)token_ids.data_ptr(), token_ids.size(0),
        (int*)output_token.data_ptr(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeights*>(layer_weights_packed.data_ptr()),
        reinterpret_cast<const PrefillFusedLayerWeights*>(prefill_fused_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
        lm_head_weight_packed.data_ptr(), lm_head_scales.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden.data_ptr(), residual.data_ptr(), normalized.data_ptr(),
        proj_buf.data_ptr(), proj_buf2.data_ptr(), proj_buf_half.data_ptr(),
        proj_act_packed.data_ptr(), proj_act_scales.data_ptr(),
        attn_buf.data_ptr(), mlp_buf.data_ptr(),
        dn_out_buf.data_ptr(), beta_buf.data_ptr(), alpha_buf.data_ptr(),
        final_normed.data_ptr(), hidden_bf16_out.data_ptr(),
        lm_bmv.data_ptr(), lm_bmi.data_ptr(),
        lm_hidden_bf16.data_ptr(), lm_hidden_packed.data_ptr(),
        lm_hidden_scales.data_ptr(), lm_logits_f16.data_ptr(),
        c10::cuda::getCurrentCUDAStream().stream());
}

void prefill_megakernel_nvfp4(
    torch::Tensor output_token, torch::Tensor token_ids,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight,
    torch::Tensor lm_head_weight_packed, torch::Tensor lm_head_scales,
    torch::Tensor lm_hidden_bf16, torch::Tensor lm_hidden_packed, torch::Tensor lm_hidden_scales, torch::Tensor lm_logits_f16,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden_buffer, torch::Tensor activations, torch::Tensor residual,
    torch::Tensor qkv_scratch, torch::Tensor kv_scratch, torch::Tensor attn_out,
    torch::Tensor mlp_inter, torch::Tensor z_scratch, torch::Tensor beta_scratch,
    torch::Tensor alpha_scratch, torch::Tensor normalized,
    torch::Tensor barrier_counter, torch::Tensor barrier_generation,
    torch::Tensor block_max_vals, torch::Tensor block_max_idxs,
    torch::Tensor lm_sync_counter, int64_t max_seq_len, int64_t group_size)
{
    TORCH_CHECK(token_ids.is_cuda(), "token_ids must be CUDA");
    TORCH_CHECK(token_ids.is_contiguous(), "token_ids must be contiguous");
    TORCH_CHECK(token_ids.scalar_type() == torch::kInt32, "token_ids must be int32");
    TORCH_CHECK(token_ids.dim() == 1, "token_ids must be 1D");

    launch_prefill_megakernel_nvfp4(
        (const int *)token_ids.data_ptr(),
        static_cast<int>(token_ids.numel()),
        (int *)output_token.data_ptr(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeightsNVFP4 *>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(),
        lm_head_weight_packed.data_ptr(),
        lm_head_scales.data_ptr(),
        lm_hidden_bf16.data_ptr(),
        lm_hidden_packed.data_ptr(),
        lm_hidden_scales.data_ptr(),
        lm_logits_f16.data_ptr(),
        fa_k_cache.data_ptr(),
        fa_v_cache.data_ptr(),
        dn_states.data_ptr(),
        conv_bufs.data_ptr(),
        hidden_buffer.data_ptr(),
        activations.data_ptr(),
        residual.data_ptr(),
        qkv_scratch.data_ptr(),
        kv_scratch.data_ptr(),
        attn_out.data_ptr(),
        mlp_inter.data_ptr(),
        z_scratch.data_ptr(),
        beta_scratch.data_ptr(),
        alpha_scratch.data_ptr(),
        normalized.data_ptr(),
        (unsigned int *)barrier_counter.data_ptr(),
        (unsigned int *)barrier_generation.data_ptr(),
        (float *)block_max_vals.data_ptr(),
        (int *)block_max_idxs.data_ptr(),
        (unsigned int *)lm_sync_counter.data_ptr(),
        (int)max_seq_len,
        (int)group_size,
        c10::cuda::getCurrentCUDAStream().stream());
}
#endif  // MEGAKERNEL_HAS_NVFP4

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("decode(Tensor output_token, int input_token_id, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden_buffer, Tensor activations, Tensor residual, "
            "Tensor qkv_scratch, Tensor kv_scratch, Tensor attn_out, "
            "Tensor mlp_inter, Tensor z_scratch, Tensor beta_scratch, "
            "Tensor alpha_scratch, Tensor normalized, "
            "Tensor barrier_counter, Tensor barrier_generation, "
            "Tensor block_max_vals, Tensor block_max_idxs, Tensor lm_sync_counter, "
            "Tensor seen_token_mask, float repetition_penalty, "
            "int position, int max_seq_len) -> ()");
    ops.impl("decode", torch::kCUDA, &decode);

    ops.def("max_safe_decode_blocks() -> int");
    ops.impl("max_safe_decode_blocks", &max_safe_decode_blocks);

    ops.def("set_decode_blocks(int blocks) -> ()");
    ops.impl("set_decode_blocks", &set_decode_blocks);

    ops.def("prefill_bf16(Tensor output_token, Tensor token_ids, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden, Tensor residual, Tensor normalized, "
            "Tensor proj_buf, Tensor proj_buf2, Tensor attn_buf, Tensor mlp_buf, "
            "Tensor dn_out_buf, Tensor beta_buf, Tensor alpha_buf, "
            "Tensor dn_pre_qkv, "
            "Tensor dn_u_scratch, Tensor dn_w_scratch, Tensor dn_cs_scratch, "
            "Tensor fused_fa_qkv, Tensor fused_gate_up, "
            "Tensor final_normed, Tensor hidden_bf16_out, "
            "Tensor lm_bmv, Tensor lm_bmi, int max_seq_len) -> ()");
    ops.impl("prefill_bf16", torch::kCUDA, &prefill_bf16);

    ops.def("prefill_bf16_qtail(Tensor output_token, Tensor token_ids, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden, Tensor residual, Tensor normalized, "
            "Tensor proj_buf, Tensor proj_buf2, Tensor attn_buf, Tensor mlp_buf, "
            "Tensor dn_out_buf, Tensor beta_buf, Tensor alpha_buf, "
            "Tensor dn_pre_qkv, "
            "Tensor dn_u_scratch, Tensor dn_w_scratch, Tensor dn_cs_scratch, "
            "Tensor fused_fa_qkv, Tensor fused_gate_up, "
            "Tensor final_normed, Tensor hidden_bf16_out, "
            "Tensor lm_bmv, Tensor lm_bmi, Tensor fa_q_tail, "
            "int max_seq_len, int q_tail_len) -> ()");
    ops.impl("prefill_bf16_qtail", torch::kCUDA, &prefill_bf16_qtail);

    ops.def("mega_pflash_score(Tensor q_tail, Tensor k_cache, "
            "Tensor logit_scratch, Tensor token_scores, Tensor chunk_scores, "
            "int seq_len, int max_seq_len, int tail_len, int chunk_size, "
            "int smooth_kernel, int reduction_mode) -> ()");
    ops.impl("mega_pflash_score", torch::kCUDA, &mega_pflash_score);

#ifdef MEGAKERNEL_HAS_NVFP4
    ops.def("decode_nvfp4(Tensor output_token, int input_token_id, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight_packed, Tensor lm_head_scales, "
            "Tensor lm_hidden_bf16, Tensor lm_hidden_packed, Tensor lm_hidden_scales, Tensor lm_logits_f16, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden_buffer, Tensor activations, Tensor residual, "
            "Tensor qkv_scratch, Tensor kv_scratch, Tensor attn_out, "
            "Tensor mlp_inter, Tensor z_scratch, Tensor beta_scratch, "
            "Tensor alpha_scratch, Tensor normalized, "
            "Tensor barrier_counter, Tensor barrier_generation, "
            "Tensor block_max_vals, Tensor block_max_idxs, Tensor lm_sync_counter, "
            "int position, int max_seq_len, int group_size) -> ()");
    ops.impl("decode_nvfp4", torch::kCUDA, &decode_nvfp4);

    ops.def("decode_many_nvfp4(Tensor output_tokens, Tensor token_buffer, int input_token_id, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight_packed, Tensor lm_head_scales, "
            "Tensor lm_hidden_bf16, Tensor lm_hidden_packed, Tensor lm_hidden_scales, Tensor lm_logits_f16, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden_buffer, Tensor activations, Tensor residual, "
            "Tensor qkv_scratch, Tensor kv_scratch, Tensor attn_out, "
            "Tensor mlp_inter, Tensor z_scratch, Tensor beta_scratch, "
            "Tensor alpha_scratch, Tensor normalized, "
            "Tensor barrier_counter, Tensor barrier_generation, "
            "Tensor block_max_vals, Tensor block_max_idxs, Tensor lm_sync_counter, "
            "int position, int max_seq_len, int group_size) -> ()");
    ops.impl("decode_many_nvfp4", torch::kCUDA, &decode_many_nvfp4);

    ops.def("prefill_bf16_mega(Tensor output_token, Tensor token_ids, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden, Tensor residual, Tensor normalized, "
            "Tensor proj_buf, Tensor proj_buf2, Tensor attn_buf, Tensor mlp_buf, "
            "Tensor dn_out_buf, Tensor beta_buf, Tensor alpha_buf, "
            "Tensor final_normed, Tensor hidden_bf16_out, "
            "Tensor lm_bmv, Tensor lm_bmi) -> ()");
    ops.impl("prefill_bf16_mega", torch::kCUDA, &prefill_bf16_mega);

    ops.def("prefill_megakernel_nvfp4(Tensor output_token, Tensor token_ids, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight_packed, Tensor lm_head_scales, "
            "Tensor lm_hidden_bf16, Tensor lm_hidden_packed, Tensor lm_hidden_scales, Tensor lm_logits_f16, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden_buffer, Tensor activations, Tensor residual, "
            "Tensor qkv_scratch, Tensor kv_scratch, Tensor attn_out, "
            "Tensor mlp_inter, Tensor z_scratch, Tensor beta_scratch, "
            "Tensor alpha_scratch, Tensor normalized, "
            "Tensor barrier_counter, Tensor barrier_generation, "
            "Tensor block_max_vals, Tensor block_max_idxs, Tensor lm_sync_counter, "
            "int max_seq_len, int group_size) -> ()");
    ops.impl("prefill_megakernel_nvfp4", torch::kCUDA, &prefill_megakernel_nvfp4);

    ops.def("prefill_bf16_nvfp4_lm(Tensor output_token, Tensor token_ids, "
            "Tensor embed_weight, Tensor layer_weights_packed, Tensor prefill_fused_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight, "
            "Tensor lm_head_weight_packed, Tensor lm_head_scales, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden, Tensor residual, Tensor normalized, "
            "Tensor proj_buf, Tensor proj_buf2, Tensor proj_buf_half, Tensor proj_act_packed, Tensor proj_act_scales, "
            "Tensor attn_buf, Tensor mlp_buf, "
            "Tensor dn_out_buf, Tensor beta_buf, Tensor alpha_buf, "
            "Tensor final_normed, Tensor hidden_bf16_out, "
            "Tensor lm_bmv, Tensor lm_bmi, "
            "Tensor lm_hidden_bf16, Tensor lm_hidden_packed, Tensor lm_hidden_scales, Tensor lm_logits_f16) -> ()");
    ops.impl("prefill_bf16_nvfp4_lm", torch::kCUDA, &prefill_bf16_nvfp4_lm);

    ops.def("quantize_nvfp4_out(Tensor packed_out, Tensor scales_out, Tensor weight, int group_size) -> ()");
    ops.impl("quantize_nvfp4_out", torch::kCUDA, &quantize_nvfp4_out);

    ops.def("quantize_nvfp4_lm_out(Tensor packed_out, Tensor scales_out, Tensor weight) -> ()");
    ops.impl("quantize_nvfp4_lm_out", torch::kCUDA, &quantize_nvfp4_lm_out);
#endif  // MEGAKERNEL_HAS_NVFP4
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
