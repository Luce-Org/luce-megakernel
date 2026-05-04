#include <cuda_runtime.h>
#include <math_constants.h>
#include <cublas_v2.h>

#include "half_type.h"

namespace {

constexpr int FA_LAYERS = 6;
constexpr int FA_Q_HEADS = 8;
constexpr int FA_KV_HEADS = 2;
constexpr int FA_HEAD_DIM = 256;
constexpr int Q_TAIL_CAPACITY = 8;

__device__ float block_sum(float v) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    smem[tid] = v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    return smem[0];
}

__device__ float block_max(float v) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    smem[tid] = v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        __syncthreads();
    }
    return smem[0];
}

__device__ float qk_dot(
    const half_t *q_tail,
    const half_t *k_cache,
    int layer,
    int tail_idx,
    int q_head,
    int kv_head,
    int key_pos,
    int max_seq_len)
{
    const half_t *q = q_tail
        + (((layer * Q_TAIL_CAPACITY + tail_idx) * FA_Q_HEADS + q_head) * FA_HEAD_DIM);
    const half_t *k = k_cache
        + (((layer * FA_KV_HEADS + kv_head) * max_seq_len + key_pos) * FA_HEAD_DIM);
    float partial = 0.0f;
    for (int d = threadIdx.x; d < FA_HEAD_DIM; d += blockDim.x) {
        partial += H2F(q[d]) * H2F(k[d]);
    }
    return block_sum(partial);
}

__global__ void mega_pflash_token_scores_kernel(
    const half_t *q_tail,
    const half_t *k_cache,
    float *token_scores,
    int seq_len,
    int max_seq_len,
    int tail_len,
    int reduction_mode)
{
    int token_idx = blockIdx.x;
    if (token_idx >= seq_len) return;

    const float scale = rsqrtf((float)FA_HEAD_DIM);
    const int gqa = FA_Q_HEADS / FA_KV_HEADS;
    float accum = 0.0f;
    float best = 0.0f;
    int count = 0;

    for (int layer = 0; layer < FA_LAYERS; ++layer) {
        for (int q_head = 0; q_head < FA_Q_HEADS; ++q_head) {
            int kv_head = q_head / gqa;
            for (int tail_idx = 0; tail_idx < tail_len; ++tail_idx) {
                int q_pos = seq_len - tail_len + tail_idx;
                float prob = 0.0f;
                if (token_idx <= q_pos) {
                    float target_logit = qk_dot(
                        q_tail, k_cache, layer, tail_idx, q_head, kv_head,
                        token_idx, max_seq_len) * scale;

                    float row_max = -CUDART_INF_F;
                    for (int key_pos = 0; key_pos <= q_pos; ++key_pos) {
                        float logit = qk_dot(
                            q_tail, k_cache, layer, tail_idx, q_head, kv_head,
                            key_pos, max_seq_len) * scale;
                        if (threadIdx.x == 0) row_max = fmaxf(row_max, logit);
                    }
                    row_max = __shfl_sync(0xffffffff, row_max, 0);

                    float denom = 0.0f;
                    for (int key_pos = 0; key_pos <= q_pos; ++key_pos) {
                        float logit = qk_dot(
                            q_tail, k_cache, layer, tail_idx, q_head, kv_head,
                            key_pos, max_seq_len) * scale;
                        if (threadIdx.x == 0) denom += expf(logit - row_max);
                    }
                    denom = __shfl_sync(0xffffffff, denom, 0);
                    if (threadIdx.x == 0 && denom > 0.0f) {
                        prob = expf(target_logit - row_max) / denom;
                    }
                    prob = __shfl_sync(0xffffffff, prob, 0);
                }

                if (reduction_mode == 1) {
                    best = fmaxf(best, prob);
                } else {
                    accum += prob;
                    ++count;
                }
            }
        }
    }

    if (threadIdx.x == 0) {
        token_scores[token_idx] = (reduction_mode == 1)
            ? best
            : (count > 0 ? accum / (float)count : 0.0f);
    }
}

__device__ unsigned int float_to_ordered_uint(float value) {
    unsigned int bits = __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ float ordered_uint_to_float(unsigned int value) {
    unsigned int bits = (value & 0x80000000u) ? (value & 0x7fffffffu) : ~value;
    return __uint_as_float(bits);
}

__device__ void atomic_max_float(float *addr, float value) {
    unsigned int *uaddr = reinterpret_cast<unsigned int *>(addr);
    unsigned int old = *uaddr;
    unsigned int assumed;
    unsigned int desired = float_to_ordered_uint(value);
    do {
        assumed = old;
        if (ordered_uint_to_float(assumed) >= value) break;
        old = atomicCAS(uaddr, assumed, desired);
    } while (assumed != old);
}

__global__ void mega_pflash_softmax_accum_kernel(
    const float *logits,
    float *token_scores,
    int seq_len,
    int tail_len,
    int total_rows,
    int reduction_mode)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= total_rows) return;

    int tail_idx = row % tail_len;
    int q_pos = seq_len - tail_len + tail_idx;
    const float scale = rsqrtf((float)FA_HEAD_DIM);
    const float *row_logits = logits + (size_t)row * seq_len;

    float local_max = -CUDART_INF_F;
    for (int i = tid; i <= q_pos; i += blockDim.x) {
        local_max = fmaxf(local_max, row_logits[i] * scale);
    }
    float row_max = block_max(local_max);

    float local_sum = 0.0f;
    for (int i = tid; i <= q_pos; i += blockDim.x) {
        local_sum += expf(row_logits[i] * scale - row_max);
    }
    float denom = block_sum(local_sum);
    if (denom <= 0.0f) return;

    for (int i = tid; i <= q_pos; i += blockDim.x) {
        float prob = expf(row_logits[i] * scale - row_max) / denom;
        if (reduction_mode == 1) {
            atomic_max_float(token_scores + i, prob);
        } else {
            atomicAdd(token_scores + i, prob);
        }
    }
}

__global__ void mega_pflash_chunk_scores_kernel(
    const float *token_scores,
    float *chunk_scores,
    int seq_len,
    int chunk_size,
    int smooth_kernel,
    int n_chunks)
{
    int chunk = blockIdx.x;
    int tid = threadIdx.x;
    if (chunk >= n_chunks) return;

    int start = chunk * chunk_size;
    int end = min(seq_len, start + chunk_size);
    int radius = smooth_kernel / 2;
    float local = -CUDART_INF_F;

    for (int pos = start + tid; pos < end; pos += blockDim.x) {
        int lo = max(0, pos - radius);
        int hi = min(seq_len - 1, pos + radius);
        float sum = 0.0f;
        int count = 0;
        for (int i = lo; i <= hi; ++i) {
            sum += token_scores[i];
            ++count;
        }
        local = fmaxf(local, count ? sum / (float)count : token_scores[pos]);
    }
    float mx = block_max(local);
    if (tid == 0) chunk_scores[chunk] = mx;
}

__global__ void mega_pflash_normalize_kernel(float *token_scores, int seq_len, float inv_scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len) token_scores[idx] *= inv_scale;
}

} // namespace

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
    cudaStream_t stream)
{
    if (seq_len <= 0 || tail_len <= 0 || chunk_size <= 0) return;
    if (tail_len > Q_TAIL_CAPACITY) tail_len = Q_TAIL_CAPACITY;
    int n_chunks = (seq_len + chunk_size - 1) / chunk_size;

    static cublasHandle_t cublas = nullptr;
    if (!cublas) cublasCreate(&cublas);
    cublasSetStream(cublas, stream);

    cudaMemsetAsync(token_scores, 0, sizeof(float) * seq_len, stream);

    const half_t *qt = static_cast<const half_t *>(q_tail);
    const half_t *kc = static_cast<const half_t *>(k_cache);
    float alpha = 1.0f;
    float beta = 0.0f;
    const int gqa = FA_Q_HEADS / FA_KV_HEADS;
    int row = 0;
    for (int layer = 0; layer < FA_LAYERS; ++layer) {
        for (int q_head = 0; q_head < FA_Q_HEADS; ++q_head) {
            int kv_head = q_head / gqa;
            const half_t *q = qt + ((layer * Q_TAIL_CAPACITY * FA_Q_HEADS + q_head) * FA_HEAD_DIM);
            const half_t *k = kc + ((layer * FA_KV_HEADS + kv_head) * max_seq_len * FA_HEAD_DIM);
            float *scores = logit_scratch + (size_t)row * tail_len * seq_len;
            cublasGemmEx(
                cublas,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                seq_len,
                tail_len,
                FA_HEAD_DIM,
                &alpha,
                k,
                CUBLAS_HALF_T,
                FA_HEAD_DIM,
                q,
                CUBLAS_HALF_T,
                FA_Q_HEADS * FA_HEAD_DIM,
                &beta,
                scores,
                CUDA_R_32F,
                seq_len,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT);
            ++row;
        }
    }

    int total_rows = FA_LAYERS * FA_Q_HEADS * tail_len;
    mega_pflash_softmax_accum_kernel<<<total_rows, 256, 0, stream>>>(
        logit_scratch,
        token_scores,
        seq_len,
        tail_len,
        total_rows,
        reduction_mode);

    if (reduction_mode == 0) {
        int total = FA_LAYERS * FA_Q_HEADS * tail_len;
        mega_pflash_normalize_kernel<<<(seq_len + 255) / 256, 256, 0, stream>>>(
            token_scores, seq_len, 1.0f / (float)total);
    }

    mega_pflash_chunk_scores_kernel<<<n_chunks, 256, 0, stream>>>(
        token_scores,
        chunk_scores,
        seq_len,
        chunk_size,
        smooth_kernel,
        n_chunks);
}
