#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace {

constexpr int DFLASH_DDTREE_TOPK = 8;
constexpr int DFLASH_DDTREE_TOPK_THREADS = 256;

__device__ bool better_topk(float a_val, int32_t a_id, float b_val, int32_t b_id) {
    return a_val > b_val || (a_val == b_val && a_id < b_id);
}

__device__ void insert_topk(float val,
                            int32_t id,
                            float (&vals)[DFLASH_DDTREE_TOPK],
                            int32_t (&ids)[DFLASH_DDTREE_TOPK]) {
    if (!better_topk(val, id, vals[DFLASH_DDTREE_TOPK - 1], ids[DFLASH_DDTREE_TOPK - 1])) {
        return;
    }
    int slot = DFLASH_DDTREE_TOPK - 1;
    while (slot > 0 && better_topk(val, id, vals[slot - 1], ids[slot - 1])) {
        vals[slot] = vals[slot - 1];
        ids[slot] = ids[slot - 1];
        slot--;
    }
    vals[slot] = val;
    ids[slot] = id;
}

__global__ void ddtree_topk_logprobs_kernel(const float * __restrict__ logits,
                                            int n_positions,
                                            int vocab,
                                            float inv_temperature,
                                            float * __restrict__ out_log_probs,
                                            int32_t * __restrict__ out_token_ids) {
    const int pos = blockIdx.x;
    if (pos >= n_positions) {
        return;
    }

    const float * row = logits + (size_t)pos * vocab;
    float local_vals[DFLASH_DDTREE_TOPK];
    int32_t local_ids[DFLASH_DDTREE_TOPK];
    #pragma unroll
    for (int i = 0; i < DFLASH_DDTREE_TOPK; i++) {
        local_vals[i] = -FLT_MAX;
        local_ids[i] = INT32_MAX;
    }

    float local_max = -FLT_MAX;
    float local_sum_exp = 0.0f;

    for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
        const float l = row[j] * inv_temperature;

        if (l > local_max) {
            local_sum_exp *= expf(local_max - l);
            local_sum_exp += 1.0f;
            local_max = l;
        } else {
            local_sum_exp += expf(l - local_max);
        }

        insert_topk(l, (int32_t)j, local_vals, local_ids);
    }

    __shared__ float s_max[DFLASH_DDTREE_TOPK_THREADS];
    __shared__ float s_sum[DFLASH_DDTREE_TOPK_THREADS];
    __shared__ float s_vals[DFLASH_DDTREE_TOPK_THREADS][DFLASH_DDTREE_TOPK];
    __shared__ int32_t s_ids[DFLASH_DDTREE_TOPK_THREADS][DFLASH_DDTREE_TOPK];

    s_max[threadIdx.x] = local_max;
    s_sum[threadIdx.x] = local_sum_exp;
    #pragma unroll
    for (int i = 0; i < DFLASH_DDTREE_TOPK; i++) {
        s_vals[threadIdx.x][i] = local_vals[i];
        s_ids[threadIdx.x][i] = local_ids[i];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float rhs_max = s_max[threadIdx.x + stride];
            const float lhs_max = s_max[threadIdx.x];
            if (rhs_max > lhs_max) {
                s_sum[threadIdx.x] =
                    s_sum[threadIdx.x] * expf(lhs_max - rhs_max) + s_sum[threadIdx.x + stride];
                s_max[threadIdx.x] = rhs_max;
            } else {
                s_sum[threadIdx.x] += s_sum[threadIdx.x + stride] * expf(rhs_max - lhs_max);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float best_vals[DFLASH_DDTREE_TOPK];
        int32_t best_ids[DFLASH_DDTREE_TOPK];
        #pragma unroll
        for (int i = 0; i < DFLASH_DDTREE_TOPK; i++) {
            best_vals[i] = -FLT_MAX;
            best_ids[i] = INT32_MAX;
        }
        for (int t = 0; t < blockDim.x; t++) {
            #pragma unroll
            for (int i = 0; i < DFLASH_DDTREE_TOPK; i++) {
                insert_topk(s_vals[t][i], s_ids[t][i], best_vals, best_ids);
            }
        }

        const float log_z = s_max[0] + logf(s_sum[0]);
        float * out_lp = out_log_probs + (size_t)pos * DFLASH_DDTREE_TOPK;
        int32_t * out_ids = out_token_ids + (size_t)pos * DFLASH_DDTREE_TOPK;
        #pragma unroll
        for (int i = 0; i < DFLASH_DDTREE_TOPK; i++) {
            out_lp[i] = best_vals[i] - log_z;
            out_ids[i] = best_ids[i];
        }
    }
}

} // namespace

extern "C" bool dflash27b_cuda_draft_topk_logprobs_f32(const float * logits,
                                                        int n_positions,
                                                        int vocab,
                                                        int K,
                                                        float temperature,
                                                        float * out_log_probs,
                                                        int32_t * out_token_ids,
                                                        cudaStream_t stream) {
    if (!logits || !out_log_probs || !out_token_ids ||
        n_positions <= 0 || vocab <= 0 || K != DFLASH_DDTREE_TOPK || temperature <= 0.0f) {
        return false;
    }

    float * d_log_probs = nullptr;
    int32_t * d_token_ids = nullptr;
    const size_t out_count = (size_t)n_positions * DFLASH_DDTREE_TOPK;
    cudaError_t err = cudaMalloc(&d_log_probs, out_count * sizeof(float));
    if (err != cudaSuccess) {
        return false;
    }
    err = cudaMalloc(&d_token_ids, out_count * sizeof(int32_t));
    if (err != cudaSuccess) {
        cudaFree(d_log_probs);
        return false;
    }

    const float inv_temperature = 1.0f / temperature;
    ddtree_topk_logprobs_kernel<<<n_positions, DFLASH_DDTREE_TOPK_THREADS, 0, stream>>>(
        logits, n_positions, vocab, inv_temperature, d_log_probs, d_token_ids);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(out_log_probs, d_log_probs, out_count * sizeof(float),
                              cudaMemcpyDeviceToHost, stream);
    }
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(out_token_ids, d_token_ids, out_count * sizeof(int32_t),
                              cudaMemcpyDeviceToHost, stream);
    }
    if (err == cudaSuccess) {
        err = cudaStreamSynchronize(stream);
    }

    cudaFree(d_token_ids);
    cudaFree(d_log_probs);
    return err == cudaSuccess;
}
