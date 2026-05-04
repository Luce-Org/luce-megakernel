// Tiny half-precision → f32 conversion kernels used by the DDtree rollback
// path and the drafter's target_feat widen. We store some tensors
// (ssm_intermediate, target_feat) at 16-bit to halve their memory footprint,
// and widen on read into f32 consumers.
//
// Exposes plain C entry points so test_dflash.cpp can call them without
// pulling in a CUDA compile unit of its own.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

static __global__ void f16_to_f32_kernel(const __half * __restrict__ src,
                                         float * __restrict__ dst,
                                         size_t n_elems) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elems) {
        dst[i] = __half2float(src[i]);
    }
}

static __global__ void bf16_to_f32_kernel(const __nv_bfloat16 * __restrict__ src,
                                          float * __restrict__ dst,
                                          size_t n_elems) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elems) {
        dst[i] = __bfloat162float(src[i]);
    }
}

extern "C" void dflash27b_launch_f16_to_f32(const void * src,
                                            void * dst,
                                            size_t n_elems,
                                            cudaStream_t stream) {
    const int threads = 256;
    const int blocks  = (int)((n_elems + threads - 1) / threads);
    f16_to_f32_kernel<<<blocks, threads, 0, stream>>>(
        (const __half *)src, (float *)dst, n_elems);
}

extern "C" void dflash27b_launch_bf16_to_f32(const void * src,
                                             void * dst,
                                             size_t n_elems,
                                             cudaStream_t stream) {
    const int threads = 256;
    const int blocks  = (int)((n_elems + threads - 1) / threads);
    bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16 *)src, (float *)dst, n_elems);
}

// DDTree rollback helpers. The accepted path can be non-contiguous in DFS
// order; compacting it with host-issued cudaMemcpyAsync loops creates many
// tiny copies. These helpers upload accepted[] once, then perform in-place
// ordered compaction on the GPU. Copies intentionally run in increasing
// destination order to preserve the old memcpy-loop semantics when a later
// source slot is also a destination slot.

static int * g_ddtree_accepted = nullptr;
static int   g_ddtree_accepted_cap = 0;

extern "C" bool dflash27b_prepare_ddtree_accepted(const int * accepted,
                                                  int count,
                                                  cudaStream_t stream) {
    if (count <= 0) return true;
    if (count > g_ddtree_accepted_cap) {
        if (g_ddtree_accepted) {
            cudaFree(g_ddtree_accepted);
            g_ddtree_accepted = nullptr;
            g_ddtree_accepted_cap = 0;
        }
        cudaError_t ce = cudaMalloc((void **)&g_ddtree_accepted,
                                    (size_t)count * sizeof(int));
        if (ce != cudaSuccess) return false;
        g_ddtree_accepted_cap = count;
    }
    cudaError_t ce = cudaMemcpyAsync(g_ddtree_accepted, accepted,
                                     (size_t)count * sizeof(int),
                                     cudaMemcpyHostToDevice, stream);
    return ce == cudaSuccess;
}

static __global__ void ddtree_compact_target_feat_kernel(
    unsigned char * __restrict__ target_feat,
    const int * __restrict__ accepted,
    int committed,
    int commit_n,
    int target_feat_cap,
    size_t col_stride,
    size_t bytes_per_col) {
    for (int d = 1; d < commit_n; d++) {
        const int src_dfs = accepted[d];
        if (src_dfs != d) {
            const int src_slot = (committed + src_dfs) % target_feat_cap;
            const int dst_slot = (committed + d)       % target_feat_cap;
            unsigned char * dst = target_feat + (size_t)dst_slot * col_stride;
            const unsigned char * src = target_feat + (size_t)src_slot * col_stride;
            for (size_t i = threadIdx.x; i < bytes_per_col; i += blockDim.x) {
                dst[i] = src[i];
            }
        }
        __syncthreads();
    }
}

extern "C" void dflash27b_launch_ddtree_compact_target_feat(
    void * target_feat,
    int committed,
    int commit_n,
    int target_feat_cap,
    size_t col_stride,
    size_t bytes_per_col,
    cudaStream_t stream) {
    if (!target_feat || !g_ddtree_accepted || commit_n <= 1 || target_feat_cap <= 0) return;
    ddtree_compact_target_feat_kernel<<<1, 256, 0, stream>>>(
        (unsigned char *)target_feat, g_ddtree_accepted,
        committed, commit_n, target_feat_cap, col_stride, bytes_per_col);
}

static __global__ void ddtree_compact_kv_pair_kernel(
    unsigned char * __restrict__ k_data,
    unsigned char * __restrict__ v_data,
    const int * __restrict__ accepted,
    int committed,
    int commit_n,
    int n_head_k,
    int n_head_v,
    size_t k_slot_stride,
    size_t v_slot_stride,
    size_t k_head_stride,
    size_t v_head_stride,
    size_t k_bytes_per_head_slot,
    size_t v_bytes_per_head_slot) {
    const size_t k_total = (size_t)n_head_k * k_bytes_per_head_slot;
    const size_t v_total = (size_t)n_head_v * v_bytes_per_head_slot;
    const size_t total   = k_total + v_total;
    for (int d = 1; d < commit_n; d++) {
        const int src_dfs = accepted[d];
        if (src_dfs != d) {
            for (size_t x = threadIdx.x; x < total; x += blockDim.x) {
                if (x < k_total) {
                    const int h = (int)(x / k_bytes_per_head_slot);
                    const size_t b = x - (size_t)h * k_bytes_per_head_slot;
                    const unsigned char * src =
                        k_data + (size_t)(committed + src_dfs) * k_slot_stride +
                        (size_t)h * k_head_stride + b;
                    unsigned char * dst =
                        k_data + (size_t)(committed + d) * k_slot_stride +
                        (size_t)h * k_head_stride + b;
                    *dst = *src;
                } else {
                    const size_t y = x - k_total;
                    const int h = (int)(y / v_bytes_per_head_slot);
                    const size_t b = y - (size_t)h * v_bytes_per_head_slot;
                    const unsigned char * src =
                        v_data + (size_t)(committed + src_dfs) * v_slot_stride +
                        (size_t)h * v_head_stride + b;
                    unsigned char * dst =
                        v_data + (size_t)(committed + d) * v_slot_stride +
                        (size_t)h * v_head_stride + b;
                    *dst = *src;
                }
            }
        }
        __syncthreads();
    }
}

extern "C" void dflash27b_launch_ddtree_compact_kv_pair(
    void * k_data,
    void * v_data,
    int committed,
    int commit_n,
    int n_head_k,
    int n_head_v,
    size_t k_slot_stride,
    size_t v_slot_stride,
    size_t k_head_stride,
    size_t v_head_stride,
    size_t k_bytes_per_head_slot,
    size_t v_bytes_per_head_slot,
    cudaStream_t stream) {
    if (!k_data || !v_data || !g_ddtree_accepted || commit_n <= 1) return;
    ddtree_compact_kv_pair_kernel<<<1, 256, 0, stream>>>(
        (unsigned char *)k_data, (unsigned char *)v_data, g_ddtree_accepted,
        committed, commit_n, n_head_k, n_head_v,
        k_slot_stride, v_slot_stride, k_head_stride, v_head_stride,
        k_bytes_per_head_slot, v_bytes_per_head_slot);
}

// DDTree verify-input prep. The tree shape is still built on the host, but this
// helper avoids materializing and uploading the large ancestor mask every step.

static int * g_ddtree_depths = nullptr;
static int * g_ddtree_parents = nullptr;
static int   g_ddtree_verify_cap = 0;

static __global__ void ddtree_prepare_verify_inputs_kernel(
    int * __restrict__ positions,
    int * __restrict__ parent_ids,
    unsigned short * __restrict__ attn_mask,
    const int * __restrict__ depths,
    const int * __restrict__ parents,
    int committed,
    int past_length,
    int win_start,
    int N,
    int kv_pad,
    int q_pad) {
    const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        const int depth = (i == 0) ? 0 : depths[i - 1];
        const int p = committed + depth;
        positions[0 * N + i] = p;
        positions[1 * N + i] = p;
        positions[2 * N + i] = p;
        positions[3 * N + i] = 0;
        parent_ids[i] = parents[i];
    }

    const int total = kv_pad * q_pad;
    for (int idx = tid; idx < total; idx += blockDim.x * gridDim.x) {
        const int k = idx % kv_pad;
        const int q = idx / kv_pad;
        unsigned short v = 0xFC00u; // f16 -inf

        if (q < N) {
            const int abs_k = win_start + k;
            if (abs_k < past_length) {
                v = 0x0000u;
            } else {
                const int j = abs_k - past_length;
                if (j >= 0 && j < N) {
                    int x = q;
                    while (x >= 0) {
                        if (x == j) {
                            v = 0x0000u;
                            break;
                        }
                        x = parents[x];
                    }
                }
            }
        }

        attn_mask[idx] = v;
    }
}

extern "C" bool dflash27b_prepare_ddtree_verify_inputs(
    int committed,
    int past_length,
    int win_start,
    int N,
    int kv_pad,
    int q_pad,
    const int * depths,
    const int * parents,
    void * positions,
    void * parent_ids,
    void * attn_mask,
    cudaStream_t stream) {
    if (!depths || !parents || !positions || !parent_ids || !attn_mask ||
        N <= 0 || kv_pad <= 0 || q_pad <= 0) {
        return false;
    }
    if (N > g_ddtree_verify_cap) {
        if (g_ddtree_depths) {
            cudaFree(g_ddtree_depths);
            cudaFree(g_ddtree_parents);
            g_ddtree_depths = nullptr;
            g_ddtree_parents = nullptr;
            g_ddtree_verify_cap = 0;
        }
        cudaError_t ce = cudaMalloc((void **)&g_ddtree_depths,
                                    (size_t)N * sizeof(int));
        if (ce != cudaSuccess) return false;
        ce = cudaMalloc((void **)&g_ddtree_parents, (size_t)N * sizeof(int));
        if (ce != cudaSuccess) return false;
        g_ddtree_verify_cap = N;
    }
    cudaError_t ce = cudaMemcpyAsync(g_ddtree_depths, depths,
                                     (size_t)(N - 1) * sizeof(int),
                                     cudaMemcpyHostToDevice, stream);
    if (ce != cudaSuccess) return false;
    ce = cudaMemcpyAsync(g_ddtree_parents, parents,
                         (size_t)N * sizeof(int),
                         cudaMemcpyHostToDevice, stream);
    if (ce != cudaSuccess) return false;

    ddtree_prepare_verify_inputs_kernel<<<1, 256, 0, stream>>>(
        (int *)positions, (int *)parent_ids, (unsigned short *)attn_mask,
        g_ddtree_depths, g_ddtree_parents,
        committed, past_length, win_start, N, kv_pad, q_pad);
    return cudaGetLastError() == cudaSuccess;
}
