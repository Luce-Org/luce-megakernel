// Torchless launcher for the Block-Sparse Attention forward kernel
// (mit-han-lab/Block-Sparse-Attention, sm_80+ FA-2 derived).
//
// Maps our (counts[B,M,H], indices[B,M,N,H]) FlashPrefill selection format
// to the BSA blockmask layout, fills `Flash_fwd_params`, then dispatches
// `run_mha_fwd_block_<bf16, 128, false>`.
//
// All scratch (blockmask, head_mask_type, softmax_lse) is held in a single
// process-lifetime cache (`BsaCache`) that grows on demand and can be freed
// explicitly via dflash_bsa_free_persistent() before the daemon swaps to
// the target-gen path that needs full VRAM.
//
// Hardcoded shape: head_dim=128, block_size=128, non-causal. Add new
// dispatch arms as we support more.

#include "device_runtime.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "namespace_config.h"
#include "flash.h"

#include <cutlass/numeric_types.h>

namespace FLASH_NAMESPACE {
template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_block_(Flash_fwd_params &params, cudaStream_t stream);
}

namespace dflash27b {
namespace flashprefill {

namespace {

constexpr int kSupportedHeadDim   = 128;
constexpr int kSupportedBlockSize = 128;
constexpr int kConvertThreads     = 64;
constexpr float kLog2E            = 1.4426950408889634f;

// Process-lifetime device buffers. Single-stream daemon use only; not
// thread-safe by design. Wrap behind a function-local accessor so we can
// reset state cleanly via dflash_bsa_free_persistent().
struct BsaCache {
    int32_t* blockmask       = nullptr;
    size_t   blockmask_bytes = 0;

    int*     head_mask_type     = nullptr;
    int      head_mask_capacity = 0;
    int      head_mask_state    = -1;  // -1 = uninitialized; 1 = filled with [1..H]

    float*   softmax_lse       = nullptr;
    size_t   softmax_lse_bytes = 0;

    cudaError_t ensure_blockmask(size_t bytes) {
        if (bytes <= blockmask_bytes) return cudaSuccess;
        if (blockmask) cudaFree(blockmask);
        cudaError_t e = cudaMalloc(&blockmask, bytes);
        if (e == cudaSuccess) blockmask_bytes = bytes;
        return e;
    }
    cudaError_t ensure_head_mask(int n_heads) {
        if (n_heads <= head_mask_capacity) return cudaSuccess;
        if (head_mask_type) cudaFree(head_mask_type);
        cudaError_t e = cudaMalloc(&head_mask_type, n_heads * sizeof(int));
        if (e == cudaSuccess) {
            head_mask_capacity = n_heads;
            head_mask_state = -1;
        }
        return e;
    }
    cudaError_t ensure_softmax_lse(size_t bytes) {
        if (bytes <= softmax_lse_bytes) return cudaSuccess;
        if (softmax_lse) cudaFree(softmax_lse);
        cudaError_t e = cudaMalloc(&softmax_lse, bytes);
        if (e == cudaSuccess) softmax_lse_bytes = bytes;
        return e;
    }
    void release() {
        if (blockmask)       { cudaFree(blockmask);       blockmask = nullptr;       blockmask_bytes = 0; }
        if (head_mask_type)  { cudaFree(head_mask_type);  head_mask_type = nullptr;  head_mask_capacity = 0; head_mask_state = -1; }
        if (softmax_lse)     { cudaFree(softmax_lse);     softmax_lse = nullptr;     softmax_lse_bytes = 0; }
    }
};

BsaCache & cache() {
    static BsaCache c;
    return c;
}

// Convert (counts[B,M,H], indices[B,M,N,H]) → BSA blockmask[B, H, Mp, Np].
// Layout: blockmask[(b * num_bs_heads + h) * (Mp * Np) + m * Np + n].
//
// Invariants:
//   - BSA's fwdBlockmask iterator does a binary search assuming a
//     descending-sorted run of valid block IDs followed by a -1 sentinel.
//     We reverse the input (which is ascending) and force at least one -1.
//   - Pad rows beyond M with all -1 so the iterator terminates.
__global__ void convert_to_blockmask_kernel(
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ blockmask_out,
    int B, int M, int N, int H,
    int Mp, int Np,
    int idx_s_b, int idx_s_m, int idx_s_n, int idx_s_h,
    int cnt_s_b, int cnt_s_m, int cnt_s_h)
{
    const int b = blockIdx.x;
    const int m = blockIdx.y;
    const int h = blockIdx.z;
    if (b >= B || h >= H) return;

    int32_t* outp = blockmask_out + ((size_t)(b * H + h) * Mp + m) * Np;

    if (m >= M) {
        for (int n = threadIdx.x; n < Np; n += blockDim.x) outp[n] = -1;
        return;
    }

    int cnt = counts[(size_t)b * cnt_s_b + (size_t)m * cnt_s_m + (size_t)h * cnt_s_h];
    if (cnt >= Np) cnt = Np - 1;  // leave at least one -1 sentinel

    for (int n = threadIdx.x; n < Np; n += blockDim.x) {
        if (n < cnt && n < N) {
            int rev = cnt - 1 - n;
            outp[n] = indices[(size_t)b * idx_s_b + (size_t)m * idx_s_m + (size_t)rev * idx_s_n + (size_t)h * idx_s_h];
        } else {
            outp[n] = -1;
        }
    }
}

}  // namespace

// Public free hook. Idempotent. Call before the daemon switches from
// drafter scoring to target generation to release the BSA scratch (~tens of
// MB at S=128K) and give the target's KV cache full headroom.
extern "C" void dflash_bsa_free_persistent() {
    cache().release();
}

extern "C" int launch_bsa_sparse_flash_forward_bf16(
    const void* Q, const void* K, const void* V, void* O,
    const int32_t* indices, const int32_t* counts,
    float scale,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int head_dim,
    int block_size,
    int s_idx_b, int s_idx_m, int s_idx_n, int s_idx_h,
    int s_cnt_b, int s_cnt_m, int s_cnt_h,
    cudaStream_t stream)
{
    if (head_dim != kSupportedHeadDim) {
        std::fprintf(stderr, "[bsa] unsupported head_dim=%d (only %d)\n",
                     head_dim, kSupportedHeadDim);
        return -1;
    }
    if (block_size != kSupportedBlockSize) {
        std::fprintf(stderr, "[bsa] unsupported block_size=%d (only %d)\n",
                     block_size, kSupportedBlockSize);
        return -1;
    }

    const int M  = (seq_len + block_size - 1) / block_size;
    const int Mp = M;
    const int Np = M;
    const int seqlen_q_rounded = M * block_size;
    const int seqlen_k_rounded = M * block_size;

    BsaCache & c = cache();

    const size_t bm_bytes  = (size_t)batch * n_q_heads * Mp * Np * sizeof(int32_t);
    const size_t lse_bytes = (size_t)batch * n_q_heads * seqlen_q_rounded * sizeof(float);

    cudaError_t err;
    if ((err = c.ensure_blockmask(bm_bytes)) != cudaSuccess)   goto fail;
    if ((err = c.ensure_head_mask(n_q_heads)) != cudaSuccess) goto fail;
    if ((err = c.ensure_softmax_lse(lse_bytes)) != cudaSuccess) goto fail;

    // head_mask_type[h] = h+1: each Q head selects its own per-head block
    // pattern (mask_type-1 indexes into params.blockmask).
    if (c.head_mask_state != 1) {
        int* h_hmt = (int*)std::malloc(n_q_heads * sizeof(int));
        for (int h = 0; h < n_q_heads; ++h) h_hmt[h] = h + 1;
        cudaMemcpyAsync(c.head_mask_type, h_hmt, n_q_heads * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
        std::free(h_hmt);
        c.head_mask_state = 1;
    }

    {
        dim3 grid(batch, Mp, n_q_heads);
        dim3 block(kConvertThreads, 1, 1);
        convert_to_blockmask_kernel<<<grid, block, 0, stream>>>(
            indices, counts, c.blockmask,
            batch, M, /*N=*/M, n_q_heads, Mp, Np,
            s_idx_b, s_idx_m, s_idx_n, s_idx_h,
            s_cnt_b, s_cnt_m, s_cnt_h);
    }

    {
        FLASH_NAMESPACE::Flash_fwd_params params{};
        params.q_ptr = const_cast<void*>(Q);
        params.k_ptr = const_cast<void*>(K);
        params.v_ptr = const_cast<void*>(V);
        params.o_ptr = O;

        params.q_batch_stride = (int64_t)seq_len * n_q_heads * head_dim;
        params.q_row_stride   = (int64_t)n_q_heads * head_dim;
        params.q_head_stride  = head_dim;

        params.k_batch_stride = (int64_t)seq_len * n_k_heads * head_dim;
        params.k_row_stride   = (int64_t)n_k_heads * head_dim;
        params.k_head_stride  = head_dim;

        params.v_batch_stride = (int64_t)seq_len * n_k_heads * head_dim;
        params.v_row_stride   = (int64_t)n_k_heads * head_dim;
        params.v_head_stride  = head_dim;

        params.o_batch_stride = (int64_t)seq_len * n_q_heads * head_dim;
        params.o_row_stride   = (int64_t)n_q_heads * head_dim;
        params.o_head_stride  = head_dim;

        params.h           = n_q_heads;
        params.h_k         = n_k_heads;
        params.h_h_k_ratio = n_q_heads / n_k_heads;

        params.b               = batch;
        params.seqlen_q        = seq_len;
        params.seqlen_k        = seq_len;
        params.d               = head_dim;
        params.seqlen_q_rounded = seqlen_q_rounded;
        params.seqlen_k_rounded = seqlen_k_rounded;
        params.d_rounded       = head_dim;

        params.scale_softmax      = scale;
        params.scale_softmax_log2 = scale * kLog2E;

        params.cu_seqlens_q   = nullptr;
        params.cu_seqlens_k   = nullptr;
        params.seqused_k      = nullptr;
        params.softmax_lse_ptr = c.softmax_lse;
        params.p_ptr          = nullptr;

        params.blockmask             = c.blockmask;
        params.streaming_info        = nullptr;
        params.head_mask_type        = c.head_mask_type;
        params.m_block_dim           = block_size;
        params.n_block_dim           = block_size;
        params.num_blocksparse_heads = n_q_heads;

        params.window_size_left      = -1;
        params.window_size_right     = -1;
        params.is_bf16               = true;
        params.is_causal             = false;
        params.is_exact_streaming    = false;
        params.is_seqlens_k_cumulative = false;
        params.is_rotary_interleaved = false;
        params.num_splits            = 1;
        params.alibi_slopes_ptr      = nullptr;
        params.alibi_slopes_batch_stride = 0;
        params.p_dropout             = 1.f;  // 1.0 = no dropout

        FLASH_NAMESPACE::run_mha_fwd_block_<cutlass::bfloat16_t,
                                            kSupportedHeadDim,
                                            /*Is_causal=*/false>(params, stream);
    }

    return 0;

fail:
    std::fprintf(stderr, "[bsa] cudaMalloc failed: %s\n", cudaGetErrorString(err));
    return -1;
}

}  // namespace flashprefill
}  // namespace dflash27b
