// CUDA port of qhfan/FlashPrefill (arXiv 2603.06199).
//
// Block-sparse attention for the pflash drafter. Replaces the upstream
// Triton implementation so the daemon can score long prompts on GPU
// in-process (no Python, no Triton runtime).
//
// Algorithm (from the FlashPrefill paper + qhfan reference impl):
//   1. compute_mean_vector_kernel  — mean K over BLOCK_SIZE blocks.
//      Output: mean_k[B, n_k_blocks, n_kv_heads, D]
//   2. compute_block_score_kernel  — per (q_block, k_block) score via
//      Q · mean_K^T.  Output: score[B, M, N, H], max[B, M, N, H]
//   3. block_select (host or CUDA) — pick top-K blocks per Q row, with
//      sink + window + dynamic alpha threshold + always-on last blocks.
//      Output: compact_indices[B, M, N, H], counts[B, M, H]
//   4. flash_forward_kernel        — sparse attention forward over the
//      SELECTED blocks only. Online softmax + output reduction.
//
// Status: all 4 kernels implemented (mean_vector, block_score, block_select, sparse_flash_forward).
// Currently dispatched only for D_HEAD=128 BLOCK_SIZE=128 (Qwen3 family).
//
// Conventions:
//   - All tensors row-major, D fastest.
//   - Q shape: [B, S, n_q_heads, D]
//   - K, V shape: [B, S, n_k_heads, D]   (n_q_heads = n_k_heads × group_size for GQA)
//   - mean_k shape: [B, ceil(S/BLOCK), n_k_heads, D]
//   - score shape: [B, M, N, n_q_heads]   (M = N = ceil(S/BLOCK))

#include <cstdint>
#include <cstdlib>
#include "device_runtime.h"
#if !defined(DFLASH27B_USE_HIP)
#include <mma.h>
#endif

namespace dflash27b {
namespace flashprefill {

// ── cp.async helpers (sm_8x) ─────────────────────────────────────────
// 16-byte (uint4) async global → shared copy. Issued by every thread that
// participates in the cooperative load. wait_all() drains all outstanding
// transfers; commit_group()/wait_group(N) supports multi-stage pipelines.
#if !defined(DFLASH27B_USE_HIP)
__device__ inline void cp_async16(void * smem_ptr, const void * gmem_ptr) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr));
}
__device__ inline void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
__device__ inline void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}
#endif


// ---- Kernel 1: compute_mean_vector ----
// Each block of (BLOCK threads, 1 K-head, 1 batch) reduces one K-block
// along the sequence dim, computing the mean per dim. Tile dims: (S/BLOCK, B*Hk).

template <int BLOCK, int D_HEAD>
__global__ void compute_mean_vector_kernel_bf16(
    const __nv_bfloat16 * __restrict__ K,
    __nv_bfloat16       * __restrict__ mean_K,
    int batch, int seq_len, int n_kv_heads,
    // strides, in elements (not bytes)
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d)
{
    const int block_idx_n = blockIdx.x;          // which K-block (0 .. M_blocks-1)
    const int zh = blockIdx.y;
    const int b = zh / n_kv_heads;
    const int h = zh % n_kv_heads;
    if (b >= batch) return;

    const int tid = threadIdx.x;
    const int dim = tid;
    if (dim >= D_HEAD) return;

    const __nv_bfloat16 * Kp = K + (size_t)b * s_K_b + (size_t)h * s_K_h;
    __nv_bfloat16       * Mp = mean_K + (size_t)b * s_mK_b + (size_t)h * s_mK_h
                                      + (size_t)block_idx_n * s_mK_m;

    const int n_lo = block_idx_n * BLOCK;
    const int n_hi = min(n_lo + BLOCK, seq_len);
    const int count = n_hi - n_lo;
    if (count <= 0) return;

    float sum = 0.0f;
    for (int n = n_lo; n < n_hi; ++n) {
        sum += __bfloat162float(Kp[(size_t)n * s_K_n + (size_t)dim * s_K_d]);
    }
    Mp[(size_t)dim * s_mK_d] = __float2bfloat16(sum / (float)count);
}

// Public launcher (called from C++).
extern "C" void launch_compute_mean_vector_bf16(
    const void * K, void * mean_K,
    int batch, int seq_len, int n_kv_heads, int head_dim, int block_size,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    cudaStream_t stream)
{
    const int n_k_blocks = (seq_len + block_size - 1) / block_size;
    dim3 grid(n_k_blocks, batch * n_kv_heads, 1);
    dim3 block(head_dim, 1, 1);
    if (head_dim == 128 && block_size == 128) {
        compute_mean_vector_kernel_bf16<128, 128><<<grid, block, 0, stream>>>(
            (const __nv_bfloat16 *)K, (__nv_bfloat16 *)mean_K,
            batch, seq_len, n_kv_heads,
            s_K_b, s_K_n, s_K_h, s_K_d,
            s_mK_b, s_mK_m, s_mK_h, s_mK_d);
    }
    // Only D_HEAD=128 BLOCK=128 dispatched here. Add other combos when new heads/blocks needed.
}

// ---- Kernel 2: compute_block_score ----
//
// Per (q_block, k_block), compute the attention score that decides whether
// the k_block makes the cut for sparse attention. Uses Q tile (full BLOCK
// rows) vs mean_K tile (one row per k_block, from kernel 1).
//
// For each (q_block_idx M, kv_head):
//   For each k_block_idx N (causal: N <= M):
//     score[M, N, h] = sum_{j in q_block} sum_{n in k_block_one_row}
//                       exp2( (Q[j,h,d] · mean_K[N,h_kv,d] * sm_scale) - max )
//     Also output max for renormalization.
//
// One CUDA block per (q_block_idx, batch×n_q_heads). Threads: BLOCK_SIZE
// rows of Q within the block. Loops over N (k_blocks) on the inner.
//
// This is a straightforward reduction. Inner dot product in registers.

template <int BLOCK, int D_HEAD, int N_BLOCKS_TILE>
__global__ void compute_block_score_kernel_bf16(
    const __nv_bfloat16 * __restrict__ Q,
    const __nv_bfloat16 * __restrict__ mean_K,
    float sm_scale,
    float * __restrict__ score,    // [B, M, N, H]
    float * __restrict__ score_max,// [B, M, N, H]
    int batch, int n_q_heads, int n_k_heads,
    int q_block_idx_max,           // total q blocks  M = ceil(S/BLOCK)
    int k_block_idx_max,           // total k blocks  N = ceil(S/BLOCK)
    // strides (in elements)
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    int s_S_b, int s_S_m, int s_S_n, int s_S_h,
    int s_M_b, int s_M_m, int s_M_n, int s_M_h)
{
    const int q_block_idx = blockIdx.x;
    const int zh = blockIdx.y;
    const int b = zh / n_q_heads;
    const int qh = zh % n_q_heads;
    if (b >= batch) return;

    const int kh = qh * n_k_heads / n_q_heads;     // GQA: q-head → kv-head

    const int tid = threadIdx.x;                    // 0..BLOCK-1
    const int q_row_local = tid;                    // which Q row in this block
    const int q_row_global = q_block_idx * BLOCK + q_row_local;

    if (q_row_local >= BLOCK || q_row_global >= q_block_idx_max * BLOCK) {
        // Out of range, no Q to load.
        return;
    }

    // Load this Q row into registers (one row per thread).
    const __nv_bfloat16 * Qp = Q + (size_t)b * s_Q_b
                                  + (size_t)q_row_global * s_Q_n
                                  + (size_t)qh * s_Q_h;
    float q_reg[D_HEAD];
    #pragma unroll
    for (int d = 0; d < D_HEAD; ++d) {
        q_reg[d] = __bfloat162float(Qp[(size_t)d * s_Q_d]);
    }

    // For each k-block, compute the score: contributions from THIS q row,
    // then reduce across BLOCK rows via warp/block reductions.
    extern __shared__ float smem[];      // [BLOCK] for per-row partial sums

    for (int n = 0; n <= q_block_idx; ++n) {       // causal
        // Load mean_K[n, kh, :] into registers (broadcast to all threads).
        const __nv_bfloat16 * mKp = mean_K + (size_t)b * s_mK_b
                                            + (size_t)n * s_mK_m
                                            + (size_t)kh * s_mK_h;
        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < D_HEAD; ++d) {
            dot += q_reg[d] * __bfloat162float(mKp[(size_t)d * s_mK_d]);
        }
        dot *= sm_scale * 1.4426950408889634f;     // log2 base for exp2

        // Find max across threads.
        smem[tid] = dot;
        __syncthreads();
        // Block-reduce max.
        for (int off = BLOCK / 2; off > 0; off >>= 1) {
            if (tid < off) smem[tid] = fmaxf(smem[tid], smem[tid + off]);
            __syncthreads();
        }
        float m_block = smem[0];
        __syncthreads();

        // Sum of exp2(dot - m_block).
        float p = exp2f(dot - m_block);
        smem[tid] = p;
        __syncthreads();
        for (int off = BLOCK / 2; off > 0; off >>= 1) {
            if (tid < off) smem[tid] += smem[tid + off];
            __syncthreads();
        }
        float p_sum = smem[0];
        __syncthreads();

        // Thread 0 writes the (score, max) for this (q_block, n).
        if (tid == 0) {
            score    [(size_t)b * s_S_b + (size_t)q_block_idx * s_S_m
                      + (size_t)n * s_S_n + (size_t)qh * s_S_h] = p_sum;
            score_max[(size_t)b * s_M_b + (size_t)q_block_idx * s_M_m
                      + (size_t)n * s_M_n + (size_t)qh * s_M_h] = m_block;
        }
    }
}

extern "C" void launch_compute_block_score_bf16(
    const void * Q, const void * mean_K, float sm_scale,
    void * score, void * score_max,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int head_dim, int block_size,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    int s_S_b, int s_S_m, int s_S_n, int s_S_h,
    int s_M_b, int s_M_m, int s_M_n, int s_M_h,
    cudaStream_t stream)
{
    const int M = (seq_len + block_size - 1) / block_size;
    dim3 grid(M, batch * n_q_heads, 1);
    dim3 block(block_size, 1, 1);
    size_t smem = block_size * sizeof(float);
    if (head_dim == 128 && block_size == 128) {
        compute_block_score_kernel_bf16<128, 128, 1><<<grid, block, smem, stream>>>(
            (const __nv_bfloat16 *)Q, (const __nv_bfloat16 *)mean_K, sm_scale,
            (float *)score, (float *)score_max,
            batch, n_q_heads, n_k_heads, M, M,
            s_Q_b, s_Q_n, s_Q_h, s_Q_d,
            s_mK_b, s_mK_m, s_mK_h, s_mK_d,
            s_S_b, s_S_m, s_S_n, s_S_h,
            s_M_b, s_M_m, s_M_n, s_M_h);
    }
}

// ---- Kernel 4: sparse_flash_forward ----
//
// FlashAttention-style online softmax over a *selected* set of K-blocks.
// For each Q tile, we iterate `counts[q_block, h]` blocks (indices listed
// in `block_index[q_block, :, h]`) and update m_i, l_i, acc per row.
//
// Inputs:
//   Q[B, S, H, D]  bf16
//   K[B, S, Hk, D] bf16
//   V[B, S, Hk, D] bf16
//   block_index[B, M, N, H] int32  (compact, padded with N=invalid)
//   counts[B, M, H] int32          (how many indices are valid per row)
// Output:
//   O[B, S, H, D]  bf16
//
// Tile: Q_TILE_SIZE rows per CTA, BLOCK_SIZE = 128 K rows per selected block.
// One CTA per (q_tile_idx, batch×n_q_heads).

// v2: shared-memory tiled. Each CTA cooperatively loads each selected
// K-block (and V-block) into shared mem, then all Q_TILE threads do their
// QK / softmax / PV reduction reading from shared mem. This avoids
// re-loading K/V from HBM Q_TILE times per row, which was the dominant
// cost of the v1 scalar kernel.
#if !defined(DFLASH27B_USE_HIP)
template <int Q_TILE, int K_TILE, int BLOCK, int D_HEAD>
__global__ void sparse_flash_forward_kernel_bf16(
    const __nv_bfloat16 * __restrict__ Q,
    const __nv_bfloat16 * __restrict__ K,
    const __nv_bfloat16 * __restrict__ V,
    __nv_bfloat16       * __restrict__ O,
    const int32_t       * __restrict__ block_index,
    const int32_t       * __restrict__ counts,
    float scale,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int M_blocks,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_V_b, int s_V_n, int s_V_h, int s_V_d,
    int s_O_b, int s_O_n, int s_O_h, int s_O_d,
    int s_idx_b, int s_idx_m, int s_idx_n, int s_idx_h,
    int s_cnt_b, int s_cnt_m, int s_cnt_h)
{
    using namespace nvcuda;
    constexpr int MMA_M = 16, MMA_N = 16, MMA_K = 16;
    constexpr int NDK = D_HEAD / MMA_K;       // 8 K-tiles along D
    constexpr int NNK = K_TILE / MMA_N;       // 4 N-tiles along K_TILE (when K_TILE=64)
    constexpr int N_INNER = BLOCK / K_TILE;   // 2 inner iters per selected block
    constexpr int NTHREADS = (Q_TILE / MMA_M) * 32;  // 1 warp per 16 Q rows

    const int q_tile_idx = blockIdx.x;
    const int zh = blockIdx.y;
    const int b  = zh / n_q_heads;
    const int qh = zh % n_q_heads;
    if (b >= batch) return;
    const int kh = qh * n_k_heads / n_q_heads;
    const int q_block_idx = q_tile_idx * Q_TILE / BLOCK;

    const int wid  = threadIdx.x / 32;        // 0..3
    const int lane = threadIdx.x & 31;
    const int tid  = threadIdx.x;             // 0..127

    // SMEM: Q + KV (K or V at a time, alias) + P + row state
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16 * Q_sh   = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16 * KV_sh  = Q_sh + (size_t)Q_TILE * D_HEAD;
    __nv_bfloat16 * P_sh   = KV_sh + (size_t)K_TILE * D_HEAD;
    float         * row_m  = reinterpret_cast<float*>(P_sh + (size_t)Q_TILE * K_TILE);
    float         * row_l  = row_m + Q_TILE;

    if (tid < Q_TILE) {
        row_m[tid] = -INFINITY;
        row_l[tid] = 0.0f;
    }

    // Load Q [Q_TILE, D_HEAD] cooperatively
    {
        const __nv_bfloat16 * Qp = Q + (size_t)b * s_Q_b + (size_t)qh * s_Q_h;
        for (int idx = tid; idx < Q_TILE * D_HEAD; idx += NTHREADS) {
            int row = idx / D_HEAD;
            int dim = idx - row * D_HEAD;
            int q_global = q_tile_idx * Q_TILE + row;
            Q_sh[row * D_HEAD + dim] = (q_global < seq_len)
                ? Qp[(size_t)q_global * s_Q_n + (size_t)dim * s_Q_d]
                : __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    // Pre-scale Q by sm_scale = scale * log2(e)
    {
        const float sm_scale = scale * 1.4426950408889634f;
        for (int idx = tid; idx < Q_TILE * D_HEAD; idx += NTHREADS) {
            float v = __bfloat162float(Q_sh[idx]);
            Q_sh[idx] = __float2bfloat16(v * sm_scale);
        }
    }
    __syncthreads();

    // O accumulator (one per D col tile per warp)
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> O_frag[NDK];
    #pragma unroll
    for (int d = 0; d < NDK; ++d) wmma::fill_fragment(O_frag[d], 0.0f);

    const int hi = counts[(size_t)b * s_cnt_b + (size_t)q_block_idx * s_cnt_m + (size_t)qh * s_cnt_h];

    for (int it = 0; it < hi; ++it) {
        int blk = block_index[(size_t)b * s_idx_b + (size_t)q_block_idx * s_idx_m
                              + (size_t)it * s_idx_n + (size_t)qh * s_idx_h];
        if (blk < 0 || blk >= M_blocks) continue;
        const int k_lo_block = blk * BLOCK;
        const bool is_diag = (blk == q_block_idx);

        #pragma unroll
        for (int inner = 0; inner < N_INNER; ++inner) {
            const int k_lo = k_lo_block + inner * K_TILE;

            // ── Load K tile [K_TILE, D_HEAD] into KV_sh via cp.async ──
            {
                const __nv_bfloat16 * Kp = K + (size_t)b * s_K_b + (size_t)kh * s_K_h;
                const bool vec_ok = (s_K_d == 1);
                if (vec_ok) {
                    int total8 = (K_TILE * D_HEAD) / 8;
                    for (int idx = tid; idx < total8; idx += NTHREADS) {
                        int row8 = idx / (D_HEAD / 8);
                        int d8   = idx - row8 * (D_HEAD / 8);
                        int j = k_lo + row8;
                        __nv_bfloat16 * dst = KV_sh + row8 * D_HEAD + d8 * 8;
                        if (j < seq_len) {
                            cp_async16(dst, Kp + (size_t)j * s_K_n + (size_t)(d8 * 8));
                        } else {
                            uint4 z = make_uint4(0, 0, 0, 0);
                            *reinterpret_cast<uint4*>(dst) = z;
                        }
                    }
                    cp_async_commit();
                    cp_async_wait_all();
                }
            }
            __syncthreads();

            // ── S = Q @ K^T in REGISTER fragments ──
            // Loop swap: dk-outer reuses Af across NNK nt-iters (saves 24/32 redundant Af SMEM reads).
            wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> S_frag[NNK];
            #pragma unroll
            for (int nt = 0; nt < NNK; ++nt) wmma::fill_fragment(S_frag[nt], 0.0f);
            {
                wmma::fragment<wmma::matrix_a,    MMA_M, MMA_N, MMA_K, __nv_bfloat16, wmma::row_major> Af;
                wmma::fragment<wmma::matrix_b,    MMA_M, MMA_N, MMA_K, __nv_bfloat16, wmma::col_major> Bf;
                #pragma unroll
                for (int dk = 0; dk < NDK; ++dk) {
                    wmma::load_matrix_sync(Af,
                        Q_sh + (size_t)(wid * MMA_M) * D_HEAD + dk * MMA_K, D_HEAD);
                    #pragma unroll
                    for (int nt = 0; nt < NNK; ++nt) {
                        wmma::load_matrix_sync(Bf,
                            KV_sh + (size_t)(nt * MMA_N) * D_HEAD + dk * MMA_K, D_HEAD);
                        wmma::mma_sync(S_frag[nt], Af, Bf, S_frag[nt]);
                    }
                }
            }

            // ── Apply causal/seq mask + per-lane rowmax for both row pairs owned ──
            // sm_8x mma.m16n16k16 acc layout: lane t holds 8 elems mapping to:
            //   e[0..3]: row = 2*(t/4)+0
            //   e[4..7]: row = 2*(t/4)+1
            //   e[0,1,4,5]: col_in_tile = 2*(t%4) + {0,1}
            //   e[2,3,6,7]: col_in_tile = 2*(t%4) + 8 + {0,1}
            const int q = lane >> 2;
            const int c = lane & 3;
            const int row0_in_warp = 2 * q + 0;
            const int row1_in_warp = 2 * q + 1;
            const int row0_g = q_tile_idx * Q_TILE + wid * MMA_M + row0_in_warp;
            const int row1_g = q_tile_idx * Q_TILE + wid * MMA_M + row1_in_warp;

            float lm0 = -INFINITY, lm1 = -INFINITY;
            #pragma unroll
            for (int nt = 0; nt < NNK; ++nt) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int col_pair = i & 1;
                    int col_off  = ((i >> 1) & 1) ? 8 : 0;
                    int col_in_tile = 2 * c + col_pair + col_off;
                    int col_g = k_lo + nt * MMA_N + col_in_tile;
                    int rg = (i < 4) ? row0_g : row1_g;
                    bool valid = (col_g < seq_len);
                    if (is_diag) valid = valid && (col_g <= rg);
                    if (!valid) S_frag[nt].x[i] = -INFINITY;
                    if (i < 4) lm0 = fmaxf(lm0, S_frag[nt].x[i]);
                    else       lm1 = fmaxf(lm1, S_frag[nt].x[i]);
                }
            }
            // Reduce rowmax across the 4 lanes of the quad (lanes with same q value).
            #pragma unroll
            for (int off = 1; off <= 2; off <<= 1) {
                lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, off));
                lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, off));
            }

            // Read m_old/l_old. Quad-leader (c==0) reads from SMEM; broadcast.
            float m_old0 = 0, m_old1 = 0, l_old0 = 0, l_old1 = 0;
            int row0_warp = wid * MMA_M + row0_in_warp;
            int row1_warp = wid * MMA_M + row1_in_warp;
            if (c == 0) {
                m_old0 = row_m[row0_warp];
                m_old1 = row_m[row1_warp];
                l_old0 = row_l[row0_warp];
                l_old1 = row_l[row1_warp];
            }
            int leader = lane & ~3;  // first lane of quad
            m_old0 = __shfl_sync(0xffffffff, m_old0, leader);
            m_old1 = __shfl_sync(0xffffffff, m_old1, leader);
            l_old0 = __shfl_sync(0xffffffff, l_old0, leader);
            l_old1 = __shfl_sync(0xffffffff, l_old1, leader);

            float m_new0 = fmaxf(m_old0, lm0);
            float m_new1 = fmaxf(m_old1, lm1);
            float alpha0 = (m_old0 == -INFINITY) ? 0.0f : exp2f(m_old0 - m_new0);
            float alpha1 = (m_old1 == -INFINITY) ? 0.0f : exp2f(m_old1 - m_new1);

            // Compute P and rowsum, write bf16 P to P_sh
            float rs0 = 0, rs1 = 0;
            #pragma unroll
            for (int nt = 0; nt < NNK; ++nt) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int col_pair = i & 1;
                    int col_off  = ((i >> 1) & 1) ? 8 : 0;
                    int col_in_tile = 2 * c + col_pair + col_off;
                    int col_in_KTILE = nt * MMA_N + col_in_tile;
                    float v = S_frag[nt].x[i];
                    float mn = (i < 4) ? m_new0 : m_new1;
                    float p = (v == -INFINITY) ? 0.0f : exp2f(v - mn);
                    if (i < 4) rs0 += p; else rs1 += p;
                    int p_row = wid * MMA_M + ((i < 4) ? row0_in_warp : row1_in_warp);
                    P_sh[(size_t)p_row * K_TILE + col_in_KTILE] = __float2bfloat16(p);
                }
            }
            // Reduce rowsum across quad
            #pragma unroll
            for (int off = 1; off <= 2; off <<= 1) {
                rs0 += __shfl_xor_sync(0xffffffff, rs0, off);
                rs1 += __shfl_xor_sync(0xffffffff, rs1, off);
            }

            // Quad-leader writes new state
            if (c == 0) {
                row_m[row0_warp] = m_new0;
                row_m[row1_warp] = m_new1;
                row_l[row0_warp] = alpha0 * l_old0 + rs0;
                row_l[row1_warp] = alpha1 * l_old1 + rs1;
            }

            // Frag-direct rescale of O accumulator
            #pragma unroll
            for (int d = 0; d < NDK; ++d) {
                O_frag[d].x[0] *= alpha0;
                O_frag[d].x[1] *= alpha0;
                O_frag[d].x[2] *= alpha0;
                O_frag[d].x[3] *= alpha0;
                O_frag[d].x[4] *= alpha1;
                O_frag[d].x[5] *= alpha1;
                O_frag[d].x[6] *= alpha1;
                O_frag[d].x[7] *= alpha1;
            }

            // Sync before V load (P_sh writes need to be visible too)
            __syncthreads();

            // ── Load V tile [K_TILE, D_HEAD] into KV_sh (overwrites K) via cp.async ──
            {
                const __nv_bfloat16 * Vp = V + (size_t)b * s_V_b + (size_t)kh * s_V_h;
                const bool vec_ok = (s_V_d == 1);
                if (vec_ok) {
                    int total8 = (K_TILE * D_HEAD) / 8;
                    for (int idx = tid; idx < total8; idx += NTHREADS) {
                        int row8 = idx / (D_HEAD / 8);
                        int d8   = idx - row8 * (D_HEAD / 8);
                        int j = k_lo + row8;
                        __nv_bfloat16 * dst = KV_sh + row8 * D_HEAD + d8 * 8;
                        if (j < seq_len) {
                            cp_async16(dst, Vp + (size_t)j * s_V_n + (size_t)(d8 * 8));
                        } else {
                            uint4 z = make_uint4(0, 0, 0, 0);
                            *reinterpret_cast<uint4*>(dst) = z;
                        }
                    }
                    cp_async_commit();
                    cp_async_wait_all();
                }
            }
            __syncthreads();

            // ── O += P @ V via WMMA ──
            // P shape [Q_TILE, K_TILE], V shape [K_TILE, D_HEAD]
            // Loop swap: kk-outer reuses Af across NDK dt-iters (saves redundant P_sh SMEM reads).
            {
                wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, __nv_bfloat16, wmma::row_major> Af;
                wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, __nv_bfloat16, wmma::row_major> Bf;
                #pragma unroll
                for (int kk = 0; kk < NNK; ++kk) {
                    wmma::load_matrix_sync(Af,
                        P_sh + (size_t)(wid * MMA_M) * K_TILE + kk * MMA_K, K_TILE);
                    #pragma unroll
                    for (int dt = 0; dt < NDK; ++dt) {
                        wmma::load_matrix_sync(Bf,
                            KV_sh + (size_t)(kk * MMA_K) * D_HEAD + dt * MMA_N, D_HEAD);
                        wmma::mma_sync(O_frag[dt], Af, Bf, O_frag[dt]);
                    }
                }
            }
            __syncthreads();
        } // inner
    } // it

    // Write O = acc / l_i. Store frag → SMEM (reuse Q_sh region as scratch), divide row-wise.
    // Q_sh is no longer needed; treat Q_sh region as f32 [Q_TILE][D_HEAD] (since 64*128*2 bf16 = 64*128*2 = 16K, but f32 needs 64*128*4 = 32K — won't fit). Use P_sh as scratch (8K, only first 4K rows needed for D=128 per warp store).
    // Simpler: store one D col tile at a time into a small scratch and write out.
    // We'll reuse the KV_sh region as f32 staging ([Q_TILE, MMA_N] = 64*16*4 = 4K) for one col tile at a time.
    float * stage_f32 = reinterpret_cast<float*>(KV_sh);  // 4 KB needed, KV_sh is 16 KB, plenty.
    #pragma unroll
    for (int d = 0; d < NDK; ++d) {
        __syncthreads();
        wmma::store_matrix_sync(stage_f32 + (size_t)(wid * MMA_M) * MMA_N,
                                O_frag[d], MMA_N, wmma::mem_row_major);
        __syncthreads();
        // Lanes 0..15 of each warp write 16 rows of MMA_N=16 D cols to global O.
        if (lane < MMA_M) {
            int row = wid * MMA_M + lane;
            int q_global = q_tile_idx * Q_TILE + row;
            if (q_global < seq_len) {
                __nv_bfloat16 * Op = O + (size_t)b * s_O_b + (size_t)q_global * s_O_n + (size_t)qh * s_O_h;
                int row_warp = wid * MMA_M + lane;
                float l = row_l[row_warp];
                float l_rec = (l > 0.0f) ? (1.0f / l) : 1.0f;
                const float * srow = stage_f32 + (size_t)row_warp * MMA_N;
                #pragma unroll
                for (int dd = 0; dd < MMA_N; ++dd) {
                    Op[(size_t)(d * MMA_N + dd) * s_O_d] = __float2bfloat16(srow[dd] * l_rec);
                }
            }
        }
    }
}
#else
// HIP tiled prototype: exact selected-block attention with wave-parallel query
// rows and shared-memory tiled K/V loads. Not used by the default launcher yet.
template <int Q_ROWS, int K_TILE, int BLOCK, int D_HEAD, int LANES>
__global__ void sparse_flash_forward_kernel_ref_bf16(
    const __nv_bfloat16 * __restrict__ Q,
    const __nv_bfloat16 * __restrict__ K,
    const __nv_bfloat16 * __restrict__ V,
    __nv_bfloat16       * __restrict__ O,
    const int32_t       * __restrict__ block_index,
    const int32_t       * __restrict__ counts,
    float scale,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int M_blocks,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_V_b, int s_V_n, int s_V_h, int s_V_d,
    int s_O_b, int s_O_n, int s_O_h, int s_O_d,
    int s_idx_b, int s_idx_m, int s_idx_n, int s_idx_h,
    int s_cnt_b, int s_cnt_m, int s_cnt_h)
{
    const int q_tile_idx = blockIdx.x;
    const int zh = blockIdx.y;
    const int b  = zh / n_q_heads;
    const int qh = zh % n_q_heads;
    if (b >= batch) return;

    const int tid = threadIdx.x;
    const int wave = tid / LANES;
    const int lane = tid - wave * LANES;
    if (wave >= Q_ROWS) return;

    const int q_global = q_tile_idx * Q_ROWS + wave;
    const bool active_q = q_global < seq_len;
    const int q_eff = active_q ? q_global : (seq_len - 1);
    const int q_tile_last = min(q_tile_idx * Q_ROWS + Q_ROWS - 1, seq_len - 1);

    const int kh = qh * n_k_heads / n_q_heads;
    const int q_block_idx = q_eff / BLOCK;
    const int hi = counts[(size_t)b * s_cnt_b + (size_t)q_block_idx * s_cnt_m + (size_t)qh * s_cnt_h];

    const __nv_bfloat16 * Qp = Q + (size_t)b * s_Q_b + (size_t)q_eff * s_Q_n + (size_t)qh * s_Q_h;
    __nv_bfloat16 * Op = O + (size_t)b * s_O_b + (size_t)q_eff * s_O_n + (size_t)qh * s_O_h;

    extern __shared__ unsigned char raw_smem[];
    __nv_bfloat16 * K_sh = reinterpret_cast<__nv_bfloat16 *>(raw_smem);
    __nv_bfloat16 * V_sh = K_sh + (size_t)K_TILE * D_HEAD;
    float * dots_sh = reinterpret_cast<float *>(V_sh + (size_t)K_TILE * D_HEAD);
    float * bcast_sh = dots_sh + (size_t)Q_ROWS * K_TILE;

    constexpr int D_PER_LANE = D_HEAD / LANES;
    float q_local[D_PER_LANE];
    float acc_local[D_PER_LANE];
    #pragma unroll
    for (int i = 0; i < D_PER_LANE; ++i) {
        const int d = lane + i * LANES;
        q_local[i] = __bfloat162float(Qp[(size_t)d * s_Q_d]);
        acc_local[i] = 0.0f;
    }

    float m_old = -INFINITY;
    float l_old = 0.0f;

    auto warp_sum = [&](float v) {
        for (int off = LANES / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, LANES);
        }
        return v;
    };

    for (int it = 0; it < hi; ++it) {
        const int blk = block_index[(size_t)b * s_idx_b + (size_t)q_block_idx * s_idx_m
                                    + (size_t)it * s_idx_n + (size_t)qh * s_idx_h];
        if (blk < 0 || blk >= M_blocks) continue;

        int k_lo = blk * BLOCK;
        int k_hi = k_lo + BLOCK;
        if (k_hi > seq_len) k_hi = seq_len;
        if (blk == q_block_idx && k_hi > q_eff + 1) k_hi = q_eff + 1;
        if (k_lo >= k_hi) continue;

        for (int tile_start = k_lo; tile_start < k_hi; tile_start += K_TILE) {
            const int tile_len = min(K_TILE, k_hi - tile_start);
            int load_hi = blk * BLOCK + BLOCK;
            if (load_hi > seq_len) load_hi = seq_len;
            if (blk == q_block_idx && load_hi > q_tile_last + 1) load_hi = q_tile_last + 1;
            const int load_len = min(K_TILE, max(0, load_hi - tile_start));

            for (int idx = tid; idx < load_len * D_HEAD; idx += blockDim.x) {
                const int kk = idx / D_HEAD;
                const int dd = idx - kk * D_HEAD;
                const int pos = tile_start + kk;
                const __nv_bfloat16 * Kp_g = K + (size_t)b * s_K_b + (size_t)pos * s_K_n + (size_t)kh * s_K_h;
                const __nv_bfloat16 * Vp_g = V + (size_t)b * s_V_b + (size_t)pos * s_V_n + (size_t)kh * s_V_h;
                K_sh[idx] = Kp_g[(size_t)dd * s_K_d];
                V_sh[idx] = Vp_g[(size_t)dd * s_V_d];
            }
            __syncthreads();

            float m_blk = -INFINITY;
            for (int kk = 0; kk < tile_len; ++kk) {
                float partial = 0.0f;
                #pragma unroll
                for (int i = 0; i < D_PER_LANE; ++i) {
                    const int d = lane + i * LANES;
                    partial += q_local[i] * __bfloat162float(K_sh[(size_t)kk * D_HEAD + d]);
                }
                float dot = warp_sum(partial);
                if (lane == 0) {
                    dot *= scale;
                    dots_sh[wave * K_TILE + kk] = dot;
                    m_blk = fmaxf(m_blk, dot);
                }
            }
            if (lane == 0) {
                for (int kk = 0; kk < tile_len; ++kk) {
                    m_blk = fmaxf(m_blk, dots_sh[wave * K_TILE + kk]);
                }
            }
            if (lane == 0) bcast_sh[wave] = m_blk;
            __syncthreads();
            m_blk = bcast_sh[wave];

            const float m_new = fmaxf(m_old, m_blk);
            const float alpha = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
            #pragma unroll
            for (int i = 0; i < D_PER_LANE; ++i) {
                acc_local[i] *= alpha;
            }

            float local_l = 0.0f;
            for (int kk = lane; kk < tile_len; kk += LANES) {
                local_l += expf(dots_sh[wave * K_TILE + kk] - m_new);
            }
            float l_sum = warp_sum(local_l);
            if (lane == 0) bcast_sh[Q_ROWS + wave] = l_sum;
            __syncthreads();
            l_sum = bcast_sh[Q_ROWS + wave];

            for (int kk = 0; kk < tile_len; ++kk) {
                const float p = expf(dots_sh[wave * K_TILE + kk] - m_new);
                #pragma unroll
                for (int i = 0; i < D_PER_LANE; ++i) {
                    const int d = lane + i * LANES;
                    acc_local[i] += p * __bfloat162float(V_sh[(size_t)kk * D_HEAD + d]);
                }
            }

            m_old = m_new;
            l_old = alpha * l_old + l_sum;
            __syncthreads();
        }
    }

    const float l_rec = (l_old > 0.0f) ? (1.0f / l_old) : 1.0f;
    #pragma unroll
    for (int i = 0; i < D_PER_LANE; ++i) {
        const int d = lane + i * LANES;
        if (active_q) {
            Op[(size_t)d * s_O_d] = __float2bfloat16(acc_local[i] * l_rec);
        }
    }
}

// Conservative HIP fallback: one CTA computes one query row. This avoids the
// fragile cross-lane bookkeeping in the tiled prototype and gives ROCm a
// numerically exact baseline while the fast MFMA/rocWMMA path is still pending.
template <int BLOCK, int D_HEAD>
__global__ void sparse_flash_forward_kernel_row_bf16(
    const __nv_bfloat16 * __restrict__ Q,
    const __nv_bfloat16 * __restrict__ K,
    const __nv_bfloat16 * __restrict__ V,
    __nv_bfloat16       * __restrict__ O,
    const int32_t       * __restrict__ block_index,
    const int32_t       * __restrict__ counts,
    float scale,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int M_blocks,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_V_b, int s_V_n, int s_V_h, int s_V_d,
    int s_O_b, int s_O_n, int s_O_h, int s_O_d,
    int s_idx_b, int s_idx_m, int s_idx_n, int s_idx_h,
    int s_cnt_b, int s_cnt_m, int s_cnt_h)
{
    const int q_global = blockIdx.x;
    const int zh = blockIdx.y;
    const int b  = zh / n_q_heads;
    const int qh = zh % n_q_heads;
    const int d = threadIdx.x;
    if (b >= batch || q_global >= seq_len || d >= D_HEAD) return;

    const int kh = qh * n_k_heads / n_q_heads;
    const int q_block_idx = q_global / BLOCK;
    const int hi = counts[(size_t)b * s_cnt_b + (size_t)q_block_idx * s_cnt_m + (size_t)qh * s_cnt_h];

    const __nv_bfloat16 * Qp = Q + (size_t)b * s_Q_b + (size_t)q_global * s_Q_n + (size_t)qh * s_Q_h;
    __nv_bfloat16 * Op = O + (size_t)b * s_O_b + (size_t)q_global * s_O_n + (size_t)qh * s_O_h;

    extern __shared__ float smem[];
    float * red = smem;          // D_HEAD floats
    float * scalars = red + D_HEAD; // alpha, p, l_rec

    const float qd = __bfloat162float(Qp[(size_t)d * s_Q_d]);
    float acc = 0.0f;
    float m_old = -INFINITY;
    float l_old = 0.0f;

    for (int it = 0; it < hi; ++it) {
        const int blk = block_index[(size_t)b * s_idx_b + (size_t)q_block_idx * s_idx_m
                                    + (size_t)it * s_idx_n + (size_t)qh * s_idx_h];
        if (blk < 0 || blk >= M_blocks) continue;

        int k_lo = blk * BLOCK;
        int k_hi = k_lo + BLOCK;
        if (k_hi > seq_len) k_hi = seq_len;
        if (blk == q_block_idx && k_hi > q_global + 1) k_hi = q_global + 1;

        for (int pos = k_lo; pos < k_hi; ++pos) {
            const __nv_bfloat16 * Kp_g = K + (size_t)b * s_K_b + (size_t)pos * s_K_n + (size_t)kh * s_K_h;
            red[d] = qd * __bfloat162float(Kp_g[(size_t)d * s_K_d]);
            __syncthreads();

            for (int stride = D_HEAD / 2; stride > 0; stride >>= 1) {
                if (d < stride) red[d] += red[d + stride];
                __syncthreads();
            }

            if (d == 0) {
                const float dot = red[0] * scale;
                const float m_new = fmaxf(m_old, dot);
                const float alpha = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
                const float p = expf(dot - m_new);
                m_old = m_new;
                l_old = alpha * l_old + p;
                scalars[0] = alpha;
                scalars[1] = p;
            }
            __syncthreads();

            const __nv_bfloat16 * Vp_g = V + (size_t)b * s_V_b + (size_t)pos * s_V_n + (size_t)kh * s_V_h;
            acc = acc * scalars[0] + scalars[1] * __bfloat162float(Vp_g[(size_t)d * s_V_d]);
        }
    }

    if (d == 0) scalars[2] = (l_old > 0.0f) ? (1.0f / l_old) : 1.0f;
    __syncthreads();
    Op[(size_t)d * s_O_d] = __float2bfloat16(acc * scalars[2]);
}
#endif
// ---- Kernel 4-TC: tensor-core sparse_flash_forward (WMMA)  [SCAFFOLD] ----
//
// FlashAttention-2 style with NVIDIA WMMA fragments. Scaffolding only —
// the online softmax l_i bookkeeping has a known bug (alpha rescale stash
// overwrites l_i, so partial sums lose the previous-block contribution).
// Kept here as a starting point for the proper rewrite. NOT compiled
// (gated out below) and NOT called from the launcher.
//
// To finish: maintain m_i, l_i, and alpha in three separate shared arrays;
// rescale acc fragments via PTX-correct per-thread fragment elementwise
// scaling (or via the store/scale/reload trick) BEFORE accumulating new PV
// contributions; ensure l_new = alpha * l_old + sum(P_new) per row.

extern "C" void launch_sparse_flash_forward_bf16(
    const void * Q, const void * K, const void * V, void * O,
    const int32_t * block_index, const int32_t * counts,
    float scale,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int head_dim, int q_tile, int block_size,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_V_b, int s_V_n, int s_V_h, int s_V_d,
    int s_O_b, int s_O_n, int s_O_h, int s_O_d,
    int s_idx_b, int s_idx_m, int s_idx_n, int s_idx_h,
    int s_cnt_b, int s_cnt_m, int s_cnt_h,
    cudaStream_t stream)
{
    const int M = (seq_len + block_size - 1) / block_size;
    const int q_tiles = (seq_len + q_tile - 1) / q_tile;
    dim3 grid(q_tiles, batch * n_q_heads, 1);
#if defined(DFLASH27B_USE_HIP)
    if (q_tile == 64 && block_size == 128 && head_dim == 128) {
        if (std::getenv("DFLASH_FP_HIP_ROW") == nullptr) {
            constexpr int HIP_WAVE = 32;
            constexpr int HIP_Q_ROWS = 4;
            constexpr int HIP_K_TILE = 32;
            dim3 tiled_grid((seq_len + HIP_Q_ROWS - 1) / HIP_Q_ROWS, batch * n_q_heads, 1);
            dim3 tiled_block(HIP_WAVE * HIP_Q_ROWS, 1, 1);
            size_t tiled_smem = sizeof(__nv_bfloat16) * HIP_K_TILE * head_dim * 2
                              + sizeof(float) * HIP_Q_ROWS * (HIP_K_TILE + 2);
            sparse_flash_forward_kernel_ref_bf16<HIP_Q_ROWS, HIP_K_TILE, 128, 128, HIP_WAVE><<<tiled_grid, tiled_block, tiled_smem, stream>>>(
                (const __nv_bfloat16 *)Q, (const __nv_bfloat16 *)K,
                (const __nv_bfloat16 *)V, (__nv_bfloat16 *)O,
                block_index, counts, scale,
                batch, n_q_heads, n_k_heads, seq_len, M,
                s_Q_b, s_Q_n, s_Q_h, s_Q_d,
                s_K_b, s_K_n, s_K_h, s_K_d,
                s_V_b, s_V_n, s_V_h, s_V_d,
                s_O_b, s_O_n, s_O_h, s_O_d,
                s_idx_b, s_idx_m, s_idx_n, s_idx_h,
                s_cnt_b, s_cnt_m, s_cnt_h);
        } else {
            dim3 row_grid(seq_len, batch * n_q_heads, 1);
            dim3 row_block(128, 1, 1);
            size_t smem_bytes = sizeof(float) * (head_dim + 3);
            sparse_flash_forward_kernel_row_bf16<128, 128><<<row_grid, row_block, smem_bytes, stream>>>(
                (const __nv_bfloat16 *)Q, (const __nv_bfloat16 *)K,
                (const __nv_bfloat16 *)V, (__nv_bfloat16 *)O,
                block_index, counts, scale,
                batch, n_q_heads, n_k_heads, seq_len, M,
                s_Q_b, s_Q_n, s_Q_h, s_Q_d,
                s_K_b, s_K_n, s_K_h, s_K_d,
                s_V_b, s_V_n, s_V_h, s_V_d,
                s_O_b, s_O_n, s_O_h, s_O_d,
                s_idx_b, s_idx_m, s_idx_n, s_idx_h,
                s_cnt_b, s_cnt_m, s_cnt_h);
        }
    }
#else
    if (q_tile == 64 && block_size == 128 && head_dim == 128) {
        // FA-2 register-resident layout @ Q_TILE=64 (4 warps, 128 threads), 2 CTAs/SM.
        constexpr int Q_TILE = 64, K_TILE = 64, BLOCK = 128, D_HEAD = 128;
        size_t smem_bytes = sizeof(__nv_bfloat16) * (Q_TILE * D_HEAD)
                           + sizeof(__nv_bfloat16) * (K_TILE * D_HEAD)
                           + sizeof(__nv_bfloat16) * (Q_TILE * K_TILE)
                           + sizeof(float)         * (2 * Q_TILE);
        dim3 block128(128, 1, 1);
        cudaFuncSetAttribute(
            (const void*)sparse_flash_forward_kernel_bf16<Q_TILE, K_TILE, BLOCK, D_HEAD>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes);
        sparse_flash_forward_kernel_bf16<Q_TILE, K_TILE, BLOCK, D_HEAD><<<grid, block128, smem_bytes, stream>>>(
            (const __nv_bfloat16 *)Q, (const __nv_bfloat16 *)K,
            (const __nv_bfloat16 *)V, (__nv_bfloat16 *)O,
            block_index, counts, scale,
            batch, n_q_heads, n_k_heads, seq_len, M,
            s_Q_b, s_Q_n, s_Q_h, s_Q_d,
            s_K_b, s_K_n, s_K_h, s_K_d,
            s_V_b, s_V_n, s_V_h, s_V_d,
            s_O_b, s_O_n, s_O_h, s_O_d,
            s_idx_b, s_idx_m, s_idx_n, s_idx_h,
            s_cnt_b, s_cnt_m, s_cnt_h);
    }
#endif
}


// ---- Kernel 3: block_select on GPU ----
//
// One warp per (B, M, H). Each warp scans n in [0, m] in chunks of 32, takes
// the max of scores[b,m,n,h] for the threshold, then re-scans applying the
// keep predicate (sink | window | last_n_full | score >= max*alpha). Surviving
// indices are compacted via warp ballot + popc so the output stays sorted by n.
//
// Replaces flashprefill_select.cpp::block_select_host. Removes the per-call
// D2H + host loop + H2D round trip (~75 MB at 140K, ~1.5 ms steady-state plus
// PCIe latency).

__global__ void block_select_kernel(
    const float * __restrict__ score,
    int B, int M, int N, int H,
    int attention_sink, int window, int last_n_full, float alpha,
    int s_b, int s_m, int s_n, int s_h,
    int idx_s_b, int idx_s_m, int idx_s_n, int idx_s_h,
    int cnt_s_b, int cnt_s_m, int cnt_s_h,
    int32_t * __restrict__ idx_out,
    int32_t * __restrict__ cnt_out)
{
    const int b = blockIdx.x;
    const int m = blockIdx.y;
    const int h = blockIdx.z;
    const int lane = threadIdx.x;  // 0..31, single warp CTA

    if (b >= B || m >= M || h >= H) return;

    const float * sp = score + (size_t)b*s_b + (size_t)m*s_m + (size_t)h*s_h;
    int32_t * idxp = idx_out + (size_t)b*idx_s_b + (size_t)m*idx_s_m + (size_t)h*idx_s_h;

    const bool last_full = (m >= M - last_n_full);
    const float NEG_INF = -INFINITY;

    // Pass 1: max score in [0, m] (warp reduce).
    float local_max = NEG_INF;
    for (int n_base = 0; n_base <= m; n_base += 32) {
        int n = n_base + lane;
        bool valid = (n <= m);
        float v = valid ? sp[(size_t)n * s_n] : NEG_INF;
        local_max = fmaxf(local_max, v);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off, 32));
    const float max_score = local_max;
    const float thresh = max_score * alpha;

    // Pass 2: predicate + warp-ballot compact, sorted by n.
    int total = 0;
    for (int n_base = 0; n_base <= m; n_base += 32) {
        int n = n_base + lane;
        bool valid = (n <= m);
        bool keep = false;
        if (valid) {
            float v = sp[(size_t)n * s_n];
            keep = last_full
                || (n < attention_sink)
                || ((m - n) < window)
                || (v >= thresh);
        }
        unsigned mask = __ballot_sync(0xffffffff, keep);
        int rank = __popc(mask & ((1u << lane) - 1u));
        if (keep) {
            idxp[(size_t)(total + rank) * idx_s_n] = (int32_t)n;
        }
        total += __popc(mask);
    }

    // Tail pad with -1 across [total, N).
    for (int n = total + lane; n < N; n += 32) {
        idxp[(size_t)n * idx_s_n] = (int32_t)-1;
    }

    if (lane == 0) {
        cnt_out[(size_t)b*cnt_s_b + (size_t)m*cnt_s_m + (size_t)h*cnt_s_h] = (int32_t)total;
    }
}

extern "C" void launch_block_select(
    const float * score,
    int B, int M, int N, int H,
    int attention_sink, int window, int last_n_full, float alpha,
    int s_b, int s_m, int s_n, int s_h,
    int idx_s_b, int idx_s_m, int idx_s_n, int idx_s_h,
    int cnt_s_b, int cnt_s_m, int cnt_s_h,
    int32_t * idx_out, int32_t * cnt_out,
    cudaStream_t stream)
{
    dim3 grid(B, M, H);
    dim3 block(32, 1, 1);
    block_select_kernel<<<grid, block, 0, stream>>>(
        score, B, M, N, H,
        attention_sink, window, last_n_full, alpha,
        s_b, s_m, s_n, s_h,
        idx_s_b, idx_s_m, idx_s_n, idx_s_h,
        cnt_s_b, cnt_s_m, cnt_s_h,
        idx_out, cnt_out);
}

} // namespace flashprefill
} // namespace dflash27b
