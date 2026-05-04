/**
 * BF16 Prefill Megakernel for NVIDIA B200 (sm_100).
 *
 * One `cudaLaunchCooperativeKernel` dispatch. All 24 layers, all S tokens,
 * processed in a single persistent kernel with `cg::this_grid().sync()`
 * between phases. No cuBLAS, no intermediate launches, no host round-trips.
 *
 * Matmuls use WMMA (wraps `mma.sync` under the hood) with bf16 operands and
 * f32 accumulate. Block tile [BTM=32, BTN=128]; warp grid 2×8. Work is
 * distributed across the 148 persistent blocks via cyclic tile assignment.
 *
 * The DeltaNet recurrence keeps its 16-block-per-layer structure from the
 * pre-mega version (hard serial dep over t) but sits inside the persistent
 * kernel so there's no per-layer relaunch.
 *
 * Assumes S is a multiple of 32 (caller pads).
 *
 * Only compiled for Blackwell (sm_120/sm_121a). Excluded from the sm_86
 * RTX 3090 build; see setup.py.
 */

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1200
#error "prefill_megakernel.cu requires CUDA arch >= sm_120 (Blackwell)"
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <mma.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;
using namespace nvcuda;

// ===== Model constants =====
constexpr int HIDDEN = 1024;
constexpr int INTER = 3584;
constexpr int VOCAB = 248320;
constexpr int NUM_LAYERS = 24;
constexpr float RMS_EPS = 1e-6f;
constexpr int MAX_SEQ = 2048;

constexpr int FA_Q_HEADS = 8;
constexpr int FA_KV_HEADS = 2;
constexpr int FA_HEAD_DIM = 256;
constexpr int FA_GQA = FA_Q_HEADS / FA_KV_HEADS;
constexpr int FA_Q_SIZE = FA_Q_HEADS * FA_HEAD_DIM;
constexpr int FA_QPROJ_SIZE = FA_Q_SIZE * 2;
constexpr int FA_KV_SIZE = FA_KV_HEADS * FA_HEAD_DIM;
constexpr int FA_ROT_DIM = 64;
constexpr float FA_ROPE_THETA = 10000000.0f;

constexpr int DN_HEADS = 16;
constexpr int DN_KEY = 128;
constexpr int DN_VAL = 128;
constexpr int DN_CONV_K = 4;
constexpr int DN_QK_SIZE = DN_HEADS * DN_KEY;
constexpr int DN_V_SIZE = DN_HEADS * DN_VAL;
constexpr int DN_CONV_CH = DN_QK_SIZE * 2 + DN_V_SIZE;  // 6144

__device__ __constant__ int MEGA_LAYER_TYPE[NUM_LAYERS] = {
    0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1
};

struct MegaLayerWeights {
    int layer_type;
    int _pad[3];
    void *ptrs[14];
};

// ===== Kernel tile sizes =====
#define MEGA_BLOCK_SIZE 512
constexpr int MEGA_WARPS = MEGA_BLOCK_SIZE / 32;

constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;
// Block tile [32, 128] with 2×8 warp grid; each warp produces a [16, 16]
// output. K is processed in BTK=64 chunks, double-buffered in shared
// memory via cp.async so the next chunk streams in while we MMA on the
// current one. This keeps parallelism high for small-N matmuls (down,
// out, z projections) while letting the async loader hide global latency.
constexpr int BTM = 32;
constexpr int BTN = 128;
constexpr int BTK = 64;                // k-chunk size (double buffered)
constexpr int N_PER_WARP = 1;          // 16-col WMMA tile per warp
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 8;
static_assert(WARPS_M * WARPS_N == MEGA_WARPS, "warp grid must match block");
static_assert(WARPS_N * WN * N_PER_WARP == BTN, "warp N coverage mismatch");
static_assert(WARPS_M * WM == BTM, "warp M coverage mismatch");

constexpr int LM_BLOCKS_MEGA = 1024;

// ===== Small helpers =====
__device__ __forceinline__ float mega_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float mega_silu(float x) { return x / (1.0f + expf(-x)); }

// cp.async.cg: copy 16 bytes global → shared, bypassing L1 (goes through L2).
// The copy is asynchronous; producer threads must issue cp.async.commit_group
// and consumers must cp.async.wait_group before reading shared.
__device__ __forceinline__ void cp_async_16(void *dst_smem, const void *src_gmem) {
    unsigned int smem_int = __cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_int), "l"(src_gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

// ===== Phase: Embedding =====
// hidden[s*H + i] = embed[ids[s]*H + i]
__device__ void phase_embed(const int *ids, const __nv_bfloat16 *embed,
                            __nv_bfloat16 *hidden, int S) {
    int total = S * HIDDEN;
    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int s = idx / HIDDEN;
        int i = idx - s * HIDDEN;
        hidden[idx] = embed[ids[s] * HIDDEN + i];
    }
}

// ===== Phase: RMSNorm =====
// normalized[s, :] = (input[s, :] * rsqrt(mean(input[s, :]^2) + eps) * (1 + w)
// Also copies input -> residual.
// Rows distributed cyclically across blocks.
__device__ void phase_rmsnorm(const __nv_bfloat16 *input, const __nv_bfloat16 *w,
                              __nv_bfloat16 *output, __nv_bfloat16 *residual,
                              int S, int D) {
    int tid = threadIdx.x, wid = tid / 32, lid = tid % 32;
    __shared__ float smem[MEGA_WARPS];

    for (int s = blockIdx.x; s < S; s += gridDim.x) {
        const __nv_bfloat16 *ri = input + s * D;
        __nv_bfloat16 *ro = output + s * D;
        __nv_bfloat16 *rr = residual + s * D;

        float sq = 0;
        for (int i = tid; i < D; i += blockDim.x) {
            float v = __bfloat162float(ri[i]);
            rr[i] = ri[i];
            sq += v * v;
        }
        sq = mega_warp_sum(sq);
        if (lid == 0) smem[wid] = sq;
        __syncthreads();
        if (wid == 0) {
            float v = (lid < MEGA_WARPS) ? smem[lid] : 0;
            v = mega_warp_sum(v);
            if (lid == 0) smem[0] = rsqrtf(v / D + RMS_EPS);
        }
        __syncthreads();
        float rstd = smem[0];
        for (int i = tid; i < D; i += blockDim.x) {
            float v = __bfloat162float(ri[i]) * rstd * (1.0f + __bfloat162float(w[i]));
            ro[i] = __float2bfloat16(v);
        }
        __syncthreads();
    }
}

// ===== Phase: bf16 matmul C[M, N] = A[M, K] @ W[N, K]^T =====
// Requires M and N multiples of BTM and BTN respectively, K multiple of WK.
// Each block owns a [BTM, BTN] output tile; warps within a block tile
// [WARPS_M, WARPS_N]×[WM, WN].
// ===== Pipelined double-buffered BF16 GEMM =====
//
// C[M, N] = A[M, K] × W[N, K]^T  (W stored row-major as [N, K]).
//
// Per block:
//   Block tile   : [BTM=32, BTN=256]
//   Warp grid    : 2 × 8
//   Per-warp out : [16, 32] via 2× WMMA m16n16k16 accumulators
//   K-chunk      : BTK=64 (double-buffered with cp.async)
//
// Shared memory layout (dynamic):
//   sA[2][BTM][BTK]  : double-buffered A staging         (2*32*64*2 =  8 KB)
//   sB[2][BTN][BTK]  : double-buffered B staging         (2*256*64*2= 64 KB)
//   sC[BTM][BTN]     : f32 output accumulator staging    (32*256*4 = 32 KB)
// Total: 104 KB dynamic shared per block (B200 carveout = 228 KB, ok).
//
// Threads-per-tile work:
//   - A chunk is BTM*BTK = 32*64 = 2048 bf16 = 4 KB per buffer.
//     Each thread copies 8 bf16 (16 B) per cp.async, so 2048/8 = 256 copies
//     across 512 threads = ½ pass. Threads >= 256 skip the A load.
//   - B chunk is BTN*BTK = 256*64 = 16384 bf16 = 32 KB per buffer.
//     512 threads × 8 bf16 = 4096 bf16 per pass → 4 passes.

__device__ void phase_matmul_bf16(const __nv_bfloat16 *A, const __nv_bfloat16 *W,
                                  __nv_bfloat16 *C, int M, int N, int K) {
    extern __shared__ __align__(16) char smem_raw[];
    __nv_bfloat16 *sA = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
    __nv_bfloat16 *sB = sA + 2 * BTM * BTK;
    float *tile_f32 = reinterpret_cast<float *>(sB + 2 * BTN * BTK);

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int wm = warp_id / WARPS_N;
    int wn = warp_id % WARPS_N;

    int num_m_tiles = (M + BTM - 1) / BTM;
    int num_n_tiles = (N + BTN - 1) / BTN;
    int total_tiles = num_m_tiles * num_n_tiles;

    // K must be a multiple of BTK for the pipeline to work cleanly.
    // All our Ks (1024, 2048, 3584) satisfy K % 64 == 0.
    int num_k_chunks = K / BTK;

    // Lambda: asynchronously stage one (BM, BTK) tile of A or B from global
    // into the `buf`-th shared buffer, using cp.async.cg (bypasses L1).
    auto issue_a_chunk = [&] (int buf, int m_start, int k_start) {
        const __nv_bfloat16 *src_base = A + m_start * K + k_start;
        __nv_bfloat16 *dst_base = sA + buf * BTM * BTK;
        // 2048 bf16 elements = 256 cp.async calls at 8 bf16 each.
        // Use 256 threads (half the block). Each thread does 1 call.
        if (tid < 256) {
            int r = tid >> 3;            // 0..31 rows
            int c = (tid & 7) * 8;        // col start within BTK
            cp_async_16(dst_base + r * BTK + c,
                         src_base + r * K + c);
        }
    };

    auto issue_b_chunk = [&] (int buf, int n_start, int k_start) {
        // W is [N, K] row-major; our tile is rows [n_start, n_start+BTN)
        // and cols [k_start, k_start+BTK). In shared we lay it out as
        // [BTN, BTK] so WMMA col_major loads get the correct N-fragment.
        const __nv_bfloat16 *src_base = W + n_start * K + k_start;
        __nv_bfloat16 *dst_base = sB + buf * BTN * BTK;
        // BTN * BTK / 8 = 128*64/8 = 1024 cp.async calls; 512 threads × 2.
        #pragma unroll
        for (int it = 0; it < 2; it++) {
            int idx = tid + it * blockDim.x;
            int r = idx >> 3;         // 0..127 rows
            int c = (idx & 7) * 8;     // col start within BTK
            if (r < BTN) {
                cp_async_16(dst_base + r * BTK + c,
                             src_base + r * K + c);
            }
        }
    };

    for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        int mt = tile / num_n_tiles;
        int nt = tile % num_n_tiles;
        int m_start = mt * BTM;
        int n_start = nt * BTN;
        bool m_in_bounds = (m_start < M);
        bool n_in_bounds = (n_start < N);

        // Per-warp accumulators: 2 × [16×16] f32 WMMA fragments.
        wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag[N_PER_WARP];
        #pragma unroll
        for (int n = 0; n < N_PER_WARP; n++) wmma::fill_fragment(c_frag[n], 0.0f);

        // Prefetch chunk 0 into buffer 0.
        if (m_in_bounds && n_in_bounds) {
            issue_a_chunk(0, m_start, 0);
            issue_b_chunk(0, n_start, 0);
        }
        cp_async_commit();

        int warp_row_in_block = wm * WM;
        int warp_col_in_block = wn * WN * N_PER_WARP;

        for (int ck = 0; ck < num_k_chunks; ck++) {
            int cur = ck & 1;
            int nxt = cur ^ 1;

            // Prefetch next chunk
            if (ck + 1 < num_k_chunks) {
                if (m_in_bounds && n_in_bounds) {
                    issue_a_chunk(nxt, m_start, (ck + 1) * BTK);
                    issue_b_chunk(nxt, n_start, (ck + 1) * BTK);
                }
                cp_async_commit();
                cp_async_wait_group<1>();
            } else {
                cp_async_wait_group<0>();
            }
            __syncthreads();

            if (m_in_bounds && n_in_bounds) {
                const __nv_bfloat16 *sA_buf = sA + cur * BTM * BTK;
                const __nv_bfloat16 *sB_buf = sB + cur * BTN * BTK;

                // MMA over this chunk: BTK/WK = 4 k-steps, each with 1 A load
                // + N_PER_WARP B loads + N_PER_WARP MMAs.
                #pragma unroll
                for (int ki = 0; ki < BTK; ki += WK) {
                    wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::load_matrix_sync(a_frag, sA_buf + warp_row_in_block * BTK + ki, BTK);

                    #pragma unroll
                    for (int n = 0; n < N_PER_WARP; n++) {
                        int col_in_block = warp_col_in_block + n * WN;
                        wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b_frag;
                        wmma::load_matrix_sync(b_frag, sB_buf + col_in_block * BTK + ki, BTK);
                        wmma::mma_sync(c_frag[n], a_frag, b_frag, c_frag[n]);
                    }
                }
            }
        }

        // Epilogue: store each accumulator to its slot in the f32 tile.
        if (m_in_bounds && n_in_bounds) {
            #pragma unroll
            for (int n = 0; n < N_PER_WARP; n++) {
                int col_in_block = warp_col_in_block + n * WN;
                wmma::store_matrix_sync(
                    tile_f32 + warp_row_in_block * BTN + col_in_block,
                    c_frag[n], BTN, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // f32 → bf16 cast and write to global, respecting M/N bounds.
        if (m_in_bounds && n_in_bounds) {
            for (int i = threadIdx.x; i < BTM * BTN; i += blockDim.x) {
                int mi = i / BTN, ni = i % BTN;
                int gm = m_start + mi, gn = n_start + ni;
                if (gm < M && gn < N) {
                    C[gm * N + gn] = __float2bfloat16(tile_f32[i]);
                }
            }
        }
        __syncthreads();
    }
}

// Same as above but output is f32 (for beta/alpha which feed the recurrence as f32).
// (unused in the current layer schedule — beta/alpha are done via
// phase_matvec_small_to_f32 below since N=16 which doesn't fit WMMA 16×16
// output fragments cleanly without padding.)

// ===== Phase: tiny matvec for beta/alpha (N=16) =====
// Output is f32.
__device__ void phase_matvec_small_to_f32(const __nv_bfloat16 *in, const __nv_bfloat16 *w,
                                          float *out, int S, int K, int N) {
    // One (s, n) per warp, cyclic assignment.
    int warps_per_block = blockDim.x / 32;
    int global_warp = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    int total_warps = gridDim.x * warps_per_block;
    int total = S * N;
    int lid = threadIdx.x & 31;

    for (int idx = global_warp; idx < total; idx += total_warps) {
        int s = idx / N, n = idx - s * N;
        const __nv_bfloat16 *ir = in + s * K;
        const __nv_bfloat16 *wr = w + n * K;
        float sum = 0;
        for (int k = lid; k < K; k += 32) {
            sum += __bfloat162float(ir[k]) * __bfloat162float(wr[k]);
        }
        sum = mega_warp_sum(sum);
        if (lid == 0) out[idx] = sum;
    }
}

// ===== Phase: elementwise bf16 residual add =====
__device__ void phase_add_residual(const __nv_bfloat16 *a, const __nv_bfloat16 *b,
                                   __nv_bfloat16 *out, int total) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
        out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
    }
}

// ===== Phase: SiLU(gate) * up =====
__device__ void phase_silu_mul(const __nv_bfloat16 *gate, const __nv_bfloat16 *up,
                               __nv_bfloat16 *out, int total) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
        float g = __bfloat162float(gate[i]);
        out[i] = __float2bfloat16(mega_silu(g) * __bfloat162float(up[i]));
    }
}

// ===== Phase: DeltaNet recurrence (16 heads, each on its own block) =====
// Reuses the optimized B200 per-head inner loop: conv ring-buffer in shared,
// conv weights in shared, s_out in shared, norm_w in shared.
__device__ void phase_deltanet_recurrence(
    const __nv_bfloat16 *qkv_proj,   // [S, DN_CONV_CH]
    const __nv_bfloat16 *z_proj,     // [S, DN_V_SIZE]
    const float *beta_proj,          // [S, DN_HEADS]
    const float *alpha_proj,         // [S, DN_HEADS]
    const __nv_bfloat16 *conv_w,     // [DN_CONV_CH, DN_CONV_K]
    const __nv_bfloat16 *a_log,      // [DN_HEADS]
    const __nv_bfloat16 *dt_bias,    // [DN_HEADS]
    const __nv_bfloat16 *norm_w,     // [DN_VAL]
    float *state,                    // [DN_HEADS, DN_KEY, DN_VAL]
    float *conv_buf,                 // [DN_CONV_CH, DN_CONV_K]
    __nv_bfloat16 *output,           // [S, DN_V_SIZE]
    int S)
{
    int h = blockIdx.x;
    if (h >= DN_HEADS) return;  // extra blocks stay idle this phase

    int tid = threadIdx.x, wid = tid / 32, lid = tid % 32;
    constexpr int NWARPS = 16;
    constexpr float Q_SCALE = 1.0f / 11.313708498984761f;
    constexpr int QKV_CH = 2 * DN_KEY + DN_VAL;

    float a_log_val = __bfloat162float(a_log[h]);
    float dt_b = __bfloat162float(dt_bias[h]);

    __shared__ float s_q[DN_KEY], s_k[DN_KEY], s_v[DN_VAL];
    __shared__ float s_beta, s_decay;
    __shared__ float s_gnorm[NWARPS];
    __shared__ float s_conv[QKV_CH * DN_CONV_K];
    __shared__ float s_conv_w[QKV_CH * DN_CONV_K];
    __shared__ float s_z[DN_VAL];
    __shared__ float s_out[DN_VAL];
    __shared__ float s_norm_w[DN_VAL];

    float *my_state = state + h * DN_KEY * DN_VAL;

    constexpr int CPW = DN_VAL / NWARPS;   // 8
    constexpr int RPL = DN_KEY / 32;       // 4
    float sreg[CPW * RPL];

    for (int jj = 0; jj < CPW; jj++) {
        int j = wid * CPW + jj;
        for (int ii = 0; ii < RPL; ii++) {
            sreg[jj * RPL + ii] = my_state[j * DN_KEY + lid + ii * 32];
        }
    }

    auto ch_global = [&] (int c) -> int {
        if (c < DN_KEY) return h * DN_KEY + c;
        if (c < 2 * DN_KEY) return DN_QK_SIZE + h * DN_KEY + (c - DN_KEY);
        return 2 * DN_QK_SIZE + h * DN_VAL + (c - 2 * DN_KEY);
    };

    for (int c = tid; c < QKV_CH; c += blockDim.x) {
        int gch = ch_global(c);
        const float *src_state = conv_buf + gch * DN_CONV_K;
        const __nv_bfloat16 *src_w = conv_w + gch * DN_CONV_K;
        float *dst_state = s_conv + c * DN_CONV_K;
        float *dst_w = s_conv_w + c * DN_CONV_K;
        #pragma unroll
        for (int k = 0; k < DN_CONV_K; k++) {
            dst_state[k] = src_state[k];
            dst_w[k] = __bfloat162float(src_w[k]);
        }
    }
    for (int i = tid; i < DN_VAL; i += blockDim.x) {
        s_norm_w[i] = __bfloat162float(norm_w[i]);
    }
    __syncthreads();

    for (int t = 0; t < S; t++) {
        // Fused conv1d+SiLU for all 384 per-head channels.
        for (int c = tid; c < QKV_CH; c += blockDim.x) {
            int gch = ch_global(c);
            float *cs = s_conv + c * DN_CONV_K;
            const float *cw = s_conv_w + c * DN_CONV_K;
            float new_x = __bfloat162float(qkv_proj[t * DN_CONV_CH + gch]);
            float h0 = cs[1], h1 = cs[2], h2 = cs[3];
            cs[0] = h0; cs[1] = h1; cs[2] = h2; cs[3] = new_x;
            float co = h0 * cw[0] + h1 * cw[1] + h2 * cw[2] + new_x * cw[3];
            float silu_out = mega_silu(co);
            if (c < DN_KEY)           s_q[c] = silu_out;
            else if (c < 2 * DN_KEY)  s_k[c - DN_KEY] = silu_out;
            else                       s_v[c - 2 * DN_KEY] = silu_out;
        }
        // Prefetch z-row in parallel.
        {
            const __nv_bfloat16 *z_h_bf = z_proj + t * DN_V_SIZE + h * DN_VAL;
            for (int i = tid; i < DN_VAL; i += blockDim.x) {
                s_z[i] = __bfloat162float(z_h_bf[i]);
            }
        }
        __syncthreads();

        // L2 normalize q (warp 0), L2 normalize k (warp 1), beta/decay
        // scalars (warp 2 lane 0) — all disjoint targets, one sync fences
        // the three in parallel (saves one sync per step vs the old
        // two-phase normalize→beta/decay sequence).
        if (wid == 0) {
            float sq = 0; for (int i = lid; i < DN_KEY; i += 32) sq += s_q[i] * s_q[i];
            sq = mega_warp_sum(sq);
            float n = rsqrtf(sq + 1e-6f) * Q_SCALE;
            n = __shfl_sync(0xffffffff, n, 0);
            for (int i = lid; i < DN_KEY; i += 32) s_q[i] *= n;
        }
        if (wid == 1) {
            float sq = 0; for (int i = lid; i < DN_KEY; i += 32) sq += s_k[i] * s_k[i];
            sq = mega_warp_sum(sq);
            float n = rsqrtf(sq + 1e-6f);
            n = __shfl_sync(0xffffffff, n, 0);
            for (int i = lid; i < DN_KEY; i += 32) s_k[i] *= n;
        }
        if (wid == 2 && lid == 0) {
            s_beta = 1.f / (1.f + expf(-beta_proj[t * DN_HEADS + h]));
            float x = alpha_proj[t * DN_HEADS + h] + dt_b;
            float sp = (x > 20.f) ? x : logf(1.f + expf(x));
            s_decay = expf(-expf(a_log_val) * sp);
        }
        __syncthreads();

        float beta = s_beta, decay = s_decay;
        __nv_bfloat16 *out_h = output + t * DN_V_SIZE + h * DN_VAL;

        // Cache per-lane s_k / s_q into registers once per step so the
        // CPW=8 inner-loop iterations read from registers instead of
        // shared memory.
        float sk_cache[RPL];
        float sq_cache[RPL];
        #pragma unroll
        for (int ii = 0; ii < RPL; ii++) {
            sk_cache[ii] = s_k[lid + ii * 32];
            sq_cache[ii] = s_q[lid + ii * 32];
        }

        #pragma unroll
        for (int jj = 0; jj < CPW; jj++) {
            int j = wid * CPW + jj;
            float kv = 0;
            #pragma unroll
            for (int ii = 0; ii < RPL; ii++) kv += sreg[jj * RPL + ii] * sk_cache[ii];
            kv = mega_warp_sum(kv);
            kv = __shfl_sync(0xffffffff, kv, 0);
            float delta = (s_v[j] - decay * kv) * beta;
            float attn = 0;
            #pragma unroll
            for (int ii = 0; ii < RPL; ii++) {
                sreg[jj * RPL + ii] = decay * sreg[jj * RPL + ii] + sk_cache[ii] * delta;
                attn += sreg[jj * RPL + ii] * sq_cache[ii];
            }
            attn = mega_warp_sum(attn);
            if (lid == 0) s_out[j] = attn;
        }
        __syncthreads();

        // Gated RMSNorm.
        //
        // After writing per-warp partials to s_gnorm[wid] and syncing, EVERY
        // thread reads all 16 warp partials and computes rstd locally — no
        // second "warp 0 reduces then everyone syncs to pick up rstd" round
        // trip through shared memory. Saves one __syncthreads per step
        // (×520 steps ×18 layers = ~9k syncs removed per prefill).
        float sq2 = 0;
        for (int i = tid; i < DN_VAL; i += blockDim.x) { float v = s_out[i]; sq2 += v * v; }
        sq2 = mega_warp_sum(sq2);
        if (lid == 0) s_gnorm[wid] = sq2;
        __syncthreads();

        float total = 0;
        #pragma unroll
        for (int w = 0; w < NWARPS; w++) total += s_gnorm[w];
        float rstd = rsqrtf(total / DN_VAL + RMS_EPS);

        for (int i = tid; i < DN_VAL; i += blockDim.x) {
            float n = s_out[i] * rstd * s_norm_w[i];
            out_h[i] = __float2bfloat16(n * mega_silu(s_z[i]));
        }
        // No end-of-iter sync: each thread's writes to shared in the next
        // iteration's conv+z phase target its own slot (stride = blockDim.x)
        // and the first cross-warp read is fenced by the existing sync
        // inside that phase.
    }

    // State writeback
    for (int jj = 0; jj < CPW; jj++) {
        int j = wid * CPW + jj;
        for (int ii = 0; ii < RPL; ii++) {
            my_state[j * DN_KEY + lid + ii * 32] = sreg[jj * RPL + ii];
        }
    }
    __syncthreads();
    for (int c = tid; c < QKV_CH; c += blockDim.x) {
        int gch = ch_global(c);
        float *dst_state = conv_buf + gch * DN_CONV_K;
        const float *src_state = s_conv + c * DN_CONV_K;
        #pragma unroll
        for (int k = 0; k < DN_CONV_K; k++) {
            dst_state[k] = src_state[k];
        }
    }
}

// ===== Phase: QK norm + RoPE + KV cache write =====
// One warp per (s, head) for q, and for k/v in sequence.
__device__ void phase_qk_norm_rope(
    __nv_bfloat16 *q,               // [S, FA_QPROJ_SIZE] (in-place)
    __nv_bfloat16 *k,               // [S, FA_KV_SIZE]    (in-place)
    const __nv_bfloat16 *v,          // [S, FA_KV_SIZE]
    const __nv_bfloat16 *qnw, const __nv_bfloat16 *knw,
    __nv_bfloat16 *k_cache,          // [FA_KV_HEADS, MAX_SEQ, FA_HEAD_DIM]
    __nv_bfloat16 *v_cache,
    int S)
{
    int warps_per_block = blockDim.x / 32;
    int global_warp = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    int total_warps = gridDim.x * warps_per_block;
    int lid = threadIdx.x & 31;

    int total_q = S * FA_Q_HEADS;
    int total_k = S * FA_KV_HEADS;
    int total = total_q + total_k;

    for (int idx = global_warp; idx < total; idx += total_warps) {
        if (idx < total_q) {
            int pos = idx / FA_Q_HEADS, head = idx - pos * FA_Q_HEADS;
            __nv_bfloat16 *qh = q + pos * FA_QPROJ_SIZE + head * FA_HEAD_DIM * 2;
            float ss = 0;
            for (int i = lid; i < FA_HEAD_DIM; i += 32) {
                float v = __bfloat162float(qh[i]); ss += v * v;
            }
            ss = mega_warp_sum(ss);
            float sc = rsqrtf(ss / FA_HEAD_DIM + RMS_EPS);
            sc = __shfl_sync(0xffffffff, sc, 0);
            for (int i = lid; i < FA_HEAD_DIM; i += 32) {
                float normed = __bfloat162float(qh[i]) * sc * (1.f + __bfloat162float(qnw[i]));
                if (i < FA_ROT_DIM) {
                    float fe = float(2 * (i % (FA_ROT_DIM / 2))) / FA_ROT_DIM;
                    float freq = float(pos) / powf(FA_ROPE_THETA, fe);
                    float cv = cosf(freq), sv = sinf(freq);
                    int p = (i < FA_ROT_DIM / 2) ? i + FA_ROT_DIM / 2 : i - FA_ROT_DIM / 2;
                    float pv = __bfloat162float(qh[p]) * sc * (1.f + __bfloat162float(qnw[p]));
                    qh[i] = __float2bfloat16((i < FA_ROT_DIM / 2)
                                              ? (normed * cv - pv * sv)
                                              : (pv * sv + normed * cv));
                } else {
                    qh[i] = __float2bfloat16(normed);
                }
            }
        } else {
            int kidx = idx - total_q;
            int pos = kidx / FA_KV_HEADS, head = kidx - pos * FA_KV_HEADS;
            __nv_bfloat16 *kh = k + pos * FA_KV_SIZE + head * FA_HEAD_DIM;
            const __nv_bfloat16 *vh = v + pos * FA_KV_SIZE + head * FA_HEAD_DIM;
            __nv_bfloat16 *kc = k_cache + head * MAX_SEQ * FA_HEAD_DIM + pos * FA_HEAD_DIM;
            __nv_bfloat16 *vc = v_cache + head * MAX_SEQ * FA_HEAD_DIM + pos * FA_HEAD_DIM;
            float ss = 0;
            for (int i = lid; i < FA_HEAD_DIM; i += 32) {
                float v = __bfloat162float(kh[i]); ss += v * v;
            }
            ss = mega_warp_sum(ss);
            float sc = rsqrtf(ss / FA_HEAD_DIM + RMS_EPS);
            sc = __shfl_sync(0xffffffff, sc, 0);
            for (int i = lid; i < FA_HEAD_DIM; i += 32) {
                float normed = __bfloat162float(kh[i]) * sc * (1.f + __bfloat162float(knw[i]));
                float fk;
                if (i < FA_ROT_DIM) {
                    float fe = float(2 * (i % (FA_ROT_DIM / 2))) / FA_ROT_DIM;
                    float freq = float(pos) / powf(FA_ROPE_THETA, fe);
                    float cv = cosf(freq), sv = sinf(freq);
                    int p = (i < FA_ROT_DIM / 2) ? i + FA_ROT_DIM / 2 : i - FA_ROT_DIM / 2;
                    float pv = __bfloat162float(kh[p]) * sc * (1.f + __bfloat162float(knw[p]));
                    fk = (i < FA_ROT_DIM / 2) ? (normed * cv - pv * sv) : (pv * sv + normed * cv);
                } else {
                    fk = normed;
                }
                kh[i] = __float2bfloat16(fk);
                kc[i] = __float2bfloat16(fk);
                vc[i] = vh[i];
            }
        }
    }
}

// ===== Phase: causal attention (per (s, q_head), single-warp online softmax) =====
__device__ void phase_causal_attn(const __nv_bfloat16 *q, const __nv_bfloat16 *k,
                                  const __nv_bfloat16 *v, __nv_bfloat16 *out, int S)
{
    int warps_per_block = blockDim.x / 32;
    int global_warp = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    int total_warps = gridDim.x * warps_per_block;
    int lid = threadIdx.x & 31;

    int total = S * FA_Q_HEADS;
    float scale = 1.0f / sqrtf(float(FA_HEAD_DIM));
    constexpr int EPL = FA_HEAD_DIM / 32;

    for (int idx = global_warp; idx < total; idx += total_warps) {
        int pos = idx / FA_Q_HEADS, qh = idx - pos * FA_Q_HEADS;
        int kvh = qh / FA_GQA;

        const __nv_bfloat16 *qv = q + pos * FA_QPROJ_SIZE + qh * FA_HEAD_DIM * 2;
        const __nv_bfloat16 *gv = qv + FA_HEAD_DIM;
        __nv_bfloat16 *ov = out + pos * FA_Q_SIZE + qh * FA_HEAD_DIM;

        float ql[EPL];
        for (int e = 0; e < EPL; e++) ql[e] = __bfloat162float(qv[lid * EPL + e]);
        float oa[EPL] = {};
        float mx = -1e30f, se = 0;

        for (int kp = 0; kp <= pos; kp++) {
            const __nv_bfloat16 *kv_ptr = k + kp * FA_KV_SIZE + kvh * FA_HEAD_DIM;
            const __nv_bfloat16 *vv = v + kp * FA_KV_SIZE + kvh * FA_HEAD_DIM;
            float sc = 0;
            for (int e = 0; e < EPL; e++) sc += ql[e] * __bfloat162float(kv_ptr[lid * EPL + e]);
            sc = mega_warp_sum(sc) * scale;
            sc = __shfl_sync(0xffffffff, sc, 0);
            float om = mx;
            mx = fmaxf(mx, sc);
            float ed = expf(om - mx);
            se = se * ed + expf(sc - mx);
            float wt = expf(sc - mx);
            for (int e = 0; e < EPL; e++) {
                oa[e] = oa[e] * ed + wt * __bfloat162float(vv[lid * EPL + e]);
            }
        }
        float rs = 1.f / se;
        for (int e = 0; e < EPL; e++) {
            int i = lid * EPL + e;
            float g = 1.f / (1.f + expf(-__bfloat162float(gv[i])));
            ov[i] = __float2bfloat16(oa[e] * rs * g);
        }
    }
}

// ===== Phase: final norm (just last row) =====
// Only block 0 works. hidden_out is embedded residual, final_normed is the
// rmsnorm'd row used by the LM head.
__device__ void phase_final_norm(const __nv_bfloat16 *hidden,
                                 const __nv_bfloat16 *w,
                                 __nv_bfloat16 *final_normed,
                                 __nv_bfloat16 *hidden_bf16_out,
                                 int S)
{
    if (blockIdx.x != 0) return;
    int tid = threadIdx.x, wid = tid / 32, lid = tid % 32;
    __shared__ float smem[MEGA_WARPS];

    const __nv_bfloat16 *row = hidden + (S - 1) * HIDDEN;
    float sq = 0;
    for (int i = tid; i < HIDDEN; i += blockDim.x) {
        float v = __bfloat162float(row[i]); sq += v * v;
    }
    sq = mega_warp_sum(sq);
    if (lid == 0) smem[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        float v = (lid < MEGA_WARPS) ? smem[lid] : 0;
        v = mega_warp_sum(v);
        if (lid == 0) smem[0] = rsqrtf(v / HIDDEN + RMS_EPS);
    }
    __syncthreads();
    float rstd = smem[0];
    for (int i = tid; i < HIDDEN; i += blockDim.x) {
        float v = __bfloat162float(row[i]);
        final_normed[i] = __float2bfloat16(v * rstd * (1.f + __bfloat162float(w[i])));
        hidden_bf16_out[i] = row[i];
    }
}

// ===== Phase: LM head (bf16 matvec over full vocab) =====
// Each block computes part of the vocab and writes (max_val, argmax) pair.
// A final reduce picks the global argmax.
__device__ void phase_lm_head(const __nv_bfloat16 *hidden,
                              const __nv_bfloat16 *weight,
                              float *block_max_vals,
                              int *block_max_idxs)
{
    __shared__ float s_hidden[HIDDEN];
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
        s_hidden[i] = __bfloat162float(hidden[i]);
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lid = threadIdx.x & 31;
    int num_warps = blockDim.x / 32;
    int total_blocks = gridDim.x;
    int rpb = (VOCAB + total_blocks - 1) / total_blocks;
    int rs = blockIdx.x * rpb;
    int re = min(rs + rpb, VOCAB);

    float local_max = -INFINITY;
    int local_max_idx = -1;
    for (int m = rs + warp_id; m < re; m += num_warps) {
        const __nv_bfloat16 *wr = weight + m * HIDDEN;
        float sum = 0;
        for (int k = lid; k < HIDDEN; k += 32) {
            sum += __bfloat162float(wr[k]) * s_hidden[k];
        }
        sum = mega_warp_sum(sum);
        if (lid == 0 && sum > local_max) { local_max = sum; local_max_idx = m; }
    }
    local_max = __shfl_sync(0xffffffff, local_max, 0);
    local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

    __shared__ float wm[32]; __shared__ int wi[32];
    if (lid == 0) { wm[warp_id] = local_max; wi[warp_id] = local_max_idx; }
    __syncthreads();
    if (warp_id == 0) {
        float mv = (lid < num_warps) ? wm[lid] : -INFINITY;
        int mi = (lid < num_warps) ? wi[lid] : -1;
        for (int o = 16; o > 0; o /= 2) {
            float ov = __shfl_down_sync(0xffffffff, mv, o);
            int oi = __shfl_down_sync(0xffffffff, mi, o);
            if (ov > mv) { mv = ov; mi = oi; }
        }
        if (lid == 0) {
            block_max_vals[blockIdx.x] = mv;
            block_max_idxs[blockIdx.x] = mi;
        }
    }
}

// ===== Phase: LM reduce (block 0 only) =====
__device__ void phase_lm_reduce(const float *block_max_vals,
                                const int *block_max_idxs,
                                int *output_token, int num_blocks)
{
    if (blockIdx.x != 0) return;
    int tid = threadIdx.x;
    __shared__ float sv[MEGA_BLOCK_SIZE];
    __shared__ int si[MEGA_BLOCK_SIZE];
    float bv = -INFINITY; int bi = -1;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float v = block_max_vals[i];
        if (v > bv) { bv = v; bi = block_max_idxs[i]; }
    }
    sv[tid] = bv; si[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sv[tid + s] > sv[tid]) { sv[tid] = sv[tid + s]; si[tid] = si[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) *output_token = si[0];
}

// ==========================================================================
// Main megakernel. All device work for a prefill lives in this one dispatch.
// ==========================================================================
__global__ void __launch_bounds__(MEGA_BLOCK_SIZE, 1)
prefill_megakernel(
    const int *token_ids, int S, int *output_token,
    const __nv_bfloat16 *embed_weight, const MegaLayerWeights *layers,
    const __nv_bfloat16 *final_norm_w, const __nv_bfloat16 *lm_head_w,
    __nv_bfloat16 *fa_k_cache, __nv_bfloat16 *fa_v_cache,
    float *dn_states, float *conv_bufs,
    __nv_bfloat16 *hidden, __nv_bfloat16 *residual, __nv_bfloat16 *normalized,
    __nv_bfloat16 *proj_buf, __nv_bfloat16 *proj_buf2,
    __nv_bfloat16 *attn_buf, __nv_bfloat16 *mlp_buf,
    __nv_bfloat16 *dn_out_buf,
    float *beta_buf, float *alpha_buf,
    __nv_bfloat16 *final_normed, __nv_bfloat16 *hidden_bf16_out,
    float *lm_bmv, int *lm_bmi)
{
    cg::grid_group grid = cg::this_grid();

    // Pad S up to the matmul block-tile multiple so WMMA fragments always
    // land inside allocated scratch. Padding rows are computed through every
    // matmul but never influence sequential phases (DN recurrence, RoPE,
    // causal attn, final norm), which use S directly.
    int S_pad = ((S + BTM - 1) / BTM) * BTM;

    int fa_stride = FA_KV_HEADS * MAX_SEQ * FA_HEAD_DIM;
    int dn_stride = DN_HEADS * DN_KEY * DN_VAL;

    // Phase 0: Embedding for real tokens; zero-fill padding rows so matmuls
    // never read uninitialised memory.
    phase_embed(token_ids, embed_weight, hidden, S);
    // Zero the padding tail of hidden.
    {
        int stride = gridDim.x * blockDim.x;
        int start = S * HIDDEN;
        int end = S_pad * HIDDEN;
        for (int i = start + blockIdx.x * blockDim.x + threadIdx.x; i < end; i += stride) {
            hidden[i] = __float2bfloat16(0.0f);
        }
    }
    grid.sync();

    int fa_idx = 0, dn_idx = 0;

    for (int li = 0; li < NUM_LAYERS; li++) {
        const MegaLayerWeights &lw = layers[li];
        int lt = MEGA_LAYER_TYPE[li];

        const __nv_bfloat16 *inp_norm_w = (const __nv_bfloat16 *)lw.ptrs[0];
        phase_rmsnorm(hidden, inp_norm_w, normalized, residual, S_pad, HIDDEN);
        grid.sync();

        if (lt == 0) {
            // DeltaNet
            const __nv_bfloat16 *qkv_w = (const __nv_bfloat16 *)lw.ptrs[1];
            const __nv_bfloat16 *z_w   = (const __nv_bfloat16 *)lw.ptrs[2];
            const __nv_bfloat16 *beta_w = (const __nv_bfloat16 *)lw.ptrs[3];
            const __nv_bfloat16 *alpha_w= (const __nv_bfloat16 *)lw.ptrs[4];
            const __nv_bfloat16 *conv_w = (const __nv_bfloat16 *)lw.ptrs[5];
            const __nv_bfloat16 *a_log  = (const __nv_bfloat16 *)lw.ptrs[6];
            const __nv_bfloat16 *dt_bias= (const __nv_bfloat16 *)lw.ptrs[7];
            const __nv_bfloat16 *dn_norm= (const __nv_bfloat16 *)lw.ptrs[8];
            const __nv_bfloat16 *out_w  = (const __nv_bfloat16 *)lw.ptrs[9];
            const __nv_bfloat16 *post_norm = (const __nv_bfloat16 *)lw.ptrs[10];
            const __nv_bfloat16 *gate_w = (const __nv_bfloat16 *)lw.ptrs[11];
            const __nv_bfloat16 *up_w   = (const __nv_bfloat16 *)lw.ptrs[12];
            const __nv_bfloat16 *down_w = (const __nv_bfloat16 *)lw.ptrs[13];

            phase_matmul_bf16(normalized, qkv_w, proj_buf, S_pad, DN_CONV_CH, HIDDEN);
            phase_matmul_bf16(normalized, z_w,   proj_buf2, S_pad, DN_V_SIZE, HIDDEN);
            phase_matvec_small_to_f32(normalized, beta_w,  beta_buf,  S, HIDDEN, DN_HEADS);
            phase_matvec_small_to_f32(normalized, alpha_w, alpha_buf, S, HIDDEN, DN_HEADS);
            grid.sync();

            phase_deltanet_recurrence(proj_buf, proj_buf2, beta_buf, alpha_buf,
                                      conv_w, a_log, dt_bias, dn_norm,
                                      dn_states + dn_idx * dn_stride,
                                      conv_bufs + dn_idx * DN_CONV_CH * DN_CONV_K,
                                      dn_out_buf, S);
            grid.sync();

            phase_matmul_bf16(dn_out_buf, out_w, proj_buf, S_pad, HIDDEN, DN_V_SIZE);
            grid.sync();
            phase_add_residual(proj_buf, residual, hidden, S_pad * HIDDEN);
            grid.sync();

            // MLP
            phase_rmsnorm(hidden, post_norm, normalized, residual, S_pad, HIDDEN);
            grid.sync();
            phase_matmul_bf16(normalized, gate_w, proj_buf, S_pad, INTER, HIDDEN);
            phase_matmul_bf16(normalized, up_w,   proj_buf2, S_pad, INTER, HIDDEN);
            grid.sync();
            phase_silu_mul(proj_buf, proj_buf2, mlp_buf, S_pad * INTER);
            grid.sync();
            phase_matmul_bf16(mlp_buf, down_w, proj_buf, S_pad, HIDDEN, INTER);
            grid.sync();
            phase_add_residual(proj_buf, residual, hidden, S_pad * HIDDEN);
            grid.sync();

            dn_idx++;
        } else {
            // Full Attention
            const __nv_bfloat16 *q_w = (const __nv_bfloat16 *)lw.ptrs[1];
            const __nv_bfloat16 *k_w = (const __nv_bfloat16 *)lw.ptrs[2];
            const __nv_bfloat16 *v_w = (const __nv_bfloat16 *)lw.ptrs[3];
            const __nv_bfloat16 *q_nw = (const __nv_bfloat16 *)lw.ptrs[4];
            const __nv_bfloat16 *k_nw = (const __nv_bfloat16 *)lw.ptrs[5];
            const __nv_bfloat16 *o_w = (const __nv_bfloat16 *)lw.ptrs[6];
            const __nv_bfloat16 *post_norm = (const __nv_bfloat16 *)lw.ptrs[7];
            const __nv_bfloat16 *gate_w = (const __nv_bfloat16 *)lw.ptrs[8];
            const __nv_bfloat16 *up_w   = (const __nv_bfloat16 *)lw.ptrs[9];
            const __nv_bfloat16 *down_w = (const __nv_bfloat16 *)lw.ptrs[10];

            phase_matmul_bf16(normalized, q_w, proj_buf, S_pad, FA_QPROJ_SIZE, HIDDEN);
            phase_matmul_bf16(normalized, k_w, proj_buf2, S_pad, FA_KV_SIZE, HIDDEN);
            phase_matmul_bf16(normalized, v_w, attn_buf, S_pad, FA_KV_SIZE, HIDDEN);
            grid.sync();

            phase_qk_norm_rope(proj_buf, proj_buf2, attn_buf, q_nw, k_nw,
                               fa_k_cache + fa_idx * fa_stride,
                               fa_v_cache + fa_idx * fa_stride, S);
            grid.sync();

            phase_causal_attn(proj_buf, proj_buf2, attn_buf, dn_out_buf, S);
            grid.sync();

            phase_matmul_bf16(dn_out_buf, o_w, proj_buf, S_pad, HIDDEN, FA_Q_SIZE);
            grid.sync();
            phase_add_residual(proj_buf, residual, hidden, S_pad * HIDDEN);
            grid.sync();

            phase_rmsnorm(hidden, post_norm, normalized, residual, S_pad, HIDDEN);
            grid.sync();
            phase_matmul_bf16(normalized, gate_w, proj_buf, S_pad, INTER, HIDDEN);
            phase_matmul_bf16(normalized, up_w,   proj_buf2, S_pad, INTER, HIDDEN);
            grid.sync();
            phase_silu_mul(proj_buf, proj_buf2, mlp_buf, S_pad * INTER);
            grid.sync();
            phase_matmul_bf16(mlp_buf, down_w, proj_buf, S_pad, HIDDEN, INTER);
            grid.sync();
            phase_add_residual(proj_buf, residual, hidden, S_pad * HIDDEN);
            grid.sync();

            fa_idx++;
        }
    }

    // Final RMSNorm (last row only) + LM head
    phase_final_norm(hidden, final_norm_w, final_normed, hidden_bf16_out, S);
    grid.sync();
    phase_lm_head(final_normed, lm_head_w, lm_bmv, lm_bmi);
    grid.sync();
    phase_lm_reduce(lm_bmv, lm_bmi, output_token, gridDim.x);
}

// ===== Launcher =====
// Dynamic shared memory = double-buffered A+B staging + f32 output tile.
//   sA: 2 × BTM × BTK × 2 B  =  8 KB
//   sB: 2 × BTN × BTK × 2 B  = 64 KB
//   sC: BTM × BTN × 4 B      = 32 KB
//   total                    = 104 KB
// B200 per-block dynamic shared max is 227 KB (228 KB carveout - driver).
constexpr int MEGA_SHMEM_BYTES =
    2 * BTM * BTK * 2   // sA
  + 2 * BTN * BTK * 2   // sB
  + BTM * BTN * 4;      // sC

static int mega_launch_blocks() {
    if (const char *override_blocks = std::getenv("MEGAKERNEL_PREFILL_MEGA_BLOCKS")) {
        int v = std::atoi(override_blocks);
        if (v > 0) return v;
    }
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    // Raise per-block dynamic shared limit (sm_100 default caps lower).
    cudaFuncSetAttribute(prefill_megakernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEGA_SHMEM_BYTES);
    int active = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, prefill_megakernel, MEGA_BLOCK_SIZE, MEGA_SHMEM_BYTES);
    return std::max(1, active * prop.multiProcessorCount);
}

extern "C" void launch_prefill_bf16_mega(
    const int *token_ids, int seq_len, int *output_token,
    const __nv_bfloat16 *embed_weight, const MegaLayerWeights *layers,
    const __nv_bfloat16 *final_norm_w, const __nv_bfloat16 *lm_head_w,
    __nv_bfloat16 *fa_k_cache, __nv_bfloat16 *fa_v_cache,
    float *dn_states, float *conv_bufs,
    __nv_bfloat16 *hidden, __nv_bfloat16 *residual, __nv_bfloat16 *normalized,
    __nv_bfloat16 *proj_buf, __nv_bfloat16 *proj_buf2,
    __nv_bfloat16 *attn_buf, __nv_bfloat16 *mlp_buf,
    __nv_bfloat16 *dn_out_buf,
    float *beta_buf, float *alpha_buf,
    __nv_bfloat16 *final_normed, __nv_bfloat16 *hidden_bf16_out,
    float *lm_bmv, int *lm_bmi,
    cudaStream_t stream)
{
    static int cached_blocks = 0;
    if (cached_blocks == 0) cached_blocks = mega_launch_blocks();

    const int *token_ids_arg = token_ids;
    int seq_len_arg = seq_len;
    int *output_token_arg = output_token;
    const __nv_bfloat16 *embed_arg = embed_weight;
    const MegaLayerWeights *layers_arg = layers;
    const __nv_bfloat16 *fn_arg = final_norm_w;
    const __nv_bfloat16 *lm_arg = lm_head_w;
    __nv_bfloat16 *fk_arg = fa_k_cache; __nv_bfloat16 *fv_arg = fa_v_cache;
    float *ds_arg = dn_states; float *cb_arg = conv_bufs;
    __nv_bfloat16 *h_arg = hidden; __nv_bfloat16 *r_arg = residual; __nv_bfloat16 *n_arg = normalized;
    __nv_bfloat16 *p1 = proj_buf; __nv_bfloat16 *p2 = proj_buf2;
    __nv_bfloat16 *a_arg = attn_buf; __nv_bfloat16 *m_arg = mlp_buf;
    __nv_bfloat16 *dn_out_arg = dn_out_buf;
    float *bb = beta_buf; float *ab = alpha_buf;
    __nv_bfloat16 *fnrm = final_normed; __nv_bfloat16 *hout = hidden_bf16_out;
    float *lv = lm_bmv; int *li_ = lm_bmi;

    void *args[] = {
        (void *)&token_ids_arg, (void *)&seq_len_arg, (void *)&output_token_arg,
        (void *)&embed_arg, (void *)&layers_arg,
        (void *)&fn_arg, (void *)&lm_arg,
        (void *)&fk_arg, (void *)&fv_arg,
        (void *)&ds_arg, (void *)&cb_arg,
        (void *)&h_arg, (void *)&r_arg, (void *)&n_arg,
        (void *)&p1, (void *)&p2,
        (void *)&a_arg, (void *)&m_arg,
        (void *)&dn_out_arg,
        (void *)&bb, (void *)&ab,
        (void *)&fnrm, (void *)&hout,
        (void *)&lv, (void *)&li_,
    };
    cudaLaunchCooperativeKernel(
        (void *)prefill_megakernel,
        dim3(cached_blocks),
        dim3(MEGA_BLOCK_SIZE),
        args, MEGA_SHMEM_BYTES, stream);
}
