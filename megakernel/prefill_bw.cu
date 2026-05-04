/**
 * BF16 prefill (Blackwell-only) — vectorized bf16 kernels + bf16 body
 * with optional NVFP4 LM head. Companion to upstream prefill.cu, which is
 * compiled and exposed unchanged on every arch. This file is only compiled
 * for sm_120+ (see setup.py) so its symbols never reach the sm_86 build.
 *
 * Internal helpers and __global__ kernels live in an anonymous namespace
 * so their names (pf_rmsnorm, pf_attention, ...) coexist with upstream's
 * versions in the same .so without ODR violation. Only the new
 * launch_prefill_bf16_nvfp4_lm entry point is exposed with extern "C" linkage.
 */

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1200
#error "prefill_bw.cu requires CUDA arch >= sm_120 (Blackwell)"
#endif
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef PREFILL_DN_BLOCK_SIZE
#define PREFILL_DN_BLOCK_SIZE 128
#endif

#ifndef PREFILL_DN_BLOCKS_PER_HEAD
#define PREFILL_DN_BLOCKS_PER_HEAD 8
#endif

#ifndef PREFILL_FA_BLOCK_SIZE
#define PREFILL_FA_BLOCK_SIZE 256
#endif


// ===== Cross-TU forward declarations — defined in kernel_gb10_nvfp4.cu =====

extern "C" bool launch_lm_head_cublaslt_bf16_top1(
    const void *hidden_bf16,
    const void *lm_head_weight_packed,
    const void *lm_head_scales,
    void *lm_hidden_bf16,
    void *lm_hidden_packed,
    void *lm_hidden_scales,
    void *lm_logits_f16,
    float *block_max_vals,
    int *block_max_idxs,
    int *output_token_id,
    cudaStream_t stream);

extern "C" void launch_quantize_nvfp4_lm_out(
    const void *weight, int rows, int cols,
    void *packed_out, void *scales_out, cudaStream_t stream);

namespace {

constexpr int HIDDEN = 1024;
constexpr int INTER = 3584;
constexpr int VOCAB = 248320;
constexpr float RMS_EPS = 1e-6f;

constexpr int FA_Q_HEADS = 8;
constexpr int FA_KV_HEADS = 2;
constexpr int FA_HEAD_DIM = 256;
constexpr int FA_GQA = FA_Q_HEADS / FA_KV_HEADS;
constexpr int FA_Q_SIZE = FA_Q_HEADS * FA_HEAD_DIM;
constexpr int FA_QPROJ_SIZE = FA_Q_SIZE * 2;
constexpr int FA_KV_SIZE = FA_KV_HEADS * FA_HEAD_DIM;
constexpr int FA_ROT_DIM = 64;
constexpr float FA_ROPE_THETA = 10000000.0f;
constexpr int FA_ROT_PAIRS = FA_ROT_DIM / 2;

constexpr int DN_HEADS = 16;
constexpr int DN_KEY = 128;
constexpr int DN_VAL = 128;
constexpr int DN_CONV_K = 4;
constexpr int DN_QK_SIZE = DN_HEADS * DN_KEY;
constexpr int DN_V_SIZE = DN_HEADS * DN_VAL;
constexpr int DN_CONV_CH = DN_QK_SIZE * 2 + DN_V_SIZE;
constexpr int DN_Z_OFFSET = DN_CONV_CH;
constexpr int DN_BETA_OFFSET = DN_Z_OFFSET + DN_V_SIZE;
constexpr int DN_ALPHA_OFFSET = DN_BETA_OFFSET + DN_HEADS;
constexpr int DN_PROJ_FUSED = DN_ALPHA_OFFSET + DN_HEADS;
constexpr int DN_PROJ_FUSED_PADDED = ((DN_PROJ_FUSED + 127) / 128) * 128;
constexpr int FA_QKV_FUSED = FA_QPROJ_SIZE + 2 * FA_KV_SIZE;
constexpr int MLP_GATE_UP_FUSED = INTER * 2;
constexpr int NVFP4_TC_ROWS_PER_TILE = 128;
constexpr int NVFP4_TC_COLS_PER_TILE = 4;
constexpr int NVFP4_TC_BLOCK_K = 16;
constexpr int NVFP4_TC_K_PER_TILE = NVFP4_TC_COLS_PER_TILE * NVFP4_TC_BLOCK_K;
constexpr size_t PREFILL_TC_MAX_WORKSPACE = 32ull * 1024ull * 1024ull;

constexpr int NUM_LAYERS = 24;
constexpr int LAYER_TYPE[24] = {0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1};

struct PFLayerWeights { int layer_type; int _pad[3]; void *ptrs[14]; };
struct PFFusedLayerWeights {
    void *proj_weight;
    void *gate_up_weight;
    void *proj_weight_packed;
    void *proj_weight_scales;
    void *gate_up_weight_packed;
    void *gate_up_weight_scales;
};


__device__ __constant__ float PF_ROPE_INV_FREQ[FA_ROT_PAIRS] = {
    1.000000000000000000e+00f, 6.042963902381328634e-01f,
    3.651741272548377215e-01f, 2.206734069084589911e-01f,
    1.333521432163324028e-01f, 8.058421877614818651e-02f,
    4.869675251658631132e-02f, 2.942727176209281731e-02f,
    1.778279410038922925e-02f, 1.074607828321317432e-02f,
    6.493816315762112983e-03f, 3.924189758484536265e-03f,
    2.371373705661655382e-03f, 1.433012570236962685e-03f,
    8.659643233600653866e-04f, 5.232991146814947340e-04f,
    3.162277660168379394e-04f, 1.910952974970440477e-04f,
    1.154781984689458215e-04f, 6.978305848598663529e-05f,
    4.216965034285822237e-05f, 2.548296747979346413e-05f,
    1.539926526059491854e-05f, 9.305720409296990429e-06f,
    5.623413251903491208e-06f, 3.398208328942559268e-06f,
    2.053525026457146066e-06f, 1.240937760751719527e-06f,
    7.498942093324558477e-07f, 4.531583637600817928e-07f,
    2.738419634264361394e-07f, 1.654817099943181354e-07f
};

__device__ __forceinline__ float pf_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o); return v;
}
__device__ __forceinline__ float pf_sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float pf_silu(float x) { return x * pf_sigmoid(x); }

// Embedding
__global__ void pf_embed(const int *ids, const __nv_bfloat16 *embed, __nv_bfloat16 *out, int S) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= S * HIDDEN) return;
    out[idx] = embed[ids[idx / HIDDEN] * HIDDEN + idx % HIDDEN];
}

// Batched RMSNorm: bf16 in → bf16 out, saves bf16 residual
__global__ void pf_rmsnorm(const __nv_bfloat16 *in, const __nv_bfloat16 *w,
    __nv_bfloat16 *out, __nv_bfloat16 *res, int S, int D) {
    int s = blockIdx.x; if (s >= S) return;
    int tid = threadIdx.x, wid = tid/32, lid = tid%32;
    __shared__ float smem[8];
    const __nv_bfloat16 *ri = in + s*D;
    __nv_bfloat16 *ro = out + s*D, *rr = res + s*D;
    const __nv_bfloat162 *ri2 = reinterpret_cast<const __nv_bfloat162 *>(ri);
    const __nv_bfloat162 *w2 = reinterpret_cast<const __nv_bfloat162 *>(w);
    __nv_bfloat162 *ro2 = reinterpret_cast<__nv_bfloat162 *>(ro);
    __nv_bfloat162 *rr2 = reinterpret_cast<__nv_bfloat162 *>(rr);
    int D2 = D / 2;
    float sq = 0;
    for (int i2 = tid; i2 < D2; i2 += blockDim.x) {
        __nv_bfloat162 rv2 = ri2[i2];
        float2 rv = __bfloat1622float2(rv2);
        rr2[i2] = rv2;
        sq += rv.x * rv.x + rv.y * rv.y;
    }
    if ((D & 1) && tid == 0) {
        float v = __bfloat162float(ri[D - 1]);
        rr[D - 1] = ri[D - 1];
        sq += v * v;
    }
    sq = pf_warp_sum(sq); if(lid==0) smem[wid]=sq; __syncthreads();
    if(wid==0){float v=(lid<blockDim.x/32)?smem[lid]:0;v=pf_warp_sum(v);if(lid==0)smem[0]=rsqrtf(v/D+RMS_EPS);}
    __syncthreads(); float rstd = smem[0];
    for (int i2 = tid; i2 < D2; i2 += blockDim.x) {
        float2 rv = __bfloat1622float2(ri2[i2]);
        float2 wv = __bfloat1622float2(w2[i2]);
        ro2[i2] = __floats2bfloat162_rn(
            rv.x * rstd * (1.0f + wv.x),
            rv.y * rstd * (1.0f + wv.y));
    }
    if ((D & 1) && tid == 0) {
        float v = __bfloat162float(ri[D - 1]) * rstd * (1.0f + __bfloat162float(w[D - 1]));
        ro[D - 1] = __float2bfloat16(v);
    }
}

// bf16 matvec for tiny projections (beta/alpha)
__global__ void pf_bf16_matvec(const __nv_bfloat16 *in, const __nv_bfloat16 *w, float *out, int S, int K, int N) {
    int idx = blockIdx.x; if (idx >= S * N) return;
    int s = idx / N, n = idx % N, lid = threadIdx.x;
    const __nv_bfloat16 *ir = in + s*K, *wr = w + n*K;
    float sum = 0;
    for (int k = lid; k < K; k += 32) sum += __bfloat162float(ir[k]) * __bfloat162float(wr[k]);
    sum = pf_warp_sum(sum);
    if (lid == 0) out[idx] = sum;
}

// bf16 result + bf16 residual → bf16 output
__global__ void pf_add_residual_bf16(const __nv_bfloat16 *a, const __nv_bfloat16 *b, __nv_bfloat16 *out, int N) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int pairs = N / 2;
    if (pair < pairs) {
        const __nv_bfloat162 *a2 = reinterpret_cast<const __nv_bfloat162 *>(a);
        const __nv_bfloat162 *b2 = reinterpret_cast<const __nv_bfloat162 *>(b);
        __nv_bfloat162 *out2 = reinterpret_cast<__nv_bfloat162 *>(out);
        float2 av = __bfloat1622float2(a2[pair]);
        float2 bv = __bfloat1622float2(b2[pair]);
        out2[pair] = __floats2bfloat162_rn(av.x + bv.x, av.y + bv.y);
    }
    if ((N & 1) && pair == 0) {
        out[N - 1] = __float2bfloat16(__bfloat162float(a[N - 1]) + __bfloat162float(b[N - 1]));
    }
}

// Fuse residual add with the immediately following RMSNorm. This avoids
// round-tripping the post-add hidden row through global memory before the MLP.
__global__ void pf_add_residual_rmsnorm_bf16(
    const __nv_bfloat16 *a,
    const __nv_bfloat16 *b,
    const __nv_bfloat16 *w,
    __nv_bfloat16 *hidden_out,
    __nv_bfloat16 *norm_out,
    __nv_bfloat16 *residual_out,
    int S, int D)
{
    int s = blockIdx.x;
    if (s >= S) return;

    int tid = threadIdx.x, wid = tid / 32, lid = tid % 32;
    __shared__ float smem[8];
    __shared__ __align__(16) __nv_bfloat162 s_sum[HIDDEN / 2];

    const __nv_bfloat16 *ar = a + s * D;
    const __nv_bfloat16 *br = b + s * D;
    __nv_bfloat16 *ho = hidden_out + s * D;
    __nv_bfloat16 *no = norm_out + s * D;
    __nv_bfloat16 *ro = residual_out + s * D;

    const __nv_bfloat162 *a2 = reinterpret_cast<const __nv_bfloat162 *>(ar);
    const __nv_bfloat162 *b2 = reinterpret_cast<const __nv_bfloat162 *>(br);
    const __nv_bfloat162 *w2 = reinterpret_cast<const __nv_bfloat162 *>(w);
    __nv_bfloat162 *ho2 = reinterpret_cast<__nv_bfloat162 *>(ho);
    __nv_bfloat162 *no2 = reinterpret_cast<__nv_bfloat162 *>(no);
    __nv_bfloat162 *ro2 = reinterpret_cast<__nv_bfloat162 *>(ro);
    int D2 = D / 2;

    float sq = 0.0f;
    for (int i2 = tid; i2 < D2; i2 += blockDim.x) {
        float2 av = __bfloat1622float2(a2[i2]);
        float2 bv = __bfloat1622float2(b2[i2]);
        float2 sv = make_float2(av.x + bv.x, av.y + bv.y);
        s_sum[i2] = __floats2bfloat162_rn(sv.x, sv.y);
        sq += sv.x * sv.x + sv.y * sv.y;
    }

    if ((D & 1) && tid == 0) {
        float sv = __bfloat162float(ar[D - 1]) + __bfloat162float(br[D - 1]);
        reinterpret_cast<__nv_bfloat16 *>(s_sum)[D - 1] = __float2bfloat16(sv);
        sq += sv * sv;
    }

    sq = pf_warp_sum(sq);
    if (lid == 0) smem[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        float v = (lid < blockDim.x / 32) ? smem[lid] : 0.0f;
        v = pf_warp_sum(v);
        if (lid == 0) smem[0] = rsqrtf(v / D + RMS_EPS);
    }
    __syncthreads();
    float rstd = smem[0];

    for (int i2 = tid; i2 < D2; i2 += blockDim.x) {
        __nv_bfloat162 sum2 = s_sum[i2];
        float2 sv = __bfloat1622float2(sum2);
        float2 wv = __bfloat1622float2(w2[i2]);
        ho2[i2] = sum2;
        ro2[i2] = sum2;
        no2[i2] = __floats2bfloat162_rn(
            sv.x * rstd * (1.0f + wv.x),
            sv.y * rstd * (1.0f + wv.y));
    }

    if ((D & 1) && tid == 0) {
        __nv_bfloat16 sumv = reinterpret_cast<__nv_bfloat16 *>(s_sum)[D - 1];
        float sv = __bfloat162float(sumv);
        ho[D - 1] = sumv;
        ro[D - 1] = sumv;
        no[D - 1] = __float2bfloat16(sv * rstd * (1.0f + __bfloat162float(w[D - 1])));
    }
}

// SiLU(gate) * up — bf16 inputs → bf16 output
__global__ void pf_silu_mul_bf16(const __nv_bfloat16 *gate, const __nv_bfloat16 *up, __nv_bfloat16 *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) { float g = __bfloat162float(gate[i]); out[i] = __float2bfloat16(pf_silu(g) * __bfloat162float(up[i])); }
}

// SiLU(gate) * up from a fused [rows, 2 * cols] row-major buffer.
__global__ void pf_silu_mul_fused_bf16(const __nv_bfloat16 *gate_up, __nv_bfloat16 *out, int rows, int cols) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int col_pairs = cols / 2;
    int total_pairs = rows * col_pairs;
    if (pair >= total_pairs) return;

    int row = pair / col_pairs;
    int col_pair = pair - row * col_pairs;
    const __nv_bfloat162 *gate_up2 =
        reinterpret_cast<const __nv_bfloat162 *>(gate_up + row * (cols * 2));
    __nv_bfloat162 *out2 = reinterpret_cast<__nv_bfloat162 *>(out + row * cols);
    float2 gate = __bfloat1622float2(gate_up2[col_pair]);
    float2 up = __bfloat1622float2(gate_up2[col_pairs + col_pair]);
    out2[col_pair] = __floats2bfloat162_rn(
        pf_silu(gate.x) * up.x,
        pf_silu(gate.y) * up.y);
}

__global__ void pf_half_to_bf16(const __half *in, __nv_bfloat16 *out, int N) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int pairs = N / 2;
    if (pair < pairs) {
        const __half2 *in2 = reinterpret_cast<const __half2 *>(in);
        __nv_bfloat162 *out2 = reinterpret_cast<__nv_bfloat162 *>(out);
        float2 fv = __half22float2(in2[pair]);
        out2[pair] = __floats2bfloat162_rn(fv.x, fv.y);
    }
    if ((N & 1) && pair == 0) {
        out[N - 1] = __float2bfloat16(__half2float(in[N - 1]));
    }
}

// Precompute DeltaNet q/k once per head so the tiled recurrence no longer
// repeats the q/k conv and normalization work across every value tile.
template <int BLOCK_THREADS>
__global__ void __launch_bounds__(BLOCK_THREADS, 1)
pf_deltanet_prepare_qk(
    const __nv_bfloat16 *__restrict__ qkv_proj,
    int qkv_stride,
    const __nv_bfloat16 *__restrict__ conv_w,
    float *__restrict__ conv_buf,
    __nv_bfloat16 *__restrict__ qk_out,
    int qk_stride,
    int S)
{
    constexpr float Q_SCALE = 1.0f / 11.313708498984761f;
    constexpr int QK_CHANNELS = DN_KEY + DN_KEY;
    constexpr int HALF_WARPS = DN_KEY / 32;
    static_assert(BLOCK_THREADS % 32 == 0, "DeltaNet qk precompute block size must be warp-aligned");
    static_assert(BLOCK_THREADS == QK_CHANNELS, "DeltaNet qk precompute expects one thread per q/k channel");

    int h = blockIdx.x;
    if (h >= DN_HEADS) return;

    int tid = threadIdx.x, wid = tid / 32, lid = tid % 32;
    const int q_base = h * DN_KEY;
    const int k_base = DN_QK_SIZE + h * DN_KEY;
    const bool is_q = tid < DN_KEY;
    const int local_c = is_q ? tid : (tid - DN_KEY);
    const int global_ch = (is_q ? q_base : k_base) + local_c;

    __shared__ float s_partials[QK_CHANNELS / 32];
    __shared__ float s_norms[2];

    float4 hist = reinterpret_cast<const float4 *>(conv_buf + global_ch * DN_CONV_K)[0];
    const __nv_bfloat16 *weight_ptr = conv_w + global_ch * DN_CONV_K;
    float4 weight = make_float4(
        __bfloat162float(weight_ptr[0]),
        __bfloat162float(weight_ptr[1]),
        __bfloat162float(weight_ptr[2]),
        __bfloat162float(weight_ptr[3]));

    for (int t = 0; t < S; t++) {
        const __nv_bfloat16 *qkv_t = qkv_proj + t * qkv_stride;
        float in_val = is_q
            ? __bfloat162float(qkv_t[q_base + local_c])
            : __bfloat162float(qkv_t[k_base + local_c]);
        hist.x = hist.y;
        hist.y = hist.z;
        hist.z = hist.w;
        hist.w = in_val;
        float act = pf_silu(fmaf(hist.x, weight.x, fmaf(hist.y, weight.y, fmaf(hist.z, weight.z, hist.w * weight.w))));

        float sq = pf_warp_sum(act * act);
        if (lid == 0) {
            s_partials[wid] = sq;
        }
        __syncthreads();

        if (wid == 0 || wid == HALF_WARPS) {
            int partial_base = (wid == 0) ? 0 : HALF_WARPS;
            float total = (lid < HALF_WARPS) ? s_partials[partial_base + lid] : 0.0f;
            total = pf_warp_sum(total);
            if (lid == 0) {
                s_norms[wid == 0 ? 0 : 1] = rsqrtf(total + 1e-6f) * ((wid == 0) ? Q_SCALE : 1.0f);
            }
        }
        __syncthreads();

        __nv_bfloat16 *qk_t = qk_out + t * qk_stride + h * QK_CHANNELS;
        qk_t[is_q ? local_c : (DN_KEY + local_c)] = __float2bfloat16(act * s_norms[is_q ? 0 : 1]);
    }

    reinterpret_cast<float4 *>(conv_buf + global_ch * DN_CONV_K)[0] = hist;
}

// ===== Standalone DeltaNet recurrence tiled over value channels =====
template <int BLOCK_THREADS, int BLOCKS_PER_HEAD, bool SEPARATE_CTRL>
__global__ void __launch_bounds__(BLOCK_THREADS, 1)
pf_deltanet_recurrence_tiled(
    const __nv_bfloat16 *__restrict__ qkv_proj,
    int qkv_stride,
    const __nv_bfloat16 *__restrict__ qk_norm,
    int qk_stride,
    const __nv_bfloat16 *__restrict__ beta_proj,
    const __nv_bfloat16 *__restrict__ alpha_proj,
    int ctrl_stride,
    const __nv_bfloat16 *__restrict__ conv_w, const __nv_bfloat16 *__restrict__ a_log,
    const __nv_bfloat16 *__restrict__ dt_bias,
    float *__restrict__ state, float *__restrict__ conv_buf, __nv_bfloat16 *__restrict__ output, int S)
{
    constexpr int NWARPS = BLOCK_THREADS / 32;
    constexpr int VAL_TILE = DN_VAL / BLOCKS_PER_HEAD;
    static_assert(BLOCK_THREADS % 32 == 0, "DeltaNet prefill block size must be warp-aligned");
    static_assert(DN_VAL % BLOCKS_PER_HEAD == 0, "DeltaNet value width must divide the tile count");
    static_assert(VAL_TILE % NWARPS == 0, "DeltaNet value tile must divide the warp count");
    static_assert(DN_KEY == DN_VAL, "DeltaNet prefill kernel assumes equal key/value widths");

    int block = blockIdx.x;
    int h = block / BLOCKS_PER_HEAD;
    if (h >= DN_HEADS) return;
    int tile = block % BLOCKS_PER_HEAD;
    int val_offset = tile * VAL_TILE;

    int tid = threadIdx.x, wid = tid/32, lid = tid%32;
    float a_log_val = __bfloat162float(a_log[h]);
    float a_scale = __expf(a_log_val);
    float dt_b = __bfloat162float(dt_bias[h]);

    __shared__ float s_beta, s_decay;

    float *my_state = state + h * DN_KEY * DN_VAL + val_offset * DN_KEY;
    const int v_base = 2 * DN_QK_SIZE + h * DN_VAL + val_offset;

    constexpr int CPW = VAL_TILE / NWARPS;
    constexpr int RPL = DN_KEY / 32;
    static_assert(CPW <= 32, "DeltaNet value channels per warp must fit in one warp");
    float sreg[CPW * RPL];
    float4 v_hist_reg = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 v_weight_reg = make_float4(0.f, 0.f, 0.f, 0.f);

#pragma unroll
    for (int jj = 0; jj < CPW; jj++) {
        int j = wid * CPW + jj;
        for (int ii = 0; ii < RPL; ii++) {
            sreg[jj * RPL + ii] = my_state[j * DN_KEY + lid + ii * 32];
        }
    }
    if (lid < CPW) {
        int j = wid * CPW + lid;
        int global_ch = v_base + j;
        const float *conv_ptr = conv_buf + global_ch * DN_CONV_K;
        const __nv_bfloat16 *weight_ptr = conv_w + global_ch * DN_CONV_K;
        v_hist_reg = make_float4(conv_ptr[0], conv_ptr[1], conv_ptr[2], conv_ptr[3]);
        v_weight_reg = make_float4(
            __bfloat162float(weight_ptr[0]),
            __bfloat162float(weight_ptr[1]),
            __bfloat162float(weight_ptr[2]),
            __bfloat162float(weight_ptr[3]));
    }

    for (int t = 0; t < S; t++) {
        const __nv_bfloat16 *qkv_t = qkv_proj + t * qkv_stride;
        const __nv_bfloat16 *qk_t = qk_norm + t * qk_stride + h * (DN_KEY * 2);
        float beta_proj_val;
        float alpha_proj_val;
        if constexpr (SEPARATE_CTRL) {
            const __nv_bfloat16 *beta_t = beta_proj + t * ctrl_stride;
            const __nv_bfloat16 *alpha_t = alpha_proj + t * ctrl_stride;
            beta_proj_val = __bfloat162float(beta_t[h]);
            alpha_proj_val = __bfloat162float(alpha_t[h]);
        } else {
            beta_proj_val = __bfloat162float(qkv_t[DN_BETA_OFFSET + h]);
            alpha_proj_val = __bfloat162float(qkv_t[DN_ALPHA_OFFSET + h]);
        }

        float v_lane = 0.0f;
        if (lid < CPW) {
            int j = wid * CPW + lid;
            float4 hist = v_hist_reg;
            float4 weight = v_weight_reg;
            hist.x = hist.y;
            hist.y = hist.z;
            hist.z = hist.w;
            hist.w = __bfloat162float(qkv_t[v_base + j]);
            v_hist_reg = hist;
            float co = fmaf(hist.x, weight.x, fmaf(hist.y, weight.y, fmaf(hist.z, weight.z, hist.w * weight.w)));
            v_lane = pf_silu(co);
        }

        float v_reg[CPW];
#pragma unroll
        for (int jj = 0; jj < CPW; jj++) {
            v_reg[jj] = __shfl_sync(0xffffffff, v_lane, jj);
        }

        float q_reg[RPL];
        float k_reg[RPL];
#pragma unroll
        for (int ii = 0; ii < RPL; ii++) {
            int key_idx = lid + ii * 32;
            q_reg[ii] = __bfloat162float(qk_t[key_idx]);
            k_reg[ii] = __bfloat162float(qk_t[DN_KEY + key_idx]);
        }
        if (tid == 0) {
            s_beta = pf_sigmoid(beta_proj_val);
            float x = alpha_proj_val + dt_b;
            float sp = (x > 20.0f) ? x : log1pf(__expf(x));
            s_decay = __expf(-a_scale * sp);
        }
        __syncthreads();

        float beta = s_beta;
        float decay = s_decay;

        __nv_bfloat16 *out_h = output + t * DN_V_SIZE + h * DN_VAL + val_offset;

#pragma unroll
        for (int jj = 0; jj < CPW; jj++) {
            int j = wid * CPW + jj;
            float kv = 0.0f;
#pragma unroll
            for (int ii = 0; ii < RPL; ii++) {
                kv = fmaf(sreg[jj * RPL + ii], k_reg[ii], kv);
            }
            kv = pf_warp_sum(kv);
            kv = __shfl_sync(0xffffffff, kv, 0);
            float delta = (v_reg[jj] - decay * kv) * beta;
            float attn = 0.0f;
#pragma unroll
            for (int ii = 0; ii < RPL; ii++) {
                float new_state = fmaf(k_reg[ii], delta, decay * sreg[jj * RPL + ii]);
                sreg[jj * RPL + ii] = new_state;
                attn = fmaf(new_state, q_reg[ii], attn);
            }
            attn = pf_warp_sum(attn);
            if (lid == 0) out_h[j] = __float2bfloat16(attn);
        }
    }

    if (lid < CPW) {
        int j = wid * CPW + lid;
        float *conv_ptr = conv_buf + (v_base + j) * DN_CONV_K;
        conv_ptr[0] = v_hist_reg.x;
        conv_ptr[1] = v_hist_reg.y;
        conv_ptr[2] = v_hist_reg.z;
        conv_ptr[3] = v_hist_reg.w;
    }

#pragma unroll
    for (int jj = 0; jj < CPW; jj++) {
        int j = wid * CPW + jj;
        for (int ii = 0; ii < RPL; ii++) {
            my_state[j * DN_KEY + lid + ii * 32] = sreg[jj * RPL + ii];
        }
    }
}

__global__ void pf_deltanet_finalize(
    __nv_bfloat16 *attn_out, const __nv_bfloat16 *proj,
    int proj_stride, int z_offset,
    const __nv_bfloat16 *norm_w, int S)
{
    int idx = blockIdx.x;
    if (idx >= S * DN_HEADS) return;
    int t = idx / DN_HEADS;
    int h = idx % DN_HEADS;
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lid = tid % 32;

    __shared__ float smem[4];
    __nv_bfloat16 *row = attn_out + t * DN_V_SIZE + h * DN_VAL;
    const __nv_bfloat16 *z_h = proj + t * proj_stride + z_offset + h * DN_VAL;
    const __nv_bfloat162 *row2 = reinterpret_cast<const __nv_bfloat162 *>(row);
    const __nv_bfloat162 *z2 = reinterpret_cast<const __nv_bfloat162 *>(z_h);
    const __nv_bfloat162 *norm2 = reinterpret_cast<const __nv_bfloat162 *>(norm_w);
    __nv_bfloat162 *out2 = reinterpret_cast<__nv_bfloat162 *>(row);

    float sq = 0.0f;
    for (int i2 = tid; i2 < DN_VAL / 2; i2 += blockDim.x) {
        float2 v = __bfloat1622float2(row2[i2]);
        sq += v.x * v.x + v.y * v.y;
    }
    sq = pf_warp_sum(sq);
    if (lid == 0) {
        smem[wid] = sq;
    }
    __syncthreads();
    if (wid == 0) {
        float v = (lid < blockDim.x / 32) ? smem[lid] : 0.0f;
        v = pf_warp_sum(v);
        if (lid == 0) {
            smem[0] = rsqrtf(v / DN_VAL + RMS_EPS);
        }
    }
    __syncthreads();
    float rstd = smem[0];
    for (int i2 = tid; i2 < DN_VAL / 2; i2 += blockDim.x) {
        float2 rv = __bfloat1622float2(row2[i2]);
        float2 zv = __bfloat1622float2(z2[i2]);
        float2 nw = __bfloat1622float2(norm2[i2]);
        out2[i2] = __floats2bfloat162_rn(
            rv.x * rstd * nw.x * pf_silu(zv.x),
            rv.y * rstd * nw.y * pf_silu(zv.y));
    }
}

// ===== QK norm + RoPE + KV cache =====
__global__ void pf_qk_norm_rope(
    __nv_bfloat16 *qkv, int qkv_stride,
    const __nv_bfloat16 *qnw, const __nv_bfloat16 *knw,
    __nv_bfloat16 *k_cache, __nv_bfloat16 *v_cache, int S, int max_seq)
{
    constexpr int HALF_DIM = FA_HEAD_DIM / 2;
    constexpr int ROT_HALF = FA_ROT_DIM / 2;
    int idx = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    int total_q = S * FA_Q_HEADS, total_k = S * FA_KV_HEADS;
    if (idx < total_q) {
        int pos = idx / FA_Q_HEADS, head = idx % FA_Q_HEADS;
        __nv_bfloat16 *qh = qkv + pos * qkv_stride + head * FA_HEAD_DIM * 2;
        const __nv_bfloat162 *qh2 = reinterpret_cast<const __nv_bfloat162 *>(qh);
        const __nv_bfloat162 *qnw2 = reinterpret_cast<const __nv_bfloat162 *>(qnw);
        __nv_bfloat162 *qh2_out = reinterpret_cast<__nv_bfloat162 *>(qh);
        float ss = 0;
        for (int i2 = lid; i2 < HALF_DIM; i2 += 32) {
            float2 v = __bfloat1622float2(qh2[i2]);
            ss += v.x * v.x + v.y * v.y;
        }
        ss = pf_warp_sum(ss);
        float sc = rsqrtf(ss / FA_HEAD_DIM + RMS_EPS);
        sc = __shfl_sync(0xffffffff, sc, 0);

        for (int pair = lid; pair < FA_ROT_PAIRS; pair += 32) {
            int lo = pair;
            int hi = pair + FA_ROT_PAIRS;
            float lo_norm = __bfloat162float(qh[lo]) * sc * (1.f + __bfloat162float(qnw[lo]));
            float hi_norm = __bfloat162float(qh[hi]) * sc * (1.f + __bfloat162float(qnw[hi]));
            float angle = float(pos) * PF_ROPE_INV_FREQ[pair];
            float sv, cv;
            __sincosf(angle, &sv, &cv);
            qh[lo] = __float2bfloat16(lo_norm * cv - hi_norm * sv);
            qh[hi] = __float2bfloat16(hi_norm * cv + lo_norm * sv);
        }
        for (int i2 = lid + ROT_HALF; i2 < HALF_DIM; i2 += 32) {
            float2 rv = __bfloat1622float2(qh2[i2]);
            float2 wv = __bfloat1622float2(qnw2[i2]);
            qh2_out[i2] = __floats2bfloat162_rn(
                rv.x * sc * (1.f + wv.x),
                rv.y * sc * (1.f + wv.y));
        }
    }
    int kidx = idx - total_q;
    if (idx >= total_q && kidx < total_k) {
        int pos = kidx / FA_KV_HEADS, head = kidx % FA_KV_HEADS;
        __nv_bfloat16 *kh = qkv + pos * qkv_stride + FA_QPROJ_SIZE + head * FA_HEAD_DIM;
        const __nv_bfloat16 *vh = qkv + pos * qkv_stride + FA_QPROJ_SIZE + FA_KV_SIZE + head * FA_HEAD_DIM;
        __nv_bfloat16 *kc = k_cache + head*max_seq*FA_HEAD_DIM + pos*FA_HEAD_DIM;
        __nv_bfloat16 *vc = v_cache + head*max_seq*FA_HEAD_DIM + pos*FA_HEAD_DIM;
        const __nv_bfloat162 *kh2 = reinterpret_cast<const __nv_bfloat162 *>(kh);
        const __nv_bfloat162 *vh2 = reinterpret_cast<const __nv_bfloat162 *>(vh);
        const __nv_bfloat162 *knw2 = reinterpret_cast<const __nv_bfloat162 *>(knw);
        __nv_bfloat162 *kh2_out = reinterpret_cast<__nv_bfloat162 *>(kh);
        __nv_bfloat162 *kc2 = reinterpret_cast<__nv_bfloat162 *>(kc);
        __nv_bfloat162 *vc2 = reinterpret_cast<__nv_bfloat162 *>(vc);
        float ss = 0;
        for (int i2 = lid; i2 < HALF_DIM; i2 += 32) {
            float2 v = __bfloat1622float2(kh2[i2]);
            ss += v.x * v.x + v.y * v.y;
        }
        ss = pf_warp_sum(ss);
        float sc = rsqrtf(ss / FA_HEAD_DIM + RMS_EPS);
        sc = __shfl_sync(0xffffffff, sc, 0);

        for (int pair = lid; pair < FA_ROT_PAIRS; pair += 32) {
            int lo = pair;
            int hi = pair + FA_ROT_PAIRS;
            float lo_norm = __bfloat162float(kh[lo]) * sc * (1.f + __bfloat162float(knw[lo]));
            float hi_norm = __bfloat162float(kh[hi]) * sc * (1.f + __bfloat162float(knw[hi]));
            float angle = float(pos) * PF_ROPE_INV_FREQ[pair];
            float sv, cv;
            __sincosf(angle, &sv, &cv);
            __nv_bfloat16 lo_out = __float2bfloat16(lo_norm * cv - hi_norm * sv);
            __nv_bfloat16 hi_out = __float2bfloat16(hi_norm * cv + lo_norm * sv);
            kh[lo] = lo_out;
            kh[hi] = hi_out;
            kc[lo] = lo_out;
            kc[hi] = hi_out;
            vc[lo] = vh[lo];
            vc[hi] = vh[hi];
        }
        for (int i2 = lid + ROT_HALF; i2 < HALF_DIM; i2 += 32) {
            float2 kv = __bfloat1622float2(kh2[i2]);
            float2 wv = __bfloat1622float2(knw2[i2]);
            __nv_bfloat162 out2 = __floats2bfloat162_rn(
                kv.x * sc * (1.f + wv.x),
                kv.y * sc * (1.f + wv.y));
            kh2_out[i2] = out2;
            kc2[i2] = out2;
            vc2[i2] = vh2[i2];
        }
    }
}

// ===== Causal attention (bf16 Q/K/V, f32 accumulation, bf16 output) =====
__global__ void pf_causal_attn_generic(
    const __nv_bfloat16 *__restrict__ qkv,
    int qkv_stride,
    __nv_bfloat16 *__restrict__ out,
    int S)
{
    int idx = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    if (idx >= S * FA_Q_HEADS) return;
    int pos = idx / FA_Q_HEADS, qh = idx % FA_Q_HEADS, kvh = qh / FA_GQA;
    float scale = 1.0f / sqrtf(float(FA_HEAD_DIM));
    constexpr int EPL = FA_HEAD_DIM / 32;
    constexpr int PAIRS = EPL / 2;
    const __nv_bfloat16 *qv = qkv + pos * qkv_stride + qh * FA_HEAD_DIM * 2;
    const __nv_bfloat16 *gv = qv + FA_HEAD_DIM;
    __nv_bfloat16 *ov = out + pos*FA_Q_SIZE + qh*FA_HEAD_DIM;
    const __nv_bfloat162 *qv2 = reinterpret_cast<const __nv_bfloat162 *>(qv + lid * EPL);
    const __nv_bfloat162 *gv2 = reinterpret_cast<const __nv_bfloat162 *>(gv + lid * EPL);
    float2 ql[PAIRS];
    float2 oa[PAIRS];
#pragma unroll
    for (int p = 0; p < PAIRS; ++p) {
        ql[p] = __bfloat1622float2(qv2[p]);
        oa[p] = make_float2(0.0f, 0.0f);
    }
    float mx=-1e30f, se=0;
    for (int kp = 0; kp <= pos; kp++) {
        const __nv_bfloat16 *kv = qkv + kp * qkv_stride + FA_QPROJ_SIZE + kvh * FA_HEAD_DIM;
        const __nv_bfloat16 *vv = qkv + kp * qkv_stride + FA_QPROJ_SIZE + FA_KV_SIZE + kvh * FA_HEAD_DIM;
        const __nv_bfloat162 *kv2 = reinterpret_cast<const __nv_bfloat162 *>(kv + lid * EPL);
        const __nv_bfloat162 *vv2 = reinterpret_cast<const __nv_bfloat162 *>(vv + lid * EPL);
        float sc = 0.0f;
#pragma unroll
        for (int p = 0; p < PAIRS; ++p) {
            float2 kf = __bfloat1622float2(kv2[p]);
            sc += ql[p].x * kf.x + ql[p].y * kf.y;
        }
        sc=pf_warp_sum(sc)*scale; sc=__shfl_sync(0xffffffff,sc,0);
        float om=mx; mx=fmaxf(mx,sc); float ed=__expf(om-mx); float wexp=__expf(sc-mx); se=se*ed+wexp;
        float wt=wexp;
#pragma unroll
        for (int p = 0; p < PAIRS; ++p) {
            float2 vf = __bfloat1622float2(vv2[p]);
            oa[p].x = oa[p].x * ed + wt * vf.x;
            oa[p].y = oa[p].y * ed + wt * vf.y;
        }
    }
    float rs=1.f/se;
#pragma unroll
    for (int p = 0; p < PAIRS; ++p) {
        int i = lid * EPL + p * 2;
        float2 gf = __bfloat1622float2(gv2[p]);
        ov[i] = __float2bfloat16(oa[p].x * rs * pf_sigmoid(gf.x));
        ov[i + 1] = __float2bfloat16(oa[p].y * rs * pf_sigmoid(gf.y));
    }
}

template <int TILE_POS>
__global__ void pf_causal_attn_posblock(
    const __nv_bfloat16 *__restrict__ qkv,
    int qkv_stride,
    __nv_bfloat16 *__restrict__ out,
    int S)
{
    static_assert(FA_Q_HEADS == 8, "optimized prefill attention assumes 8 Q heads");
    static_assert(FA_KV_HEADS == 2, "optimized prefill attention assumes 2 KV heads");

    int pos = blockIdx.x;
    if (pos >= S) return;

    int warp_id = threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    int qh = warp_id;
    int kvh = qh / FA_GQA;

    constexpr int EPL = FA_HEAD_DIM / 32;
    constexpr int PAIRS = EPL / 2;
    constexpr int HALF_DIM = FA_HEAD_DIM / 2;
    constexpr float SCALE = 1.0f / 16.0f;

    __shared__ __align__(16) __nv_bfloat162 s_k_tile[TILE_POS * FA_KV_HEADS * HALF_DIM];
    __shared__ __align__(16) __nv_bfloat162 s_v_tile[TILE_POS * FA_KV_HEADS * HALF_DIM];

    const __nv_bfloat16 *qv = qkv + pos * qkv_stride + qh * FA_HEAD_DIM * 2;
    const __nv_bfloat16 *gv = qv + FA_HEAD_DIM;
    __nv_bfloat16 *ov = out + pos * FA_Q_SIZE + qh * FA_HEAD_DIM;
    const __nv_bfloat162 *qv2 = reinterpret_cast<const __nv_bfloat162 *>(qv + lid * EPL);
    const __nv_bfloat162 *gv2 = reinterpret_cast<const __nv_bfloat162 *>(gv + lid * EPL);

    float2 ql[PAIRS];
    float2 oa[PAIRS];
#pragma unroll
    for (int p = 0; p < PAIRS; ++p) {
        ql[p] = __bfloat1622float2(qv2[p]);
        oa[p] = make_float2(0.0f, 0.0f);
    }

    float mx = -1e30f;
    float se = 0.0f;

    for (int base = 0; base <= pos; base += TILE_POS) {
        int tile = min(TILE_POS, pos - base + 1);
        int tile_pairs = tile * FA_KV_HEADS * HALF_DIM;
        for (int idx = threadIdx.x; idx < tile_pairs; idx += blockDim.x) {
            int pair = idx % HALF_DIM;
            int tmp = idx / HALF_DIM;
            int head = tmp % FA_KV_HEADS;
            int t = tmp / FA_KV_HEADS;
            const __nv_bfloat16 *k_ptr = qkv + (base + t) * qkv_stride + FA_QPROJ_SIZE + head * FA_HEAD_DIM;
            const __nv_bfloat16 *v_ptr = qkv + (base + t) * qkv_stride + FA_QPROJ_SIZE + FA_KV_SIZE + head * FA_HEAD_DIM;
            const __nv_bfloat162 *k_ptr2 = reinterpret_cast<const __nv_bfloat162 *>(k_ptr);
            const __nv_bfloat162 *v_ptr2 = reinterpret_cast<const __nv_bfloat162 *>(v_ptr);
            int tile_idx = (t * FA_KV_HEADS + head) * HALF_DIM + pair;
            s_k_tile[tile_idx] = k_ptr2[pair];
            s_v_tile[tile_idx] = v_ptr2[pair];
        }
        __syncthreads();

        for (int t = 0; t < tile; ++t) {
            int tile_idx = (t * FA_KV_HEADS + kvh) * HALF_DIM + lid * PAIRS;
            const __nv_bfloat162 *kv2 = s_k_tile + tile_idx;
            const __nv_bfloat162 *vv2 = s_v_tile + tile_idx;
            float sc = 0.0f;
#pragma unroll
            for (int p = 0; p < PAIRS; ++p) {
                float2 kf = __bfloat1622float2(kv2[p]);
                sc += ql[p].x * kf.x + ql[p].y * kf.y;
            }
            sc = pf_warp_sum(sc) * SCALE;
            sc = __shfl_sync(0xffffffff, sc, 0);
            float old_mx = mx;
            mx = fmaxf(mx, sc);
            float ed = __expf(old_mx - mx);
            float wexp = __expf(sc - mx);
            se = se * ed + wexp;
#pragma unroll
            for (int p = 0; p < PAIRS; ++p) {
                float2 vf = __bfloat1622float2(vv2[p]);
                oa[p].x = oa[p].x * ed + wexp * vf.x;
                oa[p].y = oa[p].y * ed + wexp * vf.y;
            }
        }
        __syncthreads();
    }

    float rs = 1.0f / se;
#pragma unroll
    for (int p = 0; p < PAIRS; ++p) {
        int i = lid * EPL + p * 2;
        float2 gf = __bfloat1622float2(gv2[p]);
        ov[i] = __float2bfloat16(oa[p].x * rs * pf_sigmoid(gf.x));
        ov[i + 1] = __float2bfloat16(oa[p].y * rs * pf_sigmoid(gf.y));
    }
}

// Final norm
__global__ void pf_final_norm(const __nv_bfloat16 *hidden, const __nv_bfloat16 *w,
    __nv_bfloat16 *normed, __nv_bfloat16 *hidden_out, int S) {
    int tid=threadIdx.x, wid=tid/32, lid=tid%32;
    __shared__ float smem[16];
    const __nv_bfloat16 *row = hidden + (S-1)*HIDDEN;
    float sq=0; for(int i=tid;i<HIDDEN;i+=blockDim.x){float v=__bfloat162float(row[i]);sq+=v*v;}
    sq=pf_warp_sum(sq);if(lid==0)smem[wid]=sq;__syncthreads();
    if(wid==0){float v=(lid<blockDim.x/32)?smem[lid]:0;v=pf_warp_sum(v);if(lid==0)smem[0]=rsqrtf(v/HIDDEN+RMS_EPS);}
    __syncthreads();float rstd=smem[0];
    for(int i=tid;i<HIDDEN;i+=blockDim.x){
        float v=__bfloat162float(row[i]);
        normed[i]=__float2bfloat16(v*rstd*(1.f+__bfloat162float(w[i])));
        hidden_out[i]=row[i];
    }
}

// LM head: bf16 weight × bf16 hidden
__global__ void pf_lm_head(const __nv_bfloat16 *hidden, const __nv_bfloat16 *w,
    float *bmv, int *bmi, int N) {
    __shared__ __nv_bfloat16 s_h[HIDDEN];
    for(int i=threadIdx.x;i<HIDDEN;i+=blockDim.x) s_h[i]=hidden[i];
    __syncthreads();
    int wid=threadIdx.x/32, lid=threadIdx.x%32, nw=blockDim.x/32;
    int rpb=(N+gridDim.x-1)/gridDim.x, rs=blockIdx.x*rpb, re=min(rs+rpb,N);
    float lm=-1e30f; int li=-1;
    for(int m=rs+wid;m<re;m+=nw){const __nv_bfloat16 *wr=w+m*HIDDEN;float s=0;
        for(int k=lid*8;k<HIDDEN;k+=32*8){for(int i=0;i<8;i++)s+=__bfloat162float(wr[k+i])*__bfloat162float(s_h[k+i]);}
        s=pf_warp_sum(s);if(lid==0&&s>lm){lm=s;li=m;}}
    lm=__shfl_sync(0xffffffff,lm,0);li=__shfl_sync(0xffffffff,li,0);
    __shared__ float wm[32]; __shared__ int wi[32];
    if(lid==0){wm[wid]=lm;wi[wid]=li;}__syncthreads();
    if(wid==0){float mv=(lid<nw)?wm[lid]:-1e30f;int mi=(lid<nw)?wi[lid]:-1;
        for(int o=16;o>0;o>>=1){float ov=__shfl_down_sync(0xffffffff,mv,o);int oi=__shfl_down_sync(0xffffffff,mi,o);if(ov>mv){mv=ov;mi=oi;}}
        if(lid==0){bmv[blockIdx.x]=mv;bmi[blockIdx.x]=mi;}}
}
__global__ void pf_lm_reduce(const float *bmv, const int *bmi, int *out, int nb) {
    int tid=threadIdx.x; float best=-1e30f; int bi=-1;
    for(int i=tid;i<nb;i+=blockDim.x){float v=bmv[i];if(v>best){best=v;bi=bmi[i];}}
    __shared__ float sv[256]; __shared__ int si[256];
    sv[tid]=best;si[tid]=bi;__syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){if(tid<s&&sv[tid+s]>sv[tid]){sv[tid]=sv[tid+s];si[tid]=si[tid+s];}__syncthreads();}
    if(tid==0)*out=si[0];
}

// ===== cuBLAS bf16 GEMM =====
static void cublas_bf16_gemm(cublasHandle_t h,
    const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C,
    int S, int N, int K) {
    float alpha = 1.0f, beta_val = 0.0f;
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, S, K,
        &alpha, B, CUDA_R_16BF, K, A, CUDA_R_16BF, K,
        &beta_val, C, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

static bool prefill_tc_debug_enabled() {
    static int enabled = []() {
        const char *value = std::getenv("MEGAKERNEL_DEBUG_PREFILL_TC");
        return (value != nullptr && value[0] != '\0' && value[0] != '0') ? 1 : 0;
    }();
    return enabled != 0;
}

static bool prefill_tc_compare_enabled() {
    static int enabled = []() {
        const char *value = std::getenv("MEGAKERNEL_DEBUG_PREFILL_TC_COMPARE");
        return (value != nullptr && value[0] != '\0' && value[0] != '0') ? 1 : 0;
    }();
    return enabled != 0;
}

static bool prefill_tc_proj_enabled() {
    static int enabled = []() {
        const char *master = std::getenv("MEGAKERNEL_PREFILL_TC");
        if (master == nullptr || master[0] == '\0' || master[0] == '0') {
            return 0;
        }
        const char *value = std::getenv("MEGAKERNEL_PREFILL_TC_PROJ");
        return (value == nullptr || value[0] == '\0' || value[0] != '0') ? 1 : 0;
    }();
    return enabled != 0;
}

static bool prefill_tc_gate_up_enabled() {
    static int enabled = []() {
        const char *master = std::getenv("MEGAKERNEL_PREFILL_TC");
        if (master == nullptr || master[0] == '\0' || master[0] == '0') {
            return 0;
        }
        const char *value = std::getenv("MEGAKERNEL_PREFILL_TC_GATE_UP");
        return (value != nullptr && value[0] != '\0' && value[0] != '0') ? 1 : 0;
    }();
    return enabled != 0;
}

static size_t nvfp4_tc_scale_bytes(int rows, int cols) {
    return static_cast<size_t>((rows + NVFP4_TC_ROWS_PER_TILE - 1) / NVFP4_TC_ROWS_PER_TILE) *
           static_cast<size_t>(cols / NVFP4_TC_K_PER_TILE) * 512ull;
}

static void debug_compare_prefill_tc_proj(
    cublasHandle_t cublas,
    const __nv_bfloat16 *normalized,
    const __nv_bfloat16 *weight_bf16,
    const __nv_bfloat16 *tc_out,
    int seq_len,
    int logical_rows,
    int out_stride,
    int k,
    const char *tag)
{
    static int already_dumped = 0;
    if (!prefill_tc_compare_enabled() || already_dumped) {
        return;
    }

    __nv_bfloat16 *ref = nullptr;
    if (cudaMalloc(&ref, static_cast<size_t>(seq_len) * logical_rows * sizeof(__nv_bfloat16)) != cudaSuccess) {
        return;
    }

    cudaStream_t stream = nullptr;
    cublasGetStream(cublas, &stream);
    cublas_bf16_gemm(cublas, normalized, weight_bf16, ref, seq_len, logical_rows, k);
    cudaStreamSynchronize(stream);

    int ref_count = seq_len * logical_rows;
    int tc_count = seq_len * out_stride;
    std::vector<__nv_bfloat16> tc_host(tc_count);
    std::vector<__nv_bfloat16> ref_host(ref_count);
    cudaMemcpy(tc_host.data(), tc_out, static_cast<size_t>(tc_count) * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(ref_host.data(), ref, static_cast<size_t>(ref_count) * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    auto tc_at = [&](int row, int col) { return __bfloat162float(tc_host[row * out_stride + col]); };
    auto bf16_at = [&](int row, int col) { return __bfloat162float(ref_host[row * logical_rows + col]); };

    float max_row_major = 0.0f;
    float sum_row_major = 0.0f;
    int finite_row = 0;
    int total = seq_len * logical_rows;
    for (int s = 0; s < seq_len; ++s) {
        for (int n = 0; n < logical_rows; ++n) {
            float ref_v = bf16_at(s, n);
            float row_v = tc_at(s, n);
            float diff_row = fabsf(row_v - ref_v);
            max_row_major = fmaxf(max_row_major, diff_row);
            sum_row_major += diff_row;
            finite_row += int(isfinite(row_v));
        }
    }

    std::fprintf(
        stderr,
        "prefill_tc_compare[%s]: seq=%d logical=%d stride=%d k=%d row_max=%.6f row_mean=%.6f row_finite=%d/%d ref0=%.6f row0=%.6f\n",
        tag,
        seq_len,
        logical_rows,
        out_stride,
        k,
        max_row_major,
        sum_row_major / total,
        finite_row,
        total,
        bf16_at(0, 0),
        tc_at(0, 0));

    already_dumped = 1;
    cudaFree(ref);
}

static bool cublaslt_nvfp4_bf16_gemm(
    const uint8_t *weight_packed,
    const uint8_t *weight_scales,
    const uint8_t *act_packed,
    const uint8_t *act_scales,
    __nv_bfloat16 *out,
    int out_rows,
    int seq_len,
    int k,
    cudaStream_t stream)
{
    if (!weight_packed || !weight_scales || !act_packed || !act_scales || !out) {
        return false;
    }

    static cublasLtHandle_t handle = nullptr;
    static void *workspace = nullptr;
    static size_t workspace_size = 0;
    if (!handle && cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (workspace == nullptr) {
        if (cudaMalloc(&workspace, PREFILL_TC_MAX_WORKSPACE) != cudaSuccess) {
            workspace = nullptr;
            workspace_size = 0;
            return false;
        }
        workspace_size = PREFILL_TC_MAX_WORKSPACE;
    }

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatrixLayout_t d_desc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    bool ok = false;
    cublasLtMatmulAlgo_t algo{};
    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned_results = 0;
    cublasStatus_t heuristic_status = CUBLAS_STATUS_SUCCESS;

    cublasOperation_t trans_a = CUBLAS_OP_T;
    cublasOperation_t trans_b = CUBLAS_OP_N;
    cublasLtMatmulMatrixScale_t a_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulMatrixScale_t b_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtOrder_t col_order = CUBLASLT_ORDER_COL;

    if (cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) {
        goto cleanup;
    }
    if (cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &a_scale_mode, sizeof(a_scale_mode)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &b_scale_mode, sizeof(b_scale_mode)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &weight_scales, sizeof(weight_scales)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &act_scales, sizeof(act_scales)) != CUBLAS_STATUS_SUCCESS) {
        goto cleanup;
    }

    if (cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_4F_E2M1, k, out_rows, k) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_4F_E2M1, k, seq_len, k) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, out_rows, seq_len, out_rows) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_16BF, out_rows, seq_len, out_rows) != CUBLAS_STATUS_SUCCESS) {
        goto cleanup;
    }

    cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_order, sizeof(col_order));
    cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_order, sizeof(col_order));
    cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_order, sizeof(col_order));
    cublasLtMatrixLayoutSetAttribute(d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_order, sizeof(col_order));

    if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS) {
        goto cleanup;
    }
    cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(workspace_size));

    heuristic_status = cublasLtMatmulAlgoGetHeuristic(
        handle,
        op_desc,
        a_desc,
        b_desc,
        c_desc,
        d_desc,
        preference,
        1,
        &heuristic,
        &returned_results);
    if (heuristic_status != CUBLAS_STATUS_SUCCESS || returned_results == 0 || heuristic.workspaceSize > workspace_size) {
        goto cleanup;
    }
    algo = heuristic.algo;

    {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cudaGetLastError();
        cublasStatus_t status = cublasLtMatmul(
            handle,
            op_desc,
            &alpha,
            weight_packed,
            a_desc,
            act_packed,
            b_desc,
            &beta,
            out,
            c_desc,
            out,
            d_desc,
            &algo,
            workspace,
            heuristic.workspaceSize,
            stream);
        cudaError_t last_error = cudaGetLastError();
        ok = (status == CUBLAS_STATUS_SUCCESS && last_error == cudaSuccess);
        if (prefill_tc_debug_enabled() && !ok) {
            std::fprintf(stderr,
                         "prefill_tc_cublaslt: status=%d cuda=%d m=%d n=%d k=%d\n",
                         int(status), int(last_error), out_rows, seq_len, k);
        }
    }

cleanup:
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (d_desc) cublasLtMatrixLayoutDestroy(d_desc);
    if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
    if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
    if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
    if (op_desc) cublasLtMatmulDescDestroy(op_desc);
    return ok;
}

// ===== Main orchestrator =====
static void launch_prefill_bf16_impl(
    const int *token_ids, int seq_len, int *output_token,
    const __nv_bfloat16 *embed_weight, const PFLayerWeights *layers,
    const PFFusedLayerWeights *fused_layers,
    const __nv_bfloat16 *final_norm_w, const __nv_bfloat16 *lm_head_w,
    const void *lm_head_weight_packed, const void *lm_head_scales,
    __nv_bfloat16 *fa_k_cache, __nv_bfloat16 *fa_v_cache,
    float *dn_states, float *conv_bufs,
    // Scratch (ALL bf16 except state/conv which are f32)
    __nv_bfloat16 *hidden, __nv_bfloat16 *residual, __nv_bfloat16 *normalized,
    __nv_bfloat16 *proj_buf, __nv_bfloat16 *proj_buf2, __half *proj_buf_half,
    uint8_t *proj_act_packed, uint8_t *proj_act_scales,
    __nv_bfloat16 *attn_buf, __nv_bfloat16 *mlp_buf,
    __nv_bfloat16 *dn_out_buf,
    float *beta_buf, float *alpha_buf,
    __nv_bfloat16 *final_normed, __nv_bfloat16 *hidden_bf16_out,
    float *lm_bmv, int *lm_bmi,
    __nv_bfloat16 *lm_hidden_bf16, uint8_t *lm_hidden_packed,
    uint8_t *lm_hidden_scales, __half *lm_logits_f16,
    cudaStream_t stream)
{
    static cublasHandle_t cublas = nullptr;
    if (!cublas) cublasCreate(&cublas);
    cublasSetStream(cublas, stream);

    static PFLayerWeights hl[NUM_LAYERS];
    static const PFLayerWeights *cached_layers = nullptr;
    if (layers != cached_layers) {
        cudaMemcpy(hl, layers, NUM_LAYERS * sizeof(PFLayerWeights), cudaMemcpyDeviceToHost);
        cached_layers = layers;
    }

    static PFFusedLayerWeights hfl[NUM_LAYERS];
    static const PFFusedLayerWeights *cached_fused_layers = nullptr;
    if (fused_layers != cached_fused_layers) {
        cudaMemcpy(hfl, fused_layers, NUM_LAYERS * sizeof(PFFusedLayerWeights), cudaMemcpyDeviceToHost);
        cached_fused_layers = fused_layers;
    }

    int S = seq_len;
    int bk = (S*HIDDEN+255)/256;
    int res_bk = (S*HIDDEN + 511) / 512;

    pf_embed<<<bk, 256, 0, stream>>>(token_ids, embed_weight, hidden, S);

    int fa_stride = FA_KV_HEADS * 2048 * FA_HEAD_DIM;
    int dn_stride = DN_HEADS * DN_KEY * DN_VAL;
    int fa_idx = 0, dn_idx = 0;

    for (int li = 0; li < NUM_LAYERS; li++) {
        const PFLayerWeights &lw = hl[li];
        const PFFusedLayerWeights &fw = hfl[li];
        int lt = LAYER_TYPE[li];
        bool use_tc_weights = (
            fw.proj_weight_packed != nullptr &&
            fw.proj_weight_scales != nullptr &&
            fw.gate_up_weight_packed != nullptr &&
            fw.gate_up_weight_scales != nullptr &&
            proj_act_packed != nullptr &&
            proj_act_scales != nullptr);
        bool use_tc_proj = use_tc_weights && prefill_tc_proj_enabled();
        bool use_tc_gate_up = use_tc_weights && prefill_tc_gate_up_enabled();

        const __nv_bfloat16 *norm_w = (const __nv_bfloat16 *)lw.ptrs[0];
        pf_rmsnorm<<<S, 256, 0, stream>>>(hidden, norm_w, normalized, residual, S, HIDDEN);
        if (use_tc_proj) {
            cudaMemsetAsync(proj_act_scales, 0, nvfp4_tc_scale_bytes(S, HIDDEN), stream);
            launch_quantize_nvfp4_lm_out(
                normalized,
                S,
                HIDDEN,
                proj_act_packed,
                proj_act_scales,
                stream);
        }

        if (lt == 0) {
            // DeltaNet
            constexpr int DN_QK_PRECOMP_STRIDE = DN_HEADS * DN_KEY * 2;
            constexpr int DN_CTRL_STRIDE = DN_HEADS;
            const __nv_bfloat16 *conv_w=(const __nv_bfloat16*)lw.ptrs[5];
            const __nv_bfloat16 *a_log=(const __nv_bfloat16*)lw.ptrs[6];
            const __nv_bfloat16 *dt_bias=(const __nv_bfloat16*)lw.ptrs[7];
            const __nv_bfloat16 *dn_norm=(const __nv_bfloat16*)lw.ptrs[8];
            const __nv_bfloat16 *out_w=(const __nv_bfloat16*)lw.ptrs[9];
            const __nv_bfloat16 *post_norm=(const __nv_bfloat16*)lw.ptrs[10];
            const __nv_bfloat16 *beta_w=(const __nv_bfloat16*)lw.ptrs[3];
            const __nv_bfloat16 *alpha_w=(const __nv_bfloat16*)lw.ptrs[4];
            const __nv_bfloat16 *down_w=(const __nv_bfloat16*)lw.ptrs[13];
            const __nv_bfloat16 *proj_fused_w = (const __nv_bfloat16*)fw.proj_weight;
            const __nv_bfloat16 *gate_up_fused_w = (const __nv_bfloat16*)fw.gate_up_weight;
            int dn_proj_stride = DN_PROJ_FUSED;
            __nv_bfloat16 *beta_proj_bf16 = nullptr;
            __nv_bfloat16 *alpha_proj_bf16 = nullptr;

            bool used_tc = use_tc_proj && cublaslt_nvfp4_bf16_gemm(
                (const uint8_t *)fw.proj_weight_packed,
                (const uint8_t *)fw.proj_weight_scales,
                proj_act_packed,
                proj_act_scales,
                proj_buf,
                DN_PROJ_FUSED_PADDED,
                S,
                HIDDEN,
                stream);
            if (used_tc) {
                dn_proj_stride = DN_PROJ_FUSED_PADDED;
                beta_proj_bf16 = reinterpret_cast<__nv_bfloat16 *>(beta_buf);
                alpha_proj_bf16 = reinterpret_cast<__nv_bfloat16 *>(alpha_buf);
                cublas_bf16_gemm(cublas, normalized, beta_w, beta_proj_bf16, S, DN_HEADS, HIDDEN);
                cublas_bf16_gemm(cublas, normalized, alpha_w, alpha_proj_bf16, S, DN_HEADS, HIDDEN);
                debug_compare_prefill_tc_proj(
                    cublas,
                    normalized,
                    proj_fused_w,
                    proj_buf,
                    S,
                    DN_PROJ_FUSED,
                    DN_PROJ_FUSED_PADDED,
                    HIDDEN,
                    "dn_proj");
            } else {
                cublas_bf16_gemm(cublas, normalized, proj_fused_w, proj_buf, S, DN_PROJ_FUSED, HIDDEN);
            }
            pf_deltanet_prepare_qk<256><<<DN_HEADS, 256, 0, stream>>>(
                proj_buf, dn_proj_stride,
                conv_w,
                conv_bufs + dn_idx * DN_CONV_CH * DN_CONV_K,
                proj_buf2,
                DN_QK_PRECOMP_STRIDE,
                S);

            // Standalone recurrence
            if (beta_proj_bf16 != nullptr && alpha_proj_bf16 != nullptr) {
                pf_deltanet_recurrence_tiled<PREFILL_DN_BLOCK_SIZE, PREFILL_DN_BLOCKS_PER_HEAD, true>
                    <<<DN_HEADS * PREFILL_DN_BLOCKS_PER_HEAD, PREFILL_DN_BLOCK_SIZE, 0, stream>>>(
                    proj_buf, dn_proj_stride,
                    proj_buf2, DN_QK_PRECOMP_STRIDE,
                    beta_proj_bf16, alpha_proj_bf16, DN_CTRL_STRIDE,
                    conv_w, a_log, dt_bias,
                    dn_states + dn_idx*dn_stride,
                    conv_bufs + dn_idx*DN_CONV_CH*DN_CONV_K,
                    dn_out_buf, S);
            } else {
                pf_deltanet_recurrence_tiled<PREFILL_DN_BLOCK_SIZE, PREFILL_DN_BLOCKS_PER_HEAD, false>
                    <<<DN_HEADS * PREFILL_DN_BLOCKS_PER_HEAD, PREFILL_DN_BLOCK_SIZE, 0, stream>>>(
                    proj_buf, dn_proj_stride,
                    proj_buf2, DN_QK_PRECOMP_STRIDE,
                    nullptr, nullptr, DN_CTRL_STRIDE,
                    conv_w, a_log, dt_bias,
                    dn_states + dn_idx*dn_stride,
                    conv_bufs + dn_idx*DN_CONV_CH*DN_CONV_K,
                    dn_out_buf, S);
            }
            pf_deltanet_finalize<<<S * DN_HEADS, 64, 0, stream>>>(
                dn_out_buf,
                proj_buf,
                dn_proj_stride,
                DN_CONV_CH,
                dn_norm,
                S);

            // Out projection + residual
            cublas_bf16_gemm(cublas, dn_out_buf, out_w, proj_buf2, S, HIDDEN, DN_V_SIZE);
            pf_add_residual_rmsnorm_bf16<<<S, 256, 0, stream>>>(
                proj_buf2, residual, post_norm, hidden, normalized, residual, S, HIDDEN);

            // MLP
            int mlp_bk = (S*INTER + 511) / 512;
            if (use_tc_gate_up) {
                cudaMemsetAsync(proj_act_scales, 0, nvfp4_tc_scale_bytes(S, HIDDEN), stream);
                launch_quantize_nvfp4_lm_out(
                    normalized,
                    S,
                    HIDDEN,
                    proj_act_packed,
                    proj_act_scales,
                    stream);
            }
            used_tc = use_tc_gate_up && cublaslt_nvfp4_bf16_gemm(
                (const uint8_t *)fw.gate_up_weight_packed,
                (const uint8_t *)fw.gate_up_weight_scales,
                proj_act_packed,
                proj_act_scales,
                proj_buf,
                MLP_GATE_UP_FUSED,
                S,
                HIDDEN,
                stream);
            if (!used_tc) {
                cublas_bf16_gemm(cublas, normalized, gate_up_fused_w, proj_buf, S, MLP_GATE_UP_FUSED, HIDDEN);
            }
            pf_silu_mul_fused_bf16<<<mlp_bk, 256, 0, stream>>>(proj_buf, mlp_buf, S, INTER);
            cublas_bf16_gemm(cublas, mlp_buf, down_w, proj_buf2, S, HIDDEN, INTER);
            pf_add_residual_bf16<<<res_bk, 256, 0, stream>>>(proj_buf2, residual, hidden, S*HIDDEN);

            dn_idx++;
        } else {
            // Full Attention
            const __nv_bfloat16 *q_nw=(const __nv_bfloat16*)lw.ptrs[4];
            const __nv_bfloat16 *k_nw=(const __nv_bfloat16*)lw.ptrs[5];
            const __nv_bfloat16 *o_w=(const __nv_bfloat16*)lw.ptrs[6];
            const __nv_bfloat16 *post_norm=(const __nv_bfloat16*)lw.ptrs[7];
            const __nv_bfloat16 *down_w=(const __nv_bfloat16*)lw.ptrs[10];
            const __nv_bfloat16 *proj_fused_w = (const __nv_bfloat16*)fw.proj_weight;
            const __nv_bfloat16 *gate_up_fused_w = (const __nv_bfloat16*)fw.gate_up_weight;

            bool used_tc = use_tc_proj && cublaslt_nvfp4_bf16_gemm(
                (const uint8_t *)fw.proj_weight_packed,
                (const uint8_t *)fw.proj_weight_scales,
                proj_act_packed,
                proj_act_scales,
                proj_buf,
                FA_QKV_FUSED,
                S,
                HIDDEN,
                stream);
            if (used_tc) {
                debug_compare_prefill_tc_proj(
                    cublas,
                    normalized,
                    proj_fused_w,
                    proj_buf,
                    S,
                    FA_QKV_FUSED,
                    FA_QKV_FUSED,
                    HIDDEN,
                    "fa_proj");
            } else {
                cublas_bf16_gemm(cublas, normalized, proj_fused_w, proj_buf, S, FA_QKV_FUSED, HIDDEN);
            }

            constexpr int PREFILL_FA_WARPS = PREFILL_FA_BLOCK_SIZE / 32;
            int total_heads = S*(FA_Q_HEADS+FA_KV_HEADS);
            pf_qk_norm_rope<<<(total_heads + PREFILL_FA_WARPS - 1) / PREFILL_FA_WARPS, PREFILL_FA_BLOCK_SIZE, 0, stream>>>(
                proj_buf, FA_QKV_FUSED, q_nw, k_nw,
                fa_k_cache + fa_idx*fa_stride, fa_v_cache + fa_idx*fa_stride, S, 2048);

            if constexpr (PREFILL_FA_BLOCK_SIZE == FA_Q_HEADS * 32) {
                pf_causal_attn_posblock<8><<<S, PREFILL_FA_BLOCK_SIZE, 0, stream>>>(
                    proj_buf, FA_QKV_FUSED, dn_out_buf, S);
            } else {
                pf_causal_attn_generic<<<(S * FA_Q_HEADS + PREFILL_FA_WARPS - 1) / PREFILL_FA_WARPS, PREFILL_FA_BLOCK_SIZE, 0, stream>>>(
                    proj_buf, FA_QKV_FUSED, dn_out_buf, S);
            }

            cublas_bf16_gemm(cublas, dn_out_buf, o_w, proj_buf2, S, HIDDEN, FA_Q_SIZE);
            pf_add_residual_rmsnorm_bf16<<<S, 256, 0, stream>>>(
                proj_buf2, residual, post_norm, hidden, normalized, residual, S, HIDDEN);

            // MLP
            int mlp_bk = (S*INTER + 511) / 512;
            if (use_tc_gate_up) {
                cudaMemsetAsync(proj_act_scales, 0, nvfp4_tc_scale_bytes(S, HIDDEN), stream);
                launch_quantize_nvfp4_lm_out(
                    normalized,
                    S,
                    HIDDEN,
                    proj_act_packed,
                    proj_act_scales,
                    stream);
            }
            used_tc = use_tc_gate_up && cublaslt_nvfp4_bf16_gemm(
                (const uint8_t *)fw.gate_up_weight_packed,
                (const uint8_t *)fw.gate_up_weight_scales,
                proj_act_packed,
                proj_act_scales,
                proj_buf,
                MLP_GATE_UP_FUSED,
                S,
                HIDDEN,
                stream);
            if (!used_tc) {
                cublas_bf16_gemm(cublas, normalized, gate_up_fused_w, proj_buf, S, MLP_GATE_UP_FUSED, HIDDEN);
            }
            pf_silu_mul_fused_bf16<<<mlp_bk, 256, 0, stream>>>(proj_buf, mlp_buf, S, INTER);
            cublas_bf16_gemm(cublas, mlp_buf, down_w, proj_buf2, S, HIDDEN, INTER);
            pf_add_residual_bf16<<<res_bk, 256, 0, stream>>>(proj_buf2, residual, hidden, S*HIDDEN);

            fa_idx++;
        }
    }

    pf_final_norm<<<1, 512, 0, stream>>>(hidden, final_norm_w, final_normed, hidden_bf16_out, S);

    bool used_nvfp4_lm = false;
    if (lm_head_weight_packed && lm_head_scales && lm_hidden_bf16 && lm_hidden_packed &&
        lm_hidden_scales && lm_logits_f16) {
        used_nvfp4_lm = launch_lm_head_cublaslt_bf16_top1(
            final_normed,
            lm_head_weight_packed,
            lm_head_scales,
            lm_hidden_bf16,
            lm_hidden_packed,
            lm_hidden_scales,
            lm_logits_f16,
            lm_bmv,
            lm_bmi,
            output_token,
            stream);
    }
    if (!used_nvfp4_lm) {
        int lm_blocks = 512;
        pf_lm_head<<<lm_blocks, 256, 0, stream>>>(final_normed, lm_head_w, lm_bmv, lm_bmi, VOCAB);
        pf_lm_reduce<<<1, 256, 0, stream>>>(lm_bmv, lm_bmi, output_token, lm_blocks);
    }
}

}  // anonymous namespace

// ===== Public entry point =====

extern "C" void launch_prefill_bf16_nvfp4_lm(
    const int *token_ids, int seq_len, int *output_token,
    const void *embed_weight, const PFLayerWeights *layers,
    const PFFusedLayerWeights *fused_layers,
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
    cudaStream_t stream)
{
    launch_prefill_bf16_impl(
        token_ids, seq_len, output_token,
        (const __nv_bfloat16 *)embed_weight, layers, fused_layers,
        (const __nv_bfloat16 *)final_norm_w, (const __nv_bfloat16 *)lm_head_w,
        lm_head_weight_packed, lm_head_scales,
        (__nv_bfloat16 *)fa_k_cache, (__nv_bfloat16 *)fa_v_cache,
        (float *)dn_states, (float *)conv_bufs,
        (__nv_bfloat16 *)hidden, (__nv_bfloat16 *)residual, (__nv_bfloat16 *)normalized,
        (__nv_bfloat16 *)proj_buf, (__nv_bfloat16 *)proj_buf2, (__half *)proj_buf_half,
        (uint8_t *)proj_act_packed, (uint8_t *)proj_act_scales,
        (__nv_bfloat16 *)attn_buf, (__nv_bfloat16 *)mlp_buf,
        (__nv_bfloat16 *)dn_out_buf,
        (float *)beta_buf, (float *)alpha_buf,
        (__nv_bfloat16 *)final_normed, (__nv_bfloat16 *)hidden_bf16_out,
        (float *)lm_bmv, (int *)lm_bmi,
        (__nv_bfloat16 *)lm_hidden_bf16, (uint8_t *)lm_hidden_packed,
        (uint8_t *)lm_hidden_scales, (__half *)lm_logits_f16,
        stream);
}
