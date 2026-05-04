/**
 * Fused single-kernel decode for Qwen3.5-0.8B (hybrid DeltaNet + Full Attention).
 * ALL BF16: weights bf16, activations bf16, accumulation f32.
 * DeltaNet state: f32 (recurrence needs precision).
 *
 * Optimized for: NVIDIA RTX 3090 (sm_86, 82 SMs)
 * Model:         Qwen/Qwen3.5-0.8B (bf16 weights)
 */

#include "half_type.h"
#include <cuda_runtime.h>

// =============================================================================
// Model constants
// =============================================================================

constexpr int WARP_SIZE = 32;
constexpr int HIDDEN_SIZE = 1024;
constexpr int INTERMEDIATE_SIZE = 3584;
constexpr int NUM_LAYERS = 24;
constexpr float RMS_EPS = 1e-6f;
constexpr int VOCAB_SIZE = 248320;

// Full Attention
constexpr int FA_NUM_Q_HEADS = 8;
constexpr int FA_NUM_KV_HEADS = 2;
constexpr int FA_HEAD_DIM = 256;
constexpr int FA_GQA_RATIO = FA_NUM_Q_HEADS / FA_NUM_KV_HEADS;
constexpr int FA_Q_SIZE = FA_NUM_Q_HEADS * FA_HEAD_DIM;
constexpr int FA_GATE_SIZE = FA_Q_SIZE;
constexpr int FA_QPROJ_SIZE = FA_Q_SIZE + FA_GATE_SIZE;
constexpr int FA_KV_SIZE = FA_NUM_KV_HEADS * FA_HEAD_DIM;
constexpr int FA_ROTARY_DIM = 64;
constexpr float FA_ROPE_THETA = 10000000.0f;

// DeltaNet
constexpr int DN_NUM_HEADS = 16;
constexpr int DN_KEY_DIM = 128;
constexpr int DN_VALUE_DIM = 128;
constexpr int DN_CONV_KERNEL = 4;
constexpr int DN_QK_SIZE = DN_NUM_HEADS * DN_KEY_DIM;
constexpr int DN_V_SIZE = DN_NUM_HEADS * DN_VALUE_DIM;
constexpr int DN_CONV_CHANNELS = DN_QK_SIZE + DN_QK_SIZE + DN_V_SIZE;

constexpr int MAX_ACT_DIM = (HIDDEN_SIZE > INTERMEDIATE_SIZE) ? HIDDEN_SIZE : INTERMEDIATE_SIZE;

#ifndef NUM_BLOCKS
#define NUM_BLOCKS 82
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

#ifndef LM_NUM_BLOCKS
#define LM_NUM_BLOCKS 512
#endif
#ifndef LM_BLOCK_SIZE
#define LM_BLOCK_SIZE 256
#endif

static int g_decode_blocks_override = 0;

__device__ __constant__ int LAYER_TYPE[NUM_LAYERS] = {
    0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1
};

// =============================================================================
// Weight structs — ALL BF16
// =============================================================================

struct FullAttnWeights {
    const half_t *input_layernorm_weight;   // [1024]
    const half_t *q_proj_weight;             // [4096, 1024]
    const half_t *k_proj_weight;             // [512, 1024]
    const half_t *v_proj_weight;             // [512, 1024]
    const half_t *q_norm_weight;              // [256]
    const half_t *k_norm_weight;              // [256]
    const half_t *o_proj_weight;             // [1024, 2048]
    const half_t *post_attn_layernorm_weight;
    const half_t *gate_proj_weight;          // [3584, 1024]
    const half_t *up_proj_weight;            // [3584, 1024]
    const half_t *down_proj_weight;          // [1024, 3584]
};

struct DeltaNetWeights {
    const half_t *input_layernorm_weight;
    const half_t *qkv_proj_weight;           // [6144, 1024]
    const half_t *z_proj_weight;             // [2048, 1024]
    const half_t *beta_proj_weight;          // [16, 1024]
    const half_t *alpha_proj_weight;         // [16, 1024]
    const half_t *conv1d_weight;             // [6144, 1, 4]
    const half_t *a_log;                     // [16]
    const half_t *dt_bias;                   // [16]
    const half_t *norm_weight;               // [128]
    const half_t *out_proj_weight;           // [1024, 2048]
    const half_t *post_attn_layernorm_weight;
    const half_t *gate_proj_weight;
    const half_t *up_proj_weight;
    const half_t *down_proj_weight;
};

struct LayerWeights {
    int layer_type;
    int _pad[3];
    union {
        DeltaNetWeights dn;
        FullAttnWeights fa;
    };
};

// =============================================================================
// Atomic barrier
// =============================================================================

struct AtomicGridSync {
    unsigned int *counter;
    unsigned int *generation;
    unsigned int nblocks;
    unsigned int local_gen;

    __device__ void sync() {
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int my_gen = local_gen;
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            unsigned int arrived = atomicAdd(counter, 1);
            if (arrived == nblocks - 1) {
                *counter = 0;
                asm volatile("fence.acq_rel.gpu;" ::: "memory");
                atomicAdd(generation, 1);
            } else {
                volatile unsigned int *vgen = (volatile unsigned int *)generation;
                while (*vgen <= my_gen) {}
            }
            local_gen = my_gen + 1;
        }
        __syncthreads();
    }
};

// =============================================================================
// Helpers
// =============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float fast_exp(float x) {
    float y; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.44269504088896340736f)); return y;
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    float y; asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(1.0f + fast_exp(-x))); return y;
}

__device__ __forceinline__ float fast_silu(float x) { return x * fast_sigmoid(x); }

__device__ __forceinline__ uint4 load_128bit(const uint4 *ptr) {
    uint4 out;
#if TARGET_SM >= 80
    asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w) : "l"(ptr));
#else
    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w) : "l"(ptr));
#endif
    return out;
}

// BF16 dot product: 8 bf16 weights × 8 bf16 activations → f32
__device__ __forceinline__ float dot8_bf16(const uint4 &w_u4, const half_t *act) {
    const half_t *w = reinterpret_cast<const half_t *>(&w_u4);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; i++)
        sum += H2F(w[i]) * H2F(act[i]);
    return sum;
}

// =============================================================================
// RMSNorm — reads bf16 input, writes bf16 output
// =============================================================================

__device__ void rmsnorm_redundant(
    const half_t *__restrict__ input,
    const half_t *__restrict__ weight,
    half_t *__restrict__ s_out,        // shared memory bf16
    half_t *__restrict__ g_residual)   // global bf16
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    __shared__ float smem_reduce[NUM_WARPS];

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float v = H2F(__ldg(input + i));
        s_out[i] = F2H(v);
        local_sum_sq += v * v;
    }

    if (block_id == 0) {
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE)
            g_residual[i] = s_out[i];
    }

    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0)
            smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + RMS_EPS);
    }
    __syncthreads();

    float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float w = H2F(__ldg(weight + i));
        float v = H2F(s_out[i]);
        s_out[i] = F2H(v * rstd * (1.0f + w));
    }
    __syncthreads();
}

// RMSNorm from bf16 buffer (for post-attn norm)
__device__ void rmsnorm_from_bf16(
    const half_t *__restrict__ input,
    const half_t *__restrict__ weight,
    half_t *__restrict__ s_out,
    half_t *__restrict__ g_residual)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    __shared__ float smem_reduce[NUM_WARPS];

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float v = H2F(input[i]);
        s_out[i] = F2H(v);
        local_sum_sq += v * v;
    }

    if (block_id == 0) {
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE)
            g_residual[i] = s_out[i];
    }

    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0)
            smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + RMS_EPS);
    }
    __syncthreads();

    float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float w = H2F(__ldg(weight + i));
        float v = H2F(s_out[i]);
        s_out[i] = F2H(v * rstd * (1.0f + w));
    }
    __syncthreads();
}

// =============================================================================
// BF16 Matvec: warp-per-row, activations in shared memory (bf16)
// =============================================================================

__device__ void matvec_bf16(
    const half_t *__restrict__ s_input,  // shared memory bf16 [in_dim]
    const half_t *__restrict__ weight,   // [out_dim, in_dim] bf16
    float *__restrict__ output,                  // [out_dim] f32 (accumulate in f32)
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const half_t *w_row = weight + m * in_dim;
            float sum = 0.0f;
#pragma unroll 4
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                sum += dot8_bf16(w_u4, s_input + k);
            }
            sum = warp_reduce_sum(sum);
            if (lane_id == 0) output[m] = sum;
        }
    }
}

// Fused gate+up+SiLU matvec (bf16 weights, bf16 activations)
__device__ void matvec_gate_up_silu_bf16(
    const half_t *__restrict__ s_input,
    const half_t *__restrict__ gate_weight,
    const half_t *__restrict__ up_weight,
    float *__restrict__ output,
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const half_t *g_row = gate_weight + m * in_dim;
            const half_t *u_row = up_weight + m * in_dim;
            float gate_sum = 0.0f, up_sum = 0.0f;
#pragma unroll 4
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 g_u4 = load_128bit(reinterpret_cast<const uint4 *>(g_row + k));
                uint4 u_u4 = load_128bit(reinterpret_cast<const uint4 *>(u_row + k));
                gate_sum += dot8_bf16(g_u4, s_input + k);
                up_sum += dot8_bf16(u_u4, s_input + k);
            }
            gate_sum = warp_reduce_sum(gate_sum);
            up_sum = warp_reduce_sum(up_sum);
            if (lane_id == 0)
                output[m] = fast_silu(gate_sum) * up_sum;
        }
    }
}

// Down projection + residual → bf16 hidden
__device__ void matvec_down_residual_bf16(
    const float *__restrict__ s_input,           // shared [INTER] f32
    const half_t *__restrict__ weight,    // [HIDDEN, INTER] bf16
    const half_t *__restrict__ residual,  // [HIDDEN] bf16
    half_t *__restrict__ hidden_out,      // [HIDDEN] bf16
    int in_dim, int out_dim, int num_blocks)
{
    // This needs f32 input (MLP intermediate is f32). Convert on the fly.
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const half_t *w_row = weight + m * in_dim;
            float sum = 0.0f;
            // Weight is bf16, input is f32 — convert input to bf16 on the fly
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                const half_t *w = reinterpret_cast<const half_t *>(&w_u4);
#pragma unroll
                for (int i = 0; i < 8; i++)
                    sum += H2F(w[i]) * s_input[k + i];
            }
            sum = warp_reduce_sum(sum);
            if (lane_id == 0)
                hidden_out[m] = F2H(sum + H2F(residual[m]));
        }
    }
}

// O projection + residual → bf16
__device__ void matvec_o_residual_bf16(
    const float *__restrict__ s_input,           // shared [Q_SIZE] f32
    const half_t *__restrict__ weight,
    const half_t *__restrict__ residual,
    half_t *__restrict__ hidden_out,
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const half_t *w_row = weight + m * in_dim;
            float sum = 0.0f;
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                const half_t *w = reinterpret_cast<const half_t *>(&w_u4);
#pragma unroll
                for (int i = 0; i < 8; i++)
                    sum += H2F(w[i]) * s_input[k + i];
            }
            sum = warp_reduce_sum(sum);
            if (lane_id == 0)
                hidden_out[m] = F2H(sum + H2F(residual[m]));
        }
    }
}

// =============================================================================
// Full Attention layer (bf16)
// =============================================================================

__device__ void full_attention_layer(
    AtomicGridSync &grid,
    const FullAttnWeights &w,
    const half_t *__restrict__ input,
    half_t *__restrict__ k_cache,
    half_t *__restrict__ v_cache,
    half_t *__restrict__ g_residual,  // [HIDDEN] bf16
    float *__restrict__ g_activations,        // scratch f32
    float *__restrict__ g_q,                  // [FA_QPROJ_SIZE] f32
    float *__restrict__ g_kv,                 // [FA_KV_SIZE*2] f32
    float *__restrict__ g_attn_out,           // [FA_Q_SIZE] f32
    float *__restrict__ g_mlp_inter,          // [INTER] f32
    half_t *__restrict__ hidden_out,   // [HIDDEN] bf16
    int position, int max_seq_len,
    half_t *__restrict__ shmem)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Phase 1: RMSNorm + QKV projection
    half_t *s_norm = shmem;
    rmsnorm_redundant(input, w.input_layernorm_weight, s_norm, g_residual);

    matvec_bf16(s_norm, w.q_proj_weight, g_q, HIDDEN_SIZE, FA_QPROJ_SIZE, num_blocks);
    matvec_bf16(s_norm, w.k_proj_weight, g_kv, HIDDEN_SIZE, FA_KV_SIZE, num_blocks);
    matvec_bf16(s_norm, w.v_proj_weight, g_kv + FA_KV_SIZE, HIDDEN_SIZE, FA_KV_SIZE, num_blocks);
    grid.sync();

    // Phase 2: QK norm + partial RoPE + KV cache write
    if (block_id == 0) {
        float *k_buf = g_kv, *v_buf = g_kv + FA_KV_SIZE;
        for (int h = warp_id; h < FA_NUM_KV_HEADS; h += NUM_WARPS) {
            float *kh = k_buf + h * FA_HEAD_DIM, *vh = v_buf + h * FA_HEAD_DIM;
            half_t *kc = k_cache + h * max_seq_len * FA_HEAD_DIM + position * FA_HEAD_DIM;
            half_t *vc = v_cache + h * max_seq_len * FA_HEAD_DIM + position * FA_HEAD_DIM;
            float ss = 0; for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) ss += kh[i]*kh[i];
            ss = warp_reduce_sum(ss); float sc = rsqrtf(ss / float(FA_HEAD_DIM) + RMS_EPS);
            sc = __shfl_sync(0xffffffff, sc, 0);
            for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) {
                float normed = kh[i] * sc * (1.0f + H2F(__ldg(w.k_norm_weight + i)));
                if (i < FA_ROTARY_DIM) {
                    float fe = float(2*(i%(FA_ROTARY_DIM/2))) / float(FA_ROTARY_DIM);
                    float freq = float(position) / powf(FA_ROPE_THETA, fe);
                    float cv = cosf(freq), sv = sinf(freq);
                    int p = (i < FA_ROTARY_DIM/2) ? i+FA_ROTARY_DIM/2 : i-FA_ROTARY_DIM/2;
                    float pv = kh[p]*sc*(1.0f+H2F(__ldg(w.k_norm_weight+p)));
                    float rotated = (i < FA_ROTARY_DIM/2) ? (normed*cv - pv*sv) : (pv*sv + normed*cv);
                    kc[i] = F2H(rotated);
                } else { kc[i] = F2H(normed); }
                vc[i] = F2H(vh[i]);
            }
        }
    }
    // Q norm + RoPE (all blocks)
    {
        int hpb = (FA_NUM_Q_HEADS + num_blocks - 1) / num_blocks;
        int hs = block_id * hpb, he = min(hs + hpb, FA_NUM_Q_HEADS);
        for (int qh = hs; qh < he; qh++) {
            float *qh_ptr = g_q + qh * FA_HEAD_DIM * 2;
            if (warp_id == 0) {
                float ss = 0; for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) ss += qh_ptr[i]*qh_ptr[i];
                ss = warp_reduce_sum(ss); float sc = rsqrtf(ss / float(FA_HEAD_DIM) + RMS_EPS);
                sc = __shfl_sync(0xffffffff, sc, 0);
                for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) {
                    float normed = qh_ptr[i]*sc*(1.0f+H2F(__ldg(w.q_norm_weight+i)));
                    if (i < FA_ROTARY_DIM) {
                        float fe = float(2*(i%(FA_ROTARY_DIM/2))) / float(FA_ROTARY_DIM);
                        float freq = float(position) / powf(FA_ROPE_THETA, fe);
                        float cv = cosf(freq), sv = sinf(freq);
                        int p = (i < FA_ROTARY_DIM/2) ? i+FA_ROTARY_DIM/2 : i-FA_ROTARY_DIM/2;
                        float pv = qh_ptr[p]*sc*(1.0f+H2F(__ldg(w.q_norm_weight+p)));
                        qh_ptr[i] = (i < FA_ROTARY_DIM/2) ? (normed*cv-pv*sv) : (pv*sv+normed*cv);
                    } else { qh_ptr[i] = normed; }
                }
            }
        }
    }
    grid.sync();

    // Phase 3: Attention decode (online softmax + sigmoid gate)
    {
        int cache_len = position + 1;
        float attn_scale = 1.0f / sqrtf(float(FA_HEAD_DIM));
        int hpb = (FA_NUM_Q_HEADS + num_blocks - 1) / num_blocks;
        int hs = block_id * hpb, he = min(hs + hpb, FA_NUM_Q_HEADS);
        __shared__ float s_max_score[NUM_WARPS];
        __shared__ float s_sum_exp[NUM_WARPS];
        constexpr int EPL = FA_HEAD_DIM / WARP_SIZE;

        for (int qh = hs; qh < he; qh++) {
            int kvh = qh / FA_GQA_RATIO;
            float *q_head = g_q + qh * FA_HEAD_DIM * 2;
            float *out_head = g_attn_out + qh * FA_HEAD_DIM;
            float max_score = -INFINITY, sum_exp = 0;
            float out_acc[EPL], q_local[EPL];
            for (int e = 0; e < EPL; e++) { out_acc[e] = 0; q_local[e] = q_head[lane_id*EPL+e]; }

            for (int pos = warp_id; pos < cache_len; pos += NUM_WARPS) {
                const half_t *k_pos = k_cache + kvh*max_seq_len*FA_HEAD_DIM + pos*FA_HEAD_DIM;
                const half_t *v_pos = v_cache + kvh*max_seq_len*FA_HEAD_DIM + pos*FA_HEAD_DIM;
                float score = 0;
                for (int e = 0; e < EPL; e++) score += q_local[e] * H2F(__ldg(k_pos + lane_id*EPL+e));
                score = warp_reduce_sum(score) * attn_scale;
                score = __shfl_sync(0xffffffff, score, 0);
                float old_max = max_score; max_score = fmaxf(max_score, score);
                float exp_diff = fast_exp(old_max - max_score);
                sum_exp = sum_exp * exp_diff + fast_exp(score - max_score);
                float wt = fast_exp(score - max_score);
                for (int e = 0; e < EPL; e++)
                    out_acc[e] = out_acc[e]*exp_diff + wt*H2F(__ldg(v_pos + lane_id*EPL+e));
            }
            if (lane_id == 0) { s_max_score[warp_id] = max_score; s_sum_exp[warp_id] = sum_exp; }
            for (int e = 0; e < EPL; e++) g_activations[warp_id*FA_HEAD_DIM + lane_id*EPL+e] = out_acc[e];
            __syncthreads();

            if (warp_id == 0) {
                float gm = -INFINITY; for (int ww = 0; ww < NUM_WARPS; ww++) if (s_max_score[ww] > -INFINITY) gm = fmaxf(gm, s_max_score[ww]);
                float ts = 0; float fo[EPL]; for (int e = 0; e < EPL; e++) fo[e] = 0;
                for (int ww = 0; ww < NUM_WARPS; ww++) {
                    if (s_max_score[ww] > -INFINITY) {
                        float s = fast_exp(s_max_score[ww]-gm); ts += s_sum_exp[ww]*s;
                        for (int e = 0; e < EPL; e++) fo[e] += g_activations[ww*FA_HEAD_DIM+lane_id*EPL+e]*s;
                    }
                }
                float *gate_ptr = q_head + FA_HEAD_DIM;
                float rcp = 1.0f / ts;
                for (int e = 0; e < EPL; e++) {
                    int idx = lane_id*EPL+e;
                    out_head[idx] = fo[e]*rcp * fast_sigmoid(gate_ptr[idx]);
                }
            }
            __syncthreads();
        }
    }
    grid.sync();

    // Phase 4: O projection + residual → bf16
    {
        float *s_attn = reinterpret_cast<float *>(shmem);
        for (int i = threadIdx.x; i < FA_Q_SIZE; i += BLOCK_SIZE) s_attn[i] = g_attn_out[i];
        __syncthreads();
        matvec_o_residual_bf16(s_attn, w.o_proj_weight, g_residual, hidden_out, FA_Q_SIZE, HIDDEN_SIZE, num_blocks);
    }
    grid.sync();

    // Phase 5: Post-attn norm + MLP
    half_t *s_act = shmem;
    rmsnorm_from_bf16(hidden_out, w.post_attn_layernorm_weight, s_act, g_residual);

    matvec_gate_up_silu_bf16(s_act, w.gate_proj_weight, w.up_proj_weight,
                              g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
    grid.sync();

    // Load MLP intermediate to shared (f32)
    float *s_mlp = reinterpret_cast<float *>(shmem);
    for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += BLOCK_SIZE) s_mlp[i] = g_mlp_inter[i];
    __syncthreads();

    matvec_down_residual_bf16(s_mlp, w.down_proj_weight, g_residual, hidden_out,
                               INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks);
    grid.sync();
}

// =============================================================================
// DeltaNet layer (bf16) — warp-cooperative state-in-registers recurrence
// =============================================================================

__device__ void deltanet_layer(
    AtomicGridSync &grid,
    const DeltaNetWeights &w,
    const half_t *__restrict__ input,
    half_t *__restrict__ g_residual,
    float *__restrict__ g_activations,
    float *__restrict__ g_qkv,
    float *__restrict__ g_z,
    float *__restrict__ g_beta,
    float *__restrict__ g_alpha,
    float *__restrict__ g_dn_out,
    float *__restrict__ g_mlp_inter,
    float *__restrict__ dn_state,     // [DN_NUM_HEADS, DN_KEY, DN_VAL] f32
    float *__restrict__ conv_buf,     // [DN_CONV_CH, DN_CONV_K] f32
    half_t *__restrict__ hidden_out,
    int dn_layer_idx,
    half_t *__restrict__ shmem)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Phase 1: RMSNorm + projections
    half_t *s_norm = shmem;
    rmsnorm_redundant(input, w.input_layernorm_weight, s_norm, g_residual);

    matvec_bf16(s_norm, w.qkv_proj_weight, g_qkv, HIDDEN_SIZE, DN_CONV_CHANNELS, num_blocks);
    matvec_bf16(s_norm, w.z_proj_weight, g_z, HIDDEN_SIZE, DN_V_SIZE, num_blocks);
    matvec_bf16(s_norm, w.beta_proj_weight, g_beta, HIDDEN_SIZE, DN_NUM_HEADS, num_blocks);
    matvec_bf16(s_norm, w.alpha_proj_weight, g_alpha, HIDDEN_SIZE, DN_NUM_HEADS, num_blocks);
    grid.sync();

    // Phase 2+3: Conv1d + recurrence (blocks 0-15 only)
    if (block_id < DN_NUM_HEADS) {
        int h = block_id;
        float *layer_conv = conv_buf + dn_layer_idx * DN_CONV_CHANNELS * DN_CONV_KERNEL;

        // Conv1d + SiLU
        __shared__ float s_q[DN_KEY_DIM], s_k[DN_KEY_DIM], s_v[DN_VALUE_DIM];
        int head_ch[3] = {h*DN_KEY_DIM, DN_QK_SIZE+h*DN_KEY_DIM, 2*DN_QK_SIZE+h*DN_VALUE_DIM};
        for (int region = 0; region < 3; region++) {
            int ch_base = head_ch[region], ch_count = (region < 2) ? DN_KEY_DIM : DN_VALUE_DIM;
            float *dst = (region == 0) ? s_q : (region == 1) ? s_k : s_v;
            for (int c = threadIdx.x; c < ch_count; c += BLOCK_SIZE) {
                int ch = ch_base + c;
                float h0=layer_conv[ch*DN_CONV_KERNEL+1], h1=layer_conv[ch*DN_CONV_KERNEL+2], h2=layer_conv[ch*DN_CONV_KERNEL+3];
                layer_conv[ch*DN_CONV_KERNEL]=h0; layer_conv[ch*DN_CONV_KERNEL+1]=h1;
                layer_conv[ch*DN_CONV_KERNEL+2]=h2; layer_conv[ch*DN_CONV_KERNEL+3]=g_qkv[ch];
                float co = 0;
                for (int t = 0; t < DN_CONV_KERNEL; t++)
                    co += layer_conv[ch*DN_CONV_KERNEL+t] * H2F(__ldg(w.conv1d_weight + ch*DN_CONV_KERNEL+t));
                dst[c] = fast_silu(co);
            }
        }

        // Beta/alpha activations
        if (threadIdx.x == 0) {
            g_beta[h] = fast_sigmoid(g_beta[h]);
            float a_log_val = H2F(__ldg(w.a_log + h));
            float dt_b = H2F(__ldg(w.dt_bias + h));
            float x = g_alpha[h] + dt_b;
            float sp = (x > 20.0f) ? x : logf(1.0f + fast_exp(x));
            g_alpha[h] = fast_exp(-fast_exp(a_log_val) * sp);
        }
        __syncthreads();

        // L2 normalize Q, K
        constexpr float Q_SCALE = 1.0f / 11.313708498984761f;
        if (warp_id == 0) {
            float sq = 0; for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) sq += s_q[i]*s_q[i];
            sq = warp_reduce_sum(sq); float n = rsqrtf(sq+1e-6f)*Q_SCALE;
            n = __shfl_sync(0xffffffff,n,0); for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) s_q[i] *= n;
        }
        if (warp_id == 1) {
            float sq = 0; for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) sq += s_k[i]*s_k[i];
            sq = warp_reduce_sum(sq); float n = rsqrtf(sq+1e-6f);
            n = __shfl_sync(0xffffffff,n,0); for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) s_k[i] *= n;
        }
        __syncthreads();

        float decay = g_alpha[h], beta = g_beta[h];

        // k·q dot
        __shared__ float s_kq;
        if (warp_id == 0) {
            float kq = 0; for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) kq += s_k[i]*s_q[i];
            kq = warp_reduce_sum(kq); if (lane_id == 0) s_kq = kq;
        }
        __syncthreads();
        float kq = s_kq;

        // Warp-cooperative recurrence (state in global memory — decode is 1 token, fine)
        float *state = dn_state + h * DN_KEY_DIM * DN_VALUE_DIM;
        float *out_head = g_dn_out + h * DN_VALUE_DIM;

        constexpr int J_PER_WARP = DN_VALUE_DIM / NUM_WARPS;
        constexpr int I_PER_LANE = DN_KEY_DIM / WARP_SIZE;

#pragma unroll
        for (int jj = 0; jj < J_PER_WARP; jj++) {
            int j = warp_id * J_PER_WARP + jj;
            float s_regs[I_PER_LANE], stk = 0, sqv = 0;
#pragma unroll
            for (int ii = 0; ii < I_PER_LANE; ii++) {
                int i = lane_id + ii * WARP_SIZE;
                float sv = state[j*DN_KEY_DIM+i]; s_regs[ii] = sv;
                stk += sv * s_k[i]; sqv += sv * s_q[i];
            }
            stk = warp_reduce_sum(stk); sqv = warp_reduce_sum(sqv);
            stk = __shfl_sync(0xffffffff,stk,0); sqv = __shfl_sync(0xffffffff,sqv,0);
            float error_j = (s_v[j] - decay * stk) * beta;
            float o_j = decay * sqv + error_j * kq;
            if (lane_id == 0) out_head[j] = o_j;
#pragma unroll
            for (int ii = 0; ii < I_PER_LANE; ii++) {
                int i = lane_id + ii * WARP_SIZE;
                state[j*DN_KEY_DIM+i] = s_regs[ii] * decay + s_k[i] * error_j;
            }
        }

        // Gated RMSNorm
        __syncthreads();
        {
            __shared__ float smem_gnorm[NUM_WARPS];
            float sq = 0; for (int i = threadIdx.x; i < DN_VALUE_DIM; i += BLOCK_SIZE) sq += out_head[i]*out_head[i];
            sq = warp_reduce_sum(sq); if (lane_id == 0) smem_gnorm[warp_id] = sq; __syncthreads();
            if (warp_id == 0) { float v = (lane_id < NUM_WARPS) ? smem_gnorm[lane_id] : 0; v = warp_reduce_sum(v); if (lane_id == 0) smem_gnorm[0] = rsqrtf(v/DN_VALUE_DIM + RMS_EPS); }
            __syncthreads(); float rstd = smem_gnorm[0];
            for (int i = threadIdx.x; i < DN_VALUE_DIM; i += BLOCK_SIZE) {
                float normed = out_head[i] * rstd * H2F(__ldg(w.norm_weight + i));
                float gate = fast_silu(g_z[h*DN_VALUE_DIM+i]);
                out_head[i] = normed * gate;
            }
        }
    } else {
        // Idle blocks: could prefetch weights
    }
    grid.sync();

    // Phase 4: Out projection + residual → bf16
    {
        float *s_dn = reinterpret_cast<float *>(shmem);
        for (int i = threadIdx.x; i < DN_V_SIZE; i += BLOCK_SIZE) s_dn[i] = g_dn_out[i];
        __syncthreads();
        matvec_o_residual_bf16(s_dn, w.out_proj_weight, g_residual, hidden_out, DN_V_SIZE, HIDDEN_SIZE, num_blocks);
    }
    grid.sync();

    // Phase 5: Post-attn norm + MLP
    half_t *s_act = shmem;
    rmsnorm_from_bf16(hidden_out, w.post_attn_layernorm_weight, s_act, g_residual);

    matvec_gate_up_silu_bf16(s_act, w.gate_proj_weight, w.up_proj_weight,
                              g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
    grid.sync();

    float *s_mlp = reinterpret_cast<float *>(shmem);
    for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += BLOCK_SIZE) s_mlp[i] = g_mlp_inter[i];
    __syncthreads();
    matvec_down_residual_bf16(s_mlp, w.down_proj_weight, g_residual, hidden_out,
                               INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks);
    grid.sync();
}

// =============================================================================
// LM Head: vocab projection + argmax
// =============================================================================

__global__ void lm_head_kernel(
    const float *__restrict__ hidden,
    const half_t *__restrict__ weight,   // [VOCAB, HIDDEN] bf16
    float *__restrict__ block_max_vals,
    int *__restrict__ block_max_idxs,
    int *__restrict__ output_token,
    unsigned int *__restrict__ sync_counter,
    const float *__restrict__ seen_token_mask,
    float repetition_penalty)
{
    __shared__ float s_hidden[HIDDEN_SIZE];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LM_BLOCK_SIZE) s_hidden[i] = hidden[i];
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE, lane_id = threadIdx.x % WARP_SIZE;
    int num_warps = LM_BLOCK_SIZE / WARP_SIZE;
    int rpb = (VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int rs = blockIdx.x * rpb, re = min(rs + rpb, VOCAB_SIZE);

    float local_max = -INFINITY; int local_max_idx = -1;
    for (int m = rs + warp_id; m < re; m += num_warps) {
        const half_t *w_row = weight + m * HIDDEN_SIZE;
        float sum = 0;
#pragma unroll 4
        for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
            uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
            const half_t *wp = reinterpret_cast<const half_t *>(&w_u4);
            for (int i = 0; i < 8; i++) sum += H2F(wp[i]) * s_hidden[k+i];
        }
        sum = warp_reduce_sum(sum);
        if (lane_id == 0 && repetition_penalty > 1.0f && seen_token_mask && seen_token_mask[m] > 0.0f) {
            sum = (sum > 0.0f) ? (sum / repetition_penalty) : (sum * repetition_penalty);
        }
        if (lane_id == 0 && sum > local_max) { local_max = sum; local_max_idx = m; }
    }
    local_max = __shfl_sync(0xffffffff, local_max, 0);
    local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

    __shared__ float wm[32]; __shared__ int wi[32];
    if (lane_id == 0) { wm[warp_id] = local_max; wi[warp_id] = local_max_idx; }
    __syncthreads();
    if (warp_id == 0) {
        float mv = (lane_id < num_warps) ? wm[lane_id] : -INFINITY;
        int mi = (lane_id < num_warps) ? wi[lane_id] : -1;
        for (int o = WARP_SIZE/2; o > 0; o /= 2) {
            float ov = __shfl_down_sync(0xffffffff, mv, o);
            int oi = __shfl_down_sync(0xffffffff, mi, o);
            if (ov > mv) { mv = ov; mi = oi; }
        }
        if (lane_id == 0) { block_max_vals[blockIdx.x] = mv; block_max_idxs[blockIdx.x] = mi; }
    }
    __syncthreads();
    if (threadIdx.x == 0) { __threadfence(); atomicAdd(sync_counter, 1); }
    if (blockIdx.x == 0) {
        if (threadIdx.x == 0) { volatile unsigned int *vc = (volatile unsigned int *)sync_counter; while (*vc < (unsigned int)gridDim.x) {} __threadfence(); }
        __syncthreads();
        int tid = threadIdx.x; float bv = -INFINITY; int bi = -1;
        for (int i = tid; i < gridDim.x; i += LM_BLOCK_SIZE) { float v = block_max_vals[i]; if (v > bv) { bv = v; bi = block_max_idxs[i]; } }
        __shared__ float sv[256]; __shared__ int si[256];
        sv[tid] = bv; si[tid] = bi; __syncthreads();
        for (int s = LM_BLOCK_SIZE/2; s > 0; s >>= 1) { if (tid < s && sv[tid+s] > sv[tid]) { sv[tid] = sv[tid+s]; si[tid] = si[tid+s]; } __syncthreads(); }
        if (tid == 0) *output_token = si[0];
    }
}

// =============================================================================
// Main decode kernel
// =============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
decode_kernel(
    const half_t *__restrict__ embed_weight,
    const half_t *__restrict__ final_norm_weight,
    const half_t *__restrict__ lm_head_weight,
    const LayerWeights *__restrict__ layer_weights,
    half_t *__restrict__ fa_k_cache,
    half_t *__restrict__ fa_v_cache,
    float *__restrict__ dn_states,
    float *__restrict__ conv_bufs,
    half_t *__restrict__ hidden_buffer,
    float *__restrict__ g_activations,
    half_t *__restrict__ g_residual,
    float *__restrict__ g_qkv_scratch,
    float *__restrict__ g_kv_scratch,
    float *__restrict__ g_attn_out,
    float *__restrict__ g_mlp_inter,
    float *__restrict__ g_z_scratch,
    float *__restrict__ g_beta_scratch,
    float *__restrict__ g_alpha_scratch,
    float *__restrict__ g_normalized,
    unsigned int *__restrict__ barrier_counter,
    unsigned int *__restrict__ barrier_generation,
    float *__restrict__ seen_token_mask,
    float repetition_penalty,
    int input_token_id, int position, int max_seq_len)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;

    if (block_id == 0 && threadIdx.x == 0 &&
        seen_token_mask && repetition_penalty > 1.0f &&
        input_token_id >= 0 && input_token_id < VOCAB_SIZE) {
        seen_token_mask[input_token_id] = 1.0f;
    }

    AtomicGridSync grid{barrier_counter, barrier_generation, (unsigned int)num_blocks, 0};

    // Shared memory: large enough for max(HIDDEN_SIZE bf16, INTERMEDIATE_SIZE f32)
    __shared__ __align__(16) char shmem_raw[MAX_ACT_DIM * sizeof(float)];
    half_t *shmem_bf16 = reinterpret_cast<half_t *>(shmem_raw);

    const half_t *embed_row = embed_weight + input_token_id * HIDDEN_SIZE;

    int fa_kv_stride = FA_NUM_KV_HEADS * max_seq_len * FA_HEAD_DIM;
    int dn_state_stride = DN_NUM_HEADS * DN_KEY_DIM * DN_VALUE_DIM;

    int dn_layer_idx = 0, fa_layer_idx = 0;

    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        const half_t *layer_input = (layer == 0) ? embed_row : hidden_buffer;

        if (LAYER_TYPE[layer] == 0) {
            deltanet_layer(
                grid, layer_weights[layer].dn, layer_input,
                g_residual, g_activations, g_qkv_scratch, g_z_scratch,
                g_beta_scratch, g_alpha_scratch, g_attn_out, g_mlp_inter,
                dn_states + dn_layer_idx * dn_state_stride,
                conv_bufs, hidden_buffer, dn_layer_idx, shmem_bf16);
            dn_layer_idx++;
        } else {
            full_attention_layer(
                grid, layer_weights[layer].fa, layer_input,
                fa_k_cache + fa_layer_idx * fa_kv_stride,
                fa_v_cache + fa_layer_idx * fa_kv_stride,
                g_residual, g_activations, g_qkv_scratch, g_kv_scratch,
                g_attn_out, g_mlp_inter, hidden_buffer,
                position, max_seq_len, shmem_bf16);
            fa_layer_idx++;
        }
    }

    // Final RMSNorm (block 0 only)
    if (block_id == 0) {
        __shared__ float smem_reduce[NUM_WARPS];
        int warp_id = threadIdx.x / WARP_SIZE, lane_id = threadIdx.x % WARP_SIZE;
        float local_sum_sq = 0;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
            float v = H2F(hidden_buffer[i]); g_activations[i] = v; local_sum_sq += v*v;
        }
        local_sum_sq = warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq; __syncthreads();
        if (warp_id == 0) { float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0; sum = warp_reduce_sum(sum); if (lane_id == 0) smem_reduce[0] = rsqrtf(sum/HIDDEN_SIZE + RMS_EPS); }
        __syncthreads(); float rstd = smem_reduce[0];
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
            float wt = H2F(__ldg(final_norm_weight + i));
            g_normalized[i] = g_activations[i] * rstd * (1.0f + wt);
        }
    }
}

// =============================================================================
// C entry point
// =============================================================================

static int query_max_safe_decode_blocks_impl()
{
    int device_id = 0;
    int sm_count = 0;
    int active_blocks_per_sm = 0;
    int max_safe_blocks = NUM_BLOCKS;

    if (cudaGetDevice(&device_id) == cudaSuccess &&
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id) == cudaSuccess &&
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm,
            decode_kernel,
            BLOCK_SIZE,
            0) == cudaSuccess &&
        sm_count > 0 &&
        active_blocks_per_sm > 0) {
        const int resident_blocks = sm_count * active_blocks_per_sm;
        if (resident_blocks > 0) {
            max_safe_blocks = resident_blocks;
        }
    }

    return (max_safe_blocks < 1) ? 1 : max_safe_blocks;
}

extern "C" void launch_decode(
    int input_token_id, int *output_token_id,
    const void *embed_weight, const LayerWeights *layer_weights,
    const void *final_norm_weight,
    const void *lm_head_weight,
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
    int position, int max_seq_len, cudaStream_t stream)
{
    const int max_safe_blocks = query_max_safe_decode_blocks_impl();
    int decode_blocks = NUM_BLOCKS;
    if (decode_blocks > max_safe_blocks) {
        decode_blocks = max_safe_blocks;
    }
    if (g_decode_blocks_override > 0) {
        decode_blocks = g_decode_blocks_override;
        if (decode_blocks > max_safe_blocks) {
            decode_blocks = max_safe_blocks;
        }
    }

    cudaMemsetAsync(barrier_counter, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(barrier_generation, 0, sizeof(unsigned int), stream);

    decode_kernel<<<decode_blocks, BLOCK_SIZE, 0, stream>>>(
        (const half_t *)embed_weight,
        (const half_t *)final_norm_weight,
        (const half_t *)lm_head_weight,
        layer_weights,
        (half_t *)fa_k_cache, (half_t *)fa_v_cache,
        (float *)dn_states, (float *)conv_bufs,
        (half_t *)hidden_buffer,
        (float *)g_activations, (half_t *)g_residual,
        (float *)g_qkv_scratch, (float *)g_kv_scratch,
        (float *)g_attn_out, (float *)g_mlp_inter,
        (float *)g_z_scratch, (float *)g_beta_scratch,
        (float *)g_alpha_scratch, (float *)g_normalized,
        barrier_counter, barrier_generation,
        seen_token_mask, repetition_penalty,
        input_token_id, position, max_seq_len);

    cudaMemsetAsync(lm_sync_counter, 0, sizeof(unsigned int), stream);

    lm_head_kernel<<<LM_NUM_BLOCKS, LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)g_normalized,
        (const half_t *)lm_head_weight,
        block_max_vals, block_max_idxs,
        output_token_id, lm_sync_counter,
        seen_token_mask, repetition_penalty);
}

extern "C" void set_decode_blocks_override(int blocks)
{
    g_decode_blocks_override = blocks;
}

extern "C" int query_max_safe_decode_blocks()
{
    return query_max_safe_decode_blocks_impl();
}
