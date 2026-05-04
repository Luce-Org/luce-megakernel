/**
 * GB10/Blackwell decode path: persistent CUDA megakernel with group-scaled fp4
 * hot weights. Prefill stays on the existing bf16/cuBLAS path.
 *
 * Only compiled when building for Blackwell (sm_120/sm_121a). The RTX 3090
 * (sm_86) build excludes this translation unit entirely; see setup.py.
 */

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1200
#error "kernel_gb10_nvfp4.cu requires CUDA arch >= sm_120 (Blackwell)"
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdint.h>

namespace cg = cooperative_groups;

constexpr int WARP_SIZE = 32;
constexpr int HIDDEN_SIZE = 1024;
constexpr int INTERMEDIATE_SIZE = 3584;
constexpr int NUM_LAYERS = 24;
constexpr float RMS_EPS = 1e-6f;
constexpr int VOCAB_SIZE = 248320;
constexpr int LM_HEAD_TC_N = 16;
constexpr size_t LM_HEAD_MAX_WORKSPACE = 32ull * 1024ull * 1024ull;
constexpr int LM_SCALE_ROWS_PER_TILE = 128;
constexpr int LM_SCALE_COLS_PER_TILE = 4;
constexpr int LM_SCALE_BLOCK_K = 16;
constexpr int LM_SCALE_K_PER_TILE = LM_SCALE_COLS_PER_TILE * LM_SCALE_BLOCK_K;
constexpr int LM_SCALE_THREADS = LM_SCALE_ROWS_PER_TILE * LM_SCALE_COLS_PER_TILE;
constexpr size_t LM_HEAD_HIDDEN_SCALE_BYTES =
    static_cast<size_t>((LM_HEAD_TC_N + LM_SCALE_ROWS_PER_TILE - 1) / LM_SCALE_ROWS_PER_TILE) *
    static_cast<size_t>(HIDDEN_SIZE / LM_SCALE_K_PER_TILE) * 512ull;

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

constexpr int DN_NUM_HEADS = 16;
constexpr int DN_KEY_DIM = 128;
constexpr int DN_VALUE_DIM = 128;
constexpr int DN_CONV_KERNEL = 4;
constexpr int DN_QK_SIZE = DN_NUM_HEADS * DN_KEY_DIM;
constexpr int DN_V_SIZE = DN_NUM_HEADS * DN_VALUE_DIM;
constexpr int DN_CONV_CHANNELS = DN_QK_SIZE + DN_QK_SIZE + DN_V_SIZE;

constexpr int MAX_ACT_DIM = (HIDDEN_SIZE > INTERMEDIATE_SIZE) ? HIDDEN_SIZE : INTERMEDIATE_SIZE;

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

#ifndef LM_BLOCK_SIZE
#define LM_BLOCK_SIZE 256
#endif

__device__ __constant__ int LAYER_TYPE[NUM_LAYERS] = {
    0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1
};

__device__ __constant__ float FP4_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

__device__ __constant__ float ROPE_INV_FREQ[FA_ROTARY_DIM / 2] = {
    1.000000000000000000e+00f,
    6.042963902381328634e-01f,
    3.651741272548377215e-01f,
    2.206734069084589911e-01f,
    1.333521432163324028e-01f,
    8.058421877614818651e-02f,
    4.869675251658631132e-02f,
    2.942727176209281731e-02f,
    1.778279410038922925e-02f,
    1.074607828321317432e-02f,
    6.493816315762112983e-03f,
    3.924189758484536265e-03f,
    2.371373705661655382e-03f,
    1.433012570236962685e-03f,
    8.659643233600653866e-04f,
    5.232991146814947340e-04f,
    3.162277660168379394e-04f,
    1.910952974970440477e-04f,
    1.154781984689458215e-04f,
    6.978305848598663529e-05f,
    4.216965034285822237e-05f,
    2.548296747979346413e-05f,
    1.539926526059491854e-05f,
    9.305720409296990429e-06f,
    5.623413251903491208e-06f,
    3.398208328942559268e-06f,
    2.053525026457146066e-06f,
    1.240937760751719527e-06f,
    7.498942093324558477e-07f,
    4.531583637600817928e-07f,
    2.738419634264361394e-07f,
    1.654817099943181354e-07f
};

struct PackedMatrixNVFP4 {
    const uint8_t *data;
    const __half *scales;
};

struct FullAttnWeightsNVFP4 {
    const __nv_bfloat16 *input_layernorm_weight;
    PackedMatrixNVFP4 q_proj;
    PackedMatrixNVFP4 k_proj;
    PackedMatrixNVFP4 v_proj;
    const __nv_bfloat16 *q_norm_weight;
    const __nv_bfloat16 *k_norm_weight;
    PackedMatrixNVFP4 o_proj;
    const __nv_bfloat16 *post_attn_layernorm_weight;
    PackedMatrixNVFP4 gate_proj;
    PackedMatrixNVFP4 up_proj;
    PackedMatrixNVFP4 down_proj;
};

struct DeltaNetWeightsNVFP4 {
    const __nv_bfloat16 *input_layernorm_weight;
    PackedMatrixNVFP4 qkv_proj;
    PackedMatrixNVFP4 z_proj;
    const __nv_bfloat16 *beta_proj_weight;
    const __nv_bfloat16 *alpha_proj_weight;
    const __nv_bfloat16 *conv1d_weight;
    const __nv_bfloat16 *a_log;
    const __nv_bfloat16 *dt_bias;
    const __nv_bfloat16 *norm_weight;
    PackedMatrixNVFP4 out_proj;
    const __nv_bfloat16 *post_attn_layernorm_weight;
    PackedMatrixNVFP4 gate_proj;
    PackedMatrixNVFP4 up_proj;
    PackedMatrixNVFP4 down_proj;
};

struct LayerWeightsNVFP4 {
    int layer_type;
    int group_size;
    int _pad[2];
    void *ptrs[24];
};

static __device__ __forceinline__ PackedMatrixNVFP4 make_packed_matrix(const LayerWeightsNVFP4 &w, int data_idx) {
    return {
        reinterpret_cast<const uint8_t *>(w.ptrs[data_idx]),
        reinterpret_cast<const __half *>(w.ptrs[data_idx + 1]),
    };
}

static __device__ __forceinline__ FullAttnWeightsNVFP4 make_full_attn_weights(const LayerWeightsNVFP4 &w) {
    return {
        reinterpret_cast<const __nv_bfloat16 *>(w.ptrs[0]),
        make_packed_matrix(w, 1),
        make_packed_matrix(w, 3),
        make_packed_matrix(w, 5),
        reinterpret_cast<const __nv_bfloat16 *>(w.ptrs[7]),
        reinterpret_cast<const __nv_bfloat16 *>(w.ptrs[8]),
        make_packed_matrix(w, 9),
        reinterpret_cast<const __nv_bfloat16 *>(w.ptrs[11]),
        make_packed_matrix(w, 12),
        make_packed_matrix(w, 14),
        make_packed_matrix(w, 16),
    };
}

static __device__ __forceinline__ DeltaNetWeightsNVFP4 make_deltanet_weights(const LayerWeightsNVFP4 &w) {
    return {
        reinterpret_cast<const __nv_bfloat16 *>(w.ptrs[0]),
        make_packed_matrix(w, 1),
        make_packed_matrix(w, 3),
        reinterpret_cast<const __nv_bfloat16 *>(w.ptrs[5]),
        reinterpret_cast<const __nv_bfloat16 *>(w.ptrs[6]),
        reinterpret_cast<const __nv_bfloat16 *>(w.ptrs[7]),
        reinterpret_cast<const __nv_bfloat16 *>(w.ptrs[8]),
        reinterpret_cast<const __nv_bfloat16 *>(w.ptrs[9]),
        reinterpret_cast<const __nv_bfloat16 *>(w.ptrs[10]),
        make_packed_matrix(w, 11),
        reinterpret_cast<const __nv_bfloat16 *>(w.ptrs[13]),
        make_packed_matrix(w, 14),
        make_packed_matrix(w, 16),
        make_packed_matrix(w, 18),
    };
}

struct AtomicGridSync {
    __device__ __forceinline__ void sync() {
        cg::this_grid().sync();
    }
};

namespace {

bool lm_debug_enabled() {
    static int enabled = []() {
        const char *value = std::getenv("MEGAKERNEL_DEBUG_LM");
        return (value != nullptr && value[0] != '\0' && value[0] != '0') ? 1 : 0;
    }();
    return enabled != 0;
}

struct LmHeadCublasLtPlan {
    bool initialized = false;
    bool supported = false;
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatrixLayout_t d_desc = nullptr;
    cublasLtMatmulAlgo_t algo{};
    void *dummy_a_scale = nullptr;
    void *dummy_b_scale = nullptr;
    void *workspace = nullptr;
    size_t workspace_size = 0;
};

LmHeadCublasLtPlan &lm_head_plan() {
    static LmHeadCublasLtPlan plan;
    return plan;
}

void destroy_lm_head_plan(LmHeadCublasLtPlan &plan) {
    if (plan.dummy_b_scale != nullptr) {
        cudaFree(plan.dummy_b_scale);
        plan.dummy_b_scale = nullptr;
    }
    if (plan.dummy_a_scale != nullptr) {
        cudaFree(plan.dummy_a_scale);
        plan.dummy_a_scale = nullptr;
    }
    if (plan.workspace != nullptr) {
        cudaFree(plan.workspace);
        plan.workspace = nullptr;
    }
    if (plan.d_desc != nullptr) {
        cublasLtMatrixLayoutDestroy(plan.d_desc);
        plan.d_desc = nullptr;
    }
    if (plan.c_desc != nullptr) {
        cublasLtMatrixLayoutDestroy(plan.c_desc);
        plan.c_desc = nullptr;
    }
    if (plan.b_desc != nullptr) {
        cublasLtMatrixLayoutDestroy(plan.b_desc);
        plan.b_desc = nullptr;
    }
    if (plan.a_desc != nullptr) {
        cublasLtMatrixLayoutDestroy(plan.a_desc);
        plan.a_desc = nullptr;
    }
    if (plan.op_desc != nullptr) {
        cublasLtMatmulDescDestroy(plan.op_desc);
        plan.op_desc = nullptr;
    }
    if (plan.handle != nullptr) {
        cublasLtDestroy(plan.handle);
        plan.handle = nullptr;
    }
    plan.initialized = false;
    plan.supported = false;
    plan.workspace_size = 0;
}

bool init_lm_head_plan() {
    auto &plan = lm_head_plan();
    if (plan.initialized) {
        return plan.supported;
    }
    plan.initialized = true;

    cublasOperation_t trans_a = CUBLAS_OP_T;
    cublasOperation_t trans_b = CUBLAS_OP_N;
    cublasLtMatmulMatrixScale_t a_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulMatrixScale_t b_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtOrder_t col_order = CUBLASLT_ORDER_COL;

    if (cublasLtCreate(&plan.handle) != CUBLAS_STATUS_SUCCESS) {
        if (lm_debug_enabled()) {
            std::fprintf(stderr, "lm_head_cublaslt: cublasLtCreate failed\n");
        }
        destroy_lm_head_plan(plan);
        return false;
    }
    if (cublasLtMatmulDescCreate(&plan.op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) {
        if (lm_debug_enabled()) {
            std::fprintf(stderr, "lm_head_cublaslt: MatmulDescCreate failed\n");
        }
        destroy_lm_head_plan(plan);
        return false;
    }
    if (cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &a_scale_mode, sizeof(a_scale_mode)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &b_scale_mode, sizeof(b_scale_mode)) != CUBLAS_STATUS_SUCCESS) {
        if (lm_debug_enabled()) {
            std::fprintf(stderr, "lm_head_cublaslt: MatmulDescSetAttribute failed\n");
        }
        destroy_lm_head_plan(plan);
        return false;
    }

    if (cublasLtMatrixLayoutCreate(&plan.a_desc, CUDA_R_4F_E2M1, HIDDEN_SIZE, VOCAB_SIZE, HIDDEN_SIZE) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&plan.b_desc, CUDA_R_4F_E2M1, HIDDEN_SIZE, LM_HEAD_TC_N, HIDDEN_SIZE) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&plan.c_desc, CUDA_R_16F, VOCAB_SIZE, LM_HEAD_TC_N, VOCAB_SIZE) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&plan.d_desc, CUDA_R_16F, VOCAB_SIZE, LM_HEAD_TC_N, VOCAB_SIZE) != CUBLAS_STATUS_SUCCESS) {
        if (lm_debug_enabled()) {
            std::fprintf(stderr, "lm_head_cublaslt: MatrixLayoutCreate failed\n");
        }
        destroy_lm_head_plan(plan);
        return false;
    }

    cublasLtMatrixLayoutSetAttribute(plan.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_order, sizeof(col_order));
    cublasLtMatrixLayoutSetAttribute(plan.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_order, sizeof(col_order));
    cublasLtMatrixLayoutSetAttribute(plan.c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_order, sizeof(col_order));
    cublasLtMatrixLayoutSetAttribute(plan.d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_order, sizeof(col_order));

    if (cudaMalloc(&plan.dummy_a_scale, 4096) != cudaSuccess ||
        cudaMalloc(&plan.dummy_b_scale, 4096) != cudaSuccess ||
        cublasLtMatmulDescSetAttribute(
            plan.op_desc,
            CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &plan.dummy_a_scale,
            sizeof(plan.dummy_a_scale)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(
            plan.op_desc,
            CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &plan.dummy_b_scale,
            sizeof(plan.dummy_b_scale)) != CUBLAS_STATUS_SUCCESS) {
        if (lm_debug_enabled()) {
            std::fprintf(stderr, "lm_head_cublaslt: dummy scale setup failed\n");
        }
        destroy_lm_head_plan(plan);
        return false;
    }

    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned_results = 0;
    if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS) {
        if (lm_debug_enabled()) {
            std::fprintf(stderr, "lm_head_cublaslt: PreferenceCreate failed\n");
        }
        destroy_lm_head_plan(plan);
        return false;
    }
    cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &LM_HEAD_MAX_WORKSPACE,
        sizeof(LM_HEAD_MAX_WORKSPACE));

    cublasStatus_t heuristic_status = cublasLtMatmulAlgoGetHeuristic(
        plan.handle,
        plan.op_desc,
        plan.a_desc,
        plan.b_desc,
        plan.c_desc,
        plan.d_desc,
        preference,
        1,
        &heuristic,
        &returned_results);
    cublasLtMatmulPreferenceDestroy(preference);

    if (heuristic_status != CUBLAS_STATUS_SUCCESS || returned_results == 0) {
        if (lm_debug_enabled()) {
            std::fprintf(stderr,
                         "lm_head_cublaslt: heuristic failed status=%d returned=%d n=%d\n",
                         int(heuristic_status), returned_results, LM_HEAD_TC_N);
        }
        destroy_lm_head_plan(plan);
        return false;
    }

    plan.algo = heuristic.algo;
    plan.workspace_size = heuristic.workspaceSize;
    if (plan.workspace_size != 0) {
        if (cudaMalloc(&plan.workspace, plan.workspace_size) != cudaSuccess) {
            if (lm_debug_enabled()) {
                std::fprintf(stderr,
                             "lm_head_cublaslt: workspace alloc failed size=%zu\n",
                             plan.workspace_size);
            }
            destroy_lm_head_plan(plan);
            return false;
        }
    }

    plan.supported = true;
    if (lm_debug_enabled()) {
        std::fprintf(stderr,
                     "lm_head_cublaslt: plan ready workspace=%zu n=%d\n",
                     plan.workspace_size, LM_HEAD_TC_N);
    }
    return true;
}

bool launch_lm_head_cublaslt(
    const uint8_t *weight_packed,
    const uint8_t *weight_scales,
    const uint8_t *hidden_packed,
    const uint8_t *hidden_scales,
    __half *logits_f16,
    cudaStream_t stream)
{
    auto &plan = lm_head_plan();
    if (!plan.initialized && !init_lm_head_plan()) {
        return false;
    }
    if (!plan.supported) {
        return false;
    }

    if (cublasLtMatmulDescSetAttribute(
            plan.op_desc,
            CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &weight_scales,
            sizeof(weight_scales)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(
            plan.op_desc,
            CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &hidden_scales,
            sizeof(hidden_scales)) != CUBLAS_STATUS_SUCCESS) {
        if (lm_debug_enabled()) {
            std::fprintf(stderr, "lm_head_cublaslt: scale pointer set failed\n");
        }
        return false;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudaGetLastError();
    cublasStatus_t status = cublasLtMatmul(
        plan.handle,
        plan.op_desc,
        &alpha,
        weight_packed,
        plan.a_desc,
        hidden_packed,
        plan.b_desc,
        &beta,
        logits_f16,
        plan.c_desc,
        logits_f16,
        plan.d_desc,
        &plan.algo,
        plan.workspace,
        plan.workspace_size,
        stream);
    cudaError_t last_error = cudaGetLastError();
    if (lm_debug_enabled() && (status != CUBLAS_STATUS_SUCCESS || last_error != cudaSuccess)) {
        std::fprintf(stderr,
                     "lm_head_cublaslt: matmul status=%d cuda=%d (%s)\n",
                     int(status),
                     int(last_error),
                     cudaGetErrorString(last_error));
    }
    return status == CUBLAS_STATUS_SUCCESS && last_error == cudaSuccess;
}

}  // namespace

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float fast_exp(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.44269504088896340736f));
    return y;
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    float y;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(1.0f + fast_exp(-x)));
    return y;
}

__device__ __forceinline__ float fast_silu(float x) { return x * fast_sigmoid(x); }

__device__ __forceinline__ uint4 load_128bit(const uint4 *ptr) {
    uint4 out;
    asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w) : "l"(ptr));
    return out;
}

__device__ __forceinline__ uint32_t load_32bit(const uint32_t *ptr) {
    uint32_t out;
    asm volatile("ld.global.L1::no_allocate.b32 %0, [%1];" : "=r"(out) : "l"(ptr));
    return out;
}

__device__ __forceinline__ float dot8_bf16(const uint4 &w_u4, const __nv_bfloat16 *act) {
    const __nv_bfloat16 *w = reinterpret_cast<const __nv_bfloat16 *>(&w_u4);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        sum += __bfloat162float(w[i]) * __bfloat162float(act[i]);
    }
    return sum;
}

__device__ __forceinline__ float fp4_value(unsigned int code) {
    return FP4_E2M1_LUT[code & 0xf];
}

__device__ __forceinline__ float dot8_nvfp4_bf16(uint32_t packed, float scale, const __nv_bfloat16 *act) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned int byte = (packed >> (i * 8)) & 0xff;
        sum += fp4_value(byte & 0xf) * __bfloat162float(act[i * 2 + 0]);
        sum += fp4_value(byte >> 4) * __bfloat162float(act[i * 2 + 1]);
    }
    return sum * scale;
}

__device__ __forceinline__ float dot8_nvfp4_f32(uint32_t packed, float scale, const float *act) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned int byte = (packed >> (i * 8)) & 0xff;
        sum += fp4_value(byte & 0xf) * act[i * 2 + 0];
        sum += fp4_value(byte >> 4) * act[i * 2 + 1];
    }
    return sum * scale;
}

static __device__ void rmsnorm_redundant(
    const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ s_out,
    __nv_bfloat16 *__restrict__ g_residual)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    __shared__ float smem_reduce[NUM_WARPS];

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float v = __bfloat162float(__ldg(input + i));
        s_out[i] = __float2bfloat16(v);
        local_sum_sq += v * v;
    }

    if (block_id == 0) {
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
            g_residual[i] = s_out[i];
        }
    }

    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) {
        smem_reduce[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + RMS_EPS);
        }
    }
    __syncthreads();

    float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float w = __bfloat162float(__ldg(weight + i));
        float v = __bfloat162float(s_out[i]);
        s_out[i] = __float2bfloat16(v * rstd * (1.0f + w));
    }
    __syncthreads();
}

static __device__ void rmsnorm_from_bf16(
    const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ s_out,
    __nv_bfloat16 *__restrict__ g_residual)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    __shared__ float smem_reduce[NUM_WARPS];

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float v = __bfloat162float(input[i]);
        s_out[i] = __float2bfloat16(v);
        local_sum_sq += v * v;
    }

    if (block_id == 0) {
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
            g_residual[i] = s_out[i];
        }
    }

    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) {
        smem_reduce[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + RMS_EPS);
        }
    }
    __syncthreads();

    float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float w = __bfloat162float(__ldg(weight + i));
        float v = __bfloat162float(s_out[i]);
        s_out[i] = __float2bfloat16(v * rstd * (1.0f + w));
    }
    __syncthreads();
}

static __device__ void matvec_bf16(
    const __nv_bfloat16 *__restrict__ s_input,
    const __nv_bfloat16 *__restrict__ weight,
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
            const __nv_bfloat16 *w_row = weight + m * in_dim;
            float sum = 0.0f;
#pragma unroll 4
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                sum += dot8_bf16(w_u4, s_input + k);
            }
            sum = warp_reduce_sum(sum);
            if (lane_id == 0) {
                output[m] = sum;
            }
        }
    }
}

static __device__ void matvec_nvfp4(
    const __nv_bfloat16 *__restrict__ s_input,
    PackedMatrixNVFP4 weight,
    float *__restrict__ output,
    int in_dim, int out_dim, int num_blocks, int group_size)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);
    int row_bytes = in_dim / 2;
    int row_scales = in_dim / group_size;

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const uint8_t *w_row = weight.data + m * row_bytes;
            const __half *s_row = weight.scales + m * row_scales;
            float sum = 0.0f;
#pragma unroll 4
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint32_t packed = load_32bit(reinterpret_cast<const uint32_t *>(w_row + (k / 2)));
                int scale_idx = k / group_size;
                int scale_lane = lane_id & ~1;
                float scale = 0.0f;
                if (lane_id == scale_lane) {
                    scale = __half2float(__ldg(s_row + scale_idx));
                }
                scale = __shfl_sync(0xffffffff, scale, scale_lane);
                sum += dot8_nvfp4_bf16(packed, scale, s_input + k);
            }
            sum = warp_reduce_sum(sum);
            if (lane_id == 0) {
                output[m] = sum;
            }
        }
    }
}

static __device__ void matvec_gate_up_silu_nvfp4(
    const __nv_bfloat16 *__restrict__ s_input,
    PackedMatrixNVFP4 gate_weight,
    PackedMatrixNVFP4 up_weight,
    float *__restrict__ output,
    int in_dim, int out_dim, int num_blocks, int group_size)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);
    int row_bytes = in_dim / 2;
    int row_scales = in_dim / group_size;

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const uint8_t *g_row = gate_weight.data + m * row_bytes;
            const __half *gs_row = gate_weight.scales + m * row_scales;
            const uint8_t *u_row = up_weight.data + m * row_bytes;
            const __half *us_row = up_weight.scales + m * row_scales;
            float gate_sum = 0.0f;
            float up_sum = 0.0f;
#pragma unroll 4
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint32_t g_packed = load_32bit(reinterpret_cast<const uint32_t *>(g_row + (k / 2)));
                uint32_t u_packed = load_32bit(reinterpret_cast<const uint32_t *>(u_row + (k / 2)));
                int scale_idx = k / group_size;
                int scale_lane = lane_id & ~1;
                float g_scale = 0.0f;
                float u_scale = 0.0f;
                if (lane_id == scale_lane) {
                    g_scale = __half2float(__ldg(gs_row + scale_idx));
                    u_scale = __half2float(__ldg(us_row + scale_idx));
                }
                g_scale = __shfl_sync(0xffffffff, g_scale, scale_lane);
                u_scale = __shfl_sync(0xffffffff, u_scale, scale_lane);
                gate_sum += dot8_nvfp4_bf16(g_packed, g_scale, s_input + k);
                up_sum += dot8_nvfp4_bf16(u_packed, u_scale, s_input + k);
            }
            gate_sum = warp_reduce_sum(gate_sum);
            up_sum = warp_reduce_sum(up_sum);
            if (lane_id == 0) {
                output[m] = fast_silu(gate_sum) * up_sum;
            }
        }
    }
}

static __device__ void matvec_down_residual_nvfp4(
    const float *__restrict__ s_input,
    PackedMatrixNVFP4 weight,
    const __nv_bfloat16 *__restrict__ residual,
    __nv_bfloat16 *__restrict__ hidden_out,
    int in_dim, int out_dim, int num_blocks, int group_size)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);
    int row_bytes = in_dim / 2;
    int row_scales = in_dim / group_size;

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const uint8_t *w_row = weight.data + m * row_bytes;
            const __half *s_row = weight.scales + m * row_scales;
            float sum = 0.0f;
#pragma unroll 4
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint32_t packed = load_32bit(reinterpret_cast<const uint32_t *>(w_row + (k / 2)));
                int scale_idx = k / group_size;
                int scale_lane = lane_id & ~1;
                float scale = 0.0f;
                if (lane_id == scale_lane) {
                    scale = __half2float(__ldg(s_row + scale_idx));
                }
                scale = __shfl_sync(0xffffffff, scale, scale_lane);
                sum += dot8_nvfp4_f32(packed, scale, s_input + k);
            }
            sum = warp_reduce_sum(sum);
            if (lane_id == 0) {
                hidden_out[m] = __float2bfloat16(sum + __bfloat162float(residual[m]));
            }
        }
    }
}

static __device__ void matvec_o_residual_nvfp4(
    const float *__restrict__ s_input,
    PackedMatrixNVFP4 weight,
    const __nv_bfloat16 *__restrict__ residual,
    __nv_bfloat16 *__restrict__ hidden_out,
    int in_dim, int out_dim, int num_blocks, int group_size)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);
    int row_bytes = in_dim / 2;
    int row_scales = in_dim / group_size;

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const uint8_t *w_row = weight.data + m * row_bytes;
            const __half *s_row = weight.scales + m * row_scales;
            float sum = 0.0f;
#pragma unroll 4
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint32_t packed = load_32bit(reinterpret_cast<const uint32_t *>(w_row + (k / 2)));
                int scale_idx = k / group_size;
                int scale_lane = lane_id & ~1;
                float scale = 0.0f;
                if (lane_id == scale_lane) {
                    scale = __half2float(__ldg(s_row + scale_idx));
                }
                scale = __shfl_sync(0xffffffff, scale, scale_lane);
                sum += dot8_nvfp4_f32(packed, scale, s_input + k);
            }
            sum = warp_reduce_sum(sum);
            if (lane_id == 0) {
                hidden_out[m] = __float2bfloat16(sum + __bfloat162float(residual[m]));
            }
        }
    }
}

static __device__ void full_attention_layer_nvfp4(
    AtomicGridSync &grid,
    const FullAttnWeightsNVFP4 &w,
    const __nv_bfloat16 *__restrict__ input,
    __nv_bfloat16 *__restrict__ k_cache,
    __nv_bfloat16 *__restrict__ v_cache,
    __nv_bfloat16 *__restrict__ g_residual,
    float *__restrict__ g_activations,
    float *__restrict__ g_q,
    float *__restrict__ g_kv,
    float *__restrict__ g_attn_out,
    float *__restrict__ g_mlp_inter,
    __nv_bfloat16 *__restrict__ hidden_out,
    int position, int max_seq_len,
    int group_size,
    __nv_bfloat16 *__restrict__ shmem)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    __shared__ float s_rope_cos[FA_ROTARY_DIM / 2];
    __shared__ float s_rope_sin[FA_ROTARY_DIM / 2];
    __shared__ float s_attn_partials[NUM_WARPS * FA_HEAD_DIM];

    __nv_bfloat16 *s_norm = shmem;
    rmsnorm_redundant(input, w.input_layernorm_weight, s_norm, g_residual);

    matvec_nvfp4(s_norm, w.q_proj, g_q, HIDDEN_SIZE, FA_QPROJ_SIZE, num_blocks, group_size);
    matvec_nvfp4(s_norm, w.k_proj, g_kv, HIDDEN_SIZE, FA_KV_SIZE, num_blocks, group_size);
    matvec_nvfp4(s_norm, w.v_proj, g_kv + FA_KV_SIZE, HIDDEN_SIZE, FA_KV_SIZE, num_blocks, group_size);
    grid.sync();

    if (threadIdx.x < FA_ROTARY_DIM / 2) {
        float angle = float(position) * ROPE_INV_FREQ[threadIdx.x];
        __sincosf(angle, &s_rope_sin[threadIdx.x], &s_rope_cos[threadIdx.x]);
    }
    __syncthreads();

    if (block_id == 0) {
        float *k_buf = g_kv;
        float *v_buf = g_kv + FA_KV_SIZE;
        for (int h = warp_id; h < FA_NUM_KV_HEADS; h += NUM_WARPS) {
            float *kh = k_buf + h * FA_HEAD_DIM;
            float *vh = v_buf + h * FA_HEAD_DIM;
            __nv_bfloat16 *kc = k_cache + h * max_seq_len * FA_HEAD_DIM + position * FA_HEAD_DIM;
            __nv_bfloat16 *vc = v_cache + h * max_seq_len * FA_HEAD_DIM + position * FA_HEAD_DIM;
            float ss = 0.0f;
            for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) {
                ss += kh[i] * kh[i];
            }
            ss = warp_reduce_sum(ss);
            float sc = rsqrtf(ss / float(FA_HEAD_DIM) + RMS_EPS);
            sc = __shfl_sync(0xffffffff, sc, 0);
            for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) {
                float wt = 1.0f + __bfloat162float(__ldg(w.k_norm_weight + i));
                float normed = kh[i] * sc * wt;
                if (i < FA_ROTARY_DIM) {
                    int pair = i & ((FA_ROTARY_DIM / 2) - 1);
                    int p = (i < FA_ROTARY_DIM / 2) ? i + FA_ROTARY_DIM / 2 : i - FA_ROTARY_DIM / 2;
                    float pwt = 1.0f + __bfloat162float(__ldg(w.k_norm_weight + p));
                    float pv = kh[p] * sc * pwt;
                    float cv = s_rope_cos[pair];
                    float sv = s_rope_sin[pair];
                    float rotated = (i < FA_ROTARY_DIM / 2) ? (normed * cv - pv * sv) : (pv * sv + normed * cv);
                    kc[i] = __float2bfloat16(rotated);
                } else {
                    kc[i] = __float2bfloat16(normed);
                }
                vc[i] = __float2bfloat16(vh[i]);
            }
        }
    }

    {
        int hpb = (FA_NUM_Q_HEADS + num_blocks - 1) / num_blocks;
        int hs = block_id * hpb;
        int he = min(hs + hpb, FA_NUM_Q_HEADS);
        for (int qh = hs; qh < he; qh++) {
            float *qh_ptr = g_q + qh * FA_HEAD_DIM * 2;
            if (warp_id == 0) {
                float ss = 0.0f;
                for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) {
                    ss += qh_ptr[i] * qh_ptr[i];
                }
                ss = warp_reduce_sum(ss);
                float sc = rsqrtf(ss / float(FA_HEAD_DIM) + RMS_EPS);
                sc = __shfl_sync(0xffffffff, sc, 0);
                for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) {
                    float wt = 1.0f + __bfloat162float(__ldg(w.q_norm_weight + i));
                    float normed = qh_ptr[i] * sc * wt;
                    if (i < FA_ROTARY_DIM) {
                        int pair = i & ((FA_ROTARY_DIM / 2) - 1);
                        int p = (i < FA_ROTARY_DIM / 2) ? i + FA_ROTARY_DIM / 2 : i - FA_ROTARY_DIM / 2;
                        float pwt = 1.0f + __bfloat162float(__ldg(w.q_norm_weight + p));
                        float pv = qh_ptr[p] * sc * pwt;
                        float cv = s_rope_cos[pair];
                        float sv = s_rope_sin[pair];
                        qh_ptr[i] = (i < FA_ROTARY_DIM / 2) ? (normed * cv - pv * sv) : (pv * sv + normed * cv);
                    } else {
                        qh_ptr[i] = normed;
                    }
                }
            }
        }
    }
    grid.sync();

    {
        int cache_len = position + 1;
        float attn_scale = 1.0f / sqrtf(float(FA_HEAD_DIM));
        int hpb = (FA_NUM_Q_HEADS + num_blocks - 1) / num_blocks;
        int hs = block_id * hpb;
        int he = min(hs + hpb, FA_NUM_Q_HEADS);
        __shared__ float s_max_score[NUM_WARPS];
        __shared__ float s_sum_exp[NUM_WARPS];
        constexpr int EPL = FA_HEAD_DIM / WARP_SIZE;

        for (int qh = hs; qh < he; qh++) {
            int kvh = qh / FA_GQA_RATIO;
            float *q_head = g_q + qh * FA_HEAD_DIM * 2;
            float *out_head = g_attn_out + qh * FA_HEAD_DIM;
            float max_score = -INFINITY;
            float sum_exp = 0.0f;
            float out_acc[EPL];
            float q_local[EPL];
            for (int e = 0; e < EPL; e++) {
                out_acc[e] = 0.0f;
                q_local[e] = q_head[lane_id * EPL + e];
            }

            for (int pos = warp_id; pos < cache_len; pos += NUM_WARPS) {
                const __nv_bfloat16 *k_pos = k_cache + kvh * max_seq_len * FA_HEAD_DIM + pos * FA_HEAD_DIM;
                const __nv_bfloat16 *v_pos = v_cache + kvh * max_seq_len * FA_HEAD_DIM + pos * FA_HEAD_DIM;
                float score = 0.0f;
                for (int e = 0; e < EPL; e++) {
                    score += q_local[e] * __bfloat162float(__ldg(k_pos + lane_id * EPL + e));
                }
                score = warp_reduce_sum(score) * attn_scale;
                score = __shfl_sync(0xffffffff, score, 0);
                float old_max = max_score;
                max_score = fmaxf(max_score, score);
                float exp_diff = fast_exp(old_max - max_score);
                sum_exp = sum_exp * exp_diff + fast_exp(score - max_score);
                float wt = fast_exp(score - max_score);
                for (int e = 0; e < EPL; e++) {
                    out_acc[e] = out_acc[e] * exp_diff + wt * __bfloat162float(__ldg(v_pos + lane_id * EPL + e));
                }
            }
            if (lane_id == 0) {
                s_max_score[warp_id] = max_score;
                s_sum_exp[warp_id] = sum_exp;
            }
            for (int e = 0; e < EPL; e++) {
                s_attn_partials[warp_id * FA_HEAD_DIM + lane_id * EPL + e] = out_acc[e];
            }
            __syncthreads();

            if (warp_id == 0) {
                float gm = -INFINITY;
                for (int ww = 0; ww < NUM_WARPS; ww++) {
                    if (s_max_score[ww] > -INFINITY) {
                        gm = fmaxf(gm, s_max_score[ww]);
                    }
                }
                float ts = 0.0f;
                float fo[EPL];
                for (int e = 0; e < EPL; e++) {
                    fo[e] = 0.0f;
                }
                for (int ww = 0; ww < NUM_WARPS; ww++) {
                    if (s_max_score[ww] > -INFINITY) {
                        float s = fast_exp(s_max_score[ww] - gm);
                        ts += s_sum_exp[ww] * s;
                        for (int e = 0; e < EPL; e++) {
                            fo[e] += s_attn_partials[ww * FA_HEAD_DIM + lane_id * EPL + e] * s;
                        }
                    }
                }
                float *gate_ptr = q_head + FA_HEAD_DIM;
                float rcp = 1.0f / ts;
                for (int e = 0; e < EPL; e++) {
                    int idx = lane_id * EPL + e;
                    out_head[idx] = fo[e] * rcp * fast_sigmoid(gate_ptr[idx]);
                }
            }
            __syncthreads();
        }
    }
    grid.sync();

    {
        float *s_attn = reinterpret_cast<float *>(shmem);
        for (int i = threadIdx.x; i < FA_Q_SIZE; i += BLOCK_SIZE) {
            s_attn[i] = g_attn_out[i];
        }
        __syncthreads();
        matvec_o_residual_nvfp4(s_attn, w.o_proj, g_residual, hidden_out, FA_Q_SIZE, HIDDEN_SIZE, num_blocks, group_size);
    }
    grid.sync();

    __nv_bfloat16 *s_act = shmem;
    rmsnorm_from_bf16(hidden_out, w.post_attn_layernorm_weight, s_act, g_residual);

    matvec_gate_up_silu_nvfp4(
        s_act, w.gate_proj, w.up_proj, g_mlp_inter,
        HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks, group_size);
    grid.sync();

    float *s_mlp = reinterpret_cast<float *>(shmem);
    for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += BLOCK_SIZE) {
        s_mlp[i] = g_mlp_inter[i];
    }
    __syncthreads();

    matvec_down_residual_nvfp4(
        s_mlp, w.down_proj, g_residual, hidden_out,
        INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks, group_size);
    grid.sync();
}

static __device__ void deltanet_layer_nvfp4(
    AtomicGridSync &grid,
    const DeltaNetWeightsNVFP4 &w,
    const __nv_bfloat16 *__restrict__ input,
    __nv_bfloat16 *__restrict__ g_residual,
    float *__restrict__ g_activations,
    float *__restrict__ g_qkv,
    float *__restrict__ g_z,
    float *__restrict__ g_beta,
    float *__restrict__ g_alpha,
    float *__restrict__ g_dn_out,
    float *__restrict__ g_mlp_inter,
    float *__restrict__ dn_state,
    float *__restrict__ conv_buf,
    __nv_bfloat16 *__restrict__ hidden_out,
    int dn_layer_idx,
    int group_size,
    __nv_bfloat16 *__restrict__ shmem)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    __nv_bfloat16 *s_norm = shmem;
    rmsnorm_redundant(input, w.input_layernorm_weight, s_norm, g_residual);

    matvec_nvfp4(s_norm, w.qkv_proj, g_qkv, HIDDEN_SIZE, DN_CONV_CHANNELS, num_blocks, group_size);
    matvec_nvfp4(s_norm, w.z_proj, g_z, HIDDEN_SIZE, DN_V_SIZE, num_blocks, group_size);
    matvec_bf16(s_norm, w.beta_proj_weight, g_beta, HIDDEN_SIZE, DN_NUM_HEADS, num_blocks);
    matvec_bf16(s_norm, w.alpha_proj_weight, g_alpha, HIDDEN_SIZE, DN_NUM_HEADS, num_blocks);
    grid.sync();

    if (block_id < DN_NUM_HEADS) {
        int h = block_id;
        float *layer_conv = conv_buf + dn_layer_idx * DN_CONV_CHANNELS * DN_CONV_KERNEL;

        __shared__ float s_q[DN_KEY_DIM];
        __shared__ float s_k[DN_KEY_DIM];
        __shared__ float s_v[DN_VALUE_DIM];
        int head_ch[3] = {h * DN_KEY_DIM, DN_QK_SIZE + h * DN_KEY_DIM, 2 * DN_QK_SIZE + h * DN_VALUE_DIM};
        for (int region = 0; region < 3; region++) {
            int ch_base = head_ch[region];
            int ch_count = (region < 2) ? DN_KEY_DIM : DN_VALUE_DIM;
            float *dst = (region == 0) ? s_q : (region == 1) ? s_k : s_v;
            for (int c = threadIdx.x; c < ch_count; c += BLOCK_SIZE) {
                int ch = ch_base + c;
                float h0 = layer_conv[ch * DN_CONV_KERNEL + 1];
                float h1 = layer_conv[ch * DN_CONV_KERNEL + 2];
                float h2 = layer_conv[ch * DN_CONV_KERNEL + 3];
                layer_conv[ch * DN_CONV_KERNEL + 0] = h0;
                layer_conv[ch * DN_CONV_KERNEL + 1] = h1;
                layer_conv[ch * DN_CONV_KERNEL + 2] = h2;
                layer_conv[ch * DN_CONV_KERNEL + 3] = g_qkv[ch];
                float co = 0.0f;
                for (int t = 0; t < DN_CONV_KERNEL; t++) {
                    co += layer_conv[ch * DN_CONV_KERNEL + t] * __bfloat162float(__ldg(w.conv1d_weight + ch * DN_CONV_KERNEL + t));
                }
                dst[c] = fast_silu(co);
            }
        }

        if (threadIdx.x == 0) {
            g_beta[h] = fast_sigmoid(g_beta[h]);
            float a_log_val = __bfloat162float(__ldg(w.a_log + h));
            float dt_b = __bfloat162float(__ldg(w.dt_bias + h));
            float x = g_alpha[h] + dt_b;
            float sp = (x > 20.0f) ? x : logf(1.0f + fast_exp(x));
            g_alpha[h] = fast_exp(-fast_exp(a_log_val) * sp);
        }
        __syncthreads();

        constexpr float Q_SCALE = 1.0f / 11.313708498984761f;
        if (warp_id == 0) {
            float sq = 0.0f;
            for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) {
                sq += s_q[i] * s_q[i];
            }
            sq = warp_reduce_sum(sq);
            float n = rsqrtf(sq + 1e-6f) * Q_SCALE;
            n = __shfl_sync(0xffffffff, n, 0);
            for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) {
                s_q[i] *= n;
            }
        }
        if (warp_id == 1) {
            float sq = 0.0f;
            for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) {
                sq += s_k[i] * s_k[i];
            }
            sq = warp_reduce_sum(sq);
            float n = rsqrtf(sq + 1e-6f);
            n = __shfl_sync(0xffffffff, n, 0);
            for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) {
                s_k[i] *= n;
            }
        }
        __syncthreads();

        float decay = g_alpha[h];
        float beta = g_beta[h];

        __shared__ float s_kq;
        if (warp_id == 0) {
            float kq = 0.0f;
            for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) {
                kq += s_k[i] * s_q[i];
            }
            kq = warp_reduce_sum(kq);
            if (lane_id == 0) {
                s_kq = kq;
            }
        }
        __syncthreads();
        float kq = s_kq;

        float *state = dn_state + h * DN_KEY_DIM * DN_VALUE_DIM;
        float *out_head = g_dn_out + h * DN_VALUE_DIM;

        constexpr int J_PER_WARP = DN_VALUE_DIM / NUM_WARPS;
        constexpr int I_PER_LANE = DN_KEY_DIM / WARP_SIZE;

#pragma unroll
        for (int jj = 0; jj < J_PER_WARP; jj++) {
            int j = warp_id * J_PER_WARP + jj;
            float s_regs[I_PER_LANE];
            float stk = 0.0f;
            float sqv = 0.0f;
#pragma unroll
            for (int ii = 0; ii < I_PER_LANE; ii++) {
                int i = lane_id + ii * WARP_SIZE;
                float sv = state[j * DN_KEY_DIM + i];
                s_regs[ii] = sv;
                stk += sv * s_k[i];
                sqv += sv * s_q[i];
            }
            stk = warp_reduce_sum(stk);
            sqv = warp_reduce_sum(sqv);
            stk = __shfl_sync(0xffffffff, stk, 0);
            sqv = __shfl_sync(0xffffffff, sqv, 0);
            float error_j = (s_v[j] - stk) * beta;
            float o_j = decay * sqv + error_j * kq;
            if (lane_id == 0) {
                out_head[j] = o_j;
            }
#pragma unroll
            for (int ii = 0; ii < I_PER_LANE; ii++) {
                int i = lane_id + ii * WARP_SIZE;
                state[j * DN_KEY_DIM + i] = s_regs[ii] * decay + s_k[i] * error_j;
            }
        }

        __syncthreads();
        {
            __shared__ float smem_gnorm[NUM_WARPS];
            float sq = 0.0f;
            for (int i = threadIdx.x; i < DN_VALUE_DIM; i += BLOCK_SIZE) {
                sq += out_head[i] * out_head[i];
            }
            sq = warp_reduce_sum(sq);
            if (lane_id == 0) {
                smem_gnorm[warp_id] = sq;
            }
            __syncthreads();
            if (warp_id == 0) {
                float v = (lane_id < NUM_WARPS) ? smem_gnorm[lane_id] : 0.0f;
                v = warp_reduce_sum(v);
                if (lane_id == 0) {
                    smem_gnorm[0] = rsqrtf(v / DN_VALUE_DIM + RMS_EPS);
                }
            }
            __syncthreads();
            float rstd = smem_gnorm[0];
            for (int i = threadIdx.x; i < DN_VALUE_DIM; i += BLOCK_SIZE) {
                float normed = out_head[i] * rstd * __bfloat162float(__ldg(w.norm_weight + i));
                float gate = fast_silu(g_z[h * DN_VALUE_DIM + i]);
                out_head[i] = normed * gate;
            }
        }
    }
    grid.sync();

    {
        float *s_dn = reinterpret_cast<float *>(shmem);
        for (int i = threadIdx.x; i < DN_V_SIZE; i += BLOCK_SIZE) {
            s_dn[i] = g_dn_out[i];
        }
        __syncthreads();
        matvec_o_residual_nvfp4(s_dn, w.out_proj, g_residual, hidden_out, DN_V_SIZE, HIDDEN_SIZE, num_blocks, group_size);
    }
    grid.sync();

    __nv_bfloat16 *s_act = shmem;
    rmsnorm_from_bf16(hidden_out, w.post_attn_layernorm_weight, s_act, g_residual);

    matvec_gate_up_silu_nvfp4(
        s_act, w.gate_proj, w.up_proj, g_mlp_inter,
        HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks, group_size);
    grid.sync();

    float *s_mlp = reinterpret_cast<float *>(shmem);
    for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += BLOCK_SIZE) {
        s_mlp[i] = g_mlp_inter[i];
    }
    __syncthreads();

    matvec_down_residual_nvfp4(
        s_mlp, w.down_proj, g_residual, hidden_out,
        INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks, group_size);
    grid.sync();
}

__global__ void prepare_lm_hidden_bf16_kernel(
    const float *__restrict__ hidden,
    __nv_bfloat16 *__restrict__ lm_hidden)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = LM_HEAD_TC_N * HIDDEN_SIZE;
    if (idx >= total) {
        return;
    }
    int col = idx / HIDDEN_SIZE;
    int k = idx % HIDDEN_SIZE;
    float value = (col == 0) ? hidden[k] : 0.0f;
    lm_hidden[idx] = __float2bfloat16(value);
}

__global__ void prepare_lm_hidden_bf16_from_bf16_kernel(
    const __nv_bfloat16 *__restrict__ hidden,
    __nv_bfloat16 *__restrict__ lm_hidden)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = LM_HEAD_TC_N * HIDDEN_SIZE;
    if (idx >= total) {
        return;
    }
    int row = idx / HIDDEN_SIZE;
    int k = idx % HIDDEN_SIZE;
    lm_hidden[idx] = (row == 0) ? hidden[k] : __float2bfloat16(0.0f);
}

__global__ void lm_head_argmax_half_kernel(
    const __half *__restrict__ logits,
    float *__restrict__ block_max_vals,
    int *__restrict__ block_max_idxs)
{
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int num_warps = LM_BLOCK_SIZE / WARP_SIZE;
    int rows_per_block = (VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, VOCAB_SIZE);

    float local_max = -INFINITY;
    int local_idx = -1;
    for (int row = row_start + warp_id; row < row_end; row += num_warps) {
        float value = -INFINITY;
        if (lane_id == 0) {
            value = __half2float(logits[row]);
        }
        value = __shfl_sync(0xffffffff, value, 0);
        if (lane_id == 0 && value > local_max) {
            local_max = value;
            local_idx = row;
        }
    }
    local_max = __shfl_sync(0xffffffff, local_max, 0);
    local_idx = __shfl_sync(0xffffffff, local_idx, 0);

    __shared__ float warp_max[32];
    __shared__ int warp_idx[32];
    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
        warp_idx[warp_id] = local_idx;
    }
    __syncthreads();
    if (warp_id == 0) {
        float max_value = (lane_id < num_warps) ? warp_max[lane_id] : -INFINITY;
        int max_index = (lane_id < num_warps) ? warp_idx[lane_id] : -1;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_value = __shfl_down_sync(0xffffffff, max_value, offset);
            int other_index = __shfl_down_sync(0xffffffff, max_index, offset);
            if (other_value > max_value) {
                max_value = other_value;
                max_index = other_index;
            }
        }
        if (lane_id == 0) {
            block_max_vals[blockIdx.x] = max_value;
            block_max_idxs[blockIdx.x] = max_index;
        }
    }
}

__global__ void __launch_bounds__(LM_SCALE_THREADS)
quantize_nvfp4_lm_kernel(
    const __nv_bfloat16 *__restrict__ weight,
    uint8_t *__restrict__ packed,
    uint8_t *__restrict__ scales,
    int rows, int cols, int scale_tiles)
{
    int row_block = blockIdx.x;
    int col_block = blockIdx.y;
    int tid = threadIdx.x;
    int tile_row = tid >> 2;
    int tile_col = tid & 3;

    int row = row_block * LM_SCALE_ROWS_PER_TILE + tile_row;
    int scale_col = col_block * LM_SCALE_COLS_PER_TILE + tile_col;
    int col_start = scale_col * LM_SCALE_BLOCK_K;

    bool in_bounds = (row < rows) && (col_start + LM_SCALE_BLOCK_K <= cols);
    const __nv_bfloat162 *weight_row = reinterpret_cast<const __nv_bfloat162 *>(weight + static_cast<size_t>(row) * cols);
    __nv_bfloat162 vals[8];
    float absmax = 0.0f;

    if (in_bounds) {
        const __nv_bfloat162 *src = weight_row + (col_start >> 1);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            vals[i] = src[i];
            float2 v = __bfloat1622float2(vals[i]);
            absmax = fmaxf(absmax, fmaxf(fabsf(v.x), fabsf(v.y)));
        }
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            vals[i] = __float2bfloat162_rn(0.0f);
        }
    }

    float scale_f = fmaxf(absmax * (1.0f / 6.0f), 1.0f / float(1 << 28));
    __nv_fp8_storage_t scale_fp8 = __nv_cvt_float_to_fp8(scale_f, __NV_SATFINITE, __NV_E4M3);
    __half_raw scale_half_raw = __nv_cvt_fp8_to_halfraw(scale_fp8, __NV_E4M3);
    float decoded_scale = __half2float(*reinterpret_cast<__half *>(&scale_half_raw));
    float inv_scale = (decoded_scale > 0.0f) ? (1.0f / decoded_scale) : 0.0f;

    uint64_t packed_u64 = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float2 v = __bfloat1622float2(vals[i]);
        v.x *= inv_scale;
        v.y *= inv_scale;
        __nv_fp4x2_storage_t p = __nv_cvt_float2_to_fp4x2(v, __NV_E2M1, cudaRoundNearest);
        packed_u64 |= static_cast<uint64_t>(static_cast<uint8_t>(p)) << (i * 8);
    }

    if (in_bounds) {
        *reinterpret_cast<uint64_t *>(
            packed + static_cast<size_t>(row) * (cols / 2) + (col_start >> 1)) = packed_u64;

        int row_in_tile = row & 31;
        int row_col_chunk = (row & (LM_SCALE_ROWS_PER_TILE - 1)) >> 5;
        int scale_tile = row_block * scale_tiles + (scale_col >> 2);
        int col_pos = scale_col & 3;
        int swizzled_idx = scale_tile * 512 + row_in_tile * 16 + row_col_chunk * 4 + col_pos;
        scales[swizzled_idx] = static_cast<uint8_t>(scale_fp8);
    }
}

__device__ __forceinline__ float lm_swizzled_scale_value(
    const uint8_t *__restrict__ scales,
    int row,
    int scale_idx)
{
    constexpr int scales_per_row = HIDDEN_SIZE / LM_SCALE_BLOCK_K;
    constexpr int scale_tiles_per_row = scales_per_row / LM_SCALE_COLS_PER_TILE;
    int row_block = row / LM_SCALE_ROWS_PER_TILE;
    int row_in_block = row % LM_SCALE_ROWS_PER_TILE;
    int row_in_tile = row_in_block & 31;
    int row_chunk = row_in_block >> 5;
    int tile = row_block * scale_tiles_per_row + (scale_idx >> 2);
    int swizzled_idx = tile * 512 + row_in_tile * 16 + row_chunk * 4 + (scale_idx & 3);
    __half_raw scale_half_raw = __nv_cvt_fp8_to_halfraw(
        static_cast<__nv_fp8_storage_t>(scales[swizzled_idx]),
        __NV_E4M3);
    return __half2float(*reinterpret_cast<__half *>(&scale_half_raw));
}

__global__ void lm_head_kernel_nvfp4(
    const float *__restrict__ hidden,
    const uint8_t *__restrict__ weight,
    const uint8_t *__restrict__ scales,
    float *__restrict__ block_max_vals,
    int *__restrict__ block_max_idxs,
    int group_size)
{
    __shared__ float s_hidden[HIDDEN_SIZE];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LM_BLOCK_SIZE) {
        s_hidden[i] = hidden[i];
    }
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int num_warps = LM_BLOCK_SIZE / WARP_SIZE;
    int rpb = (VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int rs = blockIdx.x * rpb;
    int re = min(rs + rpb, VOCAB_SIZE);
    int row_bytes = HIDDEN_SIZE / 2;
    (void)group_size;

    float local_max = -INFINITY;
    int local_max_idx = -1;
    for (int m = rs + warp_id; m < re; m += num_warps) {
        const uint8_t *w_row = weight + m * row_bytes;
        float sum = 0.0f;
#pragma unroll 4
        for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
            uint32_t packed = load_32bit(reinterpret_cast<const uint32_t *>(w_row + (k / 2)));
            int scale_idx = k / LM_SCALE_BLOCK_K;
            int scale_lane = lane_id & ~1;
            float scale = 0.0f;
            if (lane_id == scale_lane) {
                scale = lm_swizzled_scale_value(scales, m, scale_idx);
            }
            scale = __shfl_sync(0xffffffff, scale, scale_lane);
            sum += dot8_nvfp4_f32(packed, scale, s_hidden + k);
        }
        sum = warp_reduce_sum(sum);
        if (lane_id == 0 && sum > local_max) {
            local_max = sum;
            local_max_idx = m;
        }
    }
    local_max = __shfl_sync(0xffffffff, local_max, 0);
    local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

    __shared__ float wm[32];
    __shared__ int wi[32];
    if (lane_id == 0) {
        wm[warp_id] = local_max;
        wi[warp_id] = local_max_idx;
    }
    __syncthreads();
    if (warp_id == 0) {
        float mv = (lane_id < num_warps) ? wm[lane_id] : -INFINITY;
        int mi = (lane_id < num_warps) ? wi[lane_id] : -1;
        for (int o = WARP_SIZE / 2; o > 0; o /= 2) {
            float ov = __shfl_down_sync(0xffffffff, mv, o);
            int oi = __shfl_down_sync(0xffffffff, mi, o);
            if (ov > mv) {
                mv = ov;
                mi = oi;
            }
        }
        if (lane_id == 0) {
            block_max_vals[blockIdx.x] = mv;
            block_max_idxs[blockIdx.x] = mi;
        }
    }
}

__global__ void lm_head_reduce_kernel(
    const float *__restrict__ block_max_vals,
    const int *__restrict__ block_max_idxs,
    int *__restrict__ output_token,
    int num_blocks)
{
    int tid = threadIdx.x;
    float bv = -INFINITY;
    int bi = -1;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float v = block_max_vals[i];
        if (v > bv) {
            bv = v;
            bi = block_max_idxs[i];
        }
    }

    __shared__ float sv[LM_BLOCK_SIZE];
    __shared__ int si[LM_BLOCK_SIZE];
    sv[tid] = bv;
    si[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sv[tid + s] > sv[tid]) {
            sv[tid] = sv[tid + s];
            si[tid] = si[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *output_token = si[0];
    }
}

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
decode_kernel_nvfp4(
    const int *__restrict__ input_token_ptr,
    const __nv_bfloat16 *__restrict__ embed_weight,
    const __nv_bfloat16 *__restrict__ final_norm_weight,
    const LayerWeightsNVFP4 *__restrict__ layer_weights,
    __nv_bfloat16 *__restrict__ fa_k_cache,
    __nv_bfloat16 *__restrict__ fa_v_cache,
    float *__restrict__ dn_states,
    float *__restrict__ conv_bufs,
    __nv_bfloat16 *__restrict__ hidden_buffer,
    float *__restrict__ g_activations,
    __nv_bfloat16 *__restrict__ g_residual,
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
    int position, int max_seq_len)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    (void)barrier_counter;
    (void)barrier_generation;

    AtomicGridSync grid{};

    __shared__ __align__(16) char shmem_raw[MAX_ACT_DIM * sizeof(float)];
    __nv_bfloat16 *shmem_bf16 = reinterpret_cast<__nv_bfloat16 *>(shmem_raw);

    int input_token_id = __ldg(input_token_ptr);
    const __nv_bfloat16 *embed_row = embed_weight + input_token_id * HIDDEN_SIZE;
    int fa_kv_stride = FA_NUM_KV_HEADS * max_seq_len * FA_HEAD_DIM;
    int dn_state_stride = DN_NUM_HEADS * DN_KEY_DIM * DN_VALUE_DIM;

    int dn_layer_idx = 0;
    int fa_layer_idx = 0;

    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        const __nv_bfloat16 *layer_input = (layer == 0) ? embed_row : hidden_buffer;
        const LayerWeightsNVFP4 &weights = layer_weights[layer];

        if (LAYER_TYPE[layer] == 0) {
            DeltaNetWeightsNVFP4 dn_weights = make_deltanet_weights(weights);
            deltanet_layer_nvfp4(
                grid, dn_weights, layer_input,
                g_residual, g_activations, g_qkv_scratch, g_z_scratch,
                g_beta_scratch, g_alpha_scratch, g_attn_out, g_mlp_inter,
                dn_states + dn_layer_idx * dn_state_stride,
                conv_bufs, hidden_buffer, dn_layer_idx, weights.group_size, shmem_bf16);
            dn_layer_idx++;
        } else {
            FullAttnWeightsNVFP4 fa_weights = make_full_attn_weights(weights);
            full_attention_layer_nvfp4(
                grid, fa_weights, layer_input,
                fa_k_cache + fa_layer_idx * fa_kv_stride,
                fa_v_cache + fa_layer_idx * fa_kv_stride,
                g_residual, g_activations, g_qkv_scratch, g_kv_scratch,
                g_attn_out, g_mlp_inter, hidden_buffer,
                position, max_seq_len, weights.group_size, shmem_bf16);
            fa_layer_idx++;
        }
    }

    if (block_id == 0) {
        __shared__ float smem_reduce[NUM_WARPS];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
            float v = __bfloat162float(hidden_buffer[i]);
            local_sum_sq += v * v;
        }
        local_sum_sq = warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_sum_sq;
        }
        __syncthreads();
        if (warp_id == 0) {
            float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / HIDDEN_SIZE + RMS_EPS);
            }
        }
        __syncthreads();
        float rstd = smem_reduce[0];
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
            float v = __bfloat162float(hidden_buffer[i]);
            float wt = __bfloat162float(__ldg(final_norm_weight + i));
            g_normalized[i] = v * rstd * (1.0f + wt);
        }
    }
}

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
prefill_kernel_nvfp4(
    const int *__restrict__ token_ids,
    int seq_len,
    const __nv_bfloat16 *__restrict__ embed_weight,
    const __nv_bfloat16 *__restrict__ final_norm_weight,
    const LayerWeightsNVFP4 *__restrict__ layer_weights,
    __nv_bfloat16 *__restrict__ fa_k_cache,
    __nv_bfloat16 *__restrict__ fa_v_cache,
    float *__restrict__ dn_states,
    float *__restrict__ conv_bufs,
    __nv_bfloat16 *__restrict__ hidden_buffer,
    float *__restrict__ g_activations,
    __nv_bfloat16 *__restrict__ g_residual,
    float *__restrict__ g_qkv_scratch,
    float *__restrict__ g_kv_scratch,
    float *__restrict__ g_attn_out,
    float *__restrict__ g_mlp_inter,
    float *__restrict__ g_z_scratch,
    float *__restrict__ g_beta_scratch,
    float *__restrict__ g_alpha_scratch,
    float *__restrict__ g_normalized,
    int max_seq_len)
{
    int block_id = blockIdx.x;
    AtomicGridSync grid{};

    __shared__ __align__(16) char shmem_raw[MAX_ACT_DIM * sizeof(float)];
    __nv_bfloat16 *shmem_bf16 = reinterpret_cast<__nv_bfloat16 *>(shmem_raw);

    int fa_kv_stride = FA_NUM_KV_HEADS * max_seq_len * FA_HEAD_DIM;
    int dn_state_stride = DN_NUM_HEADS * DN_KEY_DIM * DN_VALUE_DIM;

    for (int position = 0; position < seq_len; ++position) {
        int input_token_id = __ldg(token_ids + position);
        const __nv_bfloat16 *embed_row = embed_weight + input_token_id * HIDDEN_SIZE;

        int dn_layer_idx = 0;
        int fa_layer_idx = 0;

        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            const __nv_bfloat16 *layer_input = (layer == 0) ? embed_row : hidden_buffer;
            const LayerWeightsNVFP4 &weights = layer_weights[layer];

            if (LAYER_TYPE[layer] == 0) {
                DeltaNetWeightsNVFP4 dn_weights = make_deltanet_weights(weights);
                deltanet_layer_nvfp4(
                    grid, dn_weights, layer_input,
                    g_residual, g_activations, g_qkv_scratch, g_z_scratch,
                    g_beta_scratch, g_alpha_scratch, g_attn_out, g_mlp_inter,
                    dn_states + dn_layer_idx * dn_state_stride,
                    conv_bufs, hidden_buffer, dn_layer_idx, weights.group_size, shmem_bf16);
                dn_layer_idx++;
            } else {
                FullAttnWeightsNVFP4 fa_weights = make_full_attn_weights(weights);
                full_attention_layer_nvfp4(
                    grid, fa_weights, layer_input,
                    fa_k_cache + fa_layer_idx * fa_kv_stride,
                    fa_v_cache + fa_layer_idx * fa_kv_stride,
                    g_residual, g_activations, g_qkv_scratch, g_kv_scratch,
                    g_attn_out, g_mlp_inter, hidden_buffer,
                    position, max_seq_len, weights.group_size, shmem_bf16);
                fa_layer_idx++;
            }
        }
    }

    if (block_id == 0) {
        __shared__ float smem_reduce[NUM_WARPS];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
            float v = __bfloat162float(hidden_buffer[i]);
            local_sum_sq += v * v;
        }
        local_sum_sq = warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_sum_sq;
        }
        __syncthreads();
        if (warp_id == 0) {
            float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / HIDDEN_SIZE + RMS_EPS);
            }
        }
        __syncthreads();
        float rstd = smem_reduce[0];
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
            float v = __bfloat162float(hidden_buffer[i]);
            float wt = __bfloat162float(__ldg(final_norm_weight + i));
            g_normalized[i] = v * rstd * (1.0f + wt);
        }
    }
}

__device__ __forceinline__ unsigned int quantize_fp4_code(float x) {
    float ax = fabsf(x);
    unsigned int mag = 0;
    if (ax < 0.25f) mag = 0;
    else if (ax < 0.75f) mag = 1;
    else if (ax < 1.25f) mag = 2;
    else if (ax < 1.75f) mag = 3;
    else if (ax < 2.5f) mag = 4;
    else if (ax < 3.5f) mag = 5;
    else if (ax < 5.0f) mag = 6;
    else mag = 7;
    return (x < 0.0f) ? (mag | 0x8u) : mag;
}

__global__ void quantize_nvfp4_kernel(
    const __nv_bfloat16 *__restrict__ weight,
    uint8_t *__restrict__ packed,
    __half *__restrict__ scales,
    int rows, int cols, int group_size)
{
    int groups_per_row = cols / group_size;
    int total_groups = rows * groups_per_row;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_groups) {
        return;
    }

    int row = gid / groups_per_row;
    int group = gid % groups_per_row;
    const __nv_bfloat16 *src = weight + row * cols + group * group_size;
    uint8_t *dst = packed + row * (cols / 2) + group * (group_size / 2);

    float amax = 0.0f;
    for (int i = 0; i < group_size; ++i) {
        amax = fmaxf(amax, fabsf(__bfloat162float(src[i])));
    }

    float scale = (amax > 0.0f) ? (amax / 6.0f) : 1.0f;
    scales[row * groups_per_row + group] = __float2half(scale);

    if (amax == 0.0f) {
        for (int i = 0; i < group_size / 2; ++i) {
            dst[i] = 0;
        }
        return;
    }

    float inv_scale = 1.0f / scale;
    for (int i = 0; i < group_size; i += 2) {
        unsigned int lo = quantize_fp4_code(__bfloat162float(src[i + 0]) * inv_scale);
        unsigned int hi = quantize_fp4_code(__bfloat162float(src[i + 1]) * inv_scale);
        dst[i / 2] = static_cast<uint8_t>(lo | (hi << 4));
    }
}

static int decode_launch_blocks() {
    if (const char *override_blocks = std::getenv("MEGAKERNEL_DECODE_BLOCKS")) {
        int value = std::atoi(override_blocks);
        if (value > 0) {
            return value;
        }
    }
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    int active_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, decode_kernel_nvfp4, BLOCK_SIZE, 0);
    return std::max(1, active_blocks * prop.multiProcessorCount);
}

static int prefill_launch_blocks() {
    if (const char *override_blocks = std::getenv("MEGAKERNEL_PREFILL_BLOCKS")) {
        int value = std::atoi(override_blocks);
        if (value > 0) {
            return value;
        }
    }
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    int active_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, prefill_kernel_nvfp4, BLOCK_SIZE, 0);
    int resident = std::max(1, active_blocks * prop.multiProcessorCount);
    return std::min(resident, 24);
}

static int lm_launch_blocks() {
    if (const char *override_blocks = std::getenv("MEGAKERNEL_LM_BLOCKS")) {
        int value = std::atoi(override_blocks);
        if (value > 0) {
            return value;
        }
    }
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    int active_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, lm_head_kernel_nvfp4, LM_BLOCK_SIZE, 0);
    int resident = std::max(1, active_blocks * prop.multiProcessorCount);
    int target = std::max(resident, prop.multiProcessorCount * 8);
    return std::min(1024, target);
}

extern "C" void launch_quantize_nvfp4_out(
    const void *weight, int rows, int cols, int group_size,
    void *packed_out, void *scales_out, cudaStream_t stream)
{
    int groups_per_row = cols / group_size;
    int total_groups = rows * groups_per_row;
    int threads = 256;
    int blocks = (total_groups + threads - 1) / threads;
    quantize_nvfp4_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16 *)weight,
        (uint8_t *)packed_out,
        (__half *)scales_out,
        rows, cols, group_size);
}

extern "C" void launch_quantize_nvfp4_lm_out(
    const void *weight, int rows, int cols,
    void *packed_out, void *scales_out, cudaStream_t stream)
{
    int scale_tiles = cols / LM_SCALE_K_PER_TILE;
    dim3 grid((rows + LM_SCALE_ROWS_PER_TILE - 1) / LM_SCALE_ROWS_PER_TILE, scale_tiles);
    dim3 block(LM_SCALE_THREADS);
    quantize_nvfp4_lm_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16 *)weight,
        (uint8_t *)packed_out,
        (uint8_t *)scales_out,
        rows,
        cols,
        scale_tiles);
}

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
    cudaStream_t stream)
{
    int prepare_threads = 256;
    int prepare_blocks = (LM_HEAD_TC_N * HIDDEN_SIZE + prepare_threads - 1) / prepare_threads;
    prepare_lm_hidden_bf16_from_bf16_kernel<<<prepare_blocks, prepare_threads, 0, stream>>>(
        (const __nv_bfloat16 *)hidden_bf16,
        (__nv_bfloat16 *)lm_hidden_bf16);
    cudaMemsetAsync(lm_hidden_scales, 0, LM_HEAD_HIDDEN_SCALE_BYTES, stream);
    launch_quantize_nvfp4_lm_out(
        lm_hidden_bf16,
        LM_HEAD_TC_N,
        HIDDEN_SIZE,
        lm_hidden_packed,
        lm_hidden_scales,
        stream);

    bool used_cublaslt = launch_lm_head_cublaslt(
        (const uint8_t *)lm_head_weight_packed,
        (const uint8_t *)lm_head_scales,
        (const uint8_t *)lm_hidden_packed,
        (const uint8_t *)lm_hidden_scales,
        (__half *)lm_logits_f16,
        stream);
    if (!used_cublaslt) {
        return false;
    }

    int lm_blocks = lm_launch_blocks();
    lm_head_argmax_half_kernel<<<lm_blocks, LM_BLOCK_SIZE, 0, stream>>>(
        (const __half *)lm_logits_f16,
        block_max_vals,
        block_max_idxs);
    lm_head_reduce_kernel<<<1, LM_BLOCK_SIZE, 0, stream>>>(
        block_max_vals,
        block_max_idxs,
        output_token_id,
        lm_blocks);
    return true;
}

static void launch_lm_head_from_f32_normalized(
    int lm_blocks,
    const float *normalized,
    int *output_token_id,
    const void *lm_head_weight_packed, const void *lm_head_scales,
    void *lm_hidden_bf16, void *lm_hidden_packed, void *lm_hidden_scales, void *lm_logits_f16,
    float *block_max_vals, int *block_max_idxs,
    int group_size,
    cudaStream_t stream)
{
    int prepare_threads = 256;
    int prepare_blocks = (LM_HEAD_TC_N * HIDDEN_SIZE + prepare_threads - 1) / prepare_threads;
    prepare_lm_hidden_bf16_kernel<<<prepare_blocks, prepare_threads, 0, stream>>>(
        normalized,
        (__nv_bfloat16 *)lm_hidden_bf16);
    cudaMemsetAsync(lm_hidden_scales, 0, LM_HEAD_HIDDEN_SCALE_BYTES, stream);
    launch_quantize_nvfp4_lm_out(
        lm_hidden_bf16,
        LM_HEAD_TC_N,
        HIDDEN_SIZE,
        lm_hidden_packed,
        lm_hidden_scales,
        stream);

    bool used_cublaslt = launch_lm_head_cublaslt(
        (const uint8_t *)lm_head_weight_packed,
        (const uint8_t *)lm_head_scales,
        (const uint8_t *)lm_hidden_packed,
        (const uint8_t *)lm_hidden_scales,
        (__half *)lm_logits_f16,
        stream);
    if (lm_debug_enabled()) {
        std::fprintf(stderr, "lm_head_cublaslt: used=%d\n", int(used_cublaslt));
    }
    if (used_cublaslt) {
        lm_head_argmax_half_kernel<<<lm_blocks, LM_BLOCK_SIZE, 0, stream>>>(
            (const __half *)lm_logits_f16,
            block_max_vals,
            block_max_idxs);
    } else {
        lm_head_kernel_nvfp4<<<lm_blocks, LM_BLOCK_SIZE, 0, stream>>>(
            normalized,
            (const uint8_t *)lm_head_weight_packed,
            (const uint8_t *)lm_head_scales,
            block_max_vals, block_max_idxs,
            group_size);
    }
    lm_head_reduce_kernel<<<1, LM_BLOCK_SIZE, 0, stream>>>(
        block_max_vals,
        block_max_idxs,
        output_token_id,
        lm_blocks);
}

static void launch_decode_step_nvfp4(
    int num_blocks, int lm_blocks,
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
    int position, int max_seq_len, int group_size, cudaStream_t stream)
{
    void *decode_args[] = {
        (void *)&input_token_ptr,
        (void *)&embed_weight,
        (void *)&final_norm_weight,
        (void *)&layer_weights,
        (void *)&fa_k_cache,
        (void *)&fa_v_cache,
        (void *)&dn_states,
        (void *)&conv_bufs,
        (void *)&hidden_buffer,
        (void *)&g_activations,
        (void *)&g_residual,
        (void *)&g_qkv_scratch,
        (void *)&g_kv_scratch,
        (void *)&g_attn_out,
        (void *)&g_mlp_inter,
        (void *)&g_z_scratch,
        (void *)&g_beta_scratch,
        (void *)&g_alpha_scratch,
        (void *)&g_normalized,
        (void *)&barrier_counter,
        (void *)&barrier_generation,
        (void *)&position,
        (void *)&max_seq_len,
    };
    cudaLaunchCooperativeKernel(
        (void *)decode_kernel_nvfp4,
        dim3(num_blocks),
        dim3(BLOCK_SIZE),
        decode_args,
        0,
        stream);
    launch_lm_head_from_f32_normalized(
        lm_blocks,
        (const float *)g_normalized,
        output_token_id,
        lm_head_weight_packed, lm_head_scales,
        lm_hidden_bf16, lm_hidden_packed, lm_hidden_scales, lm_logits_f16,
        block_max_vals, block_max_idxs,
        group_size,
        stream);
    (void)lm_sync_counter;
}

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
    int position, int max_seq_len, int group_size, cudaStream_t stream)
{
    int num_blocks = decode_launch_blocks();
    int lm_blocks = lm_launch_blocks();
    launch_decode_step_nvfp4(
        num_blocks,
        lm_blocks,
        input_token_ptr,
        output_token_id,
        embed_weight,
        layer_weights,
        final_norm_weight,
        lm_head_weight_packed,
        lm_head_scales,
        lm_hidden_bf16,
        lm_hidden_packed,
        lm_hidden_scales,
        lm_logits_f16,
        fa_k_cache,
        fa_v_cache,
        dn_states,
        conv_bufs,
        hidden_buffer,
        g_activations,
        g_residual,
        g_qkv_scratch,
        g_kv_scratch,
        g_attn_out,
        g_mlp_inter,
        g_z_scratch,
        g_beta_scratch,
        g_alpha_scratch,
        g_normalized,
        barrier_counter,
        barrier_generation,
        block_max_vals,
        block_max_idxs,
        lm_sync_counter,
        position,
        max_seq_len,
        group_size,
        stream);
}

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
    int max_seq_len, int group_size, cudaStream_t stream)
{
    (void)barrier_counter;
    (void)barrier_generation;
    (void)lm_sync_counter;

    int num_blocks = prefill_launch_blocks();
    int lm_blocks = lm_launch_blocks();
    void *prefill_args[] = {
        (void *)&token_ids,
        (void *)&seq_len,
        (void *)&embed_weight,
        (void *)&final_norm_weight,
        (void *)&layer_weights,
        (void *)&fa_k_cache,
        (void *)&fa_v_cache,
        (void *)&dn_states,
        (void *)&conv_bufs,
        (void *)&hidden_buffer,
        (void *)&g_activations,
        (void *)&g_residual,
        (void *)&g_qkv_scratch,
        (void *)&g_kv_scratch,
        (void *)&g_attn_out,
        (void *)&g_mlp_inter,
        (void *)&g_z_scratch,
        (void *)&g_beta_scratch,
        (void *)&g_alpha_scratch,
        (void *)&g_normalized,
        (void *)&max_seq_len,
    };
    cudaLaunchCooperativeKernel(
        (void *)prefill_kernel_nvfp4,
        dim3(num_blocks),
        dim3(BLOCK_SIZE),
        prefill_args,
        0,
        stream);

    launch_lm_head_from_f32_normalized(
        lm_blocks,
        (const float *)g_normalized,
        output_token_id,
        lm_head_weight_packed, lm_head_scales,
        lm_hidden_bf16, lm_hidden_packed, lm_hidden_scales, lm_logits_f16,
        block_max_vals, block_max_idxs,
        group_size,
        stream);
}

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
    int position, int max_seq_len, int group_size, cudaStream_t stream)
{
    int num_blocks = decode_launch_blocks();
    int lm_blocks = lm_launch_blocks();
    for (int step = 0; step < steps; ++step) {
        launch_decode_step_nvfp4(
            num_blocks,
            lm_blocks,
            token_buffer,
            token_buffer,
            embed_weight,
            layer_weights,
            final_norm_weight,
            lm_head_weight_packed,
            lm_head_scales,
            lm_hidden_bf16,
            lm_hidden_packed,
            lm_hidden_scales,
            lm_logits_f16,
            fa_k_cache,
            fa_v_cache,
            dn_states,
            conv_bufs,
            hidden_buffer,
            g_activations,
            g_residual,
            g_qkv_scratch,
            g_kv_scratch,
            g_attn_out,
            g_mlp_inter,
            g_z_scratch,
            g_beta_scratch,
            g_alpha_scratch,
            g_normalized,
            barrier_counter,
            barrier_generation,
            block_max_vals,
            block_max_idxs,
            lm_sync_counter,
            position + step,
            max_seq_len,
            group_size,
            stream);
        cudaMemcpyAsync(
            output_tokens + step,
            token_buffer,
            sizeof(int),
            cudaMemcpyDeviceToDevice,
            stream);
    }
}
