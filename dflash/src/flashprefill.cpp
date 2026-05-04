// Glue between the 4 FlashPrefill steps:
//   1. compute_mean_vector_bf16    (kernel)
//   2. compute_block_score_bf16    (kernel)  + normalise on host
//   3. block_select_host           (host)
//   4. sparse_flash_forward_bf16   (kernel)

#include "flashprefill.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "device_runtime.h"
#include <algorithm>
#include <vector>

namespace dflash27b {
namespace flashprefill {

extern "C" {
void launch_compute_mean_vector_bf16(
    const void * K, void * mean_K,
    int batch, int seq_len, int n_kv_heads, int head_dim, int block_size,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    cudaStream_t stream);

void launch_compute_block_score_bf16(
    const void * Q, const void * mean_K, float sm_scale,
    void * score, void * score_max,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int head_dim, int block_size,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    int s_S_b, int s_S_m, int s_S_n, int s_S_h,
    int s_M_b, int s_M_m, int s_M_n, int s_M_h,
    cudaStream_t stream);

void launch_block_select(
    const float * score,
    int B, int M, int N, int H,
    int attention_sink, int window, int last_n_full, float alpha,
    int s_b, int s_m, int s_n, int s_h,
    int idx_s_b, int idx_s_m, int idx_s_n, int idx_s_h,
    int cnt_s_b, int cnt_s_m, int cnt_s_h,
    int32_t * idx_out, int32_t * cnt_out,
    cudaStream_t stream);

#ifdef DFLASH27B_HAVE_BSA
int launch_bsa_sparse_flash_forward_bf16(
    const void* Q, const void* K, const void* V, void* O,
    const int32_t* indices, const int32_t* counts,
    float scale,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int head_dim, int block_size,
    int s_idx_b, int s_idx_m, int s_idx_n, int s_idx_h,
    int s_cnt_b, int s_cnt_m, int s_cnt_h,
    cudaStream_t stream);
#endif

void launch_sparse_flash_forward_bf16(
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
    cudaStream_t stream);
}

void block_select_host(
    const float * score,
    int B, int M, int N, int H,
    int attention_sink, int window, int last_n_full, float alpha,
    int32_t * idx_out, int32_t * cnt_out);

namespace {
inline int cdiv(int a, int b) { return (a + b - 1) / b; }

struct FlashPrefillScratch {
    void * dmK = nullptr; size_t dmK_bytes = 0;
    float * dS = nullptr; size_t dS_bytes = 0;
    float * dM = nullptr; size_t dM_bytes = 0;
    int32_t * dIdx = nullptr; size_t dIdx_bytes = 0;
    int32_t * dCnt = nullptr; size_t dCnt_bytes = 0;
};

static std::vector<FlashPrefillScratch> g_flashprefill_scratch;

template<typename T>
static T * ensure_buf(T ** ptr, size_t * cur_bytes, size_t need_bytes) {
    if (*ptr != nullptr && *cur_bytes >= need_bytes) {
        return *ptr;
    }
    if (*ptr != nullptr) {
        (void)cudaFree(*ptr);
        *ptr = nullptr;
        *cur_bytes = 0;
    }
    cudaError_t err = cudaMalloc((void **)ptr, need_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[flashprefill] cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    *cur_bytes = need_bytes;
    return *ptr;
}

static FlashPrefillScratch & scratch_for_current_device() {
    int device = 0;
    (void)cudaGetDevice(&device);
    if ((int)g_flashprefill_scratch.size() <= device) {
        g_flashprefill_scratch.resize(device + 1);
    }
    return g_flashprefill_scratch[(size_t)device];
}
}

int flash_prefill_forward_bf16(
    const void * Q, const void * K, const void * V, void * O,
    int batch, int seq_len, int n_q_heads, int n_k_heads, int head_dim,
    float scale,
    const FlashPrefillConfig & cfg)
{
    const int B = batch;
    const int S = seq_len;
    const int H = n_q_heads;
    const int Hk = n_k_heads;
    const int D = head_dim;
    const int BLOCK = cfg.block_size;
    const int M = cdiv(S, BLOCK);
    const int N = M;
    const int Q_TILE = 64;

    // Strides assume contiguous [B, S, H, D] / [B, S, Hk, D] row-major.
    int s_Q_b = S * H * D, s_Q_n = H * D, s_Q_h = D, s_Q_d = 1;
    int s_K_b = S * Hk * D, s_K_n = Hk * D, s_K_h = D, s_K_d = 1;
    int s_mK_b = M * Hk * D, s_mK_m = Hk * D, s_mK_h = D, s_mK_d = 1;
    int s_S_b = M * N * H, s_S_m = N * H, s_S_n = H, s_S_h = 1;
    int s_idx_b = M * N * H, s_idx_m = N * H, s_idx_n = H, s_idx_h = 1;
    int s_cnt_b = M * H, s_cnt_m = H, s_cnt_h = 1;

    FlashPrefillScratch & sc = scratch_for_current_device();
    void * dmK = ensure_buf(&sc.dmK, &sc.dmK_bytes, (size_t)B * M * Hk * D * 2); // bf16
    float * dS = ensure_buf(&sc.dS, &sc.dS_bytes, (size_t)B * M * N * H * sizeof(float));
    float * dM = ensure_buf(&sc.dM, &sc.dM_bytes, (size_t)B * M * N * H * sizeof(float));
    int32_t * dIdx = ensure_buf(&sc.dIdx, &sc.dIdx_bytes, (size_t)B * M * N * H * sizeof(int32_t));
    int32_t * dCnt = ensure_buf(&sc.dCnt, &sc.dCnt_bytes, (size_t)B * M * H * sizeof(int32_t));
    if (!dmK || !dS || !dM || !dIdx || !dCnt) {
        return -1;
    }

    static const bool prof = (std::getenv("DFLASH_FP_PROFILE") != nullptr);
    cudaEvent_t pE[5];
    if (prof) for (int i=0;i<5;i++) cudaEventCreate(&pE[i]);
    if (prof) cudaEventRecord(pE[0]);
    // 1. mean_K
    launch_compute_mean_vector_bf16(
        K, dmK, B, S, Hk, D, BLOCK,
        s_K_b, s_K_n, s_K_h, s_K_d,
        s_mK_b, s_mK_m, s_mK_h, s_mK_d, 0);

    if (prof) cudaEventRecord(pE[1]);
    // 2. block scores
    launch_compute_block_score_bf16(
        Q, dmK, scale, dS, dM,
        B, H, Hk, S, D, BLOCK,
        s_Q_b, s_Q_n, s_Q_h, s_Q_d,
        s_mK_b, s_mK_m, s_mK_h, s_mK_d,
        s_S_b, s_S_m, s_S_n, s_S_h,
        s_S_b, s_S_m, s_S_n, s_S_h, 0);

    if (prof) cudaEventRecord(pE[2]);
    // 3. block_select on GPU.
    launch_block_select(
        dS, B, M, N, H,
        cfg.attention_sink, cfg.window, cfg.last_n_full, cfg.alpha,
        s_S_b, s_S_m, s_S_n, s_S_h,
        s_idx_b, s_idx_m, s_idx_n, s_idx_h,
        s_cnt_b, s_cnt_m, s_cnt_h,
        dIdx, dCnt, 0);

    if (prof) cudaEventRecord(pE[3]);
    static const bool dump_cnt = (std::getenv("DFLASH_FP_DUMP_COUNTS") != nullptr);
    if (dump_cnt) {
        std::vector<int32_t> hcnt((size_t)B * M * H);
        cudaMemcpy(hcnt.data(), dCnt, hcnt.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
        long long sum = 0; int mn = 1<<30, mx = 0;
        for (auto c : hcnt) { sum += c; if (c<mn) mn=c; if (c>mx) mx=c; }
        std::fprintf(stderr, "[fp-cnt] S=%d M=%d H=%d  total_select=%lld avg=%.1f min=%d max=%d\n",
                     S, M, H, sum, (double)sum/(M*H*B), mn, mx);
    }
    // 4. sparse flash forward (BSA-or-WMMA)
#ifdef DFLASH27B_HAVE_BSA
    static const bool use_bsa = (std::getenv("DFLASH_FP_USE_BSA") != nullptr);
    if (use_bsa && D == 128 && BLOCK == 128) {
        launch_bsa_sparse_flash_forward_bf16(
            Q, K, V, O, dIdx, dCnt, scale,
            B, H, Hk, S, D, BLOCK,
            s_idx_b, s_idx_m, s_idx_n, s_idx_h,
            s_cnt_b, s_cnt_m, s_cnt_h, 0);
    } else
#endif
    {
        launch_sparse_flash_forward_bf16(
            Q, K, V, O, dIdx, dCnt, scale,
            B, H, Hk, S, D, Q_TILE, BLOCK,
            s_Q_b, s_Q_n, s_Q_h, s_Q_d,
            s_K_b, s_K_n, s_K_h, s_K_d,
            s_K_b, s_K_n, s_K_h, s_K_d,    // V uses K strides
            s_Q_b, s_Q_n, s_Q_h, s_Q_d,    // O uses Q strides
            s_idx_b, s_idx_m, s_idx_n, s_idx_h,
            s_cnt_b, s_cnt_m, s_cnt_h, 0);
    }

    if (prof) {
        cudaEventRecord(pE[4]);
        cudaEventSynchronize(pE[4]);
        float t1, t2, t3, t4;
        cudaEventElapsedTime(&t1, pE[0], pE[1]);
        cudaEventElapsedTime(&t2, pE[1], pE[2]);
        cudaEventElapsedTime(&t3, pE[2], pE[3]);
        cudaEventElapsedTime(&t4, pE[3], pE[4]);
        std::fprintf(stderr,
            "[fp-prof] S=%d H=%d Hk=%d  mean=%.2fms  score=%.2fms  select=%.2fms  forward=%.2fms\n",
            S, n_q_heads, n_k_heads, t1, t2, t3, t4);
        for (int i=0;i<5;i++) cudaEventDestroy(pE[i]);
    }
    return 0;
}

} // namespace flashprefill
} // namespace dflash27b
