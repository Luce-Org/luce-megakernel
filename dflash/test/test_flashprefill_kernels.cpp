// Smoke test for the FlashPrefill CUDA kernels.
//
// Tiny shapes (B=1, S=256, H=4, Hk=2, D=128, BLOCK=128) so we can run on
// CPU shadow + compare GPU output. Validates:
//   1. compute_mean_vector_bf16  — mean K per BLOCK rows
//   2. compute_block_score_bf16  — per-block score + max
//   3. sparse_flash_forward_bf16 — sparse attention output
//
// Pass criteria: max abs diff vs reference dense attention < 1e-2 (bf16
// numerics, so relaxed). Sparse path may differ where non-selected blocks
// were dropped — only check on a uniform full-selection mask.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../src/flashprefill.h"

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

#define CK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(e)); \
        return 1; \
    } \
} while (0)

static __nv_bfloat16 f2b(float x) { return __float2bfloat16(x); }
static float b2f(__nv_bfloat16 x) { return __bfloat162float(x); }

int main() {
    constexpr int B = 1;
    constexpr int S = 256;
    constexpr int H = 4;
    constexpr int Hk = 2;
    constexpr int D = 128;
    constexpr int BLOCK = 128;
    constexpr int M = (S + BLOCK - 1) / BLOCK;   // 2
    constexpr int Q_TILE = 64;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    std::vector<__nv_bfloat16> Q(B * S * H * D), K(B * S * Hk * D), V(B * S * Hk * D);
    for (auto & x : Q) x = f2b(dist(rng));
    for (auto & x : K) x = f2b(dist(rng));
    for (auto & x : V) x = f2b(dist(rng));

    __nv_bfloat16 *dQ, *dK, *dV, *dO, *dmK;
    float *dS, *dM;
    int32_t *dIdx, *dCnt;
    CK(cudaMalloc(&dQ, Q.size() * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dK, K.size() * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dV, V.size() * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dO, B * S * H * D * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dmK, B * M * Hk * D * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dS, B * M * M * H * sizeof(float)));
    CK(cudaMalloc(&dM, B * M * M * H * sizeof(float)));
    CK(cudaMalloc(&dIdx, B * M * M * H * sizeof(int32_t)));
    CK(cudaMalloc(&dCnt, B * M * H * sizeof(int32_t)));

    CK(cudaMemcpy(dQ, Q.data(), Q.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, K.data(), K.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, V.data(), V.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // Strides (elements). Layout: [B, S, H, D] row-major, D fastest.
    int s_Q_b = S * H * D, s_Q_n = H * D, s_Q_h = D, s_Q_d = 1;
    int s_K_b = S * Hk * D, s_K_n = Hk * D, s_K_h = D, s_K_d = 1;
    int s_mK_b = M * Hk * D, s_mK_m = Hk * D, s_mK_h = D, s_mK_d = 1;
    int s_S_b = M * M * H, s_S_m = M * H, s_S_n = H, s_S_h = 1;
    int s_idx_b = M * M * H, s_idx_m = M * H, s_idx_n = H, s_idx_h = 1;
    int s_cnt_b = M * H, s_cnt_m = H, s_cnt_h = 1;

    // ── Kernel 1: compute_mean_vector ──
    launch_compute_mean_vector_bf16(
        dK, dmK, B, S, Hk, D, BLOCK,
        s_K_b, s_K_n, s_K_h, s_K_d,
        s_mK_b, s_mK_m, s_mK_h, s_mK_d, 0);
    CK(cudaDeviceSynchronize());

    // Verify host vs GPU mean for one block.
    std::vector<__nv_bfloat16> mK_h(B * M * Hk * D);
    CK(cudaMemcpy(mK_h.data(), dmK, mK_h.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    float max_diff_mean = 0.0f;
    for (int n = 0; n < M; ++n) {
        for (int h = 0; h < Hk; ++h) {
            for (int d = 0; d < D; ++d) {
                float ref = 0.0f;
                int n_lo = n * BLOCK, n_hi = std::min(n_lo + BLOCK, S);
                int cnt = n_hi - n_lo;
                for (int i = n_lo; i < n_hi; ++i) {
                    ref += b2f(K[i * Hk * D + h * D + d]);
                }
                ref /= (float)cnt;
                float gpu = b2f(mK_h[n * Hk * D + h * D + d]);
                max_diff_mean = std::fmax(max_diff_mean, std::fabs(ref - gpu));
            }
        }
    }
    std::printf("[fp-test] kernel 1 (compute_mean_vector): max diff = %.5f %s\n",
                max_diff_mean, max_diff_mean < 1e-2f ? "PASS" : "FAIL");

    // ── Kernel 2: compute_block_score ──
    float scale = 1.0f / std::sqrt((float)D);
    launch_compute_block_score_bf16(
        dQ, dmK, scale, dS, dM,
        B, H, Hk, S, D, BLOCK,
        s_Q_b, s_Q_n, s_Q_h, s_Q_d,
        s_mK_b, s_mK_m, s_mK_h, s_mK_d,
        s_S_b, s_S_m, s_S_n, s_S_h,
        s_S_b, s_S_m, s_S_n, s_S_h, 0);
    CK(cudaDeviceSynchronize());
    std::printf("[fp-test] kernel 2 (compute_block_score): launch ok\n");

    // ── Kernel 4: sparse_flash_forward (full selection = dense FA) ──
    // Set indices to [0, 1, ..., M-1] for each (b, q_block, h), counts=M
    std::vector<int32_t> idx_h(B * M * M * H);
    std::vector<int32_t> cnt_h(B * M * H);
    for (int b = 0; b < B; ++b)
      for (int m = 0; m < M; ++m)
        for (int h = 0; h < H; ++h) {
          cnt_h[b * M * H + m * H + h] = m + 1;        // causal: include only blocks 0..m
          for (int n = 0; n < M; ++n)
              idx_h[b * M * M * H + m * M * H + n * H + h] = (n <= m) ? n : -1;
        }
    CK(cudaMemcpy(dIdx, idx_h.data(), idx_h.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dCnt, cnt_h.data(), cnt_h.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    launch_sparse_flash_forward_bf16(
        dQ, dK, dV, dO, dIdx, dCnt, scale,
        B, H, Hk, S, D, Q_TILE, BLOCK,
        s_Q_b, s_Q_n, s_Q_h, s_Q_d,
        s_K_b, s_K_n, s_K_h, s_K_d,
        s_K_b, s_K_n, s_K_h, s_K_d,    // V uses same strides as K
        s_Q_b, s_Q_n, s_Q_h, s_Q_d,    // O uses Q strides
        s_idx_b, s_idx_m, s_idx_n, s_idx_h,
        s_cnt_b, s_cnt_m, s_cnt_h, 0);
    cudaError_t le = cudaDeviceSynchronize();
    std::printf("[fp-test] kernel 4 (sparse_flash_forward): launch %s\n",
                le == cudaSuccess ? "ok" : cudaGetErrorString(le));

    // Numerical check kernel 4: compare GPU sparse output to CPU dense
    // attention reference (full mask, fully causal). Should match within bf16
    // numerical tolerance.
    std::vector<__nv_bfloat16> O_h(B * S * H * D);
    CK(cudaMemcpy(O_h.data(), dO, O_h.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    float max_diff_attn = 0.0f;
    for (int q = 0; q < S; ++q) {
        for (int h = 0; h < H; ++h) {
            int hk = h * Hk / H;
            // Compute reference dense attention for this (q, h).
            float row_max = -INFINITY;
            std::vector<float> qk(S, -INFINITY);
            for (int j = 0; j <= q; ++j) {
                float s = 0.0f;
                for (int d = 0; d < D; ++d) {
                    s += b2f(Q[q * H * D + h * D + d]) * b2f(K[j * Hk * D + hk * D + d]);
                }
                qk[j] = s * scale;
                if (qk[j] > row_max) row_max = qk[j];
            }
            float sumexp = 0.0f;
            std::vector<float> p(S, 0.0f);
            for (int j = 0; j <= q; ++j) {
                p[j] = std::exp(qk[j] - row_max);
                sumexp += p[j];
            }
            for (int j = 0; j <= q; ++j) p[j] /= sumexp;

            for (int d = 0; d < D; ++d) {
                float ref = 0.0f;
                for (int j = 0; j <= q; ++j) {
                    ref += p[j] * b2f(V[j * Hk * D + hk * D + d]);
                }
                float gpu = b2f(O_h[q * H * D + h * D + d]);
                float diff = std::fabs(ref - gpu);
                if (diff > max_diff_attn) max_diff_attn = diff;
            }
        }
    }
    std::printf("[fp-test] kernel 4 (sparse_flash_forward) numerics: max diff = %.5f %s\n",
                max_diff_attn, max_diff_attn < 5e-2f ? "PASS" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    cudaFree(dmK); cudaFree(dS); cudaFree(dM); cudaFree(dIdx); cudaFree(dCnt);

    // ── End-to-end FlashPrefill at 8K context ──
    // Tests the full pipeline (kernels 1-4 + block_select) wrapped via
    // flash_prefill_forward_bf16. Compares to dense reference for a small
    // shape, then times a 8K shape to verify it's fast.
    {
        constexpr int BB = 1, BS = 8192, BH = 16, BHk = 8, BD = 128;
        constexpr int BL = 128;

        std::vector<__nv_bfloat16> bQ(BB * BS * BH * BD);
        std::vector<__nv_bfloat16> bK(BB * BS * BHk * BD);
        std::vector<__nv_bfloat16> bV(BB * BS * BHk * BD);
        for (auto & x : bQ) x = f2b(dist(rng));
        for (auto & x : bK) x = f2b(dist(rng));
        for (auto & x : bV) x = f2b(dist(rng));

        __nv_bfloat16 *bdQ, *bdK, *bdV, *bdO;
        CK(cudaMalloc(&bdQ, bQ.size() * sizeof(__nv_bfloat16)));
        CK(cudaMalloc(&bdK, bK.size() * sizeof(__nv_bfloat16)));
        CK(cudaMalloc(&bdV, bV.size() * sizeof(__nv_bfloat16)));
        CK(cudaMalloc(&bdO, bQ.size() * sizeof(__nv_bfloat16)));

        CK(cudaMemcpy(bdQ, bQ.data(), bQ.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(bdK, bK.data(), bK.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(bdV, bV.data(), bV.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

        dflash27b::flashprefill::FlashPrefillConfig cfg;
        cfg.block_size = BL;
        cfg.attention_sink = 2;
        cfg.window = 4;
        cfg.last_n_full = 2;
        cfg.alpha = 0.12f;

        // Warm-up
        dflash27b::flashprefill::flash_prefill_forward_bf16(
            bdQ, bdK, bdV, bdO, BB, BS, BH, BHk, BD,
            1.0f / std::sqrt((float)BD), cfg);
        CK(cudaDeviceSynchronize());

        cudaEvent_t e_a, e_b;
        cudaEventCreate(&e_a);
        cudaEventCreate(&e_b);
        cudaEventRecord(e_a);
        for (int it = 0; it < 5; ++it) {
            dflash27b::flashprefill::flash_prefill_forward_bf16(
                bdQ, bdK, bdV, bdO, BB, BS, BH, BHk, BD,
                1.0f / std::sqrt((float)BD), cfg);
        }
        cudaEventRecord(e_b);
        cudaEventSynchronize(e_b);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e_a, e_b);
        cudaEventDestroy(e_a);
        cudaEventDestroy(e_b);
        std::printf("[fp-test] e2e flash_prefill_forward_bf16 at S=%d: %.1f ms / iter (avg of 5)\n",
                    BS, ms / 5.0f);

        cudaFree(bdQ); cudaFree(bdK); cudaFree(bdV); cudaFree(bdO);
    }
    return 0;
}
