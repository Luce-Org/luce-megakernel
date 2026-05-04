#include "flashprefill.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
    const int seq_len = argc >= 2 ? std::atoi(argv[1]) : 256;
    const int batch = 1;
    const int q_heads = 8;
    const int kv_heads = 2;
    const int head_dim = 256;
    const int elem_size = 2;

    void *q = nullptr;
    void *k = nullptr;
    void *v = nullptr;
    void *o = nullptr;
    const size_t q_bytes = (size_t)batch * seq_len * q_heads * head_dim * elem_size;
    const size_t kv_bytes = (size_t)batch * seq_len * kv_heads * head_dim * elem_size;
    if (cudaMalloc(&q, q_bytes) != cudaSuccess ||
        cudaMalloc(&k, kv_bytes) != cudaSuccess ||
        cudaMalloc(&v, kv_bytes) != cudaSuccess ||
        cudaMalloc(&o, q_bytes) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed\n");
        return 1;
    }
    cudaMemset(q, 0, q_bytes);
    cudaMemset(k, 0, kv_bytes);
    cudaMemset(v, 0, kv_bytes);
    cudaMemset(o, 0, q_bytes);

    dflash27b::flashprefill::FlashPrefillConfig cfg;
    cfg.block_size = 128;
    cfg.alpha = 0.85f;
    const float scale = 1.0f / 16.0f; // 1 / sqrt(256)
    int rc = dflash27b::flashprefill::flash_prefill_forward_bf16(
        q, k, v, o, batch, seq_len, q_heads, kv_heads, head_dim, scale, cfg);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    cudaFree(q);
    cudaFree(k);
    cudaFree(v);
    cudaFree(o);
    if (rc != 0 || err != cudaSuccess) {
        std::fprintf(stderr, "hdim256 BSA smoke failed rc=%d cuda=%s\n",
                     rc, cudaGetErrorString(err));
        return 1;
    }
    std::printf("hdim256 BSA smoke ok seq_len=%d\n", seq_len);
    return 0;
}
