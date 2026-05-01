// Public C++ entry point for the FlashPrefill block-sparse attention used by
// the in-process Qwen3-0.6B drafter (speculative prefill scoring).
//
// Wraps kernels 1-4 + GPU block_select into one call. Call signature mirrors
// the upstream `flash_prefill` from qhfan/FlashPrefill (arXiv:2603.06199).
//
// Tensor layout (all CUDA, fp16/bf16, contiguous, D fastest):
//   Q[B, S, n_q_heads, D]
//   K[B, S, n_k_heads, D]
//   V[B, S, n_k_heads, D]
//   O[B, S, n_q_heads, D]
//
// Backends:
//   - Default: WMMA m16n16k16 sparse forward (sm_70+). Functional everywhere.
//   - Set env DFLASH_FP_USE_BSA=1 to dispatch to the Block-Sparse-Attention
//     kernel (FA-2 derived, m16n8k16 PTX, sm_80+ via cuBLAS BF16 GEMM).
//     Requires building with -DDFLASH27B_ENABLE_BSA=ON. ~3x faster than WMMA
//     on RTX 3090 at S=128K.
//   - On SM75/Turing (e.g. RTX 2080 Ti), BSA is unavailable and BF16 tensor
//     cores do not exist. The drafter should use FP16 weights/compute and this
//     WMMA path; tune alpha/keep ratio and preserve quality with the PFlash
//     compressor knobs documented in qwen3_drafter.h.
//
// Tunables (env vars):
//   DFLASH_FP_USE_BSA      [0/1] enable BSA backend (default: 0).
//   DFLASH_FP_ALPHA        [float in (0,1)] override FlashPrefillConfig.alpha.
//                          Higher = stricter selection = fewer K-blocks per Q
//                          row = faster but riskier. Default 0.12. For long
//                          context with broad needles, 0.85-0.99 work well.
//   DFLASH_FP_PROFILE      [set] log per-stage timing (mean / score / select /
//                          forward) to stderr.
//   DFLASH_FP_DUMP_COUNTS  [set] log per-row select counts to stderr.
//   DFLASH_FP_K_TILE       [32/64] internal WMMA sparse-forward tile tuning.
//                          Default 32 on SM75, 64 elsewhere; 32 trades more
//                          inner iterations for lower shared/register pressure.

#pragma once

#include <cstdint>

namespace dflash27b {
namespace flashprefill {

// Algorithmic parameters for the FlashPrefill selection + sparse forward.
struct FlashPrefillConfig {
    int   block_size       = 128;   // K stride; query block size = K block size
    int   attention_sink   = 2;     // first N k-blocks always selected
    int   window           = 4;     // last `window` k-blocks before query
    int   last_n_full      = 2;     // last N q-blocks attend to all selected blocks
    float alpha            = 0.12f; // dynamic top-K threshold (score >= max_score * alpha)
};

// Runs the full FP forward (mean_K → block_score → block_select → sparse_fwd).
// Returns 0 on success, non-zero on failure (allocator OOM, bad shape, etc.).
// Output O is written in place.
//
// Scratch memory (allocated/freed per call inside): ~M*M*H*4 * 3 + M*H*4
// where M = ceil(seq_len/block_size). At S=140K, M≈1093, H=16: ~300 MB.
int flash_prefill_forward_bf16(
    const void * Q, const void * K, const void * V, void * O,
    int batch, int seq_len, int n_q_heads, int n_k_heads, int head_dim,
    float scale,
    const FlashPrefillConfig & cfg);

int flash_prefill_forward_f16(
    const void * Q, const void * K, const void * V, void * O,
    int batch, int seq_len, int n_q_heads, int n_k_heads, int head_dim,
    float scale,
    const FlashPrefillConfig & cfg);

#ifdef DFLASH27B_HAVE_BSA
// Free BSA persistent device buffers (blockmask, head_mask_type, softmax_lse).
// Safe to call any time; idempotent. Useful before unloading the drafter to
// give the daemon's target gen path the full VRAM headroom.
extern "C" void dflash_bsa_free_persistent();
#endif

} // namespace flashprefill
} // namespace dflash27b
