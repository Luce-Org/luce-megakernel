// Public C++ entry point for the FlashPrefill block-sparse attention used by
// the in-process Qwen3-0.6B drafter (speculative prefill scoring).
//
// Wraps kernels 1-4 + GPU block_select into one call. Call signature mirrors
// the upstream `flash_prefill` from qhfan/FlashPrefill (arXiv:2603.06199).
//
// Tensor layout (all CUDA, bf16, contiguous, D fastest):
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
//
// Tunables (env vars):
//   DFLASH_FP_USE_BSA      [0/1] enable BSA backend (default: 0).
//   DFLASH_FP_FUSED_SELECT [0/1] compute score+select in one CUDA kernel,
//                          avoiding full score/max scratch materialization.
//   DFLASH_FP_ALPHA        [float in (0,1)] override FlashPrefillConfig.alpha.
//                          Higher = stricter selection = fewer K-blocks per Q
//                          row = faster but riskier. Default 0.12. For long
//                          context with broad needles, 0.85-0.99 work well.
//   DFLASH_FP_PROFILE      [set] log per-stage timing (mean / score / select /
//                          forward) to stderr.
//   DFLASH_FP_DUMP_COUNTS  [set] log per-row select counts to stderr.

#pragma once

#include <cstdint>
#include "ggml-backend.h"

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
// Scratch memory (allocated/freed per call inside): default ~M*M*H*4 * 3 +
// M*H*4 where M = ceil(seq_len/block_size). At S=140K, M≈1093, H=16:
// ~300 MB. DFLASH_FP_FUSED_SELECT=1 skips the float score/max scratch.
int flash_prefill_forward_bf16(
    const void * Q, const void * K, const void * V, void * O,
    int batch, int seq_len, int n_q_heads, int n_k_heads, int head_dim,
    float scale,
    const FlashPrefillConfig & cfg);

// ggml flash_attn_ext-based implementation. Works on all SM targets (75+).
// Same interface as flash_prefill_forward_bf16 but uses ggml's FA internally
// (chunked causal attention). Accepts BF16/F16 Q/K/V tensors stored in the
// same [B, S, H, D] contiguous layout. No custom WMMA kernels needed.
//
// On SM < 80 this is the only available path (BF16 WMMA doesn't exist).
// On SM >= 80 callers may prefer flash_prefill_forward_bf16 for the
// block-sparse selection + custom WMMA kernel path.
int flash_prefill_forward_q8(
    ggml_backend_t backend,
    const void * Q, const void * K, const void * V, void * O,
    int batch, int seq_len, int n_q_heads, int n_k_heads, int head_dim,
    float scale,
    int elem_size,
    const FlashPrefillConfig & cfg);

#ifdef DFLASH27B_HAVE_BSA
// Free BSA persistent device buffers (blockmask, head_mask_type, softmax_lse).
// Safe to call any time; idempotent. Useful before unloading the drafter to
// give the daemon's target gen path the full VRAM headroom.
extern "C" void dflash_bsa_free_persistent();
#endif

} // namespace flashprefill
} // namespace dflash27b
