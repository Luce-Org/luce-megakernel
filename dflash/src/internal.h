// Internal-only shared header for dflash27b library sources.
// Not installed, not exposed in the public API.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include "dflash27b.h"

namespace dflash27b {

// Single source of truth for error reporting.
// All loaders / graph builders push into this via set_last_error(...).
void set_last_error(std::string msg);

// ─── Target weights (Qwen3.5-27B, qwen35 hybrid, Q4_K_M in ggml context) ──
//
// Qwen3.5 uses two kinds of blocks interleaved:
//   - FULL ATTENTION block  (every `full_attention_interval`-th layer, =4):
//       attn_norm, wq, wk, wv, wo, q_norm, k_norm + FFN tensors
//       (M-RoPE applied with rope_sections [11,11,10,0] — rope dims=64 of head_dim=256)
//   - GATED DELTANET block (all other layers, ~3 out of every 4):
//       attn_norm, wqkv (fused), wqkv_gate (the "z" projection),
//       delta-net per-head parameters (beta, gate, conv), plus FFN tensors.
//
// We keep ONE struct with all possible fields and leave unused ones nullptr.
// Actual tensor names in unsloth's GGUF are read via gguf_find_tensor() in
// the loader; see task #11.

struct TargetLayer {
    // Shared
    ggml_tensor * attn_norm      = nullptr;  // [hidden]
    ggml_tensor * attn_post_norm = nullptr;  // [hidden]  (post-block norm before FFN)
    ggml_tensor * ffn_norm       = nullptr;  // [hidden]
    ggml_tensor * w_gate         = nullptr;  // [hidden, intermediate]
    ggml_tensor * w_up           = nullptr;  // [hidden, intermediate]
    ggml_tensor * w_down         = nullptr;  // [intermediate, hidden]

    // Full-attention block (non-null for layers where (il+1) % 4 == 0)
    ggml_tensor * wq             = nullptr;  // [hidden, q_dim]
    ggml_tensor * wk             = nullptr;  // [hidden, kv_dim]
    ggml_tensor * wv             = nullptr;  // [hidden, kv_dim]
    ggml_tensor * wo             = nullptr;  // [q_dim, hidden]
    ggml_tensor * q_norm         = nullptr;  // [head_dim]
    ggml_tensor * k_norm         = nullptr;  // [head_dim]

    // Gated DeltaNet block (non-null for the other ~3/4 of layers)
    ggml_tensor * wqkv           = nullptr;  // fused Q/K/V projection
    ggml_tensor * wqkv_gate      = nullptr;  // the "z" projection
    ggml_tensor * ssm_conv1d     = nullptr;  // [kernel, dim]  depthwise causal conv
    ggml_tensor * ssm_beta       = nullptr;  // per-token beta input projection
    ggml_tensor * ssm_alpha      = nullptr;  // per-token alpha input projection
    ggml_tensor * ssm_a          = nullptr;  // [dt_rank] per-head -A parameter
    ggml_tensor * ssm_dt_bias    = nullptr;  // [dt_rank] per-head alpha bias
    ggml_tensor * ssm_norm       = nullptr;  // [head_v_dim]
    ggml_tensor * ssm_out        = nullptr;  // output projection after delta-net
};

// CPU-side embedder: keeps a mmap of the GGUF alive and knows how to
// dequantize individual rows of the quantized tok_embd tensor on demand.
// This matches llama.cpp's behavior of running embedding get_rows on CPU
// (because CUDA's get_rows doesn't support k-quants), so we never need to
// upload the 682 MiB token embedding to VRAM.
struct CpuEmbedder {
    void *           mmap_addr = nullptr;
    size_t           mmap_len  = 0;
#if defined(_WIN32)
    HANDLE           mmap_hfile = INVALID_HANDLE_VALUE;
    HANDLE           mmap_hmap  = nullptr;
#else
    int              mmap_fd   = -1;
#endif
    const uint8_t *  tok_embd_bytes = nullptr;  // into the mmap region
    ggml_type        tok_embd_type  = GGML_TYPE_COUNT;
    int64_t          n_embd = 0;
    int64_t          n_vocab = 0;
    size_t           row_bytes = 0;             // bytes per row in the quant format

    ~CpuEmbedder();
    // Dequantize N rows specified by `ids` into `out_f32` (shape [n_embd, n]).
    // Values are written contiguously row-major (n_embd fast axis).
    bool embed(const int32_t * ids, int n, float * out_f32) const;
};

struct TargetWeights {
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf     = nullptr;

    // CPU-side embedding table (zero GPU cost).
    CpuEmbedder           embedder;

    ggml_tensor * tok_embd = nullptr;        // [hidden, vocab] (metadata only; data NOT on GPU)
    std::vector<TargetLayer> layers;         // size = 64
    ggml_tensor * out_norm = nullptr;        // [hidden]
    ggml_tensor * output   = nullptr;        // [hidden, vocab]  (lm_head)

    // Metadata from GGUF (validated at load time)
    int full_attention_interval = 4;
    int rope_sections[4]        = {11, 11, 10, 0};
    int n_embd_head_k           = 256;  // key_length
    int n_embd_head_v           = 256;  // value_length
    int n_head                  = 24;
    int n_head_kv               = 4;
    int n_layer                 = 64;
    int n_embd                  = 5120;
    int n_ff                    = 17408;
    int ssm_d_conv              = 4;
    int ssm_d_inner             = 6144;
    int ssm_d_state             = 128;
    int ssm_dt_rank             = 48;
    int ssm_n_group             = 16;

    // EOS token ids loaded from the GGUF tokenizer metadata
    // (`tokenizer.ggml.eos_token_id` and `tokenizer.ggml.eot_token_id`).
    // -1 = key absent in this GGUF; the runtime EOS check guards both
    // comparands with `>= 0` so the sentinel never matches a real token.
    int32_t eos_id      = -1;
    int32_t eos_chat_id = -1;

    // Target layer IDs captured for the DFlash draft model.
    // Computed from n_layer at load time: step = (n_layer - 2) / (N - 1),
    // ids[k] = 1 + k * step.  E.g. 27B→{1,16,31,46,61}, 9B→{1,8,15,22,29}.
    int capture_layer_ids[DFLASH27B_DRAFT_N_TARGET_LAYERS] = {1, 16, 31, 46, 61};
};

struct TargetLoadPlan {
    int  layer_begin = 0;     // inclusive
    int  layer_end   = -1;    // exclusive; <0 means all layers
    bool load_output = true;  // output_norm + lm_head
};

// Load a Q4_K_M target model from a GGUF file on disk.
// Returns false and sets last_error on failure.
bool load_target_gguf(const std::string & path,
                      ggml_backend_t backend,
                      TargetWeights & out);

bool load_target_gguf_partial(const std::string & path,
                              ggml_backend_t backend,
                              const TargetLoadPlan & plan,
                              TargetWeights & out);

void free_target_weights(TargetWeights & w);

// ─── Draft weights (z-lab DFlash, bf16) ───────────────────────────

struct DraftLayer {
    ggml_tensor * attn_norm;
    ggml_tensor * ffn_norm;
    ggml_tensor * wq;
    ggml_tensor * wk;
    ggml_tensor * wv;
    ggml_tensor * wo;
    ggml_tensor * q_norm;
    ggml_tensor * k_norm;
    ggml_tensor * w_gate;
    ggml_tensor * w_up;
    ggml_tensor * w_down;
};

struct DraftWeights {
    ggml_context *    ctx = nullptr;
    ggml_backend_t    backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    ggml_tensor *          fc          = nullptr;   // [5*hidden, hidden]
    ggml_tensor *          hidden_norm = nullptr;   // [hidden]
    std::vector<DraftLayer> layers;                 // size = n_layer
    ggml_tensor *          out_norm    = nullptr;   // [hidden]

    // Architecture metadata (populated by loader).
    int n_layer   = DFLASH27B_DRAFT_LAYERS;           // 5
    int n_head    = DFLASH27B_TARGET_N_HEADS;          // 32
    int n_head_kv = DFLASH27B_TARGET_N_KV_HEADS;       // 8
    int head_dim  = DFLASH27B_TARGET_HEAD_DIM;         // 128
    int n_embd    = DFLASH27B_TARGET_HIDDEN;           // 5120
    int n_ff      = DFLASH27B_TARGET_INTERMEDIATE;     // 17408
};

bool load_draft_safetensors(const std::string & path,
                            ggml_backend_t backend,
                            DraftWeights & out);

// Load a Q8_0 (or F16) draft model from a GGUF file on disk.
// Alternative to load_draft_safetensors for quantized drafts.
bool load_draft_gguf(const std::string & path,
                     ggml_backend_t backend,
                     DraftWeights & out);

void free_draft_weights(DraftWeights & w);

// ─── Target cache (persistent state between forward calls) ────────

// Pre-allocated, backend-resident state that persists across decode steps.
// Created once via create_target_cache() and threaded through every
// build_qwen35_graph() call.
struct TargetCache {
    ggml_context *        base_ctx     = nullptr;
    ggml_backend_buffer_t base_buf     = nullptr;
    ggml_context *        rollback_ctx = nullptr;
    ggml_backend_buffer_t rollback_buf = nullptr;
    ggml_backend_t        backend  = nullptr;

    int max_ctx  = 0;         // max tokens in the KV cache
    int cur_pos  = 0;         // number of tokens already committed
    int last_tok = -1;        // post-prefill / post-decode argmax; decode seed.
                              // Used by prefix-cache RESTORE to bridge an
                              // empty-suffix prefill into the decode loop.

    ggml_type kv_k_type = GGML_TYPE_Q8_0;
    ggml_type kv_v_type = GGML_TYPE_Q8_0;

    // Full-attention KV cache: one K and one V per full-attention layer.
    // Layout: [head_dim, max_ctx, n_head_kv] f16, contiguous per layer.
    std::vector<ggml_tensor *> attn_k;   // size = n_full_attn_layers (16)
    std::vector<ggml_tensor *> attn_v;

    // Gated DeltaNet recurrent state: one per delta-net layer.
    // ssm_state: [S_v, S_v, H_v] f32    (head_v_dim^2 × num_v_heads)
    // conv_state: [(kernel-1), conv_channels] f32
    // where conv_channels = d_inner + 2 * n_group * d_state
    std::vector<ggml_tensor *> ssm_state;    // size = n_delta_layers (48)
    std::vector<ggml_tensor *> conv_state;

    // Snapshot buffers for speculative decoding rollback. Sized identically
    // to ssm_state/conv_state above. Populated by snapshot_ssm_state() and
    // restored by restore_ssm_state().
    std::vector<ggml_tensor *> ssm_state_snap;
    std::vector<ggml_tensor *> conv_state_snap;

    // Per-step SSM + conv inputs captured during a verify forward when
    // QwenGraphInputs::capture_delta_intermediate is true. Populated by
    // in-graph ggml_cpy ops in build_delta_net_block so their data lives in
    // persistent cache memory (not tracked by the per-call gallocr), matching
    // SGLang's mamba_caches.intermediate_ssm / intermediate_conv_window pattern.
    //
    //   ssm_intermediate: [S_v, S_v, H_v, max_q_len] f32, one per delta layer.
    //     Element t on axis 3 holds the DeltaNet recurrent state after
    //     processing verify token t. Spec decode commits t = commit_n - 1.
    //   conv_input_cache: [(kernel-1) + max_q_len, conv_channels] f32, one per
    //     delta layer. Holds the full concat(old_conv_state, qkv_new_tokens)
    //     that was fed to ggml_ssm_conv. Spec decode slices
    //     [commit_n..commit_n+kernel-2] along dim 0 for conv state rollback.
    std::vector<ggml_tensor *> ssm_intermediate;    // size = n_delta (48)
    std::vector<ggml_tensor *> conv_input_cache;    // size = n_delta (48)

    // Rolling target layer features captured during target forward passes.
    // Shape [5 * hidden, target_feat_cap] bf16. target_feat_cap is typically
    // << max_ctx (e.g. 4096) so the buffer stays small at 128K context. The
    // graph writes to slot `(kv_start + i) % target_feat_cap` so positions
    // beyond the cap wrap and overwrite older entries. Readers (draft) only
    // need the last DRAFT_CTX_MAX positions, so wrap is invisible in
    // practice. Fed into the draft graph's fc projection after a bf16→f32
    // cast (ggml_get_to_fp32_cuda).
    ggml_tensor * target_feat = nullptr;
    int target_feat_cap = 0;
};

// Snapshot the current SSM+conv state into TargetCache::*_snap tensors.
void snapshot_ssm_state(TargetCache & c);
// Restore the SSM+conv state from the snapshot.
void restore_ssm_state(TargetCache & c);

// ─── Cross-request prefix snapshot (Phase A) ──────────────────────
//
// PrefixSnapshot captures a slim copy of TargetCache state at a
// committed-token boundary so a future request sharing the same prefix
// can restore and skip re-prefilling those tokens.
//
// Slim scope:
//   - attn_k[i], attn_v[i] for every full-attn layer (the actual KV)
//   - ssm_state[i], conv_state[i] for every delta-net layer (recurrent state)
//   - target_feat ring + cur_pos
//
// NOT captured:
//   - ssm_intermediate, conv_input_cache (within-decode rollback buffers,
//     regenerated by the first decode step after restore)
//   - rollback_ctx tensors (snapshots themselves are stateless wrt rollback)
//
// All copies are device-to-device via ggml_backend_tensor_copy. The snapshot
// owns its own ggml_context + backend buffer (allocated lazily on first
// snapshot_target_cache call to a given PrefixSnapshot).
struct PrefixSnapshot {
    int       cur_pos         = 0;
    int       last_tok        = -1;                // post-prefill argmax (decode seed)
    ggml_type kv_k_type       = GGML_TYPE_COUNT;   // for hash-key validation
    int       max_ctx         = 0;                 // for sanity check at restore
    int       target_feat_cap = 0;

    // GPU-resident copies (lazy-allocated; null until first snapshot)
    std::vector<ggml_tensor *> attn_k_snap;     // size n_full_attn (16)
    std::vector<ggml_tensor *> attn_v_snap;
    std::vector<ggml_tensor *> ssm_state_snap;  // size n_delta (48)
    std::vector<ggml_tensor *> conv_state_snap;
    ggml_tensor *               target_feat_snap = nullptr;

    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    // Phase B: thin-mode snapshots cover only a KV-position range.
    bool is_thin  = false;
    int  kv_start = 0;     // inclusive (only meaningful when is_thin)
    int  kv_end   = 0;     // exclusive (only meaningful when is_thin)
    // When is_thin == true:
    //   - attn_k_snap[i] / attn_v_snap[i] are sized
    //     [HEAD_DIM, kv_end-kv_start, N_HEAD_KV] (smaller than cache).
    //   - ssm_state_snap, conv_state_snap, target_feat_snap are NOT
    //     allocated (THIN snapshots are KV-only).
};

// Snapshot the slim state of `cache` into `snap`. Allocates device buffers
// on the first call (lazy; matches the cache's own allocation pattern).
// Subsequent calls REUSE the same buffers (just refresh contents). Returns
// false on allocation failure (and sets last_error).
bool snapshot_target_cache(const TargetWeights & w,
                           const TargetCache & cache,
                           ggml_backend_t backend,
                           PrefixSnapshot & snap);

// Restore `cache` from `snap`. cache must already exist (created via
// create_target_cache) and have matching shapes. Sets cache.cur_pos =
// snap.cur_pos. Does NOT touch ssm_intermediate / conv_input_cache —
// those will be repopulated by the first decode step's verify forward.
bool restore_target_cache(const PrefixSnapshot & snap, TargetCache & cache);

// Free the snapshot's GPU buffers.
void free_prefix_snapshot(PrefixSnapshot & snap);

// Thin snapshot: capture only KV slice [kv_start, kv_end).
// SSM/conv/target_feat are not preserved (caller chains thin entries
// onto a thick base via restore_target_cache_chain).
bool snapshot_target_cache_thin(const TargetWeights & w,
                                 const TargetCache & cache,
                                 ggml_backend_t backend,
                                 int kv_start,
                                 int kv_end,
                                 PrefixSnapshot & snap);

// Restore from a thick base then layer in zero or more thin entries.
// thick may be nullptr if you only want the thin layers; in that case
// cache must already hold the right base (only safe for testing).
// Each thin's [kv_start, kv_end) range is copied into cache.attn_k[i] /
// attn_v[i] at the appropriate offset. Out-of-order thins are allowed
// (later thins overwrite earlier ones in overlapping ranges); chain
// caller must walk in time order to be deterministic.
bool restore_target_cache_chain(const PrefixSnapshot * thick,
                                 const PrefixSnapshot * const * thins,
                                 int n_thins,
                                 TargetCache & cache);

// max_verify_tokens controls the per-layer ssm_intermediate and conv_input_cache
// sizes. Default is DFLASH27B_DRAFT_BLOCK_SIZE (16) for chain verify. DDTree
// mode requires max(chain, 1 + tree_budget) to hold the flat tree + root.
// Pass 0 to use the default.
// When prefill_only is true, rollback tensors (snapshots, intermediates) are
// skipped — saving ~1.4 GB on 48 DeltaNet layers. Use migrate_prefill_cache()
// to promote the cache to a full decode cache after prefill.
bool create_target_cache(const TargetWeights & w,
                         int max_ctx,
                         int max_verify_tokens,
                         ggml_backend_t backend,
                         TargetCache & out,
                         bool prefill_only = false);

bool create_target_cache_partial(const TargetWeights & w,
                                 int max_ctx,
                                 int max_verify_tokens,
                                 ggml_backend_t backend,
                                 TargetCache & out,
                                 bool prefill_only,
                                 int layer_begin,
                                 int layer_end,
                                 bool allocate_target_feat);

void free_target_cache(TargetCache & c);

// Zero all state tensors (KV, SSM, conv, target_feat, rollback) in place
// without freeing/reallocating GPU buffers. Used by daemon mode between
// requests to avoid the ~5 s overhead of full cache destruction + recreation.
void reset_target_cache(TargetCache & c);

// Reallocate a prefill-only cache with full rollback tensors, copying all live
// state (KV, SSM, conv, target_feat) device-to-device. Frees the old cache.
bool migrate_prefill_cache(const TargetWeights & w,
                           int max_ctx,
                           int max_verify_tokens,
                           ggml_backend_t backend,
                           TargetCache & cache);

// ─── Target forward graph ─────────────────────────────────────────

// Per-delta-net-layer pointers exposed by the graph for spec-decode rollback.
// Populated when QwenGraphInputs::capture_delta_intermediate is true.
//
// Both tensors are persistent cache buffers (cache.ssm_intermediate[il] and
// cache.conv_input_cache[il]). Their ->data pointers are always valid — the
// graph just runs ggml_cpy ops to fill them during verify. Matches SGLang's
// mamba_caches.intermediate_ssm / intermediate_conv_window pattern:
// persistent memory, not managed by the per-call gallocr.
//
//   ssm_intermediate_states: [S_v, S_v, H_v, q_len] f32
//       Element t on axis 3 holds the DeltaNet state after processing verify
//       token t. Rollback reads offset (commit_n-1) * S_v*S_v*H*elt.
//   conv_input: [(kernel-1) + q_len, conv_channels, 1] f32
//       Full concat(old_conv_state, qkv_new_tokens) fed to ggml_ssm_conv.
//       Rollback reads slice [commit_n..commit_n+kernel-2] along dim 0.
struct DeltaNetCapture {
    ggml_tensor * ssm_intermediate_states = nullptr;
    ggml_tensor * conv_input              = nullptr;
};

struct QwenGraphInputs {
    ggml_tensor * inp_embed;      // [hidden, n_tokens, 1] f32 — pre-embedded by the caller
    ggml_tensor * positions;      // [4 * n_tokens] i32 (M-RoPE needs 4 per token)
    ggml_tensor * attn_mask;      // optional [kv_len, n_tokens_padded] f32 (causal); nullptr for n_tokens==1
    int           n_tokens;       // number of new tokens in this forward
    int           kv_start;       // position where the new tokens begin
    bool          capture_layers; // if true, write captured layer features into cache.target_feat
    bool          capture_delta_intermediate = false; // if true, populate out_delta_captures
    int           fa_window = 0;  // sliding window for FA layers: 0 = full attention
    ggml_tensor * parent_ids = nullptr; // [n_tokens] i32; tree mode when non-null
};

struct QwenGraphOutputs {
    ggml_tensor * logits;      // [vocab, n_tokens] f32
    // One entry per delta-net layer (48 for qwen35-27b). Only populated when
    // QwenGraphInputs::capture_delta_intermediate is true. Tensors are graph
    // views marked as ggml_set_output() so their data persists after
    // graph_compute; the spec-decode loop reads them host-side for rollback.
    std::vector<DeltaNetCapture> delta_captures;
};

QwenGraphOutputs build_qwen35_graph(
    ggml_context *         ctx,
    ggml_cgraph *          gf,
    const TargetWeights &  w,
    TargetCache &          cache,
    const QwenGraphInputs & in);

// Build a single-layer forward graph. Mirrors build_qwen35_graph but processes
// only one layer, taking `inp` as the input activation and returning the output.
// Used by layer-segmented prefill to iterate layers as the outer loop.
ggml_tensor * build_qwen35_layer(
    ggml_context *        ctx,
    ggml_cgraph *         gf,
    const TargetWeights & w,
    TargetCache &         cache,
    int                   layer_idx,
    ggml_tensor *         inp,         // [hidden, n_tokens]
    ggml_tensor *         positions,   // [4 * n_tokens] i32
    ggml_tensor *         attn_mask,   // optional
    int                   kv_start,
    int                   n_tokens,
    bool                  capture,
    int                   fa_window = 0);

} // namespace dflash27b
