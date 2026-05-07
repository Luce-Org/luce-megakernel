// Expert GPU cache for MoE models — type-major 3D tensor layout.
//
// Manages expert weights in three 3D ggml tensors (gate/up/down) on GPU,
// compatible with ggml_mul_mat_id for efficient batched expert dispatch.
// Uses LRU eviction. Expert data is loaded from the mmap'd GGUF file.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <unordered_map>

namespace dflash27b {

// Source of expert weight data in the mmap'd GGUF file.
struct MoeExpertSource {
    const uint8_t * mmap_base = nullptr;

    struct LayerOffsets {
        size_t gate_offset = 0;  // byte offset from mmap_base to blk.L.ffn_gate_exps data
        size_t up_offset   = 0;
        size_t down_offset = 0;
    };
    std::vector<LayerOffsets> layers;  // [n_layer]

    size_t gate_expert_bytes = 0;  // bytes for one expert's gate weights
    size_t up_expert_bytes   = 0;
    size_t down_expert_bytes = 0;  // bytes for primary down type (used by ExpertCache)

    ggml_type gate_type = GGML_TYPE_COUNT;
    ggml_type up_type   = GGML_TYPE_COUNT;
    ggml_type down_type = GGML_TYPE_COUNT;  // primary down type for this cache

    // Per-layer down type info (populated by loader, may vary across layers).
    std::vector<ggml_type> layer_down_types;  // [n_layer] actual per-layer types
    std::vector<size_t>    layer_down_bytes;  // [n_layer] actual per-layer expert bytes

    int hidden_dim     = 0;
    int expert_ffn_dim = 0;
    int n_experts      = 0;  // 256
    int n_layers       = 0;  // 40
};

class ExpertCache {
public:
    static constexpr int NULL_SLOT = 0;  // slot 0 is zeroed (miss sentinel)

    // Allocate n_slots expert cache slots as 3D ggml tensors on GPU.
    // If the source has mixed down types, creates a secondary down tensor
    // for the alternate type with n_alt_slots slots.
    bool init(ggml_backend_t backend, const MoeExpertSource & source,
              int n_slots, int n_alt_slots = 0);

    // Ensure (layer, expert_id) is in cache. Loads from source if not.
    // Returns the slot index (always > 0 on success).
    int ensure_cached(int layer, int expert_id, const MoeExpertSource & source);

    // Batch-ensure multiple experts for a layer using async H2D transfers.
    // All misses are loaded with cudaMemcpyAsync, synced once at the end.
    void batch_ensure_cached(int layer, const int * expert_ids, int n_experts,
                             const MoeExpertSource & source);

    // Get slot index for a cached expert. Returns NULL_SLOT if not cached.
    int get_slot(int layer, int expert_id) const;

    // Get the down_slot for a cached expert (separate namespace from gate/up slot).
    int get_down_slot(int layer, int expert_id) const;

    // Pre-populate cache with first N experts per layer.
    void warmup(const MoeExpertSource & source);

    // 3D weight tensors for ggml_mul_mat_id.
    ggml_tensor * gate_3d() const { return gate_3d_; }
    ggml_tensor * up_3d()   const { return up_3d_; }

    // Returns the correct down tensor for a given layer (primary or alt type).
    ggml_tensor * down_3d_for_layer(int layer) const {
        return is_alt_layer(layer) ? down_alt_3d_ : down_3d_;
    }
    // Legacy: returns primary down tensor.
    ggml_tensor * down_3d() const { return down_3d_; }

    bool has_alt_down() const { return down_alt_3d_ != nullptr; }
    bool is_alt_layer(int layer) const {
        return !alt_layers_.empty() &&
               std::find(alt_layers_.begin(), alt_layers_.end(), layer) != alt_layers_.end();
    }
    // Priority layers get eviction protection (first 6, last 3).
    bool is_priority_layer(int layer) const {
        return layer < 6 || layer >= n_layers_ - 3;
    }

    int n_slots() const { return n_slots_; }
    int64_t hits()   const { return hit_count_; }
    int64_t misses() const { return miss_count_; }

    void destroy();
    ~ExpertCache() { destroy(); }

private:
    int allocate_slot(int layer, int expert_id);
    int allocate_down_slot(int layer, int expert_id);
    void load_to_slot(int slot, int down_slot, int layer, int expert_id,
                      const MoeExpertSource & src);

    static int64_t make_key(int layer, int expert_id) {
        return (int64_t)layer * 65536 + expert_id;
    }

    struct Slot {
        int      layer_id  = -1;
        int      expert_id = -1;
        uint32_t lru_tick  = 0;
    };

    struct CacheEntry {
        int gate_up_slot = NULL_SLOT;
        int down_slot    = NULL_SLOT;
    };

    ggml_context *        tensor_ctx_   = nullptr;
    ggml_backend_buffer_t tensor_buf_   = nullptr;
    ggml_backend_buffer_t alt_buf_      = nullptr;  // separate buffer for alt down tensor
    ggml_tensor *         gate_3d_      = nullptr;
    ggml_tensor *         up_3d_        = nullptr;
    ggml_tensor *         down_3d_      = nullptr;  // primary down type
    ggml_tensor *         down_alt_3d_  = nullptr;  // alt down type (Q6_K if primary is Q5_K)
    ggml_backend_t        backend_      = nullptr;

    std::vector<Slot> slots_;       // gate/up slot info
    std::vector<Slot> down_slots_;  // down primary slot info
    std::vector<Slot> alt_slots_;   // down alt slot info
    std::unordered_map<int64_t, CacheEntry> cache_map_;
    std::vector<int> alt_layers_;   // layers that use the alt down type
    int n_slots_     = 0;  // gate/up slots (also primary down slots)
    int n_alt_slots_ = 0;  // alt down slots
    int n_layers_    = 0;  // total model layers (for priority calculation)

    uint32_t tick_       = 0;
    int64_t  hit_count_  = 0;
    int64_t  miss_count_ = 0;
};

// Persistent GPU buffers for MoE two-graph-per-layer execution.
// Transfers intermediate results between Graph A (attention+router) and
// Graph B (MoE FFN) without CPU round-trip for large tensors.
struct MoeState {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    ggml_tensor * post;             // [hidden, max_tokens] F32 — post-norm activation
    ggml_tensor * ffn_residual;     // [hidden, max_tokens] F32 — pre-FFN residual
    ggml_tensor * weights;          // [1, n_expert_used, max_tokens] F32 — gate weights
    ggml_tensor * slot_ids;         // [n_expert_used, max_tokens] I32 — gate/up cache slot indices
    ggml_tensor * down_slot_ids;    // [n_expert_used, max_tokens] I32 — down cache slot indices

    int max_tokens    = 0;
    int n_expert_used = 0;
};

bool create_moe_state(ggml_backend_t backend, int hidden, int n_expert_used,
                      int max_tokens, MoeState & out);
void free_moe_state(MoeState & s);

} // namespace dflash27b
