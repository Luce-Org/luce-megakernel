// Expert GPU cache implementation — type-major 3D ggml tensors + LRU.

#include "expert_cache.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <climits>

namespace dflash27b {

bool ExpertCache::init(ggml_backend_t backend, const MoeExpertSource & src, int n_slots) {
    destroy();
    backend_ = backend;
    n_slots_ = n_slots;

    // Create ggml context for the 3D weight tensors.
    ggml_init_params ip{};
    ip.mem_size   = 8 * ggml_tensor_overhead() + 16 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    tensor_ctx_ = ggml_init(ip);
    if (!tensor_ctx_) return false;

    // 3D tensors: [cols, rows, n_slots].
    // gate/up: matmul projects hidden → expert_ffn, so shape is [hidden, expert_ffn, n_slots].
    // down: matmul projects expert_ffn → hidden, so shape is [expert_ffn, hidden, n_slots].
    gate_3d_ = ggml_new_tensor_3d(tensor_ctx_, src.gate_type,
        src.hidden_dim, src.expert_ffn_dim, n_slots);
    up_3d_ = ggml_new_tensor_3d(tensor_ctx_, src.up_type,
        src.hidden_dim, src.expert_ffn_dim, n_slots);
    down_3d_ = ggml_new_tensor_3d(tensor_ctx_, src.down_type,
        src.expert_ffn_dim, src.hidden_dim, n_slots);

    ggml_set_name(gate_3d_, "ecache_gate_3d");
    ggml_set_name(up_3d_,   "ecache_up_3d");
    ggml_set_name(down_3d_, "ecache_down_3d");

    tensor_buf_ = ggml_backend_alloc_ctx_tensors(tensor_ctx_, backend);
    if (!tensor_buf_) {
        std::fprintf(stderr, "[ExpertCache] GPU alloc failed for %d slots\n", n_slots);
        destroy();
        return false;
    }

    // Zero all data (slot 0 = null sentinel stays zero permanently).
    ggml_backend_buffer_clear(tensor_buf_, 0);

    slots_.resize(n_slots);

    size_t total_bytes = ggml_nbytes(gate_3d_) + ggml_nbytes(up_3d_) + ggml_nbytes(down_3d_);
    std::printf("[ExpertCache] %d slots, %.1f MB total "
                "(gate %s [%lld,%lld], up %s [%lld,%lld], down %s [%lld,%lld])\n",
        n_slots, total_bytes / (1024.0 * 1024.0),
        ggml_type_name(src.gate_type), (long long)gate_3d_->ne[0], (long long)gate_3d_->ne[1],
        ggml_type_name(src.up_type),   (long long)up_3d_->ne[0],   (long long)up_3d_->ne[1],
        ggml_type_name(src.down_type), (long long)down_3d_->ne[0], (long long)down_3d_->ne[1]);

    return true;
}

int ExpertCache::allocate_slot(int layer, int expert_id) {
    int64_t key = make_key(layer, expert_id);

    // Check if already cached.
    auto it = cache_map_.find(key);
    if (it != cache_map_.end()) {
        slots_[it->second].lru_tick = ++tick_;
        return it->second;  // positive = hit
    }

    // Find LRU victim (skip NULL_SLOT = 0).
    int victim = 1;
    uint32_t min_tick = UINT32_MAX;
    for (int s = 1; s < n_slots_; s++) {
        if (slots_[s].layer_id < 0) {
            victim = s;  // empty slot
            break;
        }
        if (slots_[s].lru_tick < min_tick) {
            min_tick = slots_[s].lru_tick;
            victim = s;
        }
    }

    // Evict old occupant from cache_map.
    if (slots_[victim].layer_id >= 0) {
        int64_t old_key = make_key(slots_[victim].layer_id, slots_[victim].expert_id);
        cache_map_.erase(old_key);
    }

    // Assign new expert to this slot.
    slots_[victim].layer_id  = layer;
    slots_[victim].expert_id = expert_id;
    slots_[victim].lru_tick  = ++tick_;
    cache_map_[key] = victim;

    return -(victim + 1);  // negative = needs transfer
}

void ExpertCache::load_to_slot(int slot, int layer, int expert_id,
                               const MoeExpertSource & src) {
    const auto & li = src.layers[layer];

    // Gate: copy one expert's gate weights into slot position within gate_3d_.
    const uint8_t * gate_data = src.mmap_base + li.gate_offset
                              + (size_t)expert_id * src.gate_expert_bytes;
    ggml_backend_tensor_set(gate_3d_, gate_data,
        (size_t)slot * gate_3d_->nb[2], src.gate_expert_bytes);

    // Up
    const uint8_t * up_data = src.mmap_base + li.up_offset
                            + (size_t)expert_id * src.up_expert_bytes;
    ggml_backend_tensor_set(up_3d_, up_data,
        (size_t)slot * up_3d_->nb[2], src.up_expert_bytes);

    // Down
    const uint8_t * down_data = src.mmap_base + li.down_offset
                              + (size_t)expert_id * src.down_expert_bytes;
    ggml_backend_tensor_set(down_3d_, down_data,
        (size_t)slot * down_3d_->nb[2], src.down_expert_bytes);
}

int ExpertCache::ensure_cached(int layer, int expert_id, const MoeExpertSource & source) {
    int result = allocate_slot(layer, expert_id);
    if (result >= 0) {
        hit_count_++;
        return result;  // already cached
    }
    // Cache miss — load from mmap.
    miss_count_++;
    int slot = -(result + 1);
    load_to_slot(slot, layer, expert_id, source);
    return slot;
}

int ExpertCache::get_slot(int layer, int expert_id) const {
    auto it = cache_map_.find(make_key(layer, expert_id));
    return (it != cache_map_.end()) ? it->second : NULL_SLOT;
}

void ExpertCache::warmup(const MoeExpertSource & source) {
    // Distribute slots evenly across layers.
    int slots_per_layer = (n_slots_ - 1) / source.n_layers;
    if (slots_per_layer > source.n_experts) slots_per_layer = source.n_experts;

    std::printf("[ExpertCache] warmup: loading %d experts/layer × %d layers\n",
        slots_per_layer, source.n_layers);

    for (int l = 0; l < source.n_layers; l++) {
        for (int e = 0; e < slots_per_layer; e++) {
            ensure_cached(l, e, source);
        }
    }
    // Reset stats after warmup so runtime stats are meaningful.
    hit_count_  = 0;
    miss_count_ = 0;
}

void ExpertCache::destroy() {
    if (tensor_buf_) { ggml_backend_buffer_free(tensor_buf_); tensor_buf_ = nullptr; }
    if (tensor_ctx_) { ggml_free(tensor_ctx_); tensor_ctx_ = nullptr; }
    gate_3d_ = up_3d_ = down_3d_ = nullptr;
    slots_.clear();
    cache_map_.clear();
    backend_ = nullptr;
    n_slots_ = 0;
}

// ─── MoeState ────────────────────────────────────────────────────────

bool create_moe_state(ggml_backend_t backend, int hidden, int n_expert_used,
                      int max_tokens, MoeState & out) {
    free_moe_state(out);

    ggml_init_params ip{};
    ip.mem_size   = 16 * ggml_tensor_overhead() + 16 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;

    out.post         = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, hidden, max_tokens);
    out.ffn_residual = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, hidden, max_tokens);
    out.weights      = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, 1, n_expert_used, max_tokens);
    out.slot_ids     = ggml_new_tensor_2d(out.ctx, GGML_TYPE_I32, n_expert_used, max_tokens);

    ggml_set_name(out.post,         "moe_post");
    ggml_set_name(out.ffn_residual, "moe_ffn_residual");
    ggml_set_name(out.weights,      "moe_weights");
    ggml_set_name(out.slot_ids,     "moe_slot_ids");

    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        free_moe_state(out);
        return false;
    }

    out.max_tokens    = max_tokens;
    out.n_expert_used = n_expert_used;

    size_t total = ggml_backend_buffer_get_size(out.buf);
    std::printf("[MoeState] allocated %.1f KB (max_tokens=%d, n_expert_used=%d)\n",
        total / 1024.0, max_tokens, n_expert_used);
    return true;
}

void free_moe_state(MoeState & s) {
    if (s.buf) { ggml_backend_buffer_free(s.buf); s.buf = nullptr; }
    if (s.ctx) { ggml_free(s.ctx); s.ctx = nullptr; }
    s.post = s.ffn_residual = s.weights = s.slot_ids = nullptr;
    s.max_tokens = 0;
}

} // namespace dflash27b
