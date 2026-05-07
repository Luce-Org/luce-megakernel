// Expert GPU cache implementation — type-major 3D ggml tensors + LRU.
// Supports mixed down types (e.g., Q5_K + Q6_K across layers).

#include "expert_cache.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <climits>
#include <cuda_runtime.h>

namespace dflash27b {

bool ExpertCache::init(ggml_backend_t backend, const MoeExpertSource & src,
                       int n_slots, int n_alt_slots) {
    destroy();
    backend_ = backend;
    n_slots_ = n_slots;
    n_alt_slots_ = n_alt_slots;

    // Detect layers with alternate down type.
    ggml_type alt_type = GGML_TYPE_COUNT;
    if (!src.layer_down_types.empty()) {
        for (int l = 0; l < src.n_layers; l++) {
            if (src.layer_down_types[l] != src.down_type) {
                alt_type = src.layer_down_types[l];
                alt_layers_.push_back(l);
            }
        }
    }

    // Create ggml context for the 3D weight tensors.
    int n_tensors = alt_layers_.empty() ? 3 : 4;
    ggml_init_params ip{};
    ip.mem_size   = (n_tensors + 4) * ggml_tensor_overhead() + 16 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    tensor_ctx_ = ggml_init(ip);
    if (!tensor_ctx_) return false;

    // 3D tensors: [cols, rows, n_slots].
    gate_3d_ = ggml_new_tensor_3d(tensor_ctx_, src.gate_type,
        src.hidden_dim, src.expert_ffn_dim, n_slots);
    up_3d_ = ggml_new_tensor_3d(tensor_ctx_, src.up_type,
        src.hidden_dim, src.expert_ffn_dim, n_slots);
    down_3d_ = ggml_new_tensor_3d(tensor_ctx_, src.down_type,
        src.expert_ffn_dim, src.hidden_dim, n_slots);

    ggml_set_name(gate_3d_, "ecache_gate_3d");
    ggml_set_name(up_3d_,   "ecache_up_3d");
    ggml_set_name(down_3d_, "ecache_down_3d");

    // Allocate alternate down tensor if needed.
    if (!alt_layers_.empty() && n_alt_slots > 0 && alt_type != GGML_TYPE_COUNT) {
        down_alt_3d_ = ggml_new_tensor_3d(tensor_ctx_, alt_type,
            src.expert_ffn_dim, src.hidden_dim, n_alt_slots);
        ggml_set_name(down_alt_3d_, "ecache_down_alt_3d");
    }

    tensor_buf_ = ggml_backend_alloc_ctx_tensors(tensor_ctx_, backend);
    if (!tensor_buf_) {
        std::fprintf(stderr, "[ExpertCache] GPU alloc failed for %d slots\n", n_slots);
        destroy();
        return false;
    }

    // Zero all data (slot 0 = null sentinel stays zero permanently).
    ggml_backend_buffer_clear(tensor_buf_, 0);

    slots_.resize(n_slots);
    down_slots_.resize(n_slots);
    if (n_alt_slots > 0) alt_slots_.resize(n_alt_slots);

    size_t total_bytes = ggml_nbytes(gate_3d_) + ggml_nbytes(up_3d_) + ggml_nbytes(down_3d_);
    if (down_alt_3d_) total_bytes += ggml_nbytes(down_alt_3d_);

    std::printf("[ExpertCache] %d slots + %d alt_slots, %.1f MB total "
                "(gate %s [%lld,%lld], up %s [%lld,%lld], down %s [%lld,%lld]",
        n_slots, n_alt_slots, total_bytes / (1024.0 * 1024.0),
        ggml_type_name(src.gate_type), (long long)gate_3d_->ne[0], (long long)gate_3d_->ne[1],
        ggml_type_name(src.up_type),   (long long)up_3d_->ne[0],   (long long)up_3d_->ne[1],
        ggml_type_name(src.down_type), (long long)down_3d_->ne[0], (long long)down_3d_->ne[1]);
    if (down_alt_3d_) {
        std::printf(", down_alt %s [%lld,%lld]",
            ggml_type_name(alt_type),
            (long long)down_alt_3d_->ne[0], (long long)down_alt_3d_->ne[1]);
    }
    std::printf(")\n");
    if (!alt_layers_.empty()) {
        std::printf("[ExpertCache] alt layers (%zu): ", alt_layers_.size());
        for (int l : alt_layers_) std::printf("%d ", l);
        std::printf("\n");
    }

    return true;
}

int ExpertCache::allocate_slot(int layer, int expert_id) {
    // Gate/up slot allocation — uses slots_ pool.
    int victim = 1;
    uint32_t min_tick = UINT32_MAX;
    for (int s = 1; s < n_slots_; s++) {
        if (slots_[s].layer_id < 0) {
            victim = s;
            break;
        }
        if (slots_[s].lru_tick < min_tick) {
            min_tick = slots_[s].lru_tick;
            victim = s;
        }
    }

    // Evict old occupant from gate/up slot.
    if (slots_[victim].layer_id >= 0) {
        // Note: we don't clean cache_map_ here — it's done in ensure_cached.
    }

    slots_[victim].layer_id  = layer;
    slots_[victim].expert_id = expert_id;
    slots_[victim].lru_tick  = ++tick_;
    return victim;
}

int ExpertCache::allocate_down_slot(int layer, int expert_id) {
    // Pick the correct pool based on layer type.
    bool is_alt = is_alt_layer(layer);
    auto & pool = is_alt ? alt_slots_ : down_slots_;
    int pool_size = is_alt ? n_alt_slots_ : n_slots_;

    int victim = 1;
    uint32_t min_tick = UINT32_MAX;
    for (int s = 1; s < pool_size; s++) {
        if (pool[s].layer_id < 0) {
            victim = s;
            break;
        }
        if (pool[s].lru_tick < min_tick) {
            min_tick = pool[s].lru_tick;
            victim = s;
        }
    }

    pool[victim].layer_id  = layer;
    pool[victim].expert_id = expert_id;
    pool[victim].lru_tick  = tick_;  // same tick as gate/up
    return victim;
}

void ExpertCache::load_to_slot(int slot, int down_slot, int layer, int expert_id,
                               const MoeExpertSource & src) {
    const auto & li = src.layers[layer];

    // Gate
    const uint8_t * gate_data = src.mmap_base + li.gate_offset
                              + (size_t)expert_id * src.gate_expert_bytes;
    ggml_backend_tensor_set(gate_3d_, gate_data,
        (size_t)slot * gate_3d_->nb[2], src.gate_expert_bytes);

    // Up
    const uint8_t * up_data = src.mmap_base + li.up_offset
                            + (size_t)expert_id * src.up_expert_bytes;
    ggml_backend_tensor_set(up_3d_, up_data,
        (size_t)slot * up_3d_->nb[2], src.up_expert_bytes);

    // Down — use per-layer type/bytes and correct tensor.
    bool is_alt = is_alt_layer(layer);
    ggml_tensor * down_tensor = is_alt ? down_alt_3d_ : down_3d_;
    size_t down_bytes = src.layer_down_bytes.empty()
        ? src.down_expert_bytes
        : src.layer_down_bytes[layer];

    const uint8_t * down_data = src.mmap_base + li.down_offset
                              + (size_t)expert_id * down_bytes;
    ggml_backend_tensor_set(down_tensor, down_data,
        (size_t)down_slot * down_tensor->nb[2], down_bytes);
}

int ExpertCache::ensure_cached(int layer, int expert_id, const MoeExpertSource & source) {
    int64_t key = make_key(layer, expert_id);

    auto it = cache_map_.find(key);
    if (it != cache_map_.end()) {
        // Cache hit — update LRU ticks.
        hit_count_++;
        int gu_slot = it->second.gate_up_slot;
        int d_slot  = it->second.down_slot;
        slots_[gu_slot].lru_tick = ++tick_;
        bool is_alt = is_alt_layer(layer);
        auto & pool = is_alt ? alt_slots_ : down_slots_;
        pool[d_slot].lru_tick = tick_;
        return gu_slot;
    }

    // Cache miss — allocate and load.
    miss_count_++;
    int gu_slot = allocate_slot(layer, expert_id);
    int d_slot  = allocate_down_slot(layer, expert_id);

    // Evict old cache_map_ entries that pointed to these slots.
    // Gate/up eviction:
    for (auto mit = cache_map_.begin(); mit != cache_map_.end(); ) {
        if (mit->second.gate_up_slot == gu_slot && mit->first != key) {
            mit = cache_map_.erase(mit);
        } else {
            ++mit;
        }
    }
    // Down eviction:
    bool is_alt = is_alt_layer(layer);
    for (auto mit = cache_map_.begin(); mit != cache_map_.end(); ) {
        if (mit->second.down_slot == d_slot && mit->first != key) {
            // Only evict if same pool (alt vs primary).
            int64_t other_key = mit->first;
            int other_layer = (int)(other_key / 65536);
            if (is_alt_layer(other_layer) == is_alt) {
                mit = cache_map_.erase(mit);
            } else {
                ++mit;
            }
        } else {
            ++mit;
        }
    }

    cache_map_[key] = {gu_slot, d_slot};
    load_to_slot(gu_slot, d_slot, layer, expert_id, source);
    return gu_slot;
}

int ExpertCache::get_slot(int layer, int expert_id) const {
    auto it = cache_map_.find(make_key(layer, expert_id));
    return (it != cache_map_.end()) ? it->second.gate_up_slot : NULL_SLOT;
}

int ExpertCache::get_down_slot(int layer, int expert_id) const {
    auto it = cache_map_.find(make_key(layer, expert_id));
    return (it != cache_map_.end()) ? it->second.down_slot : NULL_SLOT;
}

void ExpertCache::batch_ensure_cached(int layer, const int * expert_ids, int n_experts,
                                      const MoeExpertSource & source) {
    // Phase 1: Identify misses and allocate slots (CPU-only, no GPU sync).
    struct PendingLoad {
        int expert_id;
        int gu_slot;
        int d_slot;
    };
    std::vector<PendingLoad> pending;

    for (int i = 0; i < n_experts; i++) {
        int eid = expert_ids[i];
        int64_t key = make_key(layer, eid);

        auto it = cache_map_.find(key);
        if (it != cache_map_.end()) {
            // Hit — just update LRU.
            hit_count_++;
            slots_[it->second.gate_up_slot].lru_tick = ++tick_;
            bool alt = is_alt_layer(layer);
            auto & pool = alt ? alt_slots_ : down_slots_;
            pool[it->second.down_slot].lru_tick = tick_;
            continue;
        }

        // Miss — allocate slots.
        miss_count_++;
        int gu_slot = allocate_slot(layer, eid);
        int d_slot  = allocate_down_slot(layer, eid);

        // Evict stale cache_map_ entries for these slots.
        for (auto mit = cache_map_.begin(); mit != cache_map_.end(); ) {
            if (mit->second.gate_up_slot == gu_slot && mit->first != key) {
                mit = cache_map_.erase(mit);
            } else {
                ++mit;
            }
        }
        bool alt = is_alt_layer(layer);
        for (auto mit = cache_map_.begin(); mit != cache_map_.end(); ) {
            if (mit->second.down_slot == d_slot && mit->first != key) {
                int other_layer = (int)(mit->first / 65536);
                if (is_alt_layer(other_layer) == alt) {
                    mit = cache_map_.erase(mit);
                } else {
                    ++mit;
                }
            } else {
                ++mit;
            }
        }

        cache_map_[key] = {gu_slot, d_slot};
        pending.push_back({eid, gu_slot, d_slot});
    }

    if (pending.empty()) return;

    // Phase 2: Issue all H2D copies asynchronously on a dedicated stream.
    // Using cudaStreamPerThread avoids creating/managing a custom stream.
    cudaStream_t stream = cudaStreamPerThread;
    const auto & li = source.layers[layer];

    for (const auto & p : pending) {
        // Gate
        const uint8_t * gate_data = source.mmap_base + li.gate_offset
                                  + (size_t)p.expert_id * source.gate_expert_bytes;
        char * gate_dst = (char *)gate_3d_->data + (size_t)p.gu_slot * gate_3d_->nb[2];
        cudaMemcpyAsync(gate_dst, gate_data, source.gate_expert_bytes,
                        cudaMemcpyHostToDevice, stream);

        // Up
        const uint8_t * up_data = source.mmap_base + li.up_offset
                                + (size_t)p.expert_id * source.up_expert_bytes;
        char * up_dst = (char *)up_3d_->data + (size_t)p.gu_slot * up_3d_->nb[2];
        cudaMemcpyAsync(up_dst, up_data, source.up_expert_bytes,
                        cudaMemcpyHostToDevice, stream);

        // Down
        bool alt = is_alt_layer(layer);
        ggml_tensor * down_tensor = alt ? down_alt_3d_ : down_3d_;
        size_t down_bytes = source.layer_down_bytes.empty()
            ? source.down_expert_bytes
            : source.layer_down_bytes[layer];
        const uint8_t * down_data = source.mmap_base + li.down_offset
                                  + (size_t)p.expert_id * down_bytes;
        char * down_dst = (char *)down_tensor->data + (size_t)p.d_slot * down_tensor->nb[2];
        cudaMemcpyAsync(down_dst, down_data, down_bytes,
                        cudaMemcpyHostToDevice, stream);
    }

    // Phase 3: Single sync — wait for all transfers to complete.
    cudaStreamSynchronize(stream);
}

void ExpertCache::warmup(const MoeExpertSource & source) {
    int slots_per_layer = (n_slots_ - 1) / source.n_layers;
    if (slots_per_layer > source.n_experts) slots_per_layer = source.n_experts;

    std::printf("[ExpertCache] warmup: loading %d experts/layer × %d layers\n",
        slots_per_layer, source.n_layers);

    for (int l = 0; l < source.n_layers; l++) {
        for (int e = 0; e < slots_per_layer; e++) {
            ensure_cached(l, e, source);
        }
    }
    hit_count_  = 0;
    miss_count_ = 0;
}

void ExpertCache::destroy() {
    if (alt_buf_)    { ggml_backend_buffer_free(alt_buf_);    alt_buf_    = nullptr; }
    if (tensor_buf_) { ggml_backend_buffer_free(tensor_buf_); tensor_buf_ = nullptr; }
    if (tensor_ctx_) { ggml_free(tensor_ctx_); tensor_ctx_ = nullptr; }
    gate_3d_ = up_3d_ = down_3d_ = down_alt_3d_ = nullptr;
    slots_.clear();
    down_slots_.clear();
    alt_slots_.clear();
    cache_map_.clear();
    alt_layers_.clear();
    backend_ = nullptr;
    n_slots_ = 0;
    n_alt_slots_ = 0;
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
    out.down_slot_ids = ggml_new_tensor_2d(out.ctx, GGML_TYPE_I32, n_expert_used, max_tokens);

    ggml_set_name(out.post,          "moe_post");
    ggml_set_name(out.ffn_residual,  "moe_ffn_residual");
    ggml_set_name(out.weights,       "moe_weights");
    ggml_set_name(out.slot_ids,      "moe_slot_ids");
    ggml_set_name(out.down_slot_ids, "moe_down_slot_ids");

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
    s.post = s.ffn_residual = s.weights = s.slot_ids = s.down_slot_ids = nullptr;
    s.max_tokens = 0;
}

} // namespace dflash27b
