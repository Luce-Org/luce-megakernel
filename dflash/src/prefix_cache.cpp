// Prefix cache stub implementations.
// Full prefix cache is handled by the Python server (scripts/prefix_cache.py).
// These stubs satisfy the linker for test_dflash.

#include "internal.h"
#include "device_runtime.h"

namespace dflash27b {

bool snapshot_target_cache(const TargetWeights & w,
                            const TargetCache & cache,
                            ggml_backend_t backend,
                            PrefixSnapshot & snap) {
    snap.cur_pos = 0;
    snap.last_tok = -1;
    return false;
}

bool snapshot_target_cache_thin(const TargetWeights & w,
                                 const TargetCache & cache,
                                 ggml_backend_t backend,
                                 int kv_start,
                                 int kv_end,
                                 PrefixSnapshot & snap) {
    snap.cur_pos = 0;
    snap.last_tok = -1;
    snap.is_thin = true;
    snap.kv_start = kv_start;
    snap.kv_end = kv_end;
    return false;
}

bool restore_target_cache(const PrefixSnapshot & snap, TargetCache & cache) {
    return false;
}

bool restore_target_cache_chain(const PrefixSnapshot * thick,
                                 const PrefixSnapshot * const * thins,
                                 int n_thins,
                                 TargetCache & cache) {
    return false;
}

void free_prefix_snapshot(PrefixSnapshot & snap) {
    snap.cur_pos = 0;
    snap.last_tok = -1;
    snap.is_thin = false;
    snap.kv_start = 0;
    snap.kv_end = 0;
}

} // namespace dflash27b
