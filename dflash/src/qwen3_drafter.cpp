// Qwen3-0.6B drafter for pflash speculative prefill, hosted in-process.
//
// Wires three pieces:
//   - qwen3_0p6b_loader.cpp : mmap GGUF + populate ggml tensors on backend
//   - qwen3_0p6b_graph.cpp  : custom forward (per-layer ggml + FP CUDA kernel)
//   - chunk-top-K + span merge (this file)
//
// Single-pass forward at full S using a custom Qwen3-0.6B graph with the
// FlashPrefill block-sparse attention kernel (or BSA when enabled). Tail
// attention scoring runs in a separate post-forward graph using saved Q_last
// and K_curr per layer.
//
// Result running_max [n_lookahead, S] f32 is reduced to per-token scores via
// mean-over-lookahead, smoothed with AvgPool, scored per chunk, top-K kept.

#include "qwen3_drafter.h"
#include "qwen3_0p6b_drafter.h"
#include "internal.h"
#include "pflash_chunk_select.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace dflash27b {

bool load_drafter(const std::string & gguf_path, int /*gpu_layers*/,
                  DrafterContext & out) {
    if (out.loaded) {
        set_last_error("drafter already loaded");
        return false;
    }

    // If caller didn't supply a backend, spin up our own CUDA one. Sharing
    // would be ideal but we don't have a handle to the daemon's backend
    // through this API. Same-process CUDA pools coexist fine — fragmentation
    // is the only cost, and we free everything in free_drafter.
    if (!out.backend) {
        size_t n_dev = ggml_backend_dev_count();
        for (size_t i = 0; i < n_dev; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                out.backend = ggml_backend_dev_init(dev, nullptr);
                break;
            }
        }
        if (!out.backend) {
            set_last_error("load_drafter: no GPU backend available");
            return false;
        }
    }

    if (!load_qwen3_0p6b_drafter(gguf_path, out.backend, out.weights)) {
        // last_error already set by loader
        return false;
    }

    out.loaded = true;
    std::fprintf(stderr,
        "[drafter] loaded Qwen3-0.6B %s: n_layer=%d n_head=%d n_kv=%d "
        "n_embd=%d n_ff=%d head_dim=%d vocab=%d\n",
        ggml_type_name(out.weights.compute_type),
        out.weights.n_layer, out.weights.n_head, out.weights.n_head_kv,
        out.weights.n_embd, out.weights.n_ff, out.weights.head_dim,
        out.weights.n_vocab);
    std::fflush(stderr);
    return true;
}

void free_drafter(DrafterContext & ctx) {
    if (ctx.loaded) {
        free_qwen3_0p6b_drafter(ctx.weights);
    }
    if (ctx.backend) {
        ggml_backend_free(ctx.backend);
        ctx.backend = nullptr;
    }
    ctx.loaded = false;
}

std::vector<int32_t> drafter_score_and_compress(
    DrafterContext & ctx,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int chunk_size,
    int n_lookahead,
    int pool_kernel) {
    if (!ctx.loaded) {
        set_last_error("drafter not loaded");
        return {};
    }
    const int S = (int)ids.size();

    if (const char * env = std::getenv("DFLASH_PFLASH_LOOKAHEAD")) {
        int v = std::atoi(env);
        if (v > 0 && v <= 256) {
            n_lookahead = v;
        }
    }

    if (S < n_lookahead + 1) {
        // Too short to score — return as-is.
        return ids;
    }

    // ── 1. Custom forward + GPU tail-attention scoring ────────────────
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> running_max;
    if (!forward_qwen3_0p6b_drafter(ctx.weights, ids, n_lookahead, running_max)) {
        return {};
    }
    auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[drafter] forward+score in %.2fs S=%d\n",
        std::chrono::duration<double>(t1 - t0).count(), S);
    std::fflush(stderr);

    // ── 2. Mean over lookahead → per-token score [S] ──────────────────
    std::vector<float> score((size_t)S, 0.0f);
    for (int j = 0; j < S; ++j) {
        float s = 0.0f;
        for (int t = 0; t < n_lookahead; ++t) {
            s += running_max[(size_t)t * S + j];
        }
        score[j] = s / (float)n_lookahead;
    }

    // ── 3. AvgPool 1D smoothing ───────────────────────────────────────
    std::vector<float> smooth((size_t)S, 0.0f);
    int half = pool_kernel / 2;
    for (int j = 0; j < S; ++j) {
        int lo = std::max(0, j - half);
        int hi = std::min(S - 1, j + half);
        float s = 0.0f;
        int n = 0;
        for (int k = lo; k <= hi; ++k) { s += score[k]; ++n; }
        smooth[j] = (n > 0) ? (s / (float)n) : 0.0f;
    }

    // ── 4. Chunk-top-K + lexical rescue + span merge ──────────────────
    PFlashChunkSelectOptions select_opts;
    select_opts.keep_ratio = keep_ratio;
    select_opts.chunk_size = chunk_size;
    if (const char * env = std::getenv("DFLASH_PFLASH_CHUNK_RADIUS")) {
        int v = std::atoi(env);
        if (v >= 0 && v <= 16) {
            select_opts.chunk_radius = v;
        }
    }
    if (const char * env = std::getenv("DFLASH_PFLASH_QUERY_TAIL")) {
        int v = std::atoi(env);
        if (v >= 0 && v <= 4096) {
            select_opts.query_tail = v;
        }
    }
    if (const char * env = std::getenv("DFLASH_PFLASH_RARE_MAX_FREQ")) {
        int v = std::atoi(env);
        if (v >= 0 && v <= 1024) {
            select_opts.rare_max_freq = v;
        }
    }
    if (const char * env = std::getenv("DFLASH_PFLASH_QUERY_MIN_HITS")) {
        int v = std::atoi(env);
        if (v >= 1 && v <= 64) {
            select_opts.query_min_hits = v;
        }
    }
    if (const char * env = std::getenv("DFLASH_PFLASH_QUERY_RADIUS")) {
        int v = std::atoi(env);
        if (v >= 0 && v <= 16) {
            select_opts.query_radius = v;
        }
    }
    if (const char * dump_path = std::getenv("DFLASH_PFLASH_DUMP_CHUNKS")) {
        select_opts.dump_chunks_path = dump_path;
    }
    PFlashChunkSelectStats select_stats;
    std::vector<int32_t> out =
        pflash_select_and_compress(ids, smooth, select_opts, &select_stats);

    auto t2 = std::chrono::steady_clock::now();
    std::fprintf(stderr,
        "[drafter] score_and_compress total %.2fs S=%d kept=%zu (%zu selected chunks, top=%d/%d radius=%d lexical=%d tail=%d rare<=%d min_hits=%d q_radius=%d)\n",
        std::chrono::duration<double>(t2 - t0).count(),
        S, out.size(), (size_t) select_stats.selected_chunks,
        select_stats.top_keep_chunks, select_stats.n_chunks, select_opts.chunk_radius,
        select_stats.lexical_chunks, select_opts.query_tail, select_opts.rare_max_freq,
        select_opts.query_min_hits, select_opts.query_radius);
    std::fflush(stderr);

    return out;
}

} // namespace dflash27b
