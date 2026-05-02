#include "../src/pflash_chunk_select.h"

#include <algorithm>
#include <cstdio>
#include <vector>

using namespace dflash27b;

static bool contains_token(const std::vector<int32_t> & xs, int32_t tok) {
    return std::find(xs.begin(), xs.end(), tok) != xs.end();
}

int main() {
    constexpr int S = 512;
    constexpr int chunk = 32;
    constexpr int answer_token = 424242;

    std::vector<int32_t> ids((size_t) S, 1);
    std::vector<float> scores((size_t) S, 0.01f);

    // Force the plain score top-K path to keep only chunk 1.
    for (int i = chunk; i < 2 * chunk; ++i) {
        scores[(size_t) i] = 10.0f;
    }

    // Needle is in chunk 8. The query tail repeats two rare marker tokens.
    ids[(size_t) (8 * chunk + 5)] = 777;
    ids[(size_t) (8 * chunk + 6)] = 888;
    ids[(size_t) (8 * chunk + 7)] = answer_token;
    ids[(size_t) (15 * chunk + 5)] = 777;
    ids[(size_t) (15 * chunk + 6)] = 888;

    PFlashChunkSelectOptions opts;
    opts.keep_ratio = 0.05f;
    opts.chunk_size = chunk;
    opts.chunk_radius = 0;
    opts.query_tail = 0;
    opts.rare_max_freq = 4;
    opts.query_min_hits = 2;
    opts.query_radius = 1;

    PFlashChunkSelectStats off_stats;
    std::vector<int32_t> off = pflash_select_and_compress(ids, scores, opts, &off_stats);
    if (contains_token(off, answer_token)) {
        std::fprintf(stderr, "lexical-off unexpectedly kept answer token\n");
        return 1;
    }
    if (off_stats.selected_chunks != 1 || off_stats.lexical_chunks != 0) {
        std::fprintf(stderr, "lexical-off stats mismatch: selected=%d lexical=%d\n",
                     off_stats.selected_chunks, off_stats.lexical_chunks);
        return 1;
    }

    opts.query_tail = 64;
    PFlashChunkSelectStats on_stats;
    std::vector<int32_t> on = pflash_select_and_compress(ids, scores, opts, &on_stats);
    if (!contains_token(on, answer_token)) {
        std::fprintf(stderr, "lexical rescue failed to keep answer token\n");
        return 1;
    }
    if (on_stats.lexical_chunks <= 0 || on_stats.selected_chunks <= off_stats.selected_chunks) {
        std::fprintf(stderr, "lexical-on stats mismatch: selected=%d lexical=%d\n",
                     on_stats.selected_chunks, on_stats.lexical_chunks);
        return 1;
    }

    std::printf("[pflash-chunk-select] PASS lexical rescue kept answer token "
                "(off chunks=%d, on chunks=%d, lexical=%d)\n",
                off_stats.selected_chunks, on_stats.selected_chunks, on_stats.lexical_chunks);
    return 0;
}
