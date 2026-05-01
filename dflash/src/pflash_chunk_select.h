#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash27b {

struct PFlashChunkSelectOptions {
    float keep_ratio      = 0.02f;
    int   chunk_size      = 32;
    int   chunk_radius    = 0;
    int   query_tail      = 128;
    int   rare_max_freq   = 4;
    int   query_min_hits  = 2;
    int   query_radius    = 1;
    std::string dump_chunks_path;
};

struct PFlashChunkSelectStats {
    int    n_chunks        = 0;
    int    top_keep_chunks = 0;
    int    selected_chunks = 0;
    int    lexical_chunks  = 0;
    size_t out_tokens      = 0;
};

std::vector<int32_t> pflash_select_and_compress(
    const std::vector<int32_t> & ids,
    const std::vector<float> & smooth_scores,
    const PFlashChunkSelectOptions & opts,
    PFlashChunkSelectStats * stats = nullptr);

} // namespace dflash27b
