#include "pflash_chunk_select.h"

#include <algorithm>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace dflash27b {

std::vector<int32_t> pflash_select_and_compress(
    const std::vector<int32_t> & ids,
    const std::vector<float> & smooth_scores,
    const PFlashChunkSelectOptions & opts,
    PFlashChunkSelectStats * stats) {
    const int S = (int) ids.size();
    if (stats) {
        *stats = {};
    }
    if (S <= 0 || smooth_scores.size() < ids.size() || opts.chunk_size <= 0) {
        return {};
    }

    const int n_chunks = (S + opts.chunk_size - 1) / opts.chunk_size;
    const int raw_keep = (int) ((float) n_chunks * opts.keep_ratio);
    const int n_keep = std::min(n_chunks, std::max(1, raw_keep));
    const int chunk_radius = std::max(0, opts.chunk_radius);

    std::vector<std::pair<float, int>> chunk_means;
    chunk_means.reserve((size_t) n_chunks);
    for (int c = 0; c < n_chunks; ++c) {
        const int s = c * opts.chunk_size;
        const int e = std::min(S, (c + 1) * opts.chunk_size);
        float m = 0.0f;
        for (int j = s; j < e; ++j) {
            m += smooth_scores[(size_t) j];
        }
        m /= (float) std::max(1, e - s);
        chunk_means.push_back({m, c});
    }

    std::partial_sort(chunk_means.begin(),
                      chunk_means.begin() + n_keep,
                      chunk_means.end(),
                      [](auto a, auto b) { return a.first > b.first; });

    std::vector<uint8_t> selected_mask((size_t) n_chunks, 0);
    for (int i = 0; i < n_keep; ++i) {
        const int c = chunk_means[(size_t) i].second;
        const int lo = std::max(0, c - chunk_radius);
        const int hi = std::min(n_chunks - 1, c + chunk_radius);
        for (int x = lo; x <= hi; ++x) {
            selected_mask[(size_t) x] = 1;
        }
    }

    int lexical_chunks = 0;
    if (opts.query_tail > 0 && opts.rare_max_freq > 0 && opts.query_min_hits > 0) {
        std::unordered_map<int32_t, int> freq;
        freq.reserve(ids.size());
        for (int32_t tok : ids) {
            ++freq[tok];
        }

        const int q0 = std::max(0, S - opts.query_tail);
        std::unordered_set<int32_t> rare_query_tokens;
        rare_query_tokens.reserve((size_t) std::max(1, S - q0));
        for (int i = q0; i < S; ++i) {
            const int32_t tok = ids[(size_t) i];
            const auto it = freq.find(tok);
            if (it != freq.end() && it->second <= opts.rare_max_freq) {
                rare_query_tokens.insert(tok);
            }
        }

        if (!rare_query_tokens.empty()) {
            const int query_radius = std::max(0, opts.query_radius);
            for (int c = 0; c < n_chunks; ++c) {
                const int s = c * opts.chunk_size;
                const int e = std::min(S, (c + 1) * opts.chunk_size);
                const int scan_e = std::min(e, q0);
                if (s >= scan_e) {
                    continue;
                }
                int hits = 0;
                for (int j = s; j < scan_e; ++j) {
                    if (rare_query_tokens.find(ids[(size_t) j]) != rare_query_tokens.end()) {
                        ++hits;
                        if (hits >= opts.query_min_hits) {
                            break;
                        }
                    }
                }
                if (hits >= opts.query_min_hits) {
                    const int lo = std::max(0, c - query_radius);
                    const int hi = std::min(n_chunks - 1, c + query_radius);
                    bool added = false;
                    for (int x = lo; x <= hi; ++x) {
                        if (!selected_mask[(size_t) x]) {
                            added = true;
                        }
                        selected_mask[(size_t) x] = 1;
                    }
                    if (added) {
                        ++lexical_chunks;
                    }
                }
            }
        }
    }

    std::vector<int> selected;
    selected.reserve((size_t) n_chunks);
    for (int c = 0; c < n_chunks; ++c) {
        if (selected_mask[(size_t) c]) {
            selected.push_back(c);
        }
    }

    if (!opts.dump_chunks_path.empty()) {
        FILE * fp = std::fopen(opts.dump_chunks_path.c_str(), "w");
        if (fp) {
            std::fprintf(fp, "chunk,start,end,score,selected\n");
            for (int c = 0; c < n_chunks; ++c) {
                const int s = c * opts.chunk_size;
                const int e = std::min(S, (c + 1) * opts.chunk_size);
                float m = 0.0f;
                for (int j = s; j < e; ++j) {
                    m += smooth_scores[(size_t) j];
                }
                m /= (float) std::max(1, e - s);
                std::fprintf(fp, "%d,%d,%d,%.9g,%d\n",
                             c, s, e, (double) m, selected_mask[(size_t) c] ? 1 : 0);
            }
            std::fclose(fp);
        }
    }

    std::vector<int32_t> out;
    out.reserve((size_t) n_keep * (size_t) opts.chunk_size + 16);
    int span_start = -1;
    int span_end = -1;
    for (int c : selected) {
        const int s = c * opts.chunk_size;
        const int e = std::min(S, (c + 1) * opts.chunk_size);
        if (span_start < 0) {
            span_start = s;
            span_end = e;
        } else if (s == span_end) {
            span_end = e;
        } else {
            for (int j = span_start; j < span_end; ++j) {
                out.push_back(ids[(size_t) j]);
            }
            span_start = s;
            span_end = e;
        }
    }
    if (span_start >= 0) {
        for (int j = span_start; j < span_end; ++j) {
            out.push_back(ids[(size_t) j]);
        }
    }

    if (stats) {
        stats->n_chunks = n_chunks;
        stats->top_keep_chunks = n_keep;
        stats->selected_chunks = (int) selected.size();
        stats->lexical_chunks = lexical_chunks;
        stats->out_tokens = out.size();
    }
    return out;
}

} // namespace dflash27b
