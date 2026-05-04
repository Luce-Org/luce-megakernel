#include "mega_pflash_native.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

double elapsed_ms(std::chrono::steady_clock::time_point a,
                  std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "usage: " << argv[0]
                  << " <qwen3.5-0.8b-safetensors-or-dir> <seq_len> [keep_ratio] [max_seq_len]\n";
        return 2;
    }

    const std::string model_path = argv[1];
    const int seq_len = std::max(1, std::atoi(argv[2]));
    const float keep_ratio = argc >= 4 ? std::atof(argv[3]) : 0.05f;
    const int max_seq_len = argc >= 5 ? std::max(seq_len, std::atoi(argv[4])) : seq_len;

    std::vector<int32_t> ids(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        ids[i] = 1000 + (i % 30000);
    }

    dflash27b::MegaPFlashContext ctx;
    const auto load_a = std::chrono::steady_clock::now();
    if (!dflash27b::load_mega_pflash(model_path, max_seq_len, ctx)) {
        std::cerr << "load_mega_pflash failed\n";
        return 1;
    }
    const auto load_b = std::chrono::steady_clock::now();

    const auto comp_a = std::chrono::steady_clock::now();
    const std::vector<int32_t> out =
        dflash27b::mega_pflash_score_and_compress(ctx, ids, keep_ratio);
    const auto comp_b = std::chrono::steady_clock::now();

    std::cout << "native mega-pflash smoke ok"
              << " seq_len=" << seq_len
              << " max_seq_len=" << max_seq_len
              << " keep_ratio=" << keep_ratio
              << " out_tokens=" << out.size()
              << " load_ms=" << elapsed_ms(load_a, load_b)
              << " compress_ms=" << elapsed_ms(comp_a, comp_b)
              << "\n";

    dflash27b::free_mega_pflash(ctx);
    return 0;
}
