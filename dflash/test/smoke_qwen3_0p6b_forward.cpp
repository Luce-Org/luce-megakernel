// Smoke test for the custom Qwen3-0.6B drafter forward path.
//
// Loads the BF16 GGUF, generates a synthetic token sequence at the requested
// length, runs drafter_score_and_compress end-to-end, and prints timing +
// compression ratio. Used to validate the in-process pflash integration
// without touching the daemon.
//
// Usage:
//   smoke_qwen3_0p6b_forward <gguf_path> <seq_len_or_FILE:path> [keep_ratio] [n_lookahead] [chunk_size] [pool_kernel]
// Examples:
//   smoke_qwen3_0p6b_forward .../Qwen3-0.6B-BF16.gguf 140000 0.02
//   smoke_qwen3_0p6b_forward .../Qwen3-0.6B-BF16.gguf FILE:/tmp/niah_32k.bin 0.05
//
// Token file format: little-endian u32 count, then count int32 token IDs.

#include "qwen3_drafter.h"
#include "dflash27b.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace dflash27b;

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <gguf> <seq_len_or_FILE:path> [keep_ratio=0.02] [n_lookahead=8] [chunk_size=32] [pool_kernel=13]\n",
            argv[0]);
        return 2;
    }
    const std::string gguf = argv[1];
    const std::string arg2 = argv[2];
    const float keep_ratio = (argc >= 4) ? std::atof(argv[3]) : 0.02f;
    const int n_lookahead = (argc >= 5) ? std::atoi(argv[4]) : 8;
    const int chunk_size  = (argc >= 6) ? std::atoi(argv[5]) : 32;
    const int pool_kernel = (argc >= 7) ? std::atoi(argv[6]) : 13;

    std::vector<int32_t> ids_from_file;
    int S = 0;
    bool from_file = arg2.rfind("FILE:", 0) == 0;
    if (from_file) {
        std::string path = arg2.substr(5);
        FILE * fp = std::fopen(path.c_str(), "rb");
        if (!fp) {
            std::fprintf(stderr, "open %s failed\n", path.c_str());
            return 2;
        }
        uint32_t cnt = 0;
        if (std::fread(&cnt, 4, 1, fp) != 1) { std::fclose(fp); return 2; }
        ids_from_file.resize((size_t)cnt);
        if (std::fread(ids_from_file.data(), sizeof(int32_t), cnt, fp) != cnt) {
            std::fclose(fp); return 2;
        }
        std::fclose(fp);
        S = (int)cnt;
        std::printf("[smoke] loaded %d real tokens from %s\n", S, path.c_str());
    } else {
        S = std::atoi(arg2.c_str());
        if (S < 64) {
            std::fprintf(stderr, "seq_len too small: %d\n", S);
            return 2;
        }
    }

    DrafterContext ctx;
    auto t_load0 = std::chrono::steady_clock::now();
    if (!load_drafter(gguf, /*gpu_layers=*/-1, ctx)) {
        std::fprintf(stderr, "load_drafter failed: %s\n", dflash27b_last_error());
        return 1;
    }
    auto t_load1 = std::chrono::steady_clock::now();
    std::printf("[smoke] load_drafter %.2fs vocab=%d\n",
        std::chrono::duration<double>(t_load1 - t_load0).count(),
        ctx.weights.n_vocab);

    std::vector<int32_t> ids;
    if (from_file) {
        ids = std::move(ids_from_file);
    } else {
        ids.resize((size_t)S);
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, ctx.weights.n_vocab - 1);
        for (int i = 0; i < S; ++i) ids[i] = dist(rng);
    }

    std::printf("[smoke] running drafter_score_and_compress S=%d keep_ratio=%.3f n_lookahead=%d chunk_size=%d pool_kernel=%d...\n",
        S, keep_ratio, n_lookahead, chunk_size, pool_kernel);
    std::fflush(stdout);

    auto t0 = std::chrono::steady_clock::now();
    std::vector<int32_t> out = drafter_score_and_compress(
        ctx, ids, keep_ratio,
        chunk_size, n_lookahead, pool_kernel);
    auto t1 = std::chrono::steady_clock::now();

    if (out.empty()) {
        std::fprintf(stderr, "drafter_score_and_compress returned empty: %s\n",
            dflash27b_last_error());
        free_drafter(ctx);
        return 1;
    }

    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::printf("[smoke] OK: %.2fs  in=%d  out=%zu  ratio=%.4f\n",
        secs, S, out.size(), (double)out.size() / (double)S);

    // Optional: write kept ids to a file for downstream detokenization.
    if (const char * kept_path = std::getenv("PFLASH_KEPT_OUT")) {
        FILE * fk = std::fopen(kept_path, "wb");
        if (fk) {
            uint32_t n = (uint32_t)out.size();
            std::fwrite(&n, 4, 1, fk);
            std::fwrite(out.data(), sizeof(int32_t), n, fk);
            std::fclose(fk);
            std::printf("[smoke] wrote %u kept ids to %s\n", n, kept_path);
        }
    }

    free_drafter(ctx);
    return 0;
}
