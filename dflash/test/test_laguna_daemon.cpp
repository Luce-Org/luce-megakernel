// Thin wrapper around dflash27b::run_laguna_daemon().
//
// Kept as a separate binary so scripts/laguna_pflash_niah.py can spawn the
// laguna daemon directly without going through test_dflash. The actual
// daemon implementation lives in src/laguna_daemon.cpp and is also dispatched
// by test_dflash when GGUF arch == "laguna", so the two binaries share one
// codepath.
//
// Usage:
//   test_laguna_daemon <laguna.gguf>
//       [--max-ctx N] [--kv q4_0|q5_0|q8_0|f16] [--chunk N] [--stream-fd N]

#include "laguna_daemon.h"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <laguna.gguf> [--max-ctx N] [--kv q4_0|q5_0|q8_0|f16] "
            "[--chunk N] [--stream-fd N]\n",
            argv[0]);
        return 1;
    }

    dflash27b::LagunaDaemonArgs args;
    args.target_path = argv[1];

    auto need_arg = [&](int i) {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "missing argument for %s\n", argv[i]);
            std::exit(2);
        }
    };
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--max-ctx")) { need_arg(i); args.max_ctx = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "--chunk")) { need_arg(i); args.chunk = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "--kv")) {
            need_arg(i);
            std::string s = argv[++i];
            if      (s == "q4_0") args.kv_type = GGML_TYPE_Q4_0;
            else if (s == "q5_0") args.kv_type = GGML_TYPE_Q5_0;
            else if (s == "q8_0") args.kv_type = GGML_TYPE_Q8_0;
            else if (s == "f16")  args.kv_type = GGML_TYPE_F16;
        }
        else if (!std::strncmp(argv[i], "--stream-fd=", 12)) {
            args.stream_fd = std::atoi(argv[i] + 12);
        }
        else if (!std::strcmp(argv[i], "--stream-fd")) {
            need_arg(i);
            args.stream_fd = std::atoi(argv[++i]);
        }
        else {
            std::fprintf(stderr, "[test_laguna_daemon] unknown flag: %s\n", argv[i]);
        }
    }

    return dflash27b::run_laguna_daemon(args);
}
