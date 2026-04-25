// End-to-end generation test for our qwen35 target forward.
//
// Reads a binary int32 token file (produced by scripts/tokenize_prompt.py),
// runs single-token decode over every token (no batched prefill), generates
// N new tokens via greedy argmax, and writes the resulting int32 token stream
// to an output file for Python-side detokenization.
//
// Also reports decode tok/s (generation only, prompt steps excluded).
//
// Usage:
//   test_generate <qwen35.gguf> <prompt_ids.bin> <n_gen> <out_ids.bin>

#include "dflash27b.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace dflash27b;

struct StepGraph {
    ggml_context *    ctx = nullptr;
    ggml_cgraph *     gf  = nullptr;
    ggml_gallocr_t    alloc = nullptr;
    ggml_tensor *     inp_embed = nullptr;
    ggml_tensor *     positions = nullptr;
    ggml_tensor *     logits    = nullptr;
};

// Build a fresh single-token forward graph. We rebuild per step so that
// `kv_start` updates drive the correct KV cache slot. The graph is cheap to
// rebuild — all the weights + KV cache stay persistent.
static bool build_step_graph(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int kv_start
) {
    if (sg.ctx) { ggml_free(sg.ctx); sg.ctx = nullptr; }

    ggml_init_params ip{};
    ip.mem_size   = 256 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int n_tokens = 1;
    const int hidden = w.n_embd;
    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_input(sg.inp_embed);
    ggml_set_input(sg.positions);

    sg.gf = ggml_new_graph_custom(sg.ctx, 8192, false);

    QwenGraphInputs gi{};
    gi.inp_embed      = sg.inp_embed;
    gi.positions      = sg.positions;
    gi.attn_mask      = nullptr;
    gi.n_tokens       = n_tokens;
    gi.kv_start       = kv_start;
    gi.capture_layers = false;

    QwenGraphOutputs go = build_qwen35_graph(sg.ctx, sg.gf, w, cache, gi);
    if (!go.logits) return false;
    ggml_set_output(go.logits);
    ggml_build_forward_expand(sg.gf, go.logits);
    sg.logits = go.logits;

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_reserve(sg.alloc, sg.gf);
    }
    if (!ggml_gallocr_alloc_graph(sg.alloc, sg.gf)) return false;

    return true;
}
    if (!ggml_gallocr_alloc_graph(sg.alloc, sg.gf)) return false;
    return true;
}

static std::vector<int32_t> read_int32_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<int32_t> out(sz / sizeof(int32_t));
    f.read((char *)out.data(), sz);
    return out;
}

static bool write_int32_file(const std::string & path, const std::vector<int32_t> & v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write((const char *)v.data(), v.size() * sizeof(int32_t));
    return (bool)f;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
            "usage: %s <qwen35.gguf> <prompt_ids.bin> <n_gen> <out_ids.bin>\n", argv[0]);
        return 2;
    }
    const char * gguf_path   = argv[1];
    const char * prompt_path = argv[2];
    const int    n_gen       = std::atoi(argv[3]);
    const char * out_path    = argv[4];
    int stream_fd = -1;
    for (int i = 5; i < argc; i++) {
        if (std::strncmp(argv[i], "--stream-fd=", 12) == 0) {
            stream_fd = std::atoi(argv[i] + 12);
        }
    }
    auto stream_emit = [&](int32_t tok) {
        if (stream_fd < 0) return;
        int32_t v = tok;
#if defined(_WIN32)
        DWORD written;
        WriteFile((HANDLE)(intptr_t)stream_fd, &v, sizeof(v), &written, nullptr);
#else
        ssize_t n = ::write(stream_fd, &v, sizeof(v));
        (void)n;
#endif
    };

    // ── Load model and cache
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    TargetWeights w;
    if (!load_target_gguf(gguf_path, backend, w)) {
        std::fprintf(stderr, "load: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("[target] %s\n", dflash27b_last_error());

    const int max_ctx = 4096;
    TargetCache cache;
    if (!create_target_cache(w, max_ctx, /*max_verify_tokens=*/0, backend, cache)) {
        std::fprintf(stderr, "cache: %s\n", dflash27b_last_error());
        return 1;
    }

    auto prompt = read_int32_file(prompt_path);
    if (prompt.empty()) { std::fprintf(stderr, "empty prompt bin\n"); return 1; }
    std::printf("[prompt] %zu tokens: ", prompt.size());
    for (auto t : prompt) std::printf("%d ", t);
    std::printf("\n");

    if ((int)prompt.size() + n_gen > max_ctx) {
        std::fprintf(stderr, "prompt+gen exceeds max_ctx\n");
        return 1;
    }

    std::vector<int32_t> all_tokens = prompt;
    all_tokens.reserve(prompt.size() + n_gen);

    const int hidden = w.n_embd;
    std::vector<float> embed_buf(hidden);

    StepGraph sg;

    // ── Helper: run one step given current token + absolute position
    auto run_step = [&](int32_t tok, int pos) -> int32_t {
        if (!build_step_graph(sg, w, cache, backend, pos)) {
            std::fprintf(stderr, "build_step_graph failed at pos=%d\n", pos);
            std::exit(1);
        }

        // CPU embed
        int32_t ids[1] = { tok };
        if (!w.embedder.embed(ids, 1, embed_buf.data())) {
            std::fprintf(stderr, "embed failed tok=%d\n", tok);
            std::exit(1);
        }
        ggml_backend_tensor_set(sg.inp_embed, embed_buf.data(), 0,
                                sizeof(float) * embed_buf.size());

        // M-RoPE positions: 4 copies of pos
        int32_t p4[4] = { pos, pos, pos, pos };
        ggml_backend_tensor_set(sg.positions, p4, 0, sizeof(int32_t) * 4);

        auto st = ggml_backend_graph_compute(backend, sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "compute failed at pos=%d (%d)\n", pos, (int)st);
            std::exit(1);
        }

        // argmax on logits
        const int vocab = (int)w.output->ne[1];
        std::vector<float> logits(vocab);
        ggml_backend_tensor_get(sg.logits, logits.data(), 0, sizeof(float) * vocab);
        int best = 0;
        float bv = logits[0];
        for (int i = 1; i < vocab; i++) {
            if (logits[i] > bv) { bv = logits[i]; best = i; }
        }
        return best;
    };

    // ── Prefill: feed prompt tokens one at a time (decode-only mode).
    //    We throw away the logits for all prompt tokens except the last one.
    int next = -1;
    for (int i = 0; i < (int)prompt.size(); i++) {
        next = run_step(prompt[i], i);
    }
    std::printf("[prefill] last-token argmax=%d\n", next);

    // ── Generation loop
    auto t_start = std::chrono::steady_clock::now();
    int gen_start_pos = (int)prompt.size();
    double total_build = 0, total_compute = 0;
    for (int g = 0; g < n_gen; g++) {
        int32_t tok = next;
        all_tokens.push_back(tok);
        stream_emit(tok);

        auto t0 = std::chrono::steady_clock::now();
        if (!build_step_graph(sg, w, cache, backend, gen_start_pos + g)) {
            std::fprintf(stderr, "build_step_graph failed at g=%d\n", g);
            return 1;
        }
        if (!w.embedder.embed(&tok, 1, embed_buf.data())) return 1;
        ggml_backend_tensor_set(sg.inp_embed, embed_buf.data(), 0, sizeof(float) * embed_buf.size());
        int32_t p4[4] = { gen_start_pos + g, gen_start_pos + g, gen_start_pos + g, gen_start_pos + g };
        ggml_backend_tensor_set(sg.positions, p4, 0, sizeof(int32_t) * 4);
        auto t1 = std::chrono::steady_clock::now();

        ggml_backend_graph_compute(backend, sg.gf);
        auto t2 = std::chrono::steady_clock::now();

        const int vocab = (int)w.output->ne[1];
        std::vector<float> logits(vocab);
        ggml_backend_tensor_get(sg.logits, logits.data(), 0, sizeof(float) * vocab);
        int best = 0; float bv = logits[0];
        for (int i = 1; i < vocab; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
        next = best;

        total_build += std::chrono::duration<double>(t1 - t0).count();
        total_compute += std::chrono::duration<double>(t2 - t1).count();
    }
    auto t_end = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t_end - t_start).count();
    double tps  = n_gen / std::max(1e-9, secs);

    // Also push the final next token so downstream sees it
    all_tokens.push_back(next);

    std::printf("[gen] %d new tokens in %.3f s  ->  %.2f tok/s\n", n_gen, secs, tps);
    std::printf("[gen] build=%.3f s compute=%.3f s (%.1f%% compute)\n",
        total_build, total_compute, 100.0 * total_compute / std::max(1e-12, total_build + total_compute));
    std::printf("[gen] tokens: ");
    for (int i = 0; i < n_gen; i++) std::printf("%d ", all_tokens[prompt.size() + i]);
    std::printf("\n");

    write_int32_file(out_path, all_tokens);
    std::printf("[out] wrote %zu tokens to %s\n", all_tokens.size(), out_path);

    if (sg.alloc) ggml_gallocr_free(sg.alloc);
    if (sg.ctx)   ggml_free(sg.ctx);
    free_target_cache(cache);
    free_target_weights(w);
    ggml_backend_free(backend);
    return 0;
}
