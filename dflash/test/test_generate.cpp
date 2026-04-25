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

#include <cuda_runtime.h>

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
    ggml_tensor *     attn_mask = nullptr;
    ggml_tensor *     logits    = nullptr;
    ggml_tensor *     argmax_out = nullptr;
};

static bool build_reusable_graph(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int max_ctx
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
    sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, max_ctx, n_tokens);
    ggml_set_input(sg.inp_embed);
    ggml_set_input(sg.positions);
    ggml_set_input(sg.attn_mask);

    sg.gf = ggml_new_graph_custom(sg.ctx, 8192, false);

    QwenGraphInputs gi{};
    gi.inp_embed      = sg.inp_embed;
    gi.positions      = sg.positions;
    gi.attn_mask      = sg.attn_mask;
    gi.n_tokens       = n_tokens;
    gi.kv_start       = max_ctx - 1;
    gi.capture_layers = false;

    QwenGraphOutputs go = build_qwen35_graph(sg.ctx, sg.gf, w, cache, gi);
    if (!go.logits) return false;

    sg.argmax_out = ggml_argmax(sg.ctx, go.logits);
    ggml_set_output(sg.argmax_out);
    ggml_build_forward_expand(sg.gf, sg.argmax_out);
    sg.logits = go.logits;

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    if (!ggml_gallocr_reserve(sg.alloc, sg.gf)) return false;
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

static void copy_kv_slot(
    const std::vector<ggml_tensor *> & cache_tensors,
    int src_slot, int dst_slot, int max_ctx, int n_heads
) {
    for (auto * ct : cache_tensors) {
        const size_t pos_bytes = ct->nb[1];
        const size_t head_stride = ct->nb[2];
        const char * base = (const char *) ct->data;
        cudaMemcpy2D(
            (void *)(base + (size_t)dst_slot * pos_bytes), head_stride,
            (const void *)(base + (size_t)src_slot * pos_bytes), head_stride,
            pos_bytes, n_heads, cudaMemcpyDeviceToDevice
        );
    }
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

    const int n_full_attn = w.n_layer / w.full_attention_interval;
    const int n_head_kv = w.n_head_kv;

    StepGraph sg;
    if (!build_reusable_graph(sg, w, cache, backend, max_ctx)) {
        std::fprintf(stderr, "build_reusable_graph failed\n");
        return 1;
    }

    std::vector<ggml_fp16_t> mask_buf(max_ctx);
    std::vector<int32_t> all_tokens = prompt;
    all_tokens.reserve(prompt.size() + n_gen);
    const int hidden = w.n_embd;
    std::vector<float> embed_buf(hidden);
    int32_t argmax_result = 0;

    const int write_slot = max_ctx - 1;

    auto run_step_reuse = [&](int32_t tok, int pos) -> int32_t {
        int32_t ids[1] = { tok };
        if (!w.embedder.embed(ids, 1, embed_buf.data())) {
            std::fprintf(stderr, "embed failed tok=%d\n", tok);
            std::exit(1);
        }
        ggml_backend_tensor_set(sg.inp_embed, embed_buf.data(), 0,
                                sizeof(float) * embed_buf.size());

        int32_t p4[4] = { pos, pos, pos, pos };
        ggml_backend_tensor_set(sg.positions, p4, 0, sizeof(int32_t) * 4);

        for (int i = 0; i < max_ctx; i++) {
            bool attend = (i < pos) || (i == write_slot);
            mask_buf[i] = ggml_fp32_to_fp16(attend ? 0.0f : -INFINITY);
        }
        ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0,
                                sizeof(ggml_fp16_t) * max_ctx);

        auto st = ggml_backend_graph_compute(backend, sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "compute failed at pos=%d (%d)\n", pos, (int)st);
            std::exit(1);
        }

        ggml_backend_tensor_get(sg.argmax_out, &argmax_result, 0, sizeof(int32_t));

        copy_kv_slot(cache.attn_k, write_slot, pos, max_ctx, n_head_kv);
        copy_kv_slot(cache.attn_v, write_slot, pos, max_ctx, n_head_kv);

        return argmax_result;
    };

    // ── Prefill: feed prompt tokens one at a time (decode-only mode).
    int next = -1;
    for (int i = 0; i < (int)prompt.size(); i++) {
        next = run_step_reuse(prompt[i], i);
    }
    std::printf("[prefill] last-token argmax=%d\n", next);

    // ── Generation loop (CUDA graph captures on first step)
    auto t_start = std::chrono::steady_clock::now();
    double total_compute = 0;
    int gen_start_pos = (int)prompt.size();
    for (int g = 0; g < n_gen; g++) {
        int32_t tok = next;
        all_tokens.push_back(tok);
        stream_emit(tok);

        auto t0 = std::chrono::steady_clock::now();

        if (!w.embedder.embed(&tok, 1, embed_buf.data())) return 1;
        ggml_backend_tensor_set(sg.inp_embed, embed_buf.data(), 0, sizeof(float) * embed_buf.size());
        int32_t p4[4] = { gen_start_pos + g, gen_start_pos + g, gen_start_pos + g, gen_start_pos + g };
        ggml_backend_tensor_set(sg.positions, p4, 0, sizeof(int32_t) * 4);
        for (int i = 0; i < max_ctx; i++) {
            bool attend = (i < gen_start_pos + g) || (i == write_slot);
            mask_buf[i] = ggml_fp32_to_fp16(attend ? 0.0f : -INFINITY);
        }
        ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0, sizeof(ggml_fp16_t) * max_ctx);

        ggml_backend_graph_compute(backend, sg.gf);

        ggml_backend_tensor_get(sg.argmax_out, &argmax_result, 0, sizeof(int32_t));
        next = argmax_result;

        copy_kv_slot(cache.attn_k, write_slot, gen_start_pos + g, max_ctx, n_head_kv);
        copy_kv_slot(cache.attn_v, write_slot, gen_start_pos + g, max_ctx, n_head_kv);

        auto t1 = std::chrono::steady_clock::now();
        total_compute += std::chrono::duration<double>(t1 - t0).count();
    }
    auto t_end = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t_end - t_start).count();
    double tps  = n_gen / std::max(1e-9, secs);

    all_tokens.push_back(next);

    std::printf("[gen] %d new tokens in %.3f s  ->  %.2f tok/s\n", n_gen, secs, tps);
    std::printf("[gen] compute=%.3f s (%.1f%% of total)\n",
        total_compute, 100.0 * total_compute / std::max(1e-12, secs));
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
