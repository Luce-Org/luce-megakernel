// Minimal Laguna daemon binary for end-to-end testing via stdin protocol.
//
// Loads the Laguna target once. Then for each `generate <prompt_ids.bin> <n_gen>
// <out_ids.bin>` line on stdin:
//   - reads counted-i32 prompt token IDs (already in LAGUNA vocab),
//   - chunked-prefills them through build_laguna_graph (causal F32->F16 mask),
//   - autoregressively decodes n_gen tokens via greedy argmax,
//   - writes counted-i32 generated IDs to <out_ids.bin>.
// Stops the loop on `quit` / `exit` / EOF.
//
// Reuses the same kernels as bench_laguna_pflash + bench_laguna_generate.
// Cross-tokenizer (Qwen3 -> Laguna) and PFlash compression are done by the
// caller (Python orchestrator); this binary is the TARGET-only step.
//
// Usage:
//   test_laguna_daemon <laguna.gguf> [--max-ctx N] [--kv q4_0|q8_0|f16] [--chunk N]

#include "laguna_internal.h"
#include "internal.h"
#include "dflash27b.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"

using namespace dflash27b;

static std::vector<int32_t> read_counted_i32(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    uint32_t n = 0;
    f.read(reinterpret_cast<char *>(&n), sizeof(n));
    if (!f) return {};
    std::vector<int32_t> ids((size_t)n);
    if (n > 0) {
        f.read(reinterpret_cast<char *>(ids.data()), (std::streamsize)ids.size() * sizeof(int32_t));
        if (!f) return {};
    }
    return ids;
}

static bool write_counted_i32(const std::string & path, const std::vector<int32_t> & ids) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t n = (uint32_t)ids.size();
    f.write(reinterpret_cast<const char *>(&n), sizeof(n));
    if (n > 0) f.write(reinterpret_cast<const char *>(ids.data()), (std::streamsize)ids.size() * sizeof(int32_t));
    return (bool)f;
}

// Build + run a single Laguna forward step. Returns last-token logits via host.
// Builds BOTH a full causal mask AND a sliding-window-causal mask (for SWA layers).
static bool laguna_step(
    ggml_backend_t backend,
    const LagunaTargetWeights & w,
    LagunaTargetCache & cache,
    const float * embed,
    int n_tok,
    int kv_start,
    bool no_mask,
    std::vector<float> & out_logits)
{
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead() + 16 * 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);

    ggml_tensor * ie = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w.n_embd, n_tok, 1);
    ggml_set_input(ie);
    ggml_tensor * pp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tok);
    ggml_set_input(pp);
    ggml_tensor * mk_full = nullptr, * mk_full_cnv = nullptr;
    ggml_tensor * mk_swa  = nullptr, * mk_swa_cnv  = nullptr;
    const int kv_len = kv_start + n_tok;
    if (!no_mask) {
        mk_full = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kv_len, n_tok, 1, 1);
        ggml_set_input(mk_full);
        mk_full_cnv = ggml_cast(ctx, mk_full, GGML_TYPE_F16);
        mk_swa = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kv_len, n_tok, 1, 1);
        ggml_set_input(mk_swa);
        mk_swa_cnv = ggml_cast(ctx, mk_swa, GGML_TYPE_F16);
    }

    LagunaGraphInputs gi{};
    gi.inp_embed       = ie;
    gi.positions       = pp;
    gi.attn_mask       = mk_full_cnv;
    gi.attn_mask_swa   = mk_swa_cnv;
    gi.n_tokens        = n_tok;
    gi.kv_start        = kv_start;
    gi.output_logits   = true;
    gi.output_last_only= true;

    LagunaGraphOutputs go = build_laguna_graph(ctx, gf, w, cache, gi);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc); ggml_free(ctx); return false;
    }
    ggml_backend_tensor_set(ie, embed, 0, (size_t)n_tok * w.n_embd * sizeof(float));
    std::vector<int32_t> pos(n_tok);
    for (int i = 0; i < n_tok; ++i) pos[i] = kv_start + i;
    ggml_backend_tensor_set(pp, pos.data(), 0, pos.size() * sizeof(int32_t));
    if (mk_full) {
        std::vector<float> mb((size_t)kv_len * n_tok, 0.0f);
        for (int t = 0; t < n_tok; ++t) {
            const int abs_q = kv_start + t;
            for (int j = 0; j < kv_len; ++j) {
                if (j > abs_q) mb[(size_t)t * kv_len + j] = -INFINITY;
            }
        }
        ggml_backend_tensor_set(mk_full, mb.data(), 0, mb.size() * sizeof(float));
    }
    if (mk_swa) {
        const int sw = w.sliding_window;
        std::vector<float> mb((size_t)kv_len * n_tok, 0.0f);
        for (int t = 0; t < n_tok; ++t) {
            const int abs_q = kv_start + t;
            for (int j = 0; j < kv_len; ++j) {
                // Causal AND inside sliding window: keep iff j <= abs_q AND j > abs_q - sw.
                if (j > abs_q || j <= abs_q - sw) {
                    mb[(size_t)t * kv_len + j] = -INFINITY;
                }
            }
        }
        ggml_backend_tensor_set(mk_swa, mb.data(), 0, mb.size() * sizeof(float));
    }

    ggml_status st = ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    if (st != GGML_STATUS_SUCCESS) { ggml_gallocr_free(galloc); ggml_free(ctx); return false; }
    cache.cur_pos = kv_start + n_tok;

    const int64_t vocab = go.logits->ne[0];
    out_logits.resize((size_t)vocab);
    ggml_backend_tensor_get(go.logits, out_logits.data(), 0, out_logits.size() * sizeof(float));
    ggml_gallocr_free(galloc); ggml_free(ctx);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <laguna.gguf> [--max-ctx N] [--kv q4_0|q8_0|f16] [--chunk N]\n", argv[0]);
        return 2;
    }
    const std::string laguna_path = argv[1];
    int max_ctx = 16384;
    int chunk   = 2048;
    ggml_type kv_type = GGML_TYPE_Q8_0;
    for (int i = 2; i + 1 < argc; ++i) {
        if (!std::strcmp(argv[i], "--max-ctx")) max_ctx = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--chunk")) chunk = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--kv")) {
            std::string s = argv[++i];
            if      (s == "q4_0") kv_type = GGML_TYPE_Q4_0;
            else if (s == "q5_0") kv_type = GGML_TYPE_Q5_0;
            else if (s == "q8_0") kv_type = GGML_TYPE_Q8_0;
            else if (s == "f16")  kv_type = GGML_TYPE_F16;
        }
    }
    const bool no_mask = (std::getenv("DFLASH_NO_MASK") != nullptr);

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    LagunaTargetWeights w;
    if (!load_target_gguf_laguna(laguna_path, backend, w)) {
        std::fprintf(stderr, "load failed: %s\n", dflash27b_last_error());
        ggml_backend_free(backend); return 1;
    }

    LagunaTargetCache cache;
    cache.kv_k_type = kv_type;
    cache.kv_v_type = kv_type;
    if (!create_laguna_target_cache(w, max_ctx, backend, cache)) {
        std::fprintf(stderr, "cache failed: %s\n", dflash27b_last_error());
        free_laguna_target_weights(w); ggml_backend_free(backend); return 1;
    }

    std::printf("[laguna-daemon] ready vocab=%lld eos=%d eot=%d max_ctx=%d kv=%s chunk=%d\n",
                (long long)w.embedder.n_vocab, w.eos_id, w.eos_chat_id, max_ctx,
                ggml_type_name(kv_type), chunk);
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "quit" || line == "exit") break;
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;
        if (cmd != "generate") {
            std::fprintf(stderr, "[laguna-daemon] unknown cmd: %s\n", line.c_str());
            std::printf("err unknown_command\n"); std::fflush(stdout);
            continue;
        }
        std::string in_path, out_path;
        int n_gen = 0;
        iss >> in_path >> n_gen >> out_path;
        if (in_path.empty() || out_path.empty() || n_gen <= 0) {
            std::fprintf(stderr, "[laguna-daemon] bad: %s\n", line.c_str());
            std::printf("err bad_args\n"); std::fflush(stdout);
            continue;
        }

        auto prompt = read_counted_i32(in_path);
        if (prompt.empty()) {
            std::printf("err empty_prompt\n"); std::fflush(stdout); continue;
        }
        const int N = (int)prompt.size();
        if (N + n_gen > max_ctx) {
            std::fprintf(stderr, "[laguna-daemon] N+n_gen=%d > max_ctx=%d\n", N + n_gen, max_ctx);
            std::printf("err overflow\n"); std::fflush(stdout); continue;
        }

        // Reset cache (fresh request).
        reset_laguna_target_cache(cache);

        // Embed full prompt host-side via CpuEmbedder.
        std::vector<float> embed_pf((size_t)N * w.n_embd);
        if (!w.embedder.embed(prompt.data(), N, embed_pf.data())) {
            std::printf("err embed_prefill\n"); std::fflush(stdout); continue;
        }

        // Chunked prefill.
        auto t_pf0 = std::chrono::steady_clock::now();
        std::vector<float> last_logits;
        bool ok = true;
        const int n_chunks = (N + chunk - 1) / chunk;
        for (int c = 0; c < n_chunks && ok; ++c) {
            const int kv_start = c * chunk;
            const int n_tok    = std::min(chunk, N - c * chunk);
            ok = laguna_step(backend, w, cache, embed_pf.data() + (size_t)kv_start * w.n_embd,
                              n_tok, kv_start, no_mask, last_logits);
        }
        auto t_pf1 = std::chrono::steady_clock::now();
        if (!ok) { std::printf("err prefill\n"); std::fflush(stdout); continue; }
        const double pf_s = std::chrono::duration<double>(t_pf1 - t_pf0).count();

        // Argmax sample first generated token.
        auto argmax = [&](const std::vector<float> & ll) {
            int best = 0; float bv = ll[0];
            for (size_t i = 1; i < ll.size(); ++i) if (ll[i] > bv) { bv = ll[i]; best = (int)i; }
            return best;
        };
        int next_tok = argmax(last_logits);
        std::vector<int32_t> generated;
        generated.reserve(n_gen);

        std::vector<float> embed_step((size_t)w.n_embd);
        auto t_g0 = std::chrono::steady_clock::now();
        for (int s = 0; s < n_gen; ++s) {
            if (next_tok == w.eos_id || next_tok == w.eos_chat_id) break;
            generated.push_back(next_tok);
            if (!w.embedder.embed(&next_tok, 1, embed_step.data())) {
                ok = false; break;
            }
            std::vector<float> step_logits;
            if (!laguna_step(backend, w, cache, embed_step.data(), 1, cache.cur_pos, no_mask, step_logits)) {
                ok = false; break;
            }
            next_tok = argmax(step_logits);
        }
        auto t_g1 = std::chrono::steady_clock::now();
        const double g_s = std::chrono::duration<double>(t_g1 - t_g0).count();

        if (!write_counted_i32(out_path, generated)) {
            std::printf("err write_out\n"); std::fflush(stdout); continue;
        }
        std::printf("ok N=%d gen=%zu prefill_s=%.3f decode_s=%.3f decode_tok_s=%.1f out=%s\n",
                    N, generated.size(), pf_s, g_s,
                    generated.size() / std::max(1e-9, g_s), out_path.c_str());
        std::fflush(stdout);
    }

    free_laguna_target_cache(cache);
    free_laguna_target_weights(w);
    ggml_backend_free(backend);
    return 0;
}
