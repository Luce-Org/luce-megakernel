// Laguna autoregressive decode bench. Loads Laguna, prefills N synthetic
// tokens, then generates n_gen tokens one at a time via greedy argmax and
// writes them to the cache. Reports prefill TTFT + decode tok/s.
//
// Proves the autoregressive decode kernel (n_tokens=1, kv_start=cur_pos,
// last-only logits, sample, embed, repeat) works for Laguna in dflash.
//
// Cross-tokenizer (Qwen3 -> Laguna) and string I/O are deferred; this bench
// uses a fake bos id for prefill seeding.
//
// Usage: bench_laguna_generate <laguna.gguf> [prompt_N=128] [n_gen=64]

#include "laguna_internal.h"
#include "internal.h"
#include "dflash27b.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"

using namespace dflash27b;

// Build + run a single forward step (n_tokens=1 typically) at the given
// kv_start, returns the last-token logits via host vector. Used for both
// prefill (multi-token first call) and per-token decode (n_tok=1).
static bool laguna_step_logits(
    ggml_backend_t backend,
    const LagunaTargetWeights & w,
    LagunaTargetCache & cache,
    const float * embed,        // [n_tok, n_embd] f32
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
    ggml_tensor * mk = nullptr, * mkc = nullptr;
    if (!no_mask && n_tok > 1) {
        const int kv_len = kv_start + n_tok;
        mk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kv_len, n_tok, 1, 1);
        ggml_set_input(mk);
        mkc = ggml_cast(ctx, mk, GGML_TYPE_F16);
    }

    LagunaGraphInputs gi{};
    gi.inp_embed       = ie;
    gi.positions       = pp;
    gi.attn_mask       = mkc;
    gi.n_tokens        = n_tok;
    gi.kv_start        = kv_start;
    gi.output_logits   = true;
    gi.output_last_only= true;

    LagunaGraphOutputs go = build_laguna_graph(ctx, gf, w, cache, gi);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "gallocr_alloc step n_tok=%d failed\n", n_tok);
        ggml_gallocr_free(galloc); ggml_free(ctx); return false;
    }

    ggml_backend_tensor_set(ie, embed, 0, (size_t)n_tok * w.n_embd * sizeof(float));
    std::vector<int32_t> pos(n_tok);
    for (int i = 0; i < n_tok; ++i) pos[i] = kv_start + i;
    ggml_backend_tensor_set(pp, pos.data(), 0, pos.size() * sizeof(int32_t));
    if (mk) {
        const int kv_len = kv_start + n_tok;
        std::vector<float> mb((size_t)kv_len * n_tok, 0.0f);
        for (int t = 0; t < n_tok; ++t) {
            const int abs_q = kv_start + t;
            for (int j = 0; j < kv_len; ++j) {
                if (j > abs_q) mb[(size_t)t * kv_len + j] = -INFINITY;
            }
        }
        ggml_backend_tensor_set(mk, mb.data(), 0, mb.size() * sizeof(float));
    }

    ggml_status st = ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(galloc); ggml_free(ctx); return false;
    }
    cache.cur_pos = kv_start + n_tok;

    const int64_t vocab = go.logits->ne[0];
    out_logits.resize((size_t)vocab);
    ggml_backend_tensor_get(go.logits, out_logits.data(), 0, out_logits.size() * sizeof(float));
    ggml_gallocr_free(galloc); ggml_free(ctx);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <laguna.gguf> [prompt_N=128] [n_gen=64]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int prompt_N = (argc >= 3) ? std::atoi(argv[2]) : 128;
    const int n_gen    = (argc >= 4) ? std::atoi(argv[3]) : 64;
    const bool no_mask = (std::getenv("DFLASH_NO_MASK") != nullptr);

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    LagunaTargetWeights w;
    if (!load_target_gguf_laguna(path, backend, w)) {
        std::fprintf(stderr, "load failed: %s\n", dflash27b_last_error());
        ggml_backend_free(backend); return 1;
    }

    LagunaTargetCache cache;
    if (const char * kv_t = std::getenv("DFLASH_KV_TYPE")) {
        const std::string s = kv_t;
        if      (s == "q4_0") { cache.kv_k_type = GGML_TYPE_Q4_0; cache.kv_v_type = GGML_TYPE_Q4_0; }
        else if (s == "q5_0") { cache.kv_k_type = GGML_TYPE_Q5_0; cache.kv_v_type = GGML_TYPE_Q5_0; }
        else if (s == "q8_0") { cache.kv_k_type = GGML_TYPE_Q8_0; cache.kv_v_type = GGML_TYPE_Q8_0; }
        else if (s == "f16")  { cache.kv_k_type = GGML_TYPE_F16;  cache.kv_v_type = GGML_TYPE_F16;  }
    }
    const int max_ctx = prompt_N + n_gen + 16;
    if (!create_laguna_target_cache(w, max_ctx, backend, cache)) {
        std::fprintf(stderr, "cache failed: %s\n", dflash27b_last_error());
        free_laguna_target_weights(w); ggml_backend_free(backend); return 1;
    }

    // ---- Prefill: BOS + (prompt_N - 1) fake tokens -------------------
    const int32_t bos     = (w.bos_id >= 0) ? w.bos_id : 2;
    const int32_t fake_id = 1972;  // arbitrary non-special id
    std::vector<int32_t> prompt(prompt_N);
    prompt[0] = bos;
    for (int i = 1; i < prompt_N; ++i) prompt[i] = fake_id;

    std::vector<float> embed_pf((size_t)prompt_N * w.n_embd);
    if (!w.embedder.embed(prompt.data(), prompt_N, embed_pf.data())) {
        std::fprintf(stderr, "embed prefill failed\n");
        free_laguna_target_cache(cache); free_laguna_target_weights(w); ggml_backend_free(backend); return 1;
    }

    auto t_pf0 = std::chrono::steady_clock::now();
    std::vector<float> last_logits;
    if (!laguna_step_logits(backend, w, cache, embed_pf.data(), prompt_N, 0, no_mask, last_logits)) {
        std::fprintf(stderr, "prefill failed\n");
        free_laguna_target_cache(cache); free_laguna_target_weights(w); ggml_backend_free(backend); return 1;
    }
    auto t_pf1 = std::chrono::steady_clock::now();
    const double pf_s = std::chrono::duration<double>(t_pf1 - t_pf0).count();

    // ---- Argmax sampler --------------------------------------------
    auto argmax = [&](const std::vector<float> & logits) -> int {
        int best = 0; float bv = logits[0];
        for (size_t i = 1; i < logits.size(); ++i) if (logits[i] > bv) { bv = logits[i]; best = (int)i; }
        return best;
    };

    int next_tok = argmax(last_logits);
    std::printf("[gen] prefill N=%d in %.3fs (%.1f tok/s)  first_argmax=%d\n",
                prompt_N, pf_s, prompt_N / std::max(1e-9, pf_s), next_tok);

    // ---- Autoregressive decode loop -------------------------------
    std::vector<int32_t> generated;
    generated.reserve(n_gen);
    std::vector<float> embed_step((size_t)w.n_embd);
    auto t_g0 = std::chrono::steady_clock::now();
    int n_decoded = 0;
    for (int s = 0; s < n_gen; ++s) {
        if (next_tok == w.eos_id || next_tok == w.eos_chat_id) {
            std::printf("[gen] EOS at step %d\n", s); break;
        }
        generated.push_back(next_tok);
        if (!w.embedder.embed(&next_tok, 1, embed_step.data())) {
            std::fprintf(stderr, "embed step %d (id=%d) failed\n", s, next_tok); break;
        }
        std::vector<float> step_logits;
        if (!laguna_step_logits(backend, w, cache, embed_step.data(), 1, cache.cur_pos,
                                  no_mask, step_logits)) {
            std::fprintf(stderr, "decode step %d failed\n", s); break;
        }
        next_tok = argmax(step_logits);
        ++n_decoded;
    }
    auto t_g1 = std::chrono::steady_clock::now();
    const double g_s = std::chrono::duration<double>(t_g1 - t_g0).count();

    std::printf("[gen] decoded %d tokens in %.3fs (%.1f tok/s)\n",
                n_decoded, g_s, n_decoded / std::max(1e-9, g_s));
    std::printf("[gen] generated ids:");
    for (int t : generated) std::printf(" %d", t);
    std::printf("\n");

    free_laguna_target_cache(cache);
    free_laguna_target_weights(w);
    ggml_backend_free(backend);
    return 0;
}
