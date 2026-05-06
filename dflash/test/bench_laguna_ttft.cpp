// TTFT (time-to-first-token) bench for Laguna in dflash. Loads Laguna Q4_K_M,
// allocates a cache sized for the longest prefill, then for each context
// length: builds a prefill graph for N tokens (synthetic input), runs it on
// CUDA, measures wall time. Reports TTFT @ each length.
//
// Usage:
//   bench_laguna_ttft <laguna.gguf> ["4096,16384,32768"]
//
// The synthetic input uses token id 1972 repeated N times (avoids BOS
// special-casing; any non-special id works, the bench measures wall time not
// generation quality).
//
// On RTX 3090 24 GB the practical ceiling without KV bit-reduction:
//   Q8_0 KV  + 18.77 GiB weights -> ~32K context
//   For 64K+ need Q4_0 KV (planned, not in this bench).

#include "laguna_internal.h"
#include "internal.h"
#include "dflash27b.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"

using namespace dflash27b;

static std::vector<int> parse_csv(const std::string & s, std::vector<int> dflt) {
    if (s.empty()) return dflt;
    std::vector<int> out;
    size_t start = 0;
    while (start < s.size()) {
        size_t comma = s.find(',', start);
        std::string tok = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <laguna.gguf> [\"4096,16384,32768\"]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const std::string lens_csv = (argc >= 3) ? argv[2] : "";
    std::vector<int> ctx_lens = parse_csv(lens_csv, {1024, 4096, 16384});
    int max_len = 0;
    for (int n : ctx_lens) if (n > max_len) max_len = n;

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    LagunaTargetWeights w;
    if (!load_target_gguf_laguna(path, backend, w)) {
        std::fprintf(stderr, "load failed: %s\n", dflash27b_last_error());
        ggml_backend_free(backend); return 1;
    }

    LagunaTargetCache cache;
    if (!create_laguna_target_cache(w, max_len, backend, cache)) {
        std::fprintf(stderr, "cache failed: %s\n", dflash27b_last_error());
        free_laguna_target_weights(w); ggml_backend_free(backend); return 1;
    }
    std::printf("[bench] cache max_ctx=%d  KV bytes/layer ~ %.1f MiB\n",
                max_len, (2.0 * w.head_dim * max_len * w.n_head_kv) / (1024.0 * 1024.0));

    const int32_t fake_tok = 1972;  // "hello" or whatever; just a non-special id

    // Chunked prefill: at large N a single forward exceeds the 24 GB activation
    // budget on RTX 3090 (MoE intermediate [n_embd, n_used, n_tokens] = 1 GB at
    // n_tokens=16K). Split N into CHUNK chunks, advance kv_start per chunk.
    int chunk_env = 0;
    if (const char * c = std::getenv("DFLASH_CHUNK")) chunk_env = std::atoi(c);
    const int CHUNK = chunk_env > 0 ? chunk_env : 4096;

    for (int N : ctx_lens) {
        if (N > max_len) { std::printf("[bench] skip N=%d > max_len=%d\n", N, max_len); continue; }
        reset_laguna_target_cache(cache);

        // Build embedding tensor for full N tokens upfront.
        std::vector<int32_t> ids((size_t)N, fake_tok);
        std::vector<float> embed_full((size_t)N * w.n_embd);
        if (!w.embedder.embed(ids.data(), N, embed_full.data())) {
            std::fprintf(stderr, "embed failed at N=%d\n", N);
            continue;
        }
        const int chunk = std::min(N, CHUNK);

        // Reusable per-chunk graph context. Reset/recreate per chunk; we reuse
        // gallocr across chunks since the topology is identical.
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead() + 16 * 1024 * 1024;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);

        ggml_tensor * inp_embed = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w.n_embd, chunk, 1);
        ggml_set_name(inp_embed, "inp_embed");
        ggml_set_input(inp_embed);

        ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, chunk);
        ggml_set_name(positions, "positions");
        ggml_set_input(positions);

        // Causal mask. Skip if DFLASH_NO_MASK set (semantically wrong but useful
        // to isolate FA kernel issues). ggml_flash_attn_ext expects mask shape:
        //   ne[0] = n_kv (no padding required here)
        //   ne[1] = n_tokens padded to GGML_KQ_MASK_PAD (64)
        // F16 dtype. Row-major: row stride = ne[0] elements.
        const bool no_mask = (std::getenv("DFLASH_NO_MASK") != nullptr);
        // llama.cpp build_attn_inp_kq_mask convention:
        //   ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, n_tokens, 1, n_stream)
        // Single stream so n_stream=1 -> shape [n_kv, n_tokens, 1, 1].
        // Mask shape varies per chunk: ne[0] = kv_len so far (kv_start + chunk),
        // ne[1] = chunk. Allocate the largest mask up front and use views per
        // chunk if size needs to grow. For simplicity, build per-chunk mask
        // shaped to MAX possible kv_len (= N) so the graph topology is uniform.
        // Mask cells beyond actual kv_len at any chunk are -inf.
        ggml_tensor * mask = nullptr;
        ggml_tensor * mask_cnv = nullptr;
        if (!no_mask && chunk > 1) {
            mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, N, chunk, 1, 1);
            ggml_set_name(mask, "causal_mask_f32");
            ggml_set_input(mask);
            mask_cnv = ggml_cast(ctx, mask, GGML_TYPE_F16);
            ggml_set_name(mask_cnv, "causal_mask_f16");
        }

        // We will run multiple chunks; each one uses these tensor handles. The
        // graph builder is per-chunk (we set output_last_only on the LAST chunk).
        // Build N/chunk forwards.
        const int n_chunks = (N + chunk - 1) / chunk;
        std::vector<LagunaGraphOutputs> outs(n_chunks);
        std::vector<int> chunk_kv_start(n_chunks);
        std::vector<int> chunk_n(n_chunks);
        for (int c = 0; c < n_chunks; ++c) {
            chunk_kv_start[c] = c * chunk;
            chunk_n[c]        = std::min(chunk, N - c * chunk);
        }

        // For now keep things simple: rebuild graph per chunk (gallocr alloc per
        // chunk; topology may differ in last-chunk size).
        // Time only the SUM of all chunks as TTFT.
        bool ok = true;
        ggml_gallocr_t galloc = nullptr;
        double total_pf_s = 0.0;

        // unused single-graph upload removed (handled in chunk loop below)
        // No-op: mask + inputs are filled per chunk below.

        for (int c = 0; c < n_chunks && ok; ++c) {
            const int kv_start = chunk_kv_start[c];
            const int n_tok    = chunk_n[c];
            const bool last    = (c == n_chunks - 1);

            // Per-chunk graph (rebuilt fresh; reusing gallocr saves planning cost).
            ggml_init_params ip2{};
            ip2.mem_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead() + 16 * 1024 * 1024;
            ip2.no_alloc = true;
            ggml_context * ctx2 = ggml_init(ip2);
            ggml_cgraph * gf2 = ggml_new_graph_custom(ctx2, 16384, false);

            ggml_tensor * ie = ggml_new_tensor_3d(ctx2, GGML_TYPE_F32, w.n_embd, n_tok, 1);
            ggml_set_input(ie);
            ggml_tensor * pp = ggml_new_tensor_1d(ctx2, GGML_TYPE_I32, n_tok);
            ggml_set_input(pp);
            ggml_tensor * mk = nullptr, * mkc = nullptr;
            if (!no_mask && n_tok > 1) {
                const int kv_len = kv_start + n_tok;
                mk = ggml_new_tensor_4d(ctx2, GGML_TYPE_F32, kv_len, n_tok, 1, 1);
                ggml_set_input(mk);
                mkc = ggml_cast(ctx2, mk, GGML_TYPE_F16);
            }

            LagunaGraphInputs gi2{};
            gi2.inp_embed       = ie;
            gi2.positions       = pp;
            gi2.attn_mask       = mkc;
            gi2.n_tokens        = n_tok;
            gi2.kv_start        = kv_start;
            gi2.output_logits   = last;  // only need logits at last chunk
            gi2.output_last_only= last;

            LagunaGraphOutputs go2 = build_laguna_graph(ctx2, gf2, w, cache, gi2);

            if (!galloc) galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(galloc, gf2)) {
                std::fprintf(stderr, "gallocr_alloc N=%d chunk=%d failed\n", N, c);
                ok = false; ggml_free(ctx2); break;
            }

            // Upload chunk inputs.
            ggml_backend_tensor_set(ie, embed_full.data() + (size_t)kv_start * w.n_embd,
                                     0, (size_t)n_tok * w.n_embd * sizeof(float));
            std::vector<int32_t> ppos(n_tok);
            for (int i = 0; i < n_tok; ++i) ppos[i] = kv_start + i;
            ggml_backend_tensor_set(pp, ppos.data(), 0, ppos.size() * sizeof(int32_t));
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

            // Warm + timed compute (warm only on first chunk to amortize JIT).
            if (c == 0) {
                ggml_backend_graph_compute(backend, gf2);
                ggml_backend_synchronize(backend);
            }
            auto tA = std::chrono::steady_clock::now();
            ggml_status st2 = ggml_backend_graph_compute(backend, gf2);
            ggml_backend_synchronize(backend);
            auto tB = std::chrono::steady_clock::now();
            if (st2 != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "compute N=%d chunk=%d failed status=%d\n", N, c, (int)st2);
                ok = false; ggml_free(ctx2); break;
            }
            cache.cur_pos = kv_start + n_tok;
            total_pf_s += std::chrono::duration<double>(tB - tA).count();

            if (last) outs[c] = go2;
            // Note: gf2/ctx2 keep tensor pointers; logits read happens before free below.
            if (last) {
                // Pull logits before freeing ctx (logits is a view in ctx2).
                // Stash into a local vector for the post-loop diagnostics block.
                // We just abuse the original variable scope.
                if (go2.logits) {
                    const int64_t vocab = go2.logits->ne[0];
                    std::vector<float> ll((size_t)vocab);
                    ggml_backend_tensor_get(go2.logits, ll.data(), 0, ll.size() * sizeof(float));
                    int best = 0; float bv = ll[0]; int n_inf = 0, n_nan = 0;
                    for (int i = 0; i < (int)vocab; ++i) {
                        float v = ll[i];
                        if (std::isnan(v)) ++n_nan;
                        if (std::isinf(v)) ++n_inf;
                        if (v > bv) { bv = v; best = i; }
                    }
                    std::printf("[bench] N=%6d  TTFT=%8.3f s  (%6.1f tok/s) chunks=%d  argmax=%d  logit=%.3f  nan=%d inf=%d\n",
                                N, total_pf_s, N / std::max(1e-9, total_pf_s), n_chunks, best, bv, n_nan, n_inf);
                }
            }
            ggml_free(ctx2);
        }
        if (galloc) ggml_gallocr_free(galloc);
        ggml_free(ctx);
        if (!ok) continue;
        const double pf_s = total_pf_s;
        const double sync0 = 0.0;

        (void)pf_s; (void)sync0;
    }

    free_laguna_target_cache(cache);
    free_laguna_target_weights(w);
    ggml_backend_free(backend);
    return 0;
}
