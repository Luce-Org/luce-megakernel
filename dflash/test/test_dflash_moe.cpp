// DFlash speculative decode integration test for the MoE target model.
//
// Implements the chain-verify spec-decode loop:
//   prefill → [draft → snapshot → verify → accept → restore → replay] × N
//
// Uses:
//   - Target: run_qwen35moe_forward() (imperative, layer-by-layer)
//   - Drafter: build_draft_graph() via standard ggml graph
//   - Prompt: pre-tokenized binary file (like test_dflash.cpp)
//
// Usage: test_dflash_moe <qwen35moe.gguf> <draft.gguf> <prompt_ids.bin> [n_gen] [n_cache_slots]

#include "dflash27b.h"
#include "internal.h"
#include "dflash_graph.h"
#include "expert_cache.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Debug sync flag (defined in qwen35moe_target_graph.cpp)
extern bool g_moe_debug_sync;
extern int  g_moe_debug_call;
#include <chrono>
#include <algorithm>
#include <string>
#include <fstream>

using namespace dflash27b;

// External CUDA kernel (defined in f16_convert.cu)
extern "C" void dflash27b_launch_bf16_to_f32(const void * src, void * dst,
                                              size_t n_elems, cudaStream_t stream);

// ─── Helpers ──────────────────────────────────────────────────────────────
static std::vector<int32_t> read_int32_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<int32_t> out(sz / sizeof(int32_t));
    f.read((char *)out.data(), sz);
    return out;
}

// ─── Constants ────────────────────────────────────────────────────────────
static constexpr int BLOCK_SIZE     = DFLASH27B_DRAFT_BLOCK_SIZE;  // 16
static constexpr int N_TARGET_FEATS = DFLASH27B_DRAFT_N_TARGET_LAYERS;  // 5
static constexpr int KQ_MASK_PAD    = 32;
static constexpr uint16_t F16_ZERO    = 0x0000;
static constexpr uint16_t F16_NEG_INF = 0xFC00;

static int align_up(int x, int a) { return ((x + a - 1) / a) * a; }

// ─── Causal attention mask ───────────────────────────────────────────────
// kv_len: number of active KV slots visible to queries.
// n_tokens: number of query positions (q_len).
// kv_start: absolute position of the first query token.
// win_start: start of the FA window (0 = no window).
static void build_causal_mask(std::vector<uint16_t> & out,
                              int kv_len, int n_tokens, int kv_start,
                              int win_start = 0) {
    const int kv_pad = align_up(kv_len, KQ_MASK_PAD);
    const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
    out.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    const int abs_end = win_start + kv_len;
    for (int q = 0; q < n_tokens; q++) {
        const int abs_q = kv_start + q;
        for (int k = win_start; k <= abs_q && k < abs_end; k++) {
            out[(size_t)q * kv_pad + (k - win_start)] = F16_ZERO;
        }
    }
}

static int argmax_f32(const float * data, int n) {
    int best = 0;
    float best_v = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] > best_v) { best_v = data[i]; best = i; }
    }
    return best;
}

// ─── Draft step (builds a draft forward graph) ──────────────────────────
struct DraftCtx {
    ggml_context * ctx     = nullptr;
    ggml_cgraph  * gf      = nullptr;
    ggml_gallocr_t alloc   = nullptr;
    ggml_tensor * inp_embed         = nullptr;
    ggml_tensor * target_hidden_cat = nullptr;
    ggml_tensor * positions_q       = nullptr;
    ggml_tensor * positions_k       = nullptr;
    ggml_tensor * logits            = nullptr;
    ggml_tensor * argmax_tokens     = nullptr;
};

static void draft_ctx_free(DraftCtx & d) {
    if (d.alloc) { ggml_gallocr_free(d.alloc); d.alloc = nullptr; }
    if (d.ctx)   { ggml_free(d.ctx); d.ctx = nullptr; }
    d.gf = nullptr;
}

static bool build_draft_step(DraftCtx & d,
                             const DraftWeights & dw,
                             const TargetWeights & tw,
                             ggml_backend_t backend,
                             int ctx_len) {
    draft_ctx_free(d);

    ggml_init_params ip{};
    ip.mem_size   = 256 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    d.ctx = ggml_init(ip);
    if (!d.ctx) return false;

    const int hidden = dw.n_embd;   // e.g. 2048 for MoE drafter
    const int q_len  = BLOCK_SIZE;
    const int fc_in  = N_TARGET_FEATS * tw.n_embd;  // 5 * target_hidden

    d.inp_embed = ggml_new_tensor_3d(d.ctx, GGML_TYPE_F32, hidden, q_len, 1);
    ggml_set_name(d.inp_embed, "inp_embed");
    ggml_set_input(d.inp_embed);

    d.target_hidden_cat = ggml_new_tensor_3d(d.ctx, GGML_TYPE_F32, fc_in, ctx_len, 1);
    ggml_set_name(d.target_hidden_cat, "target_hidden_cat");
    ggml_set_input(d.target_hidden_cat);

    d.positions_q = ggml_new_tensor_1d(d.ctx, GGML_TYPE_I32, q_len);
    ggml_set_name(d.positions_q, "positions_q");
    ggml_set_input(d.positions_q);

    d.positions_k = ggml_new_tensor_1d(d.ctx, GGML_TYPE_I32, ctx_len + q_len);
    ggml_set_name(d.positions_k, "positions_k");
    ggml_set_input(d.positions_k);

    d.gf = ggml_new_graph_custom(d.ctx, 4096, false);

    DraftGraphInputs gi{};
    gi.ctx_len           = ctx_len;
    gi.noise_embed       = d.inp_embed;
    gi.target_hidden_cat = d.target_hidden_cat;
    gi.positions_q       = d.positions_q;
    gi.positions_k       = d.positions_k;
    gi.lm_head           = tw.output;  // project through target lm_head
    DraftGraphOutputs go = build_draft_graph(d.ctx, dw, gi);
    d.logits = go.logits;
    if (!d.logits) {
        std::fprintf(stderr, "[draft] build_draft_graph returned null logits\n");
        draft_ctx_free(d);
        return false;
    }

    // Add argmax over logits
    d.argmax_tokens = ggml_argmax(d.ctx, d.logits);
    ggml_set_name(d.argmax_tokens, "draft_argmax");
    ggml_set_output(d.argmax_tokens);
    ggml_build_forward_expand(d.gf, d.argmax_tokens);

    // Allocate
    if (!d.alloc) {
        d.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    if (!ggml_gallocr_alloc_graph(d.alloc, d.gf)) {
        std::fprintf(stderr, "[draft] gallocr alloc failed\n");
        draft_ctx_free(d);
        return false;
    }
    return true;
}

// ─── Main ────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <qwen35moe.gguf> <draft.gguf> <prompt_ids.bin> [n_gen] [n_cache_slots]\n",
            argv[0]);
        return 2;
    }
    const char * target_path = argv[1];
    const char * draft_path  = argv[2];
    const char * prompt_path = argv[3];
    const int    n_gen       = (argc > 4) ? std::atoi(argv[4]) : 64;
    const int    n_cache_slots = (argc > 5) ? std::atoi(argv[5]) : 200;

    std::printf("=== DFlash MoE Spec-Decode Test ===\n");
    std::printf("Target: %s\nDraft:  %s\nPrompt: %s\nn_gen=%d  cache_slots=%d\n\n",
                target_path, draft_path, prompt_path, n_gen, n_cache_slots);

    // ── 1. CUDA backend ──
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "CUDA init failed\n"); return 1; }

    // ── 2. Load target (MoE) ──
    TargetWeights w;
    if (!load_target_gguf(target_path, backend, w)) {
        std::fprintf(stderr, "load_target_gguf: %s\n", dflash27b_last_error());
        return 1;
    }
    if (!w.is_moe) {
        std::fprintf(stderr, "error: target model is not MoE\n");
        return 1;
    }

    const int hidden = w.n_embd;  // 2048
    const int vocab  = (int)w.embedder.n_vocab;
    const int q_len  = BLOCK_SIZE; // 16
    std::printf("[target] hidden=%d vocab=%d n_layer=%d n_experts=%d n_active=%d\n",
                hidden, vocab, w.n_layer, w.n_experts, w.n_experts_active);

    // ── 3. Load drafter ──
    DraftWeights dw;
    if (!load_draft_gguf(draft_path, backend, dw)) {
        std::fprintf(stderr, "load_draft_gguf: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("[draft] hidden=%d n_layer=%d n_head=%d head_dim=%d\n",
                dw.n_embd, dw.n_layer, dw.n_head, dw.head_dim);

    // ── 4. ExpertCache ──
    int n_alt_layers = 0;
    const auto & es = w.expert_source;
    for (int l = 0; l < es.n_layers; l++) {
        if (!es.layer_down_types.empty() && es.layer_down_types[l] != es.down_type)
            n_alt_layers++;
    }
    // Alt slots must hold all unique experts for one layer during a verify batch.
    // Worst case: n_experts_active * max_verify_tokens unique experts per layer.
    // Use same as primary pool to avoid within-layer eviction (alt pool is shared
    // across alt layers processed sequentially, not simultaneously).
    int n_alt_slots = n_alt_layers > 0 ? n_cache_slots : 0;

    ExpertCache ecache;
    if (!ecache.init(backend, w.expert_source, n_cache_slots, n_alt_slots)) {
        std::fprintf(stderr, "ExpertCache init failed\n");
        return 1;
    }
    std::printf("[ecache] slots=%d alt_slots=%d alt_layers=%d\n",
                n_cache_slots, n_alt_slots, n_alt_layers);

    // ── 5. Target cache (prefill-only first, then migrate) ──
    const int max_ctx = 4096;
    const int max_verify = q_len + 1;  // 17 for replay

    TargetCache cache;
    if (!create_target_cache(w, max_ctx, max_verify, backend, cache, /*prefill_only=*/true)) {
        std::fprintf(stderr, "create_target_cache: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("[cache] max_ctx=%d target_feat_cap=%d\n", max_ctx, cache.target_feat_cap);

    // ── 6. MoeState ──
    MoeState moe{};
    if (!create_moe_state(backend, hidden, w.n_experts_active, max_verify, moe)) {
        std::fprintf(stderr, "create_moe_state failed\n");
        return 1;
    }

    // ── 7. Activation buffers ──
    ggml_init_params act_ip{};
    act_ip.mem_size   = 8 * ggml_tensor_overhead() + 4096;
    act_ip.mem_buffer = nullptr;
    act_ip.no_alloc   = true;
    ggml_context * act_ctx = ggml_init(act_ip);
    ggml_tensor * act_a = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, max_verify);
    ggml_tensor * act_b = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, max_verify);
    ggml_set_name(act_a, "act_a");
    ggml_set_name(act_b, "act_b");
    ggml_backend_buffer_t act_buf = ggml_backend_alloc_ctx_tensors(act_ctx, backend);
    if (!act_buf) { std::fprintf(stderr, "act alloc failed\n"); return 1; }

    // ── 8. Output tensors (logits + argmax) ──
    ggml_init_params out_ip{};
    out_ip.mem_size   = 8 * ggml_tensor_overhead() + 4096;
    out_ip.mem_buffer = nullptr;
    out_ip.no_alloc   = true;
    ggml_context * out_ctx = ggml_init(out_ip);
    ggml_tensor * logits_out = ggml_new_tensor_2d(out_ctx, GGML_TYPE_F32, vocab, max_verify);
    ggml_tensor * argmax_out = ggml_new_tensor_1d(out_ctx, GGML_TYPE_I32, max_verify);
    ggml_set_name(logits_out, "logits_out");
    ggml_set_name(argmax_out, "argmax_out");
    ggml_backend_buffer_t out_buf = ggml_backend_alloc_ctx_tensors(out_ctx, backend);
    if (!out_buf) { std::fprintf(stderr, "out alloc failed\n"); return 1; }

    // ── 9. Positions + mask tensors (persistent, reused across calls) ──
    ggml_init_params pos_ip{};
    pos_ip.mem_size   = 8 * ggml_tensor_overhead() + 4096;
    pos_ip.mem_buffer = nullptr;
    pos_ip.no_alloc   = true;
    ggml_context * pos_ctx = ggml_init(pos_ip);
    ggml_tensor * positions = ggml_new_tensor_1d(pos_ctx, GGML_TYPE_I32, 4 * max_verify);
    ggml_set_name(positions, "positions");
    // Mask sized for worst-case: kv_pad for max_ctx+max_verify, q_pad for max_verify
    ggml_tensor * attn_mask = ggml_new_tensor_2d(pos_ctx, GGML_TYPE_F16,
        align_up(max_ctx + max_verify, KQ_MASK_PAD),
        align_up(max_verify, KQ_MASK_PAD));
    ggml_set_name(attn_mask, "attn_mask");
    ggml_backend_buffer_t pos_buf = ggml_backend_alloc_ctx_tensors(pos_ctx, backend);
    if (!pos_buf) { std::fprintf(stderr, "pos alloc failed\n"); return 1; }

    // ── 10. Load prompt tokens from binary file ──
    std::vector<int32_t> prompt_tokens = read_int32_file(prompt_path);
    if (prompt_tokens.empty()) {
        std::fprintf(stderr, "failed to read prompt from: %s\n", prompt_path);
        return 1;
    }
    std::printf("[prompt] %d tokens\n", (int)prompt_tokens.size());

    // ── 11. Prefill (sequential single-token) ──
    auto t_pf0 = std::chrono::steady_clock::now();
    int committed = 0;
    int last_tok = -1;

    for (int i = 0; i < (int)prompt_tokens.size(); i++) {
        const int kv_start = i;
        const int nt = 1;

        // Embed
        std::vector<float> embed(hidden);
        if (!w.embedder.embed(&prompt_tokens[i], 1, embed.data())) {
            std::fprintf(stderr, "embed failed @%d\n", i);
            return 1;
        }
        ggml_backend_tensor_set(act_a, embed.data(), 0, sizeof(float) * hidden);

        // Positions: M-RoPE [4*1] with stride=1
        std::vector<int32_t> pos4 = {i, i, i, 0};
        positions->ne[0] = 4 * nt;  // must match 4*n_tokens for ggml_rope_multi
        ggml_backend_tensor_set(positions, pos4.data(), 0, sizeof(int32_t) * 4);

        // Mask: for n_tokens=1, only needed if kv_pad > kv_len (TBQ).
        // For simplicity and correctness, always provide mask when kv_len > 1.
        const int kv_len = kv_start + nt;
        std::vector<uint16_t> mask_data;
        ggml_tensor * mask_ptr = nullptr;
        if (kv_len > 1) {
            build_causal_mask(mask_data, kv_len, nt, kv_start);
            const int kv_pad = align_up(kv_len, KQ_MASK_PAD);
            const int q_pad  = align_up(nt, KQ_MASK_PAD);
            attn_mask->ne[0] = kv_pad;
            attn_mask->ne[1] = q_pad;
            attn_mask->nb[1] = (size_t)kv_pad * ggml_element_size(attn_mask);
            ggml_backend_tensor_set(attn_mask, mask_data.data(), 0,
                sizeof(uint16_t) * kv_pad * q_pad);
            mask_ptr = attn_mask;
        }

        bool ok = run_qwen35moe_forward(
            backend, w, cache, ecache, moe, w.expert_source,
            act_a, act_b, nt, positions, mask_ptr,
            kv_start, /*capture=*/true, /*fa_window=*/0,
            logits_out, nullptr);
        if (!ok) {
            std::fprintf(stderr, "prefill forward failed @%d\n", i);
            return 1;
        }

        // Read last position logits → argmax
        std::vector<float> logit_buf(vocab);
        ggml_backend_tensor_get(logits_out, logit_buf.data(), 0, sizeof(float) * vocab);
        last_tok = argmax_f32(logit_buf.data(), vocab);
        committed = kv_start + nt;

        if (i % 10 == 0 || i == (int)prompt_tokens.size() - 1) {
            std::printf("  prefill %d/%d last_tok=%d\r",
                i + 1, (int)prompt_tokens.size(), last_tok);
            std::fflush(stdout);
        }
    }

    auto t_pf1 = std::chrono::steady_clock::now();
    double pf_ms = std::chrono::duration<double, std::milli>(t_pf1 - t_pf0).count();
    std::printf("\n[prefill] %d tokens in %.1f ms (%.1f tok/s), last_tok=%d\n",
                committed, pf_ms, committed * 1000.0 / pf_ms, last_tok);

    // ── 12. Migrate cache for rollback ──
    if (!migrate_prefill_cache(w, max_ctx, max_verify, backend, cache)) {
        std::fprintf(stderr, "migrate_prefill_cache: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("[migrate] done\n");

    // Verify snapshot buffers exist
    {
        bool snap_ok = true;
        for (size_t i = 0; i < cache.ssm_state.size(); i++) {
            if (cache.ssm_state[i] && !cache.ssm_state_snap[i]) { snap_ok = false; break; }
        }
        if (!snap_ok) {
            std::fprintf(stderr, "ERROR: snapshot buffers not allocated after migrate\n");
            return 1;
        }
    }

    // ── 13. DFlash decode loop ──
    g_moe_debug_sync = false;  // Disable per-layer CUDA sync for perf (set true to debug)
    auto t_gen0 = std::chrono::steady_clock::now();
    int n_generated = 0, n_draft_steps = 0, n_accept_sum = 0;
    std::vector<int32_t> out_all;

    // Pre-allocate buffers
    std::vector<float>    noise_embed_buf((size_t)hidden * q_len);
    std::vector<int32_t>  noise_ids(q_len);
    std::vector<int32_t>  draft_tok(q_len);
    std::vector<int32_t>  target_tok(q_len);

    // Timing accumulators (microseconds)
    double tt_draft = 0, tt_snap = 0, tt_verify = 0, tt_accept = 0;
    double tt_restore = 0, tt_replay = 0;

    const int mask_tok = DFLASH27B_DRAFT_MASK_TOKEN_ID;  // 248070
    std::printf("[decode] mask_tok=%d\n", mask_tok);

    // Draft context (built once per ctx_len, rebuilt when ctx_len changes)
    DraftCtx dctx;
    int draft_ctx_len = -1;

    constexpr int DRAFT_CTX_MAX = 2048;

    while (n_generated < n_gen) {
        auto T0 = std::chrono::steady_clock::now();

        // ─── 1. Draft: noise block [last_tok, MASK×15] ───
        noise_ids[0] = last_tok;
        for (int i = 1; i < q_len; i++) noise_ids[i] = mask_tok;
        if (!w.embedder.embed(noise_ids.data(), q_len, noise_embed_buf.data())) {
            std::fprintf(stderr, "noise embed failed\n"); return 1;
        }

        // Draft target attention window
        const int cur_draft_ctx = std::min(committed, DRAFT_CTX_MAX);
        const int draft_start   = committed - cur_draft_ctx;

        // Rebuild draft graph if ctx_len changed
        if (cur_draft_ctx != draft_ctx_len) {
            if (!build_draft_step(dctx, dw, w, backend, cur_draft_ctx)) {
                std::fprintf(stderr, "build_draft_step failed\n"); return 1;
            }
            draft_ctx_len = cur_draft_ctx;
        }

        // Set draft inputs
        ggml_backend_tensor_set(dctx.inp_embed, noise_embed_buf.data(), 0,
                                sizeof(float) * noise_embed_buf.size());

        // Copy target_feat slice → target_hidden_cat (bf16 → f32)
        if (cache.target_feat && cur_draft_ctx > 0) {
            const int cap = cache.target_feat_cap;
            const size_t fc_in = (size_t)N_TARGET_FEATS * w.n_embd;
            const size_t elt_feat = ggml_element_size(cache.target_feat);
            const int slot0 = draft_start % cap;
            const int pre_n = std::min(cur_draft_ctx, cap - slot0);
            const int post_n = cur_draft_ctx - pre_n;

            dflash27b_launch_bf16_to_f32(
                (const char *)cache.target_feat->data + (size_t)slot0 * elt_feat * fc_in,
                dctx.target_hidden_cat->data,
                (size_t)pre_n * fc_in,
                nullptr);
            if (post_n > 0) {
                dflash27b_launch_bf16_to_f32(
                    (const char *)cache.target_feat->data,
                    (char *)dctx.target_hidden_cat->data + (size_t)pre_n * fc_in * sizeof(float),
                    (size_t)post_n * fc_in,
                    nullptr);
            }
            cudaError_t ce = cudaDeviceSynchronize();
            if (ce != cudaSuccess) {
                std::fprintf(stderr, "[DBG] bf16_to_f32 CUDA error: %s\n", cudaGetErrorString(ce));
                return 1;
            }
        }

        // Draft positions
        std::vector<int32_t> pos_q(q_len), pos_k(cur_draft_ctx + q_len);
        for (int i = 0; i < q_len; i++) pos_q[i] = cur_draft_ctx + i;
        for (int i = 0; i < cur_draft_ctx + q_len; i++) pos_k[i] = i;
        ggml_backend_tensor_set(dctx.positions_q, pos_q.data(), 0, sizeof(int32_t) * q_len);
        ggml_backend_tensor_set(dctx.positions_k, pos_k.data(), 0,
                                sizeof(int32_t) * (cur_draft_ctx + q_len));

        // Compute draft
        auto st = ggml_backend_graph_compute(backend, dctx.gf);
        {
            cudaError_t ce = cudaDeviceSynchronize();
            if (ce != cudaSuccess) {
                std::fprintf(stderr, "[DBG] draft compute CUDA error: %s\n", cudaGetErrorString(ce));
                return 1;
            }
        }
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "draft compute failed: %d\n", (int)st); return 1;
        }

        // Read draft argmax
        ggml_backend_tensor_get(dctx.argmax_tokens, draft_tok.data(), 0,
                                sizeof(int32_t) * q_len);
        draft_tok[0] = last_tok;  // pin position 0

        auto T_draft = std::chrono::steady_clock::now();
        tt_draft += std::chrono::duration<double, std::micro>(T_draft - T0).count();

        // ─── 2. Snapshot SSM state ───
        snapshot_ssm_state(cache);
        auto T_snap = std::chrono::steady_clock::now();
        tt_snap += std::chrono::duration<double, std::micro>(T_snap - T_draft).count();

        // ─── 3. Verify: run target forward on draft_tok[0..q_len-1] ───
        // Embed draft tokens
        std::vector<float> verify_embed((size_t)hidden * q_len);
        if (!w.embedder.embed(draft_tok.data(), q_len, verify_embed.data())) {
            std::fprintf(stderr, "verify embed failed\n"); return 1;
        }
        ggml_backend_tensor_set(act_a, verify_embed.data(), 0,
                                sizeof(float) * verify_embed.size());

        // M-RoPE positions (stride = q_len)
        std::vector<int32_t> pos4(4 * q_len);
        for (int i = 0; i < q_len; i++) {
            int p = committed + i;
            pos4[0 * q_len + i] = p;
            pos4[1 * q_len + i] = p;
            pos4[2 * q_len + i] = p;
            pos4[3 * q_len + i] = 0;
        }
        positions->ne[0] = 4 * q_len;
        ggml_backend_tensor_set(positions, pos4.data(), 0, sizeof(int32_t) * 4 * q_len);

        // Causal mask for verify
        const int verify_kv_len = committed + q_len;
        std::vector<uint16_t> verify_mask;
        build_causal_mask(verify_mask, verify_kv_len, q_len, committed);
        const int vkv_pad = align_up(verify_kv_len, KQ_MASK_PAD);
        const int vq_pad  = align_up(q_len, KQ_MASK_PAD);
        attn_mask->ne[0] = vkv_pad;
        attn_mask->ne[1] = vq_pad;
        attn_mask->nb[1] = (size_t)vkv_pad * ggml_element_size(attn_mask);
        ggml_backend_tensor_set(attn_mask, verify_mask.data(), 0,
            sizeof(uint16_t) * vkv_pad * vq_pad);

        bool ok = run_qwen35moe_forward(
            backend, w, cache, ecache, moe, w.expert_source,
            act_a, act_b, q_len, positions, attn_mask,
            committed, /*capture=*/true, /*fa_window=*/0,
            logits_out, argmax_out);
        if (!ok) {
            std::fprintf(stderr, "verify forward failed at step %d\n", n_draft_steps);
            return 1;
        }

        // Read target argmax
        ggml_backend_tensor_get(argmax_out, target_tok.data(), 0, sizeof(int32_t) * q_len);

        auto T_verify = std::chrono::steady_clock::now();
        tt_verify += std::chrono::duration<double, std::micro>(T_verify - T_snap).count();

        // ─── 4. Accept: longest prefix match ───
        int accept_n = 1;  // draft_tok[0] == last_tok, unconditionally accept
        for (int i = 0; i < q_len - 1; i++) {
            if (draft_tok[i + 1] == target_tok[i]) accept_n++;
            else break;
        }

        // Legacy path: bonus token from target
        int bonus_tok = -1;
        if (accept_n < q_len) {
            bonus_tok = target_tok[accept_n - 1];
        }
        int commit_n = accept_n + (bonus_tok >= 0 ? 1 : 0);

        // Clamp to budget
        const int budget = n_gen - n_generated;
        if (commit_n > budget) {
            commit_n = budget;
            if (commit_n <= accept_n) bonus_tok = -1;
        }

        auto T_accept = std::chrono::steady_clock::now();
        tt_accept += std::chrono::duration<double, std::micro>(T_accept - T_verify).count();

        std::printf("[step %d] accept_n=%d bonus=%d commit_n=%d\n",
                    n_draft_steps, accept_n, bonus_tok, commit_n);

        // ─── 5. Restore SSM state + Replay ───
        restore_ssm_state(cache);

        // Build replay tokens
        std::vector<int32_t> replay_tok(commit_n);
        for (int i = 0; i < commit_n; i++) {
            if (i < accept_n && i < (int)draft_tok.size()) {
                replay_tok[i] = draft_tok[i];
            } else {
                replay_tok[i] = bonus_tok;
            }
        }

        // Replay forward
        std::vector<float> replay_embed((size_t)hidden * commit_n);
        if (!w.embedder.embed(replay_tok.data(), commit_n, replay_embed.data())) {
            std::fprintf(stderr, "replay embed failed\n"); return 1;
        }
        ggml_backend_tensor_set(act_a, replay_embed.data(), 0,
                                sizeof(float) * replay_embed.size());

        // M-RoPE positions for replay (stride = commit_n)
        std::vector<int32_t> rpos4(4 * commit_n);
        for (int i = 0; i < commit_n; i++) {
            int p = committed + i;
            rpos4[0 * commit_n + i] = p;
            rpos4[1 * commit_n + i] = p;
            rpos4[2 * commit_n + i] = p;
            rpos4[3 * commit_n + i] = 0;
        }
        positions->ne[0] = 4 * commit_n;
        ggml_backend_tensor_set(positions, rpos4.data(), 0,
                                sizeof(int32_t) * 4 * commit_n);

        // Replay causal mask
        const int replay_kv_len = committed + commit_n;
        std::vector<uint16_t> replay_mask;
        build_causal_mask(replay_mask, replay_kv_len, commit_n, committed);
        const int rkv_pad = align_up(replay_kv_len, KQ_MASK_PAD);
        const int rq_pad  = align_up(commit_n, KQ_MASK_PAD);
        attn_mask->ne[0] = rkv_pad;
        attn_mask->ne[1] = rq_pad;
        attn_mask->nb[1] = (size_t)rkv_pad * ggml_element_size(attn_mask);
        ggml_backend_tensor_set(attn_mask, replay_mask.data(), 0,
            sizeof(uint16_t) * rkv_pad * rq_pad);

        ok = run_qwen35moe_forward(
            backend, w, cache, ecache, moe, w.expert_source,
            act_a, act_b, commit_n, positions, attn_mask,
            committed, /*capture=*/true, /*fa_window=*/0,
            logits_out, nullptr);
        if (!ok) {
            std::fprintf(stderr, "replay forward failed at step %d\n", n_draft_steps);
            return 1;
        }

        // Read last-position logits from replay to get next last_tok
        std::vector<float> replay_logits(vocab);
        ggml_backend_tensor_get(logits_out, replay_logits.data(),
                                sizeof(float) * (size_t)vocab * (commit_n - 1),
                                sizeof(float) * vocab);
        last_tok = argmax_f32(replay_logits.data(), vocab);

        auto T_replay = std::chrono::steady_clock::now();
        tt_restore += std::chrono::duration<double, std::micro>(T_accept - T_verify).count();
        tt_replay += std::chrono::duration<double, std::micro>(T_replay - T_accept).count();

        // ─── 6. Commit ───
        bool hit_eos = false;
        for (int i = 0; i < commit_n; i++) {
            out_all.push_back(replay_tok[i]);
            if (replay_tok[i] == w.eos_id || replay_tok[i] == w.eos_chat_id) {
                hit_eos = true;
            }
        }
        committed += commit_n;
        n_generated += commit_n;
        n_accept_sum += accept_n;
        n_draft_steps++;

        if (hit_eos) {
            std::printf("[eos] hit EOS after %d tokens\n", n_generated);
            break;
        }
    }

    auto t_gen1 = std::chrono::steady_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t_gen1 - t_gen0).count();

    // ── 14. Print results ──
    std::printf("\n=== Results ===\n");
    std::printf("Generated %d tokens in %.1f ms (%.1f tok/s)\n",
                n_generated, gen_ms, n_generated * 1000.0 / gen_ms);
    std::printf("Draft steps: %d, avg accept: %.2f\n",
                n_draft_steps, n_draft_steps > 0 ? (double)n_accept_sum / n_draft_steps : 0.0);

    // Phase timing
    auto avg_ms = [&](double us) { return n_draft_steps > 0 ? us / n_draft_steps / 1000.0 : 0.0; };
    std::printf("\nPer-step timing (ms):\n");
    std::printf("  draft:   %.2f\n", avg_ms(tt_draft));
    std::printf("  snap:    %.2f\n", avg_ms(tt_snap));
    std::printf("  verify:  %.2f\n", avg_ms(tt_verify));
    std::printf("  accept:  %.2f\n", avg_ms(tt_accept));
    std::printf("  replay:  %.2f\n", avg_ms(tt_replay));

    // Print generated token IDs (no built-in detokenizer)
    std::printf("\n--- Generated token IDs (%d) ---\n", (int)out_all.size());
    for (int i = 0; i < std::min((int)out_all.size(), 64); i++) {
        std::printf("%d ", out_all[i]);
    }
    if ((int)out_all.size() > 64) std::printf("...");
    std::printf("\n---\n");

    // Cleanup
    draft_ctx_free(dctx);
    ggml_backend_buffer_free(pos_buf);
    ggml_free(pos_ctx);
    ggml_backend_buffer_free(out_buf);
    ggml_free(out_ctx);
    ggml_backend_buffer_free(act_buf);
    ggml_free(act_ctx);
    free_draft_weights(dw);
    free_target_weights(w);
    ggml_backend_free(backend);

    std::printf("\nDone.\n");
    return 0;
}
