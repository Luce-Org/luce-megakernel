// Loads Poolside Laguna-XS.2 from a GGUF file on disk into a ggml context
// on the CUDA backend. Mirrors gguf_target_loader.cpp's qwen35 path but for
// the laguna arch (iSWA + sigmoid-routed MoE + per-head softplus attn gate +
// per-layer-varying head count + per-layer-type partial RoPE with YaRN).
//
// Tensor naming (matches gguf-py MODEL_ARCH.LAGUNA list, see
// rnd_experiments/gguf_converter_quantizer/patches/laguna/arch_patch_summary.md):
//
//   Top-level:
//     token_embd.weight              [n_embd, n_vocab]                Q4_K_M (kept on CPU)
//     output_norm.weight             [n_embd]                          F32
//     output.weight                  [n_embd, n_vocab]                 Q6_K
//
//   Per layer blk.<i> (shared, all layers):
//     attn_norm.weight               [n_embd]                          F32
//     ffn_norm.weight                [n_embd]                          F32
//     attn_q.weight                  [n_embd, n_head[il] * head_dim]   Q4_K
//     attn_k.weight                  [n_embd, n_head_kv * head_dim]    Q8_0
//     attn_v.weight                  [n_embd, n_head_kv * head_dim]    Q8_0
//     attn_output.weight             [n_head[il] * head_dim, n_embd]   Q5_K
//     attn_q_norm.weight             [head_dim]                        F32
//     attn_k_norm.weight             [head_dim]                        F32
//     attn_gate.weight               [n_embd, n_head[il]]              Q4_K   (per-head softplus gate)
//
//   Layer 0 (dense MLP only):
//     ffn_gate.weight                [n_embd, n_ff]                    Q4_K
//     ffn_up.weight                  [n_embd, n_ff]                    Q4_K
//     ffn_down.weight                [n_ff, n_embd]                    Q5_K
//
//   Layers 1..n-1 (sparse MoE only):
//     ffn_gate_inp.weight            [n_embd, n_expert]                F32  (sigmoid router)
//     exp_probs_b.bias               [n_expert]                        F32  (DeepSeek-style score-correction bias)
//     ffn_gate_exps.weight           [n_embd, n_ff_exp, n_expert]      Q4_K (stacked)
//     ffn_up_exps.weight             [n_embd, n_ff_exp, n_expert]      Q4_K
//     ffn_down_exps.weight           [n_ff_exp, n_embd, n_expert]      Q5_K (or Q4_K)
//     ffn_gate_shexp.weight          [n_embd, n_ff_shexp]              Q4_K (always-on shared expert)
//     ffn_up_shexp.weight            [n_embd, n_ff_shexp]              Q4_K
//     ffn_down_shexp.weight          [n_ff_shexp, n_embd]              Q5_K

#include "laguna_internal.h"
#include "internal.h"
#include "dflash27b.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#if !defined(_WIN32)
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash27b {

namespace {

// Same Mmap shape as gguf_target_loader.cpp's local helper. Duplicated locally
// to keep the loader self-contained without exporting internals.
struct LagunaMmap {
    void *  addr = nullptr;
    size_t  len  = 0;
#if defined(_WIN32)
    HANDLE  hFile = INVALID_HANDLE_VALUE;
    HANDLE  hMap  = nullptr;
#else
    int     fd   = -1;
#endif

    bool open_ro(const std::string & path, std::string & err) {
#if defined(_WIN32)
        hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hFile == INVALID_HANDLE_VALUE) {
            err = "CreateFileA: " + path; return false;
        }
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(hFile, &sz)) { err = "GetFileSizeEx"; return false; }
        len = (size_t)sz.QuadPart;
        hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMap) { err = "CreateFileMappingA"; return false; }
        addr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!addr) { err = "MapViewOfFile"; return false; }
#else
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { err = "open: " + path + ": " + std::strerror(errno); return false; }
        struct stat st;
        if (::fstat(fd, &st) < 0) { err = "fstat: " + std::string(std::strerror(errno)); return false; }
        len = (size_t)st.st_size;
        addr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) { err = "mmap: " + std::string(std::strerror(errno)); addr = nullptr; return false; }
#endif
        return true;
    }
    void release() {
        addr = nullptr; len = 0;
#if defined(_WIN32)
        hFile = INVALID_HANDLE_VALUE; hMap = nullptr;
#else
        fd = -1;
#endif
    }
    ~LagunaMmap() {
#if defined(_WIN32)
        if (addr)                          UnmapViewOfFile(addr);
        if (hMap)                          CloseHandle(hMap);
        if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
#else
        if (addr) ::munmap(addr, len);
        if (fd >= 0) ::close(fd);
#endif
    }
};

int32_t  get_i32_or(const gguf_context * g, const char * key, int32_t fallback) {
    int64_t id = gguf_find_key(g, key); return (id < 0) ? fallback : gguf_get_val_i32(g, id);
}
uint32_t get_u32_or(const gguf_context * g, const char * key, uint32_t fallback) {
    int64_t id = gguf_find_key(g, key); return (id < 0) ? fallback : gguf_get_val_u32(g, id);
}
float    get_f32_or(const gguf_context * g, const char * key, float fallback) {
    int64_t id = gguf_find_key(g, key); return (id < 0) ? fallback : gguf_get_val_f32(g, id);
}
bool     get_bool_or(const gguf_context * g, const char * key, bool fallback) {
    int64_t id = gguf_find_key(g, key); return (id < 0) ? fallback : (bool)gguf_get_val_bool(g, id);
}

} // namespace

bool load_target_gguf_laguna(const std::string & path,
                              ggml_backend_t       backend,
                              LagunaTargetWeights & out) {

    // ── 1. Parse metadata ────────────────────────────────────────────────
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx      = &meta_ctx;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gip);
    if (!gctx) { set_last_error("gguf_init_from_file failed: " + path); return false; }

    // Validate arch.
    {
        int64_t arch_id = gguf_find_key(gctx, "general.architecture");
        if (arch_id < 0) { set_last_error("missing general.architecture"); gguf_free(gctx); return false; }
        const char * arch = gguf_get_val_str(gctx, arch_id);
        if (std::string(arch) != "laguna") {
            set_last_error(std::string("unexpected arch: ") + arch + " (expected laguna)");
            gguf_free(gctx);
            return false;
        }
    }

    // Read scalar hparams.
    const uint32_t n_layer       = get_u32_or(gctx, "laguna.block_count",                     0);
    const uint32_t n_embd        = get_u32_or(gctx, "laguna.embedding_length",                0);
    const uint32_t n_ff          = get_u32_or(gctx, "laguna.feed_forward_length",             0);
    const uint32_t n_ff_exp      = get_u32_or(gctx, "laguna.expert_feed_forward_length",      0);
    const uint32_t n_ff_shexp    = get_u32_or(gctx, "laguna.expert_shared_feed_forward_length",0);
    const uint32_t n_head_kv     = get_u32_or(gctx, "laguna.attention.head_count_kv",         0);
    const uint32_t key_length    = get_u32_or(gctx, "laguna.attention.key_length",            0);
    const uint32_t value_length  = get_u32_or(gctx, "laguna.attention.value_length",          0);
    const uint32_t n_expert      = get_u32_or(gctx, "laguna.expert_count",                    0);
    const uint32_t n_expert_used = get_u32_or(gctx, "laguna.expert_used_count",               0);
    const uint32_t n_dense_lead  = get_u32_or(gctx, "laguna.leading_dense_block_count",       1);
    const uint32_t sliding_win   = get_u32_or(gctx, "laguna.attention.sliding_window",        0);
    const uint32_t n_rot_full    = get_u32_or(gctx, "laguna.rope.dimension_count",            0);
    const uint32_t n_rot_swa     = get_u32_or(gctx, "laguna.rope.dimension_count_swa",        0);
    const uint32_t n_vocab       = get_u32_or(gctx, "laguna.vocab_size",                      0);

    const float  rope_base_full   = get_f32_or(gctx, "laguna.rope.freq_base",     0.0f);
    const float  rope_base_swa    = get_f32_or(gctx, "laguna.rope.freq_base_swa", 0.0f);
    const float  yarn_factor      = get_f32_or(gctx, "laguna.rope.scaling.factor",          0.0f);
    const float  yarn_attn_factor = get_f32_or(gctx, "laguna.rope.scaling.yarn_attn_factor", 1.0f);
    const float  yarn_beta_fast   = get_f32_or(gctx, "laguna.rope.scaling.yarn_beta_fast",  64.0f);
    const float  yarn_beta_slow   = get_f32_or(gctx, "laguna.rope.scaling.yarn_beta_slow",   1.0f);
    const uint32_t yarn_orig_ctx  = get_u32_or(gctx, "laguna.rope.scaling.original_context_length", 4096);

    const float  exp_w_scale      = get_f32_or(gctx, "laguna.expert_weights_scale", 1.0f);
    const bool   exp_w_norm       = get_bool_or(gctx, "laguna.expert_weights_norm", true);
    // expert_gating_func: 0=NONE, 1=SOFTMAX, 2=SIGMOID (LLAMA_EXPERT_GATING_FUNC_TYPE_*)
    const uint32_t exp_gate_fn    = get_u32_or(gctx, "laguna.expert_gating_func", 2);
    (void)yarn_attn_factor; // currently unused at load time; consumed by the graph builder

    if (n_layer == 0 || n_embd == 0 || n_head_kv == 0 || key_length == 0 || value_length == 0 ||
        n_ff == 0 || n_ff_exp == 0 || n_ff_shexp == 0 || n_expert == 0 || n_expert_used == 0 ||
        sliding_win == 0 || n_rot_full == 0 || n_rot_swa == 0 || n_vocab == 0) {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "missing or zero hparams: n_layer=%u n_embd=%u n_head_kv=%u key=%u val=%u "
            "n_ff=%u n_ff_exp=%u n_ff_shexp=%u n_expert=%u used=%u sw=%u n_rot{full=%u swa=%u} vocab=%u",
            n_layer, n_embd, n_head_kv, key_length, value_length,
            n_ff, n_ff_exp, n_ff_shexp, n_expert, n_expert_used,
            sliding_win, n_rot_full, n_rot_swa, n_vocab);
        set_last_error(buf);
        gguf_free(gctx);
        return false;
    }
    if (key_length != value_length) {
        set_last_error("laguna: key_length != value_length not supported");
        gguf_free(gctx); return false;
    }

    // Per-layer head count (ARRAY of length n_layer).
    std::vector<uint32_t> heads_per_layer((size_t)n_layer, 0);
    {
        int64_t aid = gguf_find_key(gctx, "laguna.attention.head_count");
        if (aid < 0) { set_last_error("missing laguna.attention.head_count"); gguf_free(gctx); return false; }
        const enum gguf_type kt = gguf_get_kv_type(gctx, aid);
        if (kt == GGUF_TYPE_ARRAY) {
            const size_t n = gguf_get_arr_n(gctx, aid);
            if (n != (size_t)n_layer) {
                char b[160];
                std::snprintf(b, sizeof(b),
                    "laguna.attention.head_count array len %zu != n_layer %u", n, n_layer);
                set_last_error(b);
                gguf_free(gctx); return false;
            }
            const enum gguf_type elt = gguf_get_arr_type(gctx, aid);
            if (elt != GGUF_TYPE_INT32 && elt != GGUF_TYPE_UINT32) {
                set_last_error("laguna.attention.head_count array element type must be i32 or u32");
                gguf_free(gctx); return false;
            }
            const void * p = gguf_get_arr_data(gctx, aid);
            for (uint32_t i = 0; i < n_layer; ++i) {
                heads_per_layer[i] = (elt == GGUF_TYPE_INT32)
                    ? (uint32_t)((const int32_t *)p)[i]
                    : ((const uint32_t *)p)[i];
            }
        } else {
            // Some GGUF writers may emit a scalar even when override exists; fall back.
            const uint32_t scalar = gguf_get_val_u32(gctx, aid);
            for (uint32_t i = 0; i < n_layer; ++i) heads_per_layer[i] = scalar;
        }
    }

    // Tokenizer special tokens.
    const uint32_t kEosKeyMissing = 0xFFFFFFFFu;
    const uint32_t raw_bos      = get_u32_or(gctx, "tokenizer.ggml.bos_token_id",     kEosKeyMissing);
    const uint32_t raw_eos      = get_u32_or(gctx, "tokenizer.ggml.eos_token_id",     kEosKeyMissing);
    const uint32_t raw_eot      = get_u32_or(gctx, "tokenizer.ggml.eot_token_id",     kEosKeyMissing);
    const uint32_t raw_pad      = get_u32_or(gctx, "tokenizer.ggml.padding_token_id", kEosKeyMissing);

    // Populate metadata
    out.ctx     = meta_ctx;
    out.backend = backend;
    out.n_layer            = (int)n_layer;
    out.n_embd             = (int)n_embd;
    out.n_ff               = (int)n_ff;
    out.n_ff_exp           = (int)n_ff_exp;
    out.n_ff_shexp         = (int)n_ff_shexp;
    out.n_head_kv          = (int)n_head_kv;
    out.head_dim           = (int)key_length;
    out.n_expert           = (int)n_expert;
    out.n_expert_used      = (int)n_expert_used;
    out.n_layer_dense_lead = (int)n_dense_lead;
    out.sliding_window     = (int)sliding_win;
    out.swa_pattern        = 4;  // (full, sw, sw, sw); fixed by Laguna design
    out.n_rot_full         = (int)n_rot_full;
    out.n_rot_swa          = (int)n_rot_swa;
    out.rope_freq_base_full= rope_base_full > 0 ? rope_base_full : 500000.0f;
    out.rope_freq_base_swa = rope_base_swa  > 0 ? rope_base_swa  :  10000.0f;
    out.yarn_factor        = yarn_factor    > 0 ? yarn_factor    :     32.0f;
    out.yarn_beta_fast     = yarn_beta_fast;
    out.yarn_beta_slow     = yarn_beta_slow;
    out.yarn_orig_ctx      = (int)yarn_orig_ctx;
    out.expert_weights_scale  = exp_w_scale > 0 ? exp_w_scale : 2.5f;
    out.expert_weights_norm   = exp_w_norm;
    out.expert_gating_sigmoid = (exp_gate_fn == 2);
    out.bos_id      = (raw_bos == kEosKeyMissing) ? -1 : (int32_t)raw_bos;
    out.eos_id      = (raw_eos == kEosKeyMissing) ? -1 : (int32_t)raw_eos;
    out.eos_chat_id = (raw_eot == kEosKeyMissing) ? -1 : (int32_t)raw_eot;
    out.pad_id      = (raw_pad == kEosKeyMissing) ? -1 : (int32_t)raw_pad;
    if ((int)n_layer <= (int)(sizeof(out.n_head_arr)/sizeof(out.n_head_arr[0]))) {
        for (uint32_t i = 0; i < n_layer; ++i) out.n_head_arr[i] = (int)heads_per_layer[i];
    } else {
        set_last_error("laguna: n_layer exceeds compiled-in n_head_arr capacity (40)");
        gguf_free(gctx); return false;
    }

    // Diagnostic.
    std::printf("[laguna-loader] n_layer=%u n_embd=%u head_dim=%u n_head_kv=%u\n",
                n_layer, n_embd, key_length, n_head_kv);
    std::printf("[laguna-loader] dense_lead=%u sliding_window=%u (pattern fwd,swa,swa,swa)\n",
                n_dense_lead, sliding_win);
    std::printf("[laguna-loader] rope full=%g swa=%g  n_rot full=%u swa=%u  yarn factor=%g orig_ctx=%u\n",
                rope_base_full, rope_base_swa, n_rot_full, n_rot_swa, yarn_factor, yarn_orig_ctx);
    std::printf("[laguna-loader] MoE n_expert=%u used=%u  ff_exp=%u  ff_shexp=%u  scale=%g  sigmoid_router=%d\n",
                n_expert, n_expert_used, n_ff_exp, n_ff_shexp, exp_w_scale, (int)(exp_gate_fn == 2));
    std::printf("[laguna-loader] specials bos=%d eos=%d eot=%d pad=%d  vocab=%u\n",
                out.bos_id, out.eos_id, out.eos_chat_id, out.pad_id, n_vocab);

    out.layers.assign((size_t)n_layer, LagunaTargetLayer{});

    // ── 2. Resolve tensor pointers ───────────────────────────────────────
    auto g = [&](const char * name) -> ggml_tensor * { return ggml_get_tensor(meta_ctx, name); };
    out.tok_embd = g("token_embd.weight");
    out.out_norm = g("output_norm.weight");
    out.output   = g("output.weight");
    if (!out.tok_embd || !out.out_norm) {
        set_last_error("missing top-level tensors (token_embd / output_norm)");
        gguf_free(gctx); return false;
    }
    // output (lm_head) may be tied to token_embd in some quants; we always need the data
    // to be uploadable. Laguna's converter tells parent NOT to tie, so this must exist.
    if (!out.output) {
        set_last_error("missing output.weight (Laguna does not tie embeddings)");
        gguf_free(gctx); return false;
    }

    for (uint32_t il = 0; il < n_layer; ++il) {
        char name[160];
        auto fnd = [&](const char * suffix) -> ggml_tensor * {
            std::snprintf(name, sizeof(name), "blk.%u.%s", il, suffix);
            return ggml_get_tensor(meta_ctx, name);
        };
        LagunaTargetLayer & L = out.layers[il];

        // Always present:
        L.attn_norm = fnd("attn_norm.weight");
        L.ffn_norm  = fnd("ffn_norm.weight");
        L.wq        = fnd("attn_q.weight");
        L.wk        = fnd("attn_k.weight");
        L.wv        = fnd("attn_v.weight");
        L.wo        = fnd("attn_output.weight");
        L.q_norm    = fnd("attn_q_norm.weight");
        L.k_norm    = fnd("attn_k_norm.weight");
        L.wqkv_gate = fnd("attn_gate.weight");

        if (!L.attn_norm || !L.ffn_norm || !L.wq || !L.wk || !L.wv || !L.wo ||
            !L.q_norm || !L.k_norm || !L.wqkv_gate) {
            char b[160];
            std::snprintf(b, sizeof(b), "layer %u missing required attention tensor(s)", il);
            set_last_error(b);
            gguf_free(gctx); return false;
        }

        const bool is_dense = (il < n_dense_lead);
        if (is_dense) {
            L.w_gate = fnd("ffn_gate.weight");
            L.w_up   = fnd("ffn_up.weight");
            L.w_down = fnd("ffn_down.weight");
            if (!L.w_gate || !L.w_up || !L.w_down) {
                char b[160];
                std::snprintf(b, sizeof(b), "dense layer %u missing ffn_gate/up/down", il);
                set_last_error(b);
                gguf_free(gctx); return false;
            }
        } else {
            // Sparse MoE
            L.ffn_gate_inp     = fnd("ffn_gate_inp.weight");
            L.ffn_exp_probs_b  = fnd("exp_probs_b.bias");
            L.ffn_gate_exps    = fnd("ffn_gate_exps.weight");
            L.ffn_up_exps      = fnd("ffn_up_exps.weight");
            L.ffn_down_exps    = fnd("ffn_down_exps.weight");
            L.ffn_gate_shexp   = fnd("ffn_gate_shexp.weight");
            L.ffn_up_shexp     = fnd("ffn_up_shexp.weight");
            L.ffn_down_shexp   = fnd("ffn_down_shexp.weight");
            if (!L.ffn_gate_inp || !L.ffn_exp_probs_b || !L.ffn_gate_exps ||
                !L.ffn_up_exps  || !L.ffn_down_exps  || !L.ffn_gate_shexp ||
                !L.ffn_up_shexp || !L.ffn_down_shexp) {
                char b[200];
                std::snprintf(b, sizeof(b),
                    "sparse layer %u missing MoE tensor(s) "
                    "(gate_inp=%p probs_b=%p gate_exps=%p up_exps=%p down_exps=%p gate_sh=%p up_sh=%p down_sh=%p)",
                    il,
                    (void*)L.ffn_gate_inp, (void*)L.ffn_exp_probs_b,
                    (void*)L.ffn_gate_exps, (void*)L.ffn_up_exps, (void*)L.ffn_down_exps,
                    (void*)L.ffn_gate_shexp, (void*)L.ffn_up_shexp, (void*)L.ffn_down_shexp);
                set_last_error(b);
                gguf_free(gctx); return false;
            }
        }
    }

    // ── 3. Allocate CUDA buffer for tensors (excludes tok_embd — stays on host) ─
    out.buf = ggml_backend_alloc_ctx_tensors(meta_ctx, backend);
    if (!out.buf) {
        set_last_error("ggml_backend_alloc_ctx_tensors failed (laguna target)");
        gguf_free(gctx); return false;
    }

    // ── 4. mmap + copy tensors to GPU ────────────────────────────────────
    LagunaMmap mm;
    std::string err;
    if (!mm.open_ro(path, err)) { set_last_error(err); gguf_free(gctx); return false; }
    const size_t  data_start = gguf_get_data_offset(gctx);
    const int64_t n_tensors  = gguf_get_n_tensors(gctx);

    size_t total = 0;
    size_t tok_embd_off = 0, tok_embd_sz = 0;
    ggml_type tok_embd_type = GGML_TYPE_COUNT;
    for (int64_t tid = 0; tid < n_tensors; ++tid) {
        const char * tname = gguf_get_tensor_name(gctx, tid);
        ggml_tensor * t = ggml_get_tensor(meta_ctx, tname);
        if (!t) continue;
        const size_t off = data_start + gguf_get_tensor_offset(gctx, tid);
        const size_t sz  = gguf_get_tensor_size(gctx, tid);
        if (off + sz > mm.len) {
            set_last_error(std::string("tensor '") + tname + "' overflows file");
            gguf_free(gctx); return false;
        }
        if (std::string(tname) == "token_embd.weight") {
            tok_embd_off  = off;
            tok_embd_sz   = sz;
            tok_embd_type = gguf_get_tensor_type(gctx, tid);
            continue;
        }
        ggml_backend_tensor_set(t, (const uint8_t *)mm.addr + off, 0, sz);
        total += sz;
    }

    gguf_free(gctx);

    if (tok_embd_off == 0 || tok_embd_type == GGML_TYPE_COUNT) {
        set_last_error("token_embd.weight not found or invalid type");
        return false;
    }

    // ── 5. Hand mmap to CpuEmbedder ──────────────────────────────────────
    out.embedder.mmap_addr      = mm.addr;
    out.embedder.mmap_len       = mm.len;
#if defined(_WIN32)
    out.embedder.mmap_hfile     = mm.hFile;
    out.embedder.mmap_hmap      = mm.hMap;
#else
    out.embedder.mmap_fd        = mm.fd;
#endif
    out.embedder.tok_embd_bytes = (const uint8_t *)mm.addr + tok_embd_off;
    out.embedder.tok_embd_type  = tok_embd_type;
    out.embedder.n_embd         = out.n_embd;
    out.embedder.n_vocab        = (int64_t)n_vocab;
    out.embedder.row_bytes      = tok_embd_sz / (size_t)n_vocab;
    mm.release();

    char summary[224];
    std::snprintf(summary, sizeof(summary),
        "laguna target loaded: %" PRId64 " tensors on GPU %.2f GiB, tok_embd %.0f MiB CPU-only (%s)",
        n_tensors, total / (1024.0 * 1024.0 * 1024.0),
        tok_embd_sz / (1024.0 * 1024.0), ggml_type_name(tok_embd_type));
    set_last_error(summary);
    std::printf("[laguna-loader] %s\n", summary);
    return true;
}

void free_laguna_target_weights(LagunaTargetWeights & w) {
    if (w.buf) { ggml_backend_buffer_free(w.buf); w.buf = nullptr; }
    if (w.ctx) { ggml_free(w.ctx);                w.ctx = nullptr; }
    // CpuEmbedder destructor handles mmap.
    w.layers.clear();
    w.tok_embd = nullptr;
    w.out_norm = nullptr;
    w.output   = nullptr;
}

} // namespace dflash27b
