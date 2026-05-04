// Loads Qwen3.5-27B qwen35 hybrid from a GGUF file on disk into a ggml
// context on the CUDA backend.
//
// The file is expected to use arch "qwen35" (NOT plain "qwen3"). See
// unsloth/Qwen3.5-27B-GGUF or ddh0/Qwen3.5-GGUF for reference.
//
// Tensor naming convention (from real inspection of ddh0's Qwen3.5-27B-4.71.gguf):
//
//   Top-level:
//     token_embd.weight              [hidden, vocab]
//     output_norm.weight             [hidden]                  F32
//     output.weight                  [hidden, vocab]           Q6_K (lm_head)
//
//   Per layer blk.<i> (full-attention layers, i.e. i % 4 == 3):
//     attn_norm.weight               [hidden]                  F32
//     post_attention_norm.weight     [hidden]                  F32
//     attn_q.weight                  [hidden, 2*q_dim]         Q4_K   (Q || gate packed)
//     attn_k.weight                  [hidden, kv_dim]          Q8_0
//     attn_v.weight                  [hidden, kv_dim]          Q8_0
//     attn_output.weight             [q_dim,  hidden]          Q5_K
//     attn_q_norm.weight             [head_dim]                F32
//     attn_k_norm.weight             [head_dim]                F32
//     ffn_gate.weight                [hidden, intermediate]    IQ4_XS
//     ffn_up.weight                  [hidden, intermediate]    IQ4_XS
//     ffn_down.weight                [intermediate, hidden]    IQ4_XS
//
//   Per layer blk.<i> (Gated DeltaNet layers, i.e. i % 4 != 3):
//     attn_norm.weight               [hidden]                  F32
//     post_attention_norm.weight     [hidden]                  F32
//     attn_qkv.weight                [hidden, 10240]           Q5_K   (q/k/v/beta fused)
//     attn_gate.weight               [hidden, inner=6144]      Q5_K   (z projection)
//     ssm_conv1d.weight              [inner, 4]                F32
//     ssm_a                          [dt_rank=48]              F32
//     ssm_alpha.weight               [dt_rank, hidden]         F32
//     ssm_beta.weight                [dt_rank, hidden]         F32
//     ssm_dt.bias                    [dt_rank]                 F32
//     ssm_norm.weight                [state=128]               F32
//     ssm_out.weight                 [inner, hidden]           Q5_K
//     ffn_gate/up/down              (same as full-attn)
//
// This loader reads the file via ggml's built-in GGUF API, which returns a
// ggml_context pre-populated with tensors. We then wire that context onto
// the CUDA backend (via ggml_backend_alloc_ctx_tensors) and copy each
// tensor's bytes from the mmap'd file.

#include "internal.h"

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

// CpuEmbedder destructor + embed() method
CpuEmbedder::~CpuEmbedder() {
#if defined(_WIN32)
    if (mmap_addr)                         UnmapViewOfFile(mmap_addr);
    if (mmap_hmap)                         CloseHandle(mmap_hmap);
    if (mmap_hfile != INVALID_HANDLE_VALUE) CloseHandle(mmap_hfile);
#else
    if (mmap_addr) ::munmap(mmap_addr, mmap_len);
    if (mmap_fd >= 0) ::close(mmap_fd);
#endif
}

bool CpuEmbedder::embed(const int32_t * ids, int n, float * out_f32) const {
    if (!tok_embd_bytes || tok_embd_type == GGML_TYPE_COUNT) return false;
    const ggml_type_traits * tr = ggml_get_type_traits(tok_embd_type);
    if (!tr || !tr->to_float) return false;
    for (int i = 0; i < n; i++) {
        int32_t id = ids[i];
        if (id < 0 || id >= n_vocab) return false;
        const uint8_t * row = tok_embd_bytes + (size_t)id * row_bytes;
        tr->to_float(row, out_f32 + (size_t)i * n_embd, n_embd);
    }
    return true;
}

namespace {

// Local Mmap used only during load (separate from the one kept alive inside
// TargetWeights::embedder). We don't call munmap on this one when we want
// to hand ownership to the CpuEmbedder — see end of load_target_gguf.
struct Mmap {
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
            err = "CreateFileA: " + path + ": error " + std::to_string(GetLastError());
            return false;
        }
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(hFile, &sz)) {
            err = "GetFileSizeEx: error " + std::to_string(GetLastError());
            return false;
        }
        len = (size_t)sz.QuadPart;
        hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMap) {
            err = "CreateFileMappingA: error " + std::to_string(GetLastError());
            return false;
        }
        addr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!addr) {
            err = "MapViewOfFile: error " + std::to_string(GetLastError());
            return false;
        }
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
    // Ownership transfer: release handles without unmapping.
    void release() {
        addr = nullptr;
        len  = 0;
#if defined(_WIN32)
        hFile = INVALID_HANDLE_VALUE;
        hMap  = nullptr;
#else
        fd = -1;
#endif
    }
    ~Mmap() {
#if defined(_WIN32)
        if (addr)                        UnmapViewOfFile(addr);
        if (hMap)                        CloseHandle(hMap);
        if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
#else
        if (addr) ::munmap(addr, len);
        if (fd >= 0) ::close(fd);
#endif
    }
};

// Required uint32 metadata key → bound check. Aborts load on mismatch.
bool expect_u32(const gguf_context * g, const char * key, uint32_t expected, std::string & err) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) { err = std::string("missing gguf key: ") + key; return false; }
    uint32_t v = gguf_get_val_u32(g, id);
    if (v != expected) {
        char b[256];
        std::snprintf(b, sizeof(b), "gguf key %s=%u expected %u", key, v, expected);
        err = b;
        return false;
    }
    return true;
}

int32_t get_i32_or(const gguf_context * g, const char * key, int32_t fallback) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return fallback;
    return gguf_get_val_i32(g, id);
}

uint32_t get_u32_or(const gguf_context * g, const char * key, uint32_t fallback) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return fallback;
    return gguf_get_val_u32(g, id);
}

} // namespace

bool load_target_gguf(const std::string & path,
                      ggml_backend_t       backend,
                      TargetWeights &      out) {

    // ── 1. Parse metadata + create a ggml_context holding tensor descriptors ─
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx      = &meta_ctx;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gip);
    if (!gctx) {
        set_last_error("gguf_init_from_file failed: " + path);
        return false;
    }

    // Validate arch + the dimensions we hardcode everywhere.
    {
        int64_t arch_id = gguf_find_key(gctx, "general.architecture");
        if (arch_id < 0) {
            set_last_error("missing general.architecture");
            gguf_free(gctx);
            return false;
        }
        const char * arch = gguf_get_val_str(gctx, arch_id);
        if (std::string(arch) != "qwen35") {
            set_last_error(std::string("unexpected arch: ") + arch + " (expected qwen35)");
            gguf_free(gctx);
            return false;
        }
    }

    std::string err;
    const uint32_t n_embd = get_u32_or(gctx, "qwen35.embedding_length",    0);
    const uint32_t n_ff   = get_u32_or(gctx, "qwen35.feed_forward_length", 0);
    const uint32_t n_layer= get_u32_or(gctx, "qwen35.block_count",         0);
    const uint32_t n_head = get_u32_or(gctx, "qwen35.attention.head_count",0);
    const uint32_t n_headkv=get_u32_or(gctx, "qwen35.attention.head_count_kv",0);
    const uint32_t kl     = get_u32_or(gctx, "qwen35.attention.key_length",   0);
    const uint32_t vl     = get_u32_or(gctx, "qwen35.attention.value_length", 0);
    const uint32_t fai    = get_u32_or(gctx, "qwen35.full_attention_interval",0);
    const uint32_t ssm_conv  = get_u32_or(gctx, "qwen35.ssm.conv_kernel",  0);
    const uint32_t ssm_inner = get_u32_or(gctx, "qwen35.ssm.inner_size",   0);
    const uint32_t ssm_state = get_u32_or(gctx, "qwen35.ssm.state_size",   0);
    const uint32_t ssm_dt    = get_u32_or(gctx, "qwen35.ssm.time_step_rank",0);
    const uint32_t ssm_grp   = get_u32_or(gctx, "qwen35.ssm.group_count",  0);

    if (n_embd == 0 || n_layer == 0 || n_head == 0 || n_headkv == 0 ||
        kl == 0 || vl == 0 || n_ff == 0 || fai == 0 ||
        ssm_conv == 0 || ssm_inner == 0 || ssm_state == 0 ||
        ssm_dt == 0 || ssm_grp == 0) {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "missing or zero hparams: n_embd=%u n_layer=%u n_head=%u n_head_kv=%u "
            "kl=%u vl=%u n_ff=%u fai=%u ssm{conv=%u inner=%u state=%u dt=%u grp=%u}",
            n_embd, n_layer, n_head, n_headkv, kl, vl, n_ff, fai,
            ssm_conv, ssm_inner, ssm_state, ssm_dt, ssm_grp);
        set_last_error(buf);
        gguf_free(gctx);
        return false;
    }

    // Structural invariants required by the graph builder.
    if (kl != vl) {
        set_last_error("key_length != value_length not supported");
        gguf_free(gctx); return false;
    }
    if (ssm_inner % ssm_dt != 0) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "ssm.inner_size=%u not divisible by ssm.time_step_rank=%u", ssm_inner, ssm_dt);
        set_last_error(buf);
        gguf_free(gctx); return false;
    }
    if (n_layer % fai != 0) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "block_count=%u not divisible by full_attention_interval=%u", n_layer, fai);
        set_last_error(buf);
        gguf_free(gctx); return false;
    }

    // rope dimension_sections (array of 4 uint32)
    int rope_sections[4] = {0, 0, 0, 0};
    {
        int64_t rid = gguf_find_key(gctx, "qwen35.rope.dimension_sections");
        if (rid < 0) {
            set_last_error("missing qwen35.rope.dimension_sections");
            gguf_free(gctx); return false;
        }
        size_t n = gguf_get_arr_n(gctx, rid);
        if (n < 4) {
            set_last_error("qwen35.rope.dimension_sections has < 4 entries");
            gguf_free(gctx); return false;
        }
        const int32_t * arr = (const int32_t *)gguf_get_arr_data(gctx, rid);
        for (int k = 0; k < 4; k++) rope_sections[k] = arr[k];
    }

    // Validate rope_sections against head_dim. n_rot = 2 * sum(sections) is
    // the number of dims rotated by ggml_rope_multi; it must be even, > 0,
    // and ≤ head_dim, otherwise rope reads/writes out of bounds.
    {
        long sum = 0;
        for (int k = 0; k < 4; k++) {
            if (rope_sections[k] < 0) {
                char buf[160];
                std::snprintf(buf, sizeof(buf),
                    "rope_sections[%d]=%d is negative", k, rope_sections[k]);
                set_last_error(buf);
                gguf_free(gctx); return false;
            }
            sum += rope_sections[k];
        }
        const long n_rot = 2 * sum;
        if (n_rot <= 0 || n_rot > (long)kl) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "rope_sections {%d,%d,%d,%d} → n_rot=%ld invalid for head_dim=%u",
                rope_sections[0], rope_sections[1], rope_sections[2], rope_sections[3],
                n_rot, kl);
            set_last_error(buf);
            gguf_free(gctx); return false;
        }
    }

    out.ctx     = meta_ctx;
    out.backend = backend;
    out.n_layer = (int)n_layer;
    out.n_embd  = (int)n_embd;
    out.n_ff    = (int)n_ff;
    out.n_head  = (int)n_head;
    out.n_head_kv = (int)n_headkv;
    out.n_embd_head_k = (int)kl;
    out.n_embd_head_v = (int)vl;
    out.full_attention_interval = (int)fai;
    for (int k = 0; k < 4; k++) out.rope_sections[k] = rope_sections[k];
    out.ssm_d_conv = (int)ssm_conv;
    out.ssm_d_inner= (int)ssm_inner;
    out.ssm_d_state= (int)ssm_state;
    out.ssm_dt_rank= (int)ssm_dt;
    out.ssm_n_group= (int)ssm_grp;

    // EOS token ids from GGUF tokenizer metadata (stored as UINT32 by the
    // GGUF spec; we use the u32 helper and cast). UINT32_MAX is the
    // missing-key sentinel and maps to int32_t -1, which the runtime EOS
    // check rejects via the `>= 0` guard.
    {
        const uint32_t kEosKeyMissing = 0xFFFFFFFFu;
        const uint32_t raw_eos      = get_u32_or(gctx, "tokenizer.ggml.eos_token_id", kEosKeyMissing);
        const uint32_t raw_eos_chat = get_u32_or(gctx, "tokenizer.ggml.eot_token_id", kEosKeyMissing);
        out.eos_id      = (raw_eos      == kEosKeyMissing) ? -1 : (int32_t)raw_eos;
        out.eos_chat_id = (raw_eos_chat == kEosKeyMissing) ? -1 : (int32_t)raw_eos_chat;
        std::printf("[loader] eos_id=%d eos_chat_id=%d\n", out.eos_id, out.eos_chat_id);
    }

    // Compute capture layer IDs: evenly spaced through the target layers.
    // step = (n_layer - 2) / (N - 1), ids[k] = 1 + k * step.
    {
        const int N = DFLASH27B_DRAFT_N_TARGET_LAYERS;
        const int step = ((int)n_layer - 2) / (N - 1);
        for (int k = 0; k < N; k++) out.capture_layer_ids[k] = 1 + k * step;
    }

    out.layers.assign((size_t)n_layer, TargetLayer{});

    // ── 2. Wire our layer pointers to tensors inside meta_ctx ─────────
    auto g = [&](const char * name) -> ggml_tensor * {
        return ggml_get_tensor(meta_ctx, name);
    };
    out.tok_embd = g("token_embd.weight");
    out.out_norm = g("output_norm.weight");
    out.output   = g("output.weight");
    if (!out.tok_embd || !out.out_norm || !out.output) {
        set_last_error("missing top-level tensors (token_embd/output_norm/output)");
        gguf_free(gctx);
        return false;
    }

    for (int il = 0; il < (int)n_layer; il++) {
        char name[128];
        auto fnd = [&](const char * suffix) -> ggml_tensor * {
            std::snprintf(name, sizeof(name), "blk.%d.%s", il, suffix);
            return ggml_get_tensor(meta_ctx, name);
        };
        TargetLayer & L = out.layers[il];

        // Always-present tensors
        L.attn_norm      = fnd("attn_norm.weight");
        L.attn_post_norm = fnd("post_attention_norm.weight");
        L.w_gate         = fnd("ffn_gate.weight");
        L.w_up           = fnd("ffn_up.weight");
        L.w_down         = fnd("ffn_down.weight");
        if (!L.attn_norm || !L.attn_post_norm || !L.w_gate || !L.w_up || !L.w_down) {
            char b[128];
            std::snprintf(b, sizeof(b), "layer %d: missing shared tensor", il);
            set_last_error(b);
            gguf_free(gctx);
            return false;
        }

        // Full-attention tensors (only on layers where (il+1)%fai == 0,
        // i.e. il%4 == 3 for fai=4). May be null on deltanet layers.
        L.wq     = fnd("attn_q.weight");
        L.wk     = fnd("attn_k.weight");
        L.wv     = fnd("attn_v.weight");
        L.wo     = fnd("attn_output.weight");
        L.q_norm = fnd("attn_q_norm.weight");
        L.k_norm = fnd("attn_k_norm.weight");

        // Gated DeltaNet tensors (null on full-attention layers)
        L.wqkv         = fnd("attn_qkv.weight");
        L.wqkv_gate    = fnd("attn_gate.weight");
        L.ssm_conv1d   = fnd("ssm_conv1d.weight");
        L.ssm_beta     = fnd("ssm_beta.weight");
        L.ssm_alpha    = fnd("ssm_alpha.weight");
        L.ssm_a        = fnd("ssm_a");
        L.ssm_dt_bias  = fnd("ssm_dt.bias");
        L.ssm_norm     = fnd("ssm_norm.weight");
        L.ssm_out      = fnd("ssm_out.weight");

        // Sanity: each layer must be EITHER full-attn OR deltanet, not both, not neither.
        const bool has_attn = L.wq && L.wk && L.wv && L.wo && L.q_norm && L.k_norm;
        const bool has_ssm  = L.wqkv && L.wqkv_gate && L.ssm_conv1d && L.ssm_out;
        const bool is_full_attn_layer = (((il + 1) % out.full_attention_interval) == 0);
        if (is_full_attn_layer && !has_attn) {
            char b[128];
            std::snprintf(b, sizeof(b), "layer %d expected full-attn, missing tensors", il);
            set_last_error(b);
            gguf_free(gctx);
            return false;
        }
        if (!is_full_attn_layer && !has_ssm) {
            char b[128];
            std::snprintf(b, sizeof(b), "layer %d expected deltanet, missing tensors", il);
            set_last_error(b);
            gguf_free(gctx);
            return false;
        }
    }

    // ── 3. Allocate CUDA buffer for all tensors in meta_ctx ───────────
    out.buf = ggml_backend_alloc_ctx_tensors(meta_ctx, backend);
    if (!out.buf) {
        set_last_error("ggml_backend_alloc_ctx_tensors failed (target)");
        gguf_free(gctx);
        return false;
    }

    // ── 4. mmap the file and copy tensor bytes to CUDA ────────────────
    //
    // SKIP uploading token_embd.weight — it stays on CPU for embedding
    // lookup (CUDA get_rows doesn't support k-quants). We hand the mmap
    // ownership to TargetWeights::embedder at the end.
    Mmap mm;
    if (!mm.open_ro(path, err)) { set_last_error(err); gguf_free(gctx); return false; }
    const size_t data_start = gguf_get_data_offset(gctx);
    const int64_t n_tensors = gguf_get_n_tensors(gctx);

    size_t total = 0;
    size_t tok_embd_off = 0, tok_embd_sz = 0;
    ggml_type tok_embd_type = GGML_TYPE_COUNT;
    for (int64_t tid = 0; tid < n_tensors; tid++) {
        const char * tname = gguf_get_tensor_name(gctx, tid);
        ggml_tensor * t = ggml_get_tensor(meta_ctx, tname);
        if (!t) continue;
        const size_t off = data_start + gguf_get_tensor_offset(gctx, tid);
        const size_t sz  = gguf_get_tensor_size(gctx, tid);
        if (off + sz > mm.len) {
            set_last_error(std::string("tensor '") + tname + "' overflows file");
            gguf_free(gctx);
            return false;
        }
        if (std::string(tname) == "token_embd.weight") {
            // Remember offset + size for the CPU embedder; don't upload to GPU.
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

    // ── 5. Transfer mmap ownership to the CpuEmbedder so it can dequantize
    //       rows on demand without uploading the full embedding table to GPU.
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
    out.embedder.n_vocab        = DFLASH27B_TARGET_VOCAB;
    out.embedder.row_bytes      = tok_embd_sz / DFLASH27B_TARGET_VOCAB;
    mm.release();  // don't munmap on Mmap dtor — now owned by the embedder

    // Stash the total for callers that want to print it
    char summary[192];
    std::snprintf(summary, sizeof(summary),
        "target loaded: %" PRId64 " tensors on GPU %.2f GiB, tok_embd %.0f MiB CPU-only (%s)",
        n_tensors, total / (1024.0 * 1024.0 * 1024.0),
        tok_embd_sz / (1024.0 * 1024.0), ggml_type_name(tok_embd_type));
    set_last_error(summary);

    return true;
}

void free_target_weights(TargetWeights & w) {
    if (w.buf) { ggml_backend_buffer_free(w.buf); w.buf = nullptr; }
    if (w.ctx) { ggml_free(w.ctx);                w.ctx = nullptr; }
    // CpuEmbedder destructor handles the mmap automatically.
    w.layers.clear();
    w.tok_embd = nullptr;
    w.out_norm = nullptr;
    w.output   = nullptr;
}

} // namespace dflash27b
