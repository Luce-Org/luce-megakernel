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

    // Validate arch — accept both qwen35 (dense) and qwen35moe (MoE)
    std::string arch;
    {
        int64_t arch_id = gguf_find_key(gctx, "general.architecture");
        if (arch_id < 0) {
            set_last_error("missing general.architecture");
            gguf_free(gctx);
            return false;
        }
        arch = gguf_get_val_str(gctx, arch_id);
        if (arch != "qwen35" && arch != "qwen35moe") {
            set_last_error(std::string("unexpected arch: ") + arch + " (expected qwen35 or qwen35moe)");
            gguf_free(gctx);
            return false;
        }
    }

    std::string err;
    const uint32_t n_embd = get_u32_or(gctx, (arch + ".embedding_length").c_str(),    0);
    const uint32_t n_ff   = get_u32_or(gctx, (arch + ".feed_forward_length").c_str(), 0);
    const uint32_t n_layer= get_u32_or(gctx, (arch + ".block_count").c_str(),         0);
    const uint32_t n_head = get_u32_or(gctx, (arch + ".attention.head_count").c_str(),0);
    const uint32_t n_headkv=get_u32_or(gctx, (arch + ".attention.head_count_kv").c_str(),0);
    const uint32_t kl     = get_u32_or(gctx, (arch + ".attention.key_length").c_str(),   0);
    const uint32_t vl     = get_u32_or(gctx, (arch + ".attention.value_length").c_str(), 0);
    const uint32_t fai    = get_u32_or(gctx, (arch + ".full_attention_interval").c_str(),0);
    const uint32_t ssm_conv  = get_u32_or(gctx, (arch + ".ssm.conv_kernel").c_str(),  0);
    const uint32_t ssm_inner = get_u32_or(gctx, (arch + ".ssm.inner_size").c_str(),   0);
    const uint32_t ssm_state = get_u32_or(gctx, (arch + ".ssm.state_size").c_str(),   0);
    const uint32_t ssm_dt    = get_u32_or(gctx, (arch + ".ssm.time_step_rank").c_str(),0);
    const uint32_t ssm_grp   = get_u32_or(gctx, (arch + ".ssm.group_count").c_str(),  0);

    // MoE fields (zero for dense qwen35)
    const uint32_t n_expert      = get_u32_or(gctx, (arch + ".expert_count").c_str(), 0);
    const uint32_t n_expert_used = get_u32_or(gctx, (arch + ".expert_used_count").c_str(), 0);
    const uint32_t expert_ff     = get_u32_or(gctx, (arch + ".expert_feed_forward_length").c_str(), 0);
    const uint32_t shared_ff     = get_u32_or(gctx, (arch + ".expert_shared_feed_forward_length").c_str(), 0);

    const bool is_moe = (arch == "qwen35moe");

    if (is_moe) {
        // Validate qwen35moe hparams: 40 layers, 2048 hidden, 16 heads, 2 kv heads,
        // 256 experts, 8 active, 512 expert ff, 512 shared ff
        if (n_embd != 2048 || n_layer != 40 || n_head != 16 || n_headkv != 2 ||
            kl != 256 || vl != 256 || fai != 4 ||
            ssm_conv != 4 || ssm_inner != 4096 || ssm_state != 128 ||
            ssm_dt != 32 || ssm_grp != 16 ||
            n_expert != 256 || n_expert_used != 8 ||
            expert_ff != 512 || shared_ff != 512) {
            char buf[512];
            std::snprintf(buf, sizeof(buf),
                "unexpected qwen35moe hparams: n_embd=%u n_layer=%u n_head=%u n_head_kv=%u "
                "kl=%u vl=%u fai=%u ssm{conv=%u inner=%u state=%u dt=%u grp=%u} "
                "n_expert=%u n_expert_used=%u expert_ff=%u shared_ff=%u",
                n_embd, n_layer, n_head, n_headkv, kl, vl, fai,
                ssm_conv, ssm_inner, ssm_state, ssm_dt, ssm_grp,
                n_expert, n_expert_used, expert_ff, shared_ff);
            set_last_error(buf);
            gguf_free(gctx);
            return false;
        }
    } else {
        // Validate qwen35 dense hparams (unchanged)
        if (n_embd != 5120 || n_layer != 64 || n_head != 24 || n_headkv != 4 ||
            kl != 256 || vl != 256 || n_ff != 17408 || fai != 4 ||
            ssm_conv != 4 || ssm_inner != 6144 || ssm_state != 128 ||
            ssm_dt != 48 || ssm_grp != 16) {
            char buf[512];
            std::snprintf(buf, sizeof(buf),
                "unexpected hparams: n_embd=%u n_layer=%u n_head=%u n_head_kv=%u "
                "kl=%u vl=%u n_ff=%u fai=%u ssm{conv=%u inner=%u state=%u dt=%u grp=%u}",
                n_embd, n_layer, n_head, n_headkv, kl, vl, n_ff, fai,
                ssm_conv, ssm_inner, ssm_state, ssm_dt, ssm_grp);
            set_last_error(buf);
            gguf_free(gctx);
            return false;
        }
    }

    // rope dimension_sections (array of 4 uint32)
    int rope_sections[4] = {0, 0, 0, 0};
    {
        int64_t rid = gguf_find_key(gctx, (arch + ".rope.dimension_sections").c_str());
        if (rid >= 0) {
            size_t n = gguf_get_arr_n(gctx, rid);
            if (n >= 4) {
                const int32_t * arr = (const int32_t *)gguf_get_arr_data(gctx, rid);
                for (int k = 0; k < 4; k++) rope_sections[k] = arr[k];
            }
        }
    }

    out.ctx     = meta_ctx;
    out.backend = backend;
    out.n_layer = (int)n_layer;
    out.n_embd  = (int)n_embd;
    out.n_ff    = is_moe ? 0 : (int)n_ff;
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
    out.n_expert       = (int)n_expert;
    out.n_expert_used  = (int)n_expert_used;
    out.expert_ff_dim  = (int)expert_ff;
    out.shared_ff_dim  = (int)shared_ff;
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

        // Dense FFN tensors (qwen35) or MoE tensors (qwen35moe)
        L.w_gate         = fnd("ffn_gate.weight");
        L.w_up           = fnd("ffn_up.weight");
        L.w_down         = fnd("ffn_down.weight");

        // MoE FFN tensors (qwen35moe only; null for dense)
        L.ffn_gate_inp       = fnd("ffn_gate_inp.weight");
        L.ffn_up_exps        = fnd("ffn_up_exps.weight");
        L.ffn_gate_exps      = fnd("ffn_gate_exps.weight");
        L.ffn_down_exps      = fnd("ffn_down_exps.weight");
        L.ffn_up_shexp       = fnd("ffn_up_shexp.weight");
        L.ffn_gate_shexp     = fnd("ffn_gate_shexp.weight");
        L.ffn_down_shexp     = fnd("ffn_down_shexp.weight");
        L.ffn_gate_inp_shexp = fnd("ffn_gate_inp_shexp.weight");

        if (!is_moe) {
            // Dense model: gate/up/down required
            if (!L.attn_norm || !L.attn_post_norm || !L.w_gate || !L.w_up || !L.w_down) {
                char b[128];
                std::snprintf(b, sizeof(b), "layer %d: missing shared tensor", il);
                set_last_error(b);
                gguf_free(gctx);
                return false;
            }
        } else {
            // MoE model: attn_norm + post_norm + expert tensors required
            if (!L.attn_norm || !L.attn_post_norm) {
                char b[128];
                std::snprintf(b, sizeof(b), "layer %d: missing attn_norm/post_norm", il);
                set_last_error(b);
                gguf_free(gctx);
                return false;
            }
            if (!L.ffn_gate_inp || !L.ffn_up_exps || !L.ffn_gate_exps || !L.ffn_down_exps ||
                !L.ffn_up_shexp || !L.ffn_gate_shexp || !L.ffn_down_shexp || !L.ffn_gate_inp_shexp) {
                char b[256];
                std::snprintf(b, sizeof(b), "layer %d: missing MoE tensors", il);
                set_last_error(b);
                gguf_free(gctx);
                return false;
            }
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
