// GGUF loader for Qwen3-0.6B drafter. Reads weights from a BF16 GGUF file
// produced by `convert_hf_to_gguf.py Qwen/Qwen3-0.6B`. Sets up ggml tensors
// on the requested backend.
//
// Tensor layout (verified via gguf reader):
//   token_embd.weight                 BF16 [hidden=1024, vocab=151936]
//   output_norm.weight                F32  [hidden]
//   output.weight                     BF16 [hidden, vocab] (lm_head)
//
//   blk.<i>.attn_norm.weight          F32  [hidden]
//   blk.<i>.attn_q.weight             BF16 [hidden, q_dim=2048]
//   blk.<i>.attn_k.weight             BF16 [hidden, kv_dim=1024]
//   blk.<i>.attn_v.weight             BF16 [hidden, kv_dim]
//   blk.<i>.attn_output.weight        BF16 [q_dim, hidden]
//   blk.<i>.attn_q_norm.weight        F32  [head_dim=128]
//   blk.<i>.attn_k_norm.weight        F32  [head_dim]
//   blk.<i>.ffn_norm.weight           F32  [hidden]
//   blk.<i>.ffn_gate.weight           BF16 [hidden, ffn=3072]
//   blk.<i>.ffn_up.weight             BF16 [hidden, ffn]
//   blk.<i>.ffn_down.weight           BF16 [ffn, hidden]
//
// We mmap the GGUF file and copy each tensor's bytes to the backend buffer
// (mirrors the dflash gguf_target_loader pattern).

#include "qwen3_0p6b_drafter.h"
#include "internal.h"

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash27b {

namespace {

bool copy_tensor_from_file(gguf_context * gctx, const char * name,
                           const void * mmap_base, size_t data_offset,
                           ggml_tensor * dst) {
    int idx = gguf_find_tensor(gctx, name);
    if (idx < 0) {
        std::fprintf(stderr, "[qwen3-0.6b] missing tensor: %s\n", name);
        return false;
    }
    const ggml_type src_type = gguf_get_tensor_type(gctx, idx);
    const size_t src_bytes = gguf_get_tensor_size(gctx, idx);
    const size_t dst_bytes = ggml_nbytes(dst);
    if (src_type != dst->type || src_bytes != dst_bytes) {
        std::fprintf(stderr,
                     "[qwen3-0.6b] incompatible tensor %s: src_type=%s src_bytes=%zu dst_type=%s dst_bytes=%zu\n",
                     name, ggml_type_name(src_type), src_bytes,
                     ggml_type_name(dst->type), dst_bytes);
        return false;
    }
    const size_t off = gguf_get_tensor_offset(gctx, idx);
    const uint8_t * src = (const uint8_t *)mmap_base + data_offset + off;
    ggml_backend_tensor_set(dst, src, 0, dst_bytes);
    return true;
}

uint32_t get_u32(gguf_context * g, const char * key, uint32_t def) {
    int k = gguf_find_key(g, key);
    if (k < 0) return def;
    return gguf_get_val_u32(g, k);
}

float get_f32(gguf_context * g, const char * key, float def) {
    int k = gguf_find_key(g, key);
    if (k < 0) return def;
    return gguf_get_val_f32(g, k);
}

} // namespace

bool load_qwen3_0p6b_drafter(const std::string & path,
                              ggml_backend_t backend,
                              Qwen3DrafterWeights & out) {
    out.backend = backend;

    gguf_init_params iparams{ /*no_alloc=*/ false, /*ctx=*/ nullptr };
    gguf_context * gctx = gguf_init_from_file(path.c_str(), iparams);
    if (!gctx) {
        set_last_error("gguf_init_from_file failed: " + path);
        return false;
    }

    out.n_embd     = (int)get_u32(gctx, "qwen3.embedding_length", 1024);
    out.n_ff       = (int)get_u32(gctx, "qwen3.feed_forward_length", 3072);
    out.n_head     = (int)get_u32(gctx, "qwen3.attention.head_count", 16);
    out.n_head_kv  = (int)get_u32(gctx, "qwen3.attention.head_count_kv", 8);
    out.n_layer    = (int)get_u32(gctx, "qwen3.block_count", 28);
    out.n_ctx_max  = (int)get_u32(gctx, "qwen3.context_length", 40960);
    out.head_dim   = (int)get_u32(gctx, "qwen3.attention.key_length", 128);
    out.rope_theta = get_f32(gctx, "qwen3.rope.freq_base", 1000000.0f);

    // Compute total tensor metadata size for context allocation.
    const int n_layer = out.n_layer;
    const int n_tensors_per_layer = 11;
    const int n_top_tensors = 3;
    const int total_tensors = n_top_tensors + n_layer * n_tensors_per_layer;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * total_tensors + 16 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);

    const int n_embd = out.n_embd;
    const int n_ff   = out.n_ff;
    const int n_head = out.n_head;
    const int n_head_kv = out.n_head_kv;
    const int head_dim  = out.head_dim;
    const int n_vocab   = out.n_vocab;
    const int q_dim     = n_head * head_dim;
    const int kv_dim    = n_head_kv * head_dim;

    // Top-level tensors.
    out.tok_embd = ggml_new_tensor_2d(out.ctx, GGML_TYPE_BF16, n_embd, n_vocab);
    out.out_norm = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, n_embd);
    out.output   = ggml_new_tensor_2d(out.ctx, GGML_TYPE_BF16, n_embd, n_vocab);
    ggml_set_name(out.tok_embd, "token_embd.weight");
    ggml_set_name(out.out_norm, "output_norm.weight");
    ggml_set_name(out.output,   "output.weight");

    out.layers.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        auto & L = out.layers[il];
        L.attn_norm = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, n_embd);
        L.wq        = ggml_new_tensor_2d(out.ctx, GGML_TYPE_BF16, n_embd, q_dim);
        L.wk        = ggml_new_tensor_2d(out.ctx, GGML_TYPE_BF16, n_embd, kv_dim);
        L.wv        = ggml_new_tensor_2d(out.ctx, GGML_TYPE_BF16, n_embd, kv_dim);
        L.wo        = ggml_new_tensor_2d(out.ctx, GGML_TYPE_BF16, q_dim, n_embd);
        L.q_norm    = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, head_dim);
        L.k_norm    = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, head_dim);
        L.ffn_norm  = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, n_embd);
        L.ffn_gate  = ggml_new_tensor_2d(out.ctx, GGML_TYPE_BF16, n_embd, n_ff);
        L.ffn_up    = ggml_new_tensor_2d(out.ctx, GGML_TYPE_BF16, n_embd, n_ff);
        L.ffn_down  = ggml_new_tensor_2d(out.ctx, GGML_TYPE_BF16, n_ff, n_embd);
    }

    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        set_last_error("ggml_backend_alloc_ctx_tensors failed for Qwen3-0.6B drafter");
        gguf_free(gctx);
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }

    // mmap the GGUF data section.
    const size_t data_off = gguf_get_data_offset(gctx);
#if defined(_WIN32)
    std::wstring wpath;
    {
        const int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        if (wlen <= 0) {
            set_last_error("MultiByteToWideChar failed for " + path);
            gguf_free(gctx);
            return false;
        }
        wpath.resize(wlen - 1);
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);
    }
    HANDLE hFile = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        set_last_error("CreateFileW failed for " + path);
        gguf_free(gctx);
        return false;
    }
    HANDLE hMapping = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(hFile);
    if (!hMapping) {
        set_last_error("CreateFileMappingA failed for " + path);
        gguf_free(gctx);
        return false;
    }
    void * mm = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMapping);
    if (!mm) {
        set_last_error("MapViewOfFile failed for " + path);
        gguf_free(gctx);
        return false;
    }
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    struct stat st; ::fstat(fd, &st);
    void * mm = ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (mm == MAP_FAILED) {
        set_last_error("mmap failed for " + path);
        gguf_free(gctx);
        return false;
    }
#endif

    bool ok = true;
    ok &= copy_tensor_from_file(gctx, "token_embd.weight", mm, data_off, out.tok_embd);
    ok &= copy_tensor_from_file(gctx, "output_norm.weight", mm, data_off, out.out_norm);
    // Qwen3-0.6B ties lm_head to embed; output.weight is optional.
    if (gguf_find_tensor(gctx, "output.weight") >= 0) {
        ok &= copy_tensor_from_file(gctx, "output.weight", mm, data_off, out.output);
    }
    char nm[128];
    for (int il = 0; il < n_layer; ++il) {
        const auto & L = out.layers[il];
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight",   il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.attn_norm);
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight",      il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.wq);
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight",      il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.wk);
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight",      il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.wv);
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.wo);
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.q_norm);
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.k_norm);
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_norm.weight",    il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ffn_norm);
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight",    il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ffn_gate);
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight",      il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ffn_up);
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight",    il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ffn_down);
    }
#if defined(_WIN32)
    UnmapViewOfFile(mm);
#else
    ::munmap(mm, st.st_size);
#endif
    gguf_free(gctx);

    if (!ok) {
        set_last_error("one or more Qwen3-0.6B tensors failed to load");
        ggml_backend_buffer_free(out.buf);
        ggml_free(out.ctx);
        out.buf = nullptr;
        out.ctx = nullptr;
        return false;
    }
    return true;
}

void free_qwen3_0p6b_drafter(Qwen3DrafterWeights & w) {
    if (w.buf) { ggml_backend_buffer_free(w.buf); w.buf = nullptr; }
    if (w.ctx) { ggml_free(w.ctx); w.ctx = nullptr; }
    w.layers.clear();
    w.tok_embd = w.out_norm = w.output = nullptr;
    w.backend = nullptr;
}

} // namespace dflash27b
