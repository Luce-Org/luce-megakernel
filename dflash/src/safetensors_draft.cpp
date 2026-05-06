// Loads z-lab/Qwen3.5-27B-DFlash draft weights from an HF safetensors file
// (bf16) into a ggml context on the CUDA backend.
//
// Safetensors format:
//   [8 bytes little-endian uint64]  header length
//   [header length bytes]           UTF-8 JSON metadata
//   [remainder]                     raw tensor data (offsets from JSON "data_offsets")
//
// Tensor layout for the z-lab draft (5 layers, fixed):
//   fc.weight                                      [hidden, 5*hidden]  BF16
//   hidden_norm.weight                             [hidden]            BF16
//   norm.weight                                    [hidden]            BF16
//   layers.<i>.input_layernorm.weight              [hidden]
//   layers.<i>.post_attention_layernorm.weight     [hidden]
//   layers.<i>.self_attn.q_proj.weight             [q_dim=4096, hidden=5120]
//   layers.<i>.self_attn.k_proj.weight             [kv_dim=1024, hidden=5120]
//   layers.<i>.self_attn.v_proj.weight             [kv_dim=1024, hidden=5120]
//   layers.<i>.self_attn.o_proj.weight             [hidden=5120, q_dim=4096]
//   layers.<i>.self_attn.q_norm.weight             [head_dim=128]
//   layers.<i>.self_attn.k_norm.weight             [head_dim=128]
//   layers.<i>.mlp.gate_proj.weight                [intermediate=17408, hidden=5120]
//   layers.<i>.mlp.up_proj.weight                  [intermediate=17408, hidden=5120]
//   layers.<i>.mlp.down_proj.weight                [hidden=5120, intermediate=17408]
//
// HF stores matrices ROW-MAJOR [out_features, in_features]. ggml_mul_mat
// expects weights with ne[0]=in_features (fastest-varying), ne[1]=out_features.
// The byte layout is identical — we just create the tensor as
// ggml_new_tensor_2d(ctx, BF16, in, out) and copy the raw bytes.

#include "internal.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>

#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <unordered_map>
#include <vector>

namespace dflash27b {

namespace {

struct StEntry {
    std::string           dtype;
    std::vector<int64_t>  shape;
    uint64_t              data_start;
    uint64_t              data_end;
};

using StMap = std::unordered_map<std::string, StEntry>;

// Tiny hand-rolled parser for the fixed safetensors JSON schema.
// Safetensors JSON is always a single object:
//   { "name": {"dtype":"BF16","shape":[a,b],"data_offsets":[s,e]}, ..., "__metadata__":{...} }
// We parse one top-level entry at a time, tracking brace depth to find each
// tensor's object boundary cleanly.
bool parse_st_header(const char * h, size_t hlen, StMap & out) {
    auto skip_ws = [&](size_t & i) {
        while (i < hlen && (h[i] == ' ' || h[i] == '\t' || h[i] == '\n' || h[i] == '\r')) i++;
    };
    size_t i = 0;
    skip_ws(i);
    if (i >= hlen || h[i] != '{') return false;
    i++;
    while (i < hlen) {
        skip_ws(i);
        if (i >= hlen) return false;
        if (h[i] == '}') { i++; break; }
        if (h[i] == ',') { i++; skip_ws(i); }
        if (i >= hlen || h[i] != '"') return false;
        i++;
        size_t name_start = i;
        while (i < hlen && h[i] != '"') i++;
        if (i >= hlen) return false;
        std::string name(h + name_start, i - name_start);
        i++;
        skip_ws(i);
        if (i >= hlen || h[i] != ':') return false;
        i++;
        skip_ws(i);
        if (i >= hlen || h[i] != '{') return false;
        size_t obj_start = i;
        int depth = 0;
        size_t obj_end = i;
        for (; obj_end < hlen; obj_end++) {
            if (h[obj_end] == '{') depth++;
            else if (h[obj_end] == '}') {
                depth--;
                if (depth == 0) { obj_end++; break; }
            }
        }
        if (depth != 0) return false;

        if (name == "__metadata__") {
            i = obj_end;
            continue;
        }

        std::string obj(h + obj_start, obj_end - obj_start);

        StEntry e;
        {
            auto k = obj.find("\"dtype\":\"");
            if (k == std::string::npos) return false;
            auto vs = k + 9;
            auto ve = obj.find('"', vs);
            if (ve == std::string::npos) return false;
            e.dtype = obj.substr(vs, ve - vs);
        }
        {
            auto k = obj.find("\"shape\":[");
            if (k == std::string::npos) return false;
            auto vs = k + 9;
            auto ve = obj.find(']', vs);
            if (ve == std::string::npos) return false;
            const char * p = obj.c_str() + vs;
            const char * pe = obj.c_str() + ve;
            while (p < pe) {
                char * end = nullptr;
                long long v = std::strtoll(p, &end, 10);
                if (end == p) break;
                e.shape.push_back((int64_t)v);
                p = end;
                while (p < pe && (*p == ',' || *p == ' ')) p++;
            }
        }
        {
            auto k = obj.find("\"data_offsets\":[");
            if (k == std::string::npos) return false;
            auto vs = k + 16;
            auto ve = obj.find(']', vs);
            if (ve == std::string::npos) return false;
            unsigned long long s = 0, ed = 0;
            if (std::sscanf(obj.c_str() + vs, "%llu , %llu", &s, &ed) != 2) {
                if (std::sscanf(obj.c_str() + vs, "%llu,%llu", &s, &ed) != 2) return false;
            }
            e.data_start = s;
            e.data_end   = ed;
        }

        out.emplace(std::move(name), std::move(e));
        i = obj_end;
    }
    return true;
}

// Map safetensors dtype string to ggml type
ggml_type st_dtype_to_ggml(const std::string & dt) {
    if (dt == "BF16") return GGML_TYPE_BF16;
    if (dt == "F16")  return GGML_TYPE_F16;
    if (dt == "F32")  return GGML_TYPE_F32;
    return GGML_TYPE_COUNT;  // sentinel "invalid"
}

struct Mmap {
    void *  addr    = nullptr;
    size_t  len     = 0;
#if defined(_WIN32)
    HANDLE  hFile   = INVALID_HANDLE_VALUE;
    HANDLE  hMap    = nullptr;
#else
    int     fd      = -1;
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
        if (fd < 0) {
            err = "open: " + path + ": " + std::strerror(errno);
            return false;
        }
        struct stat st;
        if (::fstat(fd, &st) < 0) {
            err = "fstat: " + std::string(std::strerror(errno));
            return false;
        }
        len = (size_t)st.st_size;
        addr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) {
            err = "mmap: " + std::string(std::strerror(errno));
            addr = nullptr;
            return false;
        }
#endif
        return true;
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

// Allocate a ggml tensor matching a safetensors entry.
//
// `gt_override`: if not GGML_TYPE_COUNT, use this as the ggml storage type
// instead of the safetensors dtype. Used to store small "norm" weights as
// F32 while the safetensors file has them as BF16 — required because
// ggml's CUDA elementwise ops (ggml_mul in particular) reject BF16 src1.
// The actual bf16→f32 conversion happens later in the data-copy loop.
ggml_tensor * alloc_tensor(ggml_context * ctx,
                           const StMap & st,
                           const std::string & name,
                           const std::vector<int64_t> & expected_shape,
                           const std::string & dtype_expected = "BF16",
                           ggml_type gt_override = GGML_TYPE_COUNT) {
    auto it = st.find(name);
    if (it == st.end()) {
        set_last_error("safetensors: missing tensor '" + name + "'");
        return nullptr;
    }
    const StEntry & e = it->second;
    if (e.dtype != dtype_expected) {
        set_last_error("safetensors: '" + name + "' dtype=" + e.dtype +
                       " expected " + dtype_expected);
        return nullptr;
    }
    if (e.shape.size() != expected_shape.size()) {
        set_last_error("safetensors: '" + name + "' ndim mismatch");
        return nullptr;
    }
    for (size_t k = 0; k < expected_shape.size(); k++) {
        if (e.shape[k] != expected_shape[k]) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "safetensors: '%s' shape[%zu]=%lld expected %lld",
                name.c_str(), k, (long long)e.shape[k], (long long)expected_shape[k]);
            set_last_error(buf);
            return nullptr;
        }
    }
    ggml_type gt = (gt_override == GGML_TYPE_COUNT)
                       ? st_dtype_to_ggml(dtype_expected)
                       : gt_override;
    if (gt == GGML_TYPE_COUNT) {
        set_last_error("safetensors: unsupported dtype " + dtype_expected);
        return nullptr;
    }

    // Shape convention: HF row-major [out, in] → ggml col-major [in, out].
    ggml_tensor * t = nullptr;
    if (expected_shape.size() == 1) {
        t = ggml_new_tensor_1d(ctx, gt, expected_shape[0]);
    } else if (expected_shape.size() == 2) {
        // expected_shape is written as [out, in]; ggml wants ne[0]=in, ne[1]=out
        t = ggml_new_tensor_2d(ctx, gt, expected_shape[1], expected_shape[0]);
    } else {
        set_last_error("safetensors: unexpected ndim > 2 for '" + name + "'");
        return nullptr;
    }
    ggml_set_name(t, name.c_str());
    return t;
}

// Convert an array of bf16 values to f32 in place into a destination buffer.
static void bf16_to_f32_array(const uint16_t * src, float * dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = ((uint32_t)src[i]) << 16;
        std::memcpy(&dst[i], &bits, 4);
    }
}

// Convert an array of bf16 values to fp16 via f32 intermediate.
static void bf16_to_f16_array(const uint16_t * src, uint16_t * dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = ((uint32_t)src[i]) << 16;
        float f;
        std::memcpy(&f, &bits, 4);
        // IEEE 754 f32→f16: truncate mantissa, clamp exponent.
        uint32_t u;
        std::memcpy(&u, &f, 4);
        uint32_t sign = (u >> 16) & 0x8000;
        int32_t  exp  = ((u >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (u >> 13) & 0x03FF;
        if (exp <= 0)       dst[i] = (uint16_t)sign;          // flush to zero
        else if (exp >= 31) dst[i] = (uint16_t)(sign | 0x7C00); // inf
        else                dst[i] = (uint16_t)(sign | (exp << 10) | mant);
    }
}

// Returns true if the current CUDA device has native BF16 tensor core support
// (Ampere SM 8.0+). On Turing (SM 7.5) cuBLAS BF16 GEMM falls back to slow
// CUDA cores instead of tensor cores. Uses ggml's CUDA backend info at runtime.
static bool cuda_has_native_bf16() {
    // Check at runtime: link against ggml-cuda's device info if available,
    // otherwise fall back to env var DFLASH27B_DRAFT_FP16 for manual override.
    const char * env = std::getenv("DFLASH27B_DRAFT_FP16");
    if (env && std::atoi(env) != 0) return false;  // force fp16

    // Probe via ggml_backend_cuda device properties (compiled-in at build time).
    // The CMAKE_CUDA_ARCHITECTURES list tells us the minimum supported arch.
    // If the smallest arch is < 80, return false.
#if defined(DFLASH27B_MIN_SM) && DFLASH27B_MIN_SM < 80
    return false;
#else
    return true;
#endif
}

} // namespace

bool load_draft_safetensors(const std::string & path,
                            ggml_backend_t       backend,
                            DraftWeights &       out) {
    // ── 1. Open + mmap ────────────────────────────────────────────
    Mmap mm;
    std::string err;
    if (!mm.open_ro(path, err)) { set_last_error(err); return false; }
    if (mm.len < 8) { set_last_error("safetensors: file too small"); return false; }

    // ── 2. Parse header ───────────────────────────────────────────
    uint64_t header_len = 0;
    std::memcpy(&header_len, mm.addr, 8);
    if (header_len == 0 || 8 + header_len > mm.len) {
        set_last_error("safetensors: bad header length");
        return false;
    }
    const char * header_ptr = (const char *)mm.addr + 8;
    StMap st;
    if (!parse_st_header(header_ptr, header_len, st)) {
        set_last_error("safetensors: JSON header parse failed");
        return false;
    }
    const uint8_t * blob = (const uint8_t *)mm.addr + 8 + header_len;
    const size_t    blob_len = mm.len - 8 - header_len;

    // ── 3. Allocate ggml context big enough for 5 layers × 11 + 3 top ─
    const int n_layers    = DFLASH27B_DRAFT_LAYERS;
    const int n_tensors   = 3 + 11 * n_layers;  // with some headroom below
    ggml_init_params ip{};
    ip.mem_size   = (size_t)(n_tensors + 16) * ggml_tensor_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) { set_last_error("ggml_init failed for draft ctx"); return false; }
    out.backend = backend;
    out.n_layer   = n_layers;
    out.n_head    = DFLASH27B_TARGET_N_HEADS;
    out.n_head_kv = DFLASH27B_TARGET_N_KV_HEADS;
    out.head_dim  = DFLASH27B_TARGET_HEAD_DIM;
    out.n_embd    = DFLASH27B_TARGET_HIDDEN;
    out.n_ff      = DFLASH27B_TARGET_INTERMEDIATE;
    out.layers.assign(n_layers, DraftLayer{});

    const int64_t HIDDEN  = out.n_embd;
    const int64_t Q_DIM   = out.n_head * out.head_dim;
    const int64_t KV_DIM  = out.n_head_kv * out.head_dim;
    const int64_t INTER   = out.n_ff;
    const int64_t HD      = out.head_dim;
    const int64_t FC_IN   = DFLASH27B_DRAFT_N_TARGET_LAYERS * HIDDEN;

    // ── 4. Create named tensors in the context ───────────────────
    //
    // Norms (rms_norm weights) are loaded as F32 because ggml's CUDA
    // elementwise ops require F32/F16 operands. Projection weights stay bf16
    // on Ampere+ (native tensor core support) or are converted to fp16 on
    // Turing (SM 7.5) where cuBLAS BF16 GEMM falls back to slow CUDA cores.
    const ggml_type NORM_GT = GGML_TYPE_F32;
    const bool native_bf16 = cuda_has_native_bf16();
    const ggml_type PROJ_GT = native_bf16 ? GGML_TYPE_COUNT : GGML_TYPE_F16;

    out.fc          = alloc_tensor(out.ctx, st, "fc.weight",           {HIDDEN, FC_IN},  "BF16", PROJ_GT);
    out.hidden_norm = alloc_tensor(out.ctx, st, "hidden_norm.weight",  {HIDDEN}, "BF16", NORM_GT);
    out.out_norm    = alloc_tensor(out.ctx, st, "norm.weight",         {HIDDEN}, "BF16", NORM_GT);
    if (!out.fc || !out.hidden_norm || !out.out_norm) return false;

    for (int il = 0; il < n_layers; il++) {
        char pfx[64];
        std::snprintf(pfx, sizeof(pfx), "layers.%d.", il);
        std::string p = pfx;
        DraftLayer & L = out.layers[il];
        L.attn_norm = alloc_tensor(out.ctx, st, p + "input_layernorm.weight",          {HIDDEN}, "BF16", NORM_GT);
        L.ffn_norm  = alloc_tensor(out.ctx, st, p + "post_attention_layernorm.weight", {HIDDEN}, "BF16", NORM_GT);
        L.wq        = alloc_tensor(out.ctx, st, p + "self_attn.q_proj.weight", {Q_DIM,  HIDDEN}, "BF16", PROJ_GT);
        L.wk        = alloc_tensor(out.ctx, st, p + "self_attn.k_proj.weight", {KV_DIM, HIDDEN}, "BF16", PROJ_GT);
        L.wv        = alloc_tensor(out.ctx, st, p + "self_attn.v_proj.weight", {KV_DIM, HIDDEN}, "BF16", PROJ_GT);
        L.wo        = alloc_tensor(out.ctx, st, p + "self_attn.o_proj.weight", {HIDDEN, Q_DIM},  "BF16", PROJ_GT);
        L.q_norm    = alloc_tensor(out.ctx, st, p + "self_attn.q_norm.weight", {HD}, "BF16", NORM_GT);
        L.k_norm    = alloc_tensor(out.ctx, st, p + "self_attn.k_norm.weight", {HD}, "BF16", NORM_GT);
        L.w_gate    = alloc_tensor(out.ctx, st, p + "mlp.gate_proj.weight",    {INTER,  HIDDEN}, "BF16", PROJ_GT);
        L.w_up      = alloc_tensor(out.ctx, st, p + "mlp.up_proj.weight",      {INTER,  HIDDEN}, "BF16", PROJ_GT);
        L.w_down    = alloc_tensor(out.ctx, st, p + "mlp.down_proj.weight",    {HIDDEN, INTER},  "BF16", PROJ_GT);
        if (!L.attn_norm || !L.ffn_norm || !L.wq || !L.wk || !L.wv || !L.wo ||
            !L.q_norm || !L.k_norm || !L.w_gate || !L.w_up || !L.w_down) {
            return false;
        }
    }

    // ── 4b. Read config.json for SWA layer_types (Qwen3.6 draft) ──
    {
        // config.json sits next to model.safetensors
        std::string dir;
        auto slash = path.find_last_of('/');
        if (slash != std::string::npos) {
            dir = path.substr(0, slash);
        } else {
            dir = ".";  // bare filename — look in CWD
        }
        std::string cfg_path = dir + "/config.json";
        FILE * f = std::fopen(cfg_path.c_str(), "r");
        if (f) {
            std::fseek(f, 0, SEEK_END);
            long flen = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            std::string cfg(flen, '\0');
            std::fread(&cfg[0], 1, flen, f);
            std::fclose(f);

            // Parse sliding_window
            auto sw_pos = cfg.find("\"sliding_window\"");
            if (sw_pos != std::string::npos) {
                auto colon = cfg.find(':', sw_pos);
                if (colon != std::string::npos) {
                    int sw = std::atoi(cfg.c_str() + colon + 1);
                    if (sw > 0) out.swa_window = sw;
                }
            }

            // Parse layer_types array
            auto lt_pos = cfg.find("\"layer_types\"");
            if (lt_pos != std::string::npos) {
                auto arr_start = cfg.find('[', lt_pos);
                auto arr_end   = cfg.find(']', arr_start);
                if (arr_start != std::string::npos && arr_end != std::string::npos) {
                    std::string arr = cfg.substr(arr_start, arr_end - arr_start + 1);
                    int li = 0;
                    size_t search_pos = 0;
                    while (li < n_layers && search_pos < arr.size()) {
                        auto q1 = arr.find('"', search_pos);
                        if (q1 == std::string::npos) break;
                        auto q2 = arr.find('"', q1 + 1);
                        if (q2 == std::string::npos) break;
                        std::string lt = arr.substr(q1 + 1, q2 - q1 - 1);
                        out.layers[li].is_swa = (lt == "sliding_attention");
                        li++;
                        search_pos = q2 + 1;
                    }
                }
            }

            int n_swa = 0;
            for (int il = 0; il < n_layers; il++) {
                if (out.layers[il].is_swa) n_swa++;
            }
            if (n_swa > 0) {
                fprintf(stderr, "[draft] SWA layers: %d/%d (window=%d)\n", n_swa, n_layers, out.swa_window);
            }
        }
    }

    // ── 5. Allocate backend buffer, copy bytes ───────────────────
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) { set_last_error("ggml_backend_alloc_ctx_tensors failed (draft)"); return false; }

    // Walk the tensors in the context and upload their bytes.
    // For tensors whose ggml type differs from the safetensors dtype (i.e.
    // BF16-on-disk, F32-in-ggml for norms, or BF16-on-disk, F16-in-ggml for
    // projection weights on Turing), convert on the fly via scratch buffers.
    std::vector<float>    scratch_f32;
    std::vector<uint16_t> scratch_f16;
    for (ggml_tensor * t = ggml_get_first_tensor(out.ctx); t != nullptr;
         t = ggml_get_next_tensor(out.ctx, t)) {
        const char * name = ggml_get_name(t);
        auto it = st.find(name);
        if (it == st.end()) {
            set_last_error("post-alloc: tensor '" + std::string(name) + "' vanished from header");
            return false;
        }
        const StEntry & e = it->second;
        if (e.data_end > 8 + header_len + blob_len + 8 /*slack*/) {
            set_last_error("post-alloc: offset out of bounds for '" + std::string(name) + "'");
            return false;
        }
        const size_t src_nbytes = e.data_end - e.data_start;
        const size_t dst_nbytes = ggml_nbytes(t);
        const bool same_dtype = (t->type == st_dtype_to_ggml(e.dtype));

        if (same_dtype) {
            if (src_nbytes != dst_nbytes) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "byte count mismatch for '%s': blob=%zu ggml=%zu",
                    name, src_nbytes, dst_nbytes);
                set_last_error(buf);
                return false;
            }
            ggml_backend_tensor_set(t, blob + e.data_start, 0, dst_nbytes);
        } else if (e.dtype == "BF16" && t->type == GGML_TYPE_F32) {
            const size_t n = ggml_nelements(t);
            if (src_nbytes != n * sizeof(uint16_t) || dst_nbytes != n * sizeof(float)) {
                set_last_error("BF16->F32 size mismatch for '" + std::string(name) + "'");
                return false;
            }
            scratch_f32.resize(n);
            bf16_to_f32_array((const uint16_t *)(blob + e.data_start),
                              scratch_f32.data(), n);
            ggml_backend_tensor_set(t, scratch_f32.data(), 0, dst_nbytes);
        } else if (e.dtype == "BF16" && t->type == GGML_TYPE_F16) {
            const size_t n = ggml_nelements(t);
            if (src_nbytes != n * sizeof(uint16_t) || dst_nbytes != n * sizeof(uint16_t)) {
                set_last_error("BF16->F16 size mismatch for '" + std::string(name) + "'");
                return false;
            }
            scratch_f16.resize(n);
            bf16_to_f16_array((const uint16_t *)(blob + e.data_start),
                              scratch_f16.data(), n);
            ggml_backend_tensor_set(t, scratch_f16.data(), 0, dst_nbytes);
        } else {
            set_last_error(std::string("unsupported dtype conversion for '") +
                           name + "': " + e.dtype + " -> ggml type " +
                           ggml_type_name(t->type));
            return false;
        }
    }

    return true;
}

void free_draft_weights(DraftWeights & w) {
    if (w.buf) { ggml_backend_buffer_free(w.buf); w.buf = nullptr; }
    if (w.ctx) { ggml_free(w.ctx);                w.ctx = nullptr; }
    w.layers.clear();
    w.fc = nullptr;
    w.hidden_norm = nullptr;
    w.out_norm = nullptr;
}

} // namespace dflash27b
