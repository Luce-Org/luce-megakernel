#include "mega_pflash_native.h"
#include "internal.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unordered_map>
#include <unistd.h>

namespace dflash27b {

namespace {

constexpr int NUM_LAYERS = 24;
constexpr int HIDDEN = 1024;
constexpr int INTER = 3584;
constexpr int VOCAB = 248320;
constexpr int FA_Q_HEADS = 8;
constexpr int FA_KV_HEADS = 2;
constexpr int FA_HEAD_DIM = 256;
constexpr int FA_Q_SIZE = FA_Q_HEADS * FA_HEAD_DIM;
constexpr int FA_QPROJ_SIZE = FA_Q_SIZE * 2;
constexpr int FA_KV_SIZE = FA_KV_HEADS * FA_HEAD_DIM;
constexpr int DN_HEADS = 16;
constexpr int DN_KEY = 128;
constexpr int DN_VAL = 128;
constexpr int DN_CONV_K = 4;
constexpr int DN_QK_SIZE = DN_HEADS * DN_KEY;
constexpr int DN_V_SIZE = DN_HEADS * DN_VAL;
constexpr int DN_CONV_CH = DN_QK_SIZE * 2 + DN_V_SIZE;
constexpr int DN_CHUNK_C = 16;
constexpr int LAYER_TYPE[NUM_LAYERS] = {
    0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1
};

struct NativeLayerWeights {
    int layer_type;
    int pad[3];
    void *ptrs[14];
};

struct StEntry {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t data_start = 0;
    uint64_t data_end = 0;
};

using StMap = std::unordered_map<std::string, StEntry>;

bool parse_st_header(const char *h, size_t hlen, StMap &out) {
    auto skip_ws = [&](size_t &i) {
        while (i < hlen && (h[i] == ' ' || h[i] == '\t' || h[i] == '\n' || h[i] == '\r')) i++;
    };
    size_t i = 0;
    skip_ws(i);
    if (i >= hlen || h[i] != '{') return false;
    i++;
    while (i < hlen) {
        skip_ws(i);
        if (i >= hlen) return false;
        if (h[i] == '}') break;
        if (h[i] == ',') { i++; skip_ws(i); }
        if (i >= hlen || h[i] != '"') return false;
        i++;
        size_t ns = i;
        while (i < hlen && h[i] != '"') i++;
        if (i >= hlen) return false;
        std::string name(h + ns, i - ns);
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
        if (name == "__metadata__") { i = obj_end; continue; }
        std::string obj(h + obj_start, obj_end - obj_start);
        StEntry e;
        auto dk = obj.find("\"dtype\":\"");
        if (dk == std::string::npos) return false;
        auto ds = dk + 9;
        auto de = obj.find('"', ds);
        if (de == std::string::npos) return false;
        e.dtype = obj.substr(ds, de - ds);
        auto sk = obj.find("\"shape\":[");
        if (sk == std::string::npos) return false;
        auto ss = sk + 9;
        auto se = obj.find(']', ss);
        if (se == std::string::npos) return false;
        const char *p = obj.c_str() + ss;
        const char *pe = obj.c_str() + se;
        while (p < pe) {
            char *end = nullptr;
            long long v = std::strtoll(p, &end, 10);
            if (end == p) break;
            e.shape.push_back((int64_t)v);
            p = end;
            while (p < pe && (*p == ',' || *p == ' ')) p++;
        }
        auto ok = obj.find("\"data_offsets\":[");
        if (ok == std::string::npos) return false;
        auto os = ok + 16;
        unsigned long long a = 0, b = 0;
        if (std::sscanf(obj.c_str() + os, "%llu,%llu", &a, &b) != 2 &&
            std::sscanf(obj.c_str() + os, "%llu , %llu", &a, &b) != 2) {
            return false;
        }
        e.data_start = a;
        e.data_end = b;
        out.emplace(std::move(name), std::move(e));
        i = obj_end;
    }
    return true;
}

struct MmapFile {
    int fd = -1;
    void *addr = nullptr;
    size_t len = 0;
    ~MmapFile() {
        if (addr && addr != MAP_FAILED) ::munmap(addr, len);
        if (fd >= 0) ::close(fd);
    }
    bool open(const std::string &path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st{};
        if (::fstat(fd, &st) != 0) return false;
        len = (size_t)st.st_size;
        addr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        return addr != MAP_FAILED;
    }
};

std::string resolve_model_file(const std::string &model_path) {
    namespace fs = std::filesystem;
    fs::path p(model_path);
    if (fs::is_regular_file(p)) return p.string();
    if (fs::is_directory(p)) {
        fs::path shard = p / "model.safetensors-00001-of-00001.safetensors";
        if (fs::is_regular_file(shard)) return shard.string();
        fs::path plain = p / "model.safetensors";
        if (fs::is_regular_file(plain)) return plain.string();
    }
    return model_path;
}

uint16_t f32_to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}

bool same_shape(const StEntry &e, std::initializer_list<int64_t> shape) {
    return e.shape.size() == shape.size() &&
           std::equal(e.shape.begin(), e.shape.end(), shape.begin());
}

void *alloc_dev(MegaPFlashContext &ctx, size_t bytes) {
    void *p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
    ctx.owned_device_ptrs.push_back(p);
    return p;
}

bool upload_tensor(MegaPFlashContext &ctx, const StMap &st, const uint8_t *blob,
                   const std::string &name, std::initializer_list<int64_t> shape,
                   void **out) {
    auto it = st.find(name);
    if (it == st.end()) {
        set_last_error("mega-pflash: missing tensor '" + name + "'");
        return false;
    }
    const StEntry &e = it->second;
    if (!same_shape(e, shape)) {
        set_last_error("mega-pflash: shape mismatch for '" + name + "'");
        return false;
    }
    size_t n = 1;
    for (int64_t d : e.shape) n *= (size_t)d;
    size_t bytes = n * sizeof(uint16_t);
    void *dst = alloc_dev(ctx, bytes);
    if (!dst) {
        set_last_error("mega-pflash: cudaMalloc failed for '" + name + "'");
        return false;
    }
    const uint8_t *src = blob + e.data_start;
    if (e.dtype == "BF16") {
        if ((e.data_end - e.data_start) != bytes) {
            set_last_error("mega-pflash: byte mismatch for '" + name + "'");
            return false;
        }
        if (cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            set_last_error("mega-pflash: cudaMemcpy failed for '" + name + "'");
            return false;
        }
    } else if (e.dtype == "F32") {
        if ((e.data_end - e.data_start) != n * sizeof(float)) {
            set_last_error("mega-pflash: F32 byte mismatch for '" + name + "'");
            return false;
        }
        std::vector<uint16_t> tmp(n);
        const float *f = reinterpret_cast<const float *>(src);
        for (size_t i = 0; i < n; ++i) tmp[i] = f32_to_bf16(f[i]);
        if (cudaMemcpy(dst, tmp.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            set_last_error("mega-pflash: cudaMemcpy(converted) failed for '" + name + "'");
            return false;
        }
    } else {
        set_last_error("mega-pflash: unsupported dtype " + e.dtype + " for '" + name + "'");
        return false;
    }
    *out = dst;
    return true;
}

bool copy_tensor_to_device(const StMap &st, const uint8_t *blob,
                           const std::string &name, std::initializer_list<int64_t> shape,
                           void *dst) {
    auto it = st.find(name);
    if (it == st.end()) {
        set_last_error("mega-pflash: missing tensor '" + name + "'");
        return false;
    }
    const StEntry &e = it->second;
    if (!same_shape(e, shape)) {
        set_last_error("mega-pflash: shape mismatch for '" + name + "'");
        return false;
    }
    size_t n = 1;
    for (int64_t d : e.shape) n *= (size_t)d;
    const size_t bytes = n * sizeof(uint16_t);
    const uint8_t *src = blob + e.data_start;
    if (e.dtype == "BF16") {
        if ((e.data_end - e.data_start) != bytes) {
            set_last_error("mega-pflash: byte mismatch for '" + name + "'");
            return false;
        }
        if (cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            set_last_error("mega-pflash: cudaMemcpy failed for '" + name + "'");
            return false;
        }
    } else if (e.dtype == "F32") {
        if ((e.data_end - e.data_start) != n * sizeof(float)) {
            set_last_error("mega-pflash: F32 byte mismatch for '" + name + "'");
            return false;
        }
        std::vector<uint16_t> tmp(n);
        const float *f = reinterpret_cast<const float *>(src);
        for (size_t i = 0; i < n; ++i) tmp[i] = f32_to_bf16(f[i]);
        if (cudaMemcpy(dst, tmp.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            set_last_error("mega-pflash: cudaMemcpy(converted) failed for '" + name + "'");
            return false;
        }
    } else {
        set_last_error("mega-pflash: unsupported dtype " + e.dtype + " for '" + name + "'");
        return false;
    }
    return true;
}

bool upload_tensor_alias(MegaPFlashContext &ctx, const StMap &st, const uint8_t *blob,
                         const std::string &name, std::initializer_list<int64_t> shape,
                         void *base, size_t &offset_bytes, void **out) {
    auto it = st.find(name);
    if (it == st.end()) {
        set_last_error("mega-pflash: missing tensor '" + name + "'");
        return false;
    }
    size_t n = 1;
    for (int64_t d : it->second.shape) n *= (size_t)d;
    const size_t bytes = n * sizeof(uint16_t);
    void *dst = (char *)base + offset_bytes;
    if (!copy_tensor_to_device(st, blob, name, shape, dst)) return false;
    *out = dst;
    offset_bytes += bytes;
    return true;
}

bool upload_concat(MegaPFlashContext &ctx, const StMap &st, const uint8_t *blob,
                   const std::vector<std::pair<std::string, std::initializer_list<int64_t>>> &parts,
                   void *dst, size_t &offset_bytes) {
    for (const auto &part : parts) {
        void *tmp = nullptr;
        if (!upload_tensor(ctx, st, blob, part.first, part.second, &tmp)) return false;
        auto it = st.find(part.first);
        size_t n = 1;
        for (int64_t d : it->second.shape) n *= (size_t)d;
        size_t bytes = n * sizeof(uint16_t);
        if (cudaMemcpy((char *)dst + offset_bytes, tmp, bytes, cudaMemcpyDeviceToDevice) != cudaSuccess) {
            set_last_error("mega-pflash: cudaMemcpy concat failed for '" + part.first + "'");
            return false;
        }
        offset_bytes += bytes;
    }
    return true;
}

int div_up(int a, int b) { return (a + b - 1) / b; }

bool env_enabled(const char *name) {
    const char *v = std::getenv(name);
    return v && std::atoi(v) != 0;
}

} // namespace

extern "C" void launch_prefill_bf16(
    const int *token_ids, int seq_len, int *output_token,
    const void *embed_weight, const void *layers,
    const void *final_norm_w, const void *lm_head_w,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden, void *residual, void *normalized,
    void *proj_buf, void *proj_buf2, void *attn_buf, void *mlp_buf,
    void *dn_out_buf,
    void *beta_buf, void *alpha_buf, void *dn_pre_qkv,
    void *dn_u_scratch, void *dn_w_scratch, void *dn_cs_scratch,
    const void *fused_fa_qkv_base, const void *fused_gate_up_base,
    void *final_normed, void *hidden_bf16_out,
    void *lm_bmv, void *lm_bmi,
    void *fa_q_tail,
    int max_seq_len, int q_tail_len,
    cudaStream_t stream);

extern "C" void launch_mega_pflash_score(
    const void *q_tail,
    const void *k_cache,
    float *logit_scratch,
    float *token_scores,
    float *chunk_scores,
    int seq_len,
    int max_seq_len,
    int tail_len,
    int chunk_size,
    int smooth_kernel,
    int reduction_mode,
    cudaStream_t stream);

bool load_mega_pflash(const std::string & model_path,
                      int max_seq_len,
                      MegaPFlashContext & out) {
    if (out.loaded) {
        set_last_error("mega-pflash: already loaded");
        return false;
    }
    if (max_seq_len <= 0) {
        set_last_error("mega-pflash: max_seq_len must be positive");
        return false;
    }

    const std::string file = resolve_model_file(model_path);
    MmapFile mm;
    if (!mm.open(file)) {
        set_last_error("mega-pflash: mmap failed for " + file);
        return false;
    }
    if (mm.len < 8) {
        set_last_error("mega-pflash: safetensors file too small");
        return false;
    }
    uint64_t header_len = 0;
    std::memcpy(&header_len, mm.addr, 8);
    if (header_len == 0 || 8 + header_len >= mm.len) {
        set_last_error("mega-pflash: bad safetensors header length");
        return false;
    }
    StMap st;
    if (!parse_st_header((const char *)mm.addr + 8, header_len, st)) {
        set_last_error("mega-pflash: safetensors header parse failed");
        return false;
    }
    const uint8_t *blob = (const uint8_t *)mm.addr + 8 + header_len;
    const std::string pfx = "model.language_model.";

    out.model_path = file;
    out.max_seq_len = max_seq_len;

    std::vector<NativeLayerWeights> host_layers(NUM_LAYERS);
    auto up = [&](const std::string &name, std::initializer_list<int64_t> shape, void **ptr) {
        return upload_tensor(out, st, blob, pfx + name, shape, ptr);
    };
    auto up_alias = [&](const std::string &name, std::initializer_list<int64_t> shape,
                        void *base, size_t &off, void **ptr) {
        return upload_tensor_alias(out, st, blob, pfx + name, shape, base, off, ptr);
    };

    if (!up("embed_tokens.weight", {VOCAB, HIDDEN}, &out.embed_weight)) return false;
    if (!up("norm.weight", {HIDDEN}, &out.final_norm_weight)) return false;
    out.lm_head_weight = out.embed_weight;

    const size_t fa_qkv_bytes = (size_t)6 * (FA_QPROJ_SIZE + 2 * FA_KV_SIZE) * HIDDEN * sizeof(uint16_t);
    const size_t gate_up_bytes = (size_t)NUM_LAYERS * (2 * INTER) * HIDDEN * sizeof(uint16_t);
    out.fused_fa_qkv = alloc_dev(out, fa_qkv_bytes);
    out.fused_gate_up = alloc_dev(out, gate_up_bytes);
    if (!out.fused_fa_qkv || !out.fused_gate_up) {
        set_last_error("mega-pflash: fused weight cudaMalloc failed");
        return false;
    }

    size_t fa_fused_off = 0;
    for (int li = 0; li < NUM_LAYERS; ++li) {
        NativeLayerWeights &lw = host_layers[li];
        lw.layer_type = LAYER_TYPE[li];
        std::fill(std::begin(lw.ptrs), std::end(lw.ptrs), nullptr);
        char base[128];
        std::snprintf(base, sizeof(base), "layers.%d.", li);
        std::string b = base;
        if (!up(b + "input_layernorm.weight", {HIDDEN}, &lw.ptrs[0])) return false;
        if (LAYER_TYPE[li] == 0) {
            if (!up(b + "linear_attn.in_proj_qkv.weight", {DN_CONV_CH, HIDDEN}, &lw.ptrs[1])) return false;
            if (!up(b + "linear_attn.in_proj_z.weight", {DN_V_SIZE, HIDDEN}, &lw.ptrs[2])) return false;
            if (!up(b + "linear_attn.in_proj_b.weight", {DN_HEADS, HIDDEN}, &lw.ptrs[3])) return false;
            if (!up(b + "linear_attn.in_proj_a.weight", {DN_HEADS, HIDDEN}, &lw.ptrs[4])) return false;
            if (!up(b + "linear_attn.conv1d.weight", {DN_CONV_CH, 1, DN_CONV_K}, &lw.ptrs[5])) return false;
            if (!up(b + "linear_attn.A_log", {DN_HEADS}, &lw.ptrs[6])) return false;
            if (!up(b + "linear_attn.dt_bias", {DN_HEADS}, &lw.ptrs[7])) return false;
            if (!up(b + "linear_attn.norm.weight", {DN_VAL}, &lw.ptrs[8])) return false;
            if (!up(b + "linear_attn.out_proj.weight", {HIDDEN, DN_V_SIZE}, &lw.ptrs[9])) return false;
            if (!up(b + "post_attention_layernorm.weight", {HIDDEN}, &lw.ptrs[10])) return false;
            size_t gu_off = (size_t)li * (2 * INTER) * HIDDEN * sizeof(uint16_t);
            if (!up_alias(b + "mlp.gate_proj.weight", {INTER, HIDDEN},
                          out.fused_gate_up, gu_off, &lw.ptrs[11])) return false;
            if (!up_alias(b + "mlp.up_proj.weight", {INTER, HIDDEN},
                          out.fused_gate_up, gu_off, &lw.ptrs[12])) return false;
            if (!up(b + "mlp.down_proj.weight", {HIDDEN, INTER}, &lw.ptrs[13])) return false;
        } else {
            if (!up_alias(b + "self_attn.q_proj.weight", {FA_QPROJ_SIZE, HIDDEN},
                          out.fused_fa_qkv, fa_fused_off, &lw.ptrs[1])) return false;
            if (!up_alias(b + "self_attn.k_proj.weight", {FA_KV_SIZE, HIDDEN},
                          out.fused_fa_qkv, fa_fused_off, &lw.ptrs[2])) return false;
            if (!up_alias(b + "self_attn.v_proj.weight", {FA_KV_SIZE, HIDDEN},
                          out.fused_fa_qkv, fa_fused_off, &lw.ptrs[3])) return false;
            if (!up(b + "self_attn.q_norm.weight", {FA_HEAD_DIM}, &lw.ptrs[4])) return false;
            if (!up(b + "self_attn.k_norm.weight", {FA_HEAD_DIM}, &lw.ptrs[5])) return false;
            if (!up(b + "self_attn.o_proj.weight", {HIDDEN, FA_Q_SIZE}, &lw.ptrs[6])) return false;
            if (!up(b + "post_attention_layernorm.weight", {HIDDEN}, &lw.ptrs[7])) return false;
            size_t gu_off = (size_t)li * (2 * INTER) * HIDDEN * sizeof(uint16_t);
            if (!up_alias(b + "mlp.gate_proj.weight", {INTER, HIDDEN},
                          out.fused_gate_up, gu_off, &lw.ptrs[8])) return false;
            if (!up_alias(b + "mlp.up_proj.weight", {INTER, HIDDEN},
                          out.fused_gate_up, gu_off, &lw.ptrs[9])) return false;
            if (!up(b + "mlp.down_proj.weight", {HIDDEN, INTER}, &lw.ptrs[10])) return false;
        }
    }

    out.layer_weights_dev = alloc_dev(out, host_layers.size() * sizeof(NativeLayerWeights));
    if (!out.layer_weights_dev) {
        set_last_error("mega-pflash: layer pack cudaMalloc failed");
        return false;
    }
    if (cudaMemcpy(out.layer_weights_dev, host_layers.data(),
                   host_layers.size() * sizeof(NativeLayerWeights),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        set_last_error("mega-pflash: layer pack cudaMemcpy failed");
        return false;
    }

    const int S = max_seq_len;
    const int S_pad = div_up(S, 32) * 32;
    const int mx = std::max({DN_CONV_CH, FA_QPROJ_SIZE, INTER});
    auto alloc = [&](void **p, size_t bytes, const char *label) -> bool {
        *p = alloc_dev(out, bytes);
        if (!*p) {
            set_last_error(std::string("mega-pflash: cudaMalloc failed for ") + label);
            return false;
        }
        return true;
    };
    const bool use_bsa = env_enabled("DFLASH_FP_USE_BSA");
    if (!alloc(&out.fa_k_cache, (size_t)6 * FA_KV_HEADS * S * FA_HEAD_DIM * 2, "fa_k_cache")) return false;
    if (!use_bsa) {
        if (!alloc(&out.fa_v_cache, (size_t)6 * FA_KV_HEADS * S * FA_HEAD_DIM * 2, "fa_v_cache")) return false;
    }
    if (!alloc(&out.fa_q_tail, (size_t)6 * 8 * FA_Q_HEADS * FA_HEAD_DIM * 2, "fa_q_tail")) return false;
    if (!alloc(&out.dn_states, (size_t)18 * DN_HEADS * DN_KEY * DN_VAL * sizeof(float), "dn_states")) return false;
    if (!alloc(&out.conv_bufs, (size_t)18 * DN_CONV_CH * DN_CONV_K * sizeof(float), "conv_bufs")) return false;
    if (!alloc(&out.hidden, (size_t)S * HIDDEN * 2, "hidden")) return false;
    if (!alloc(&out.residual, (size_t)S * HIDDEN * 2, "residual")) return false;
    if (!alloc(&out.normalized, (size_t)S * HIDDEN * 2, "normalized")) return false;
    if (!alloc(&out.proj_buf, (size_t)S * mx * 2, "proj_buf")) return false;
    if (!alloc(&out.proj_buf2, (size_t)S * mx * 2, "proj_buf2")) return false;
    if (!alloc(&out.attn_buf, (size_t)S * std::max(FA_Q_SIZE, FA_KV_SIZE) * 2, "attn_buf")) return false;
    if (!alloc(&out.mlp_buf, (size_t)S * INTER * 2, "mlp_buf")) return false;
    if (!alloc(&out.dn_out_buf, (size_t)S * DN_V_SIZE * 2, "dn_out_buf")) return false;
    if (!alloc(&out.beta_buf, (size_t)S * DN_HEADS * sizeof(float), "beta_buf")) return false;
    if (!alloc(&out.alpha_buf, (size_t)S * DN_HEADS * sizeof(float), "alpha_buf")) return false;
    if (!alloc(&out.dn_pre_qkv, (size_t)S * DN_CONV_CH * sizeof(float), "dn_pre_qkv")) return false;
    if (!alloc(&out.dn_u_scratch, (size_t)S_pad * DN_HEADS * 128 * sizeof(float), "dn_u_scratch")) return false;
    if (!alloc(&out.dn_w_scratch, (size_t)S_pad * DN_HEADS * 128 * sizeof(float), "dn_w_scratch")) return false;
    if (!alloc(&out.dn_cs_scratch, (size_t)S_pad * DN_HEADS * sizeof(float), "dn_cs_scratch")) return false;
    if (!env_enabled("DFLASH_MEGA_PFLASH_WITH_LM_HEAD")) {
        out.final_normed = nullptr;
        out.hidden_bf16_out = nullptr;
        out.lm_bmv = nullptr;
        out.lm_bmi = nullptr;
        out.output_token = nullptr;
    } else {
        if (!alloc(&out.final_normed, HIDDEN * 2, "final_normed")) return false;
        if (!alloc(&out.hidden_bf16_out, HIDDEN * 2, "hidden_bf16_out")) return false;
        if (!alloc(&out.lm_bmv, 1024 * sizeof(float), "lm_bmv")) return false;
        if (!alloc(&out.lm_bmi, 1024 * sizeof(int), "lm_bmi")) return false;
        if (!alloc(&out.output_token, sizeof(int), "output_token")) return false;
    }

    cudaMemset(out.dn_states, 0, (size_t)18 * DN_HEADS * DN_KEY * DN_VAL * sizeof(float));
    cudaMemset(out.conv_bufs, 0, (size_t)18 * DN_CONV_CH * DN_CONV_K * sizeof(float));
    cudaMemset(out.fa_k_cache, 0, (size_t)6 * FA_KV_HEADS * S * FA_HEAD_DIM * 2);
    if (out.fa_v_cache) {
        cudaMemset(out.fa_v_cache, 0, (size_t)6 * FA_KV_HEADS * S * FA_HEAD_DIM * 2);
    }
    cudaDeviceSynchronize();

    out.loaded = true;
    std::fprintf(stderr, "[mega-pflash] native loaded %s max_seq_len=%d\n",
                 file.c_str(), max_seq_len);
    std::fflush(stderr);
    return true;
}

void free_mega_pflash(MegaPFlashContext & ctx) {
    for (void *p : ctx.owned_device_ptrs) {
        if (p) cudaFree(p);
    }
    ctx.owned_device_ptrs.clear();
    ctx = MegaPFlashContext{};
}

std::vector<int32_t> mega_pflash_score_and_compress(
    MegaPFlashContext & ctx,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int chunk_size,
    int n_lookahead,
    int pool_kernel) {
    if (!ctx.loaded) {
        set_last_error("mega-pflash: context is not loaded");
        return {};
    }
    const int S = (int)ids.size();
    if (S <= 0) return {};
    if (S > ctx.max_seq_len) {
        set_last_error("mega-pflash: prompt exceeds max_seq_len");
        return {};
    }
    if (S <= chunk_size || keep_ratio >= 1.0f) return ids;
    int *ids_dev = nullptr;
    float *logit_scratch = nullptr;
    float *token_scores = nullptr;
    float *chunk_scores = nullptr;
    cudaMalloc(&ids_dev, (size_t)S * sizeof(int));
    cudaMemcpy(ids_dev, ids.data(), (size_t)S * sizeof(int), cudaMemcpyHostToDevice);
    int tail_len = std::min({n_lookahead, 8, S});
    int n_chunks = div_up(S, chunk_size);
    int total_rows = 6 * FA_Q_HEADS * tail_len;
    cudaMalloc(&logit_scratch, (size_t)total_rows * S * sizeof(float));
    cudaMalloc(&token_scores, (size_t)S * sizeof(float));
    cudaMalloc(&chunk_scores, (size_t)n_chunks * sizeof(float));

    auto t0 = std::chrono::steady_clock::now();
    cudaStream_t stream = 0;
    cudaMemset(ctx.dn_states, 0, (size_t)18 * DN_HEADS * DN_KEY * DN_VAL * sizeof(float));
    cudaMemset(ctx.conv_bufs, 0, (size_t)18 * DN_CONV_CH * DN_CONV_K * sizeof(float));
    launch_prefill_bf16(
        ids_dev, S, (int *)ctx.output_token,
        ctx.embed_weight, ctx.layer_weights_dev,
        ctx.final_norm_weight, ctx.lm_head_weight,
        ctx.fa_k_cache, ctx.fa_v_cache, ctx.dn_states, ctx.conv_bufs,
        ctx.hidden, ctx.residual, ctx.normalized,
        ctx.proj_buf, ctx.proj_buf2, ctx.attn_buf, ctx.mlp_buf,
        ctx.dn_out_buf,
        ctx.beta_buf, ctx.alpha_buf, ctx.dn_pre_qkv,
        ctx.dn_u_scratch, ctx.dn_w_scratch, ctx.dn_cs_scratch,
        ctx.fused_fa_qkv, ctx.fused_gate_up,
        ctx.final_normed, ctx.hidden_bf16_out,
        ctx.lm_bmv, ctx.lm_bmi,
        ctx.fa_q_tail,
        ctx.max_seq_len, tail_len,
        stream);
    launch_mega_pflash_score(
        ctx.fa_q_tail, ctx.fa_k_cache, logit_scratch,
        token_scores, chunk_scores,
        S, ctx.max_seq_len, tail_len, chunk_size, pool_kernel, 0, stream);
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_last_error(std::string("mega-pflash: CUDA error: ") + cudaGetErrorString(err));
        cudaFree(ids_dev); cudaFree(logit_scratch); cudaFree(token_scores); cudaFree(chunk_scores);
        return {};
    }
    std::vector<float> h_chunks((size_t)n_chunks);
    cudaMemcpy(h_chunks.data(), chunk_scores, (size_t)n_chunks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(ids_dev); cudaFree(logit_scratch); cudaFree(token_scores); cudaFree(chunk_scores);

    int keep_tokens = std::max(1, (int)std::ceil(S * keep_ratio));
    int keep_chunks = std::max(1, std::min(n_chunks, (int)std::ceil((float)keep_tokens / chunk_size)));
    std::vector<int> selected;
    selected.reserve((size_t)keep_chunks);
    selected.push_back(0);
    if (n_chunks > 1 && keep_chunks > 1) selected.push_back(n_chunks - 1);

    std::vector<std::pair<float, int>> ranked;
    ranked.reserve((size_t)n_chunks);
    for (int c = 0; c < n_chunks; ++c) ranked.push_back({h_chunks[c], c});
    std::sort(ranked.begin(), ranked.end(), [](auto a, auto b) { return a.first > b.first; });
    for (auto [_, c] : ranked) {
        if ((int)selected.size() >= keep_chunks) break;
        if (std::find(selected.begin(), selected.end(), c) == selected.end()) selected.push_back(c);
    }
    std::sort(selected.begin(), selected.end());

    std::vector<int32_t> out;
    out.reserve((size_t)keep_chunks * chunk_size);
    for (int c : selected) {
        int start = c * chunk_size;
        int end = std::min(S, start + chunk_size);
        for (int i = start; i < end; ++i) out.push_back(ids[i]);
    }
    auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[mega-pflash] native compress %.3fs S=%d kept=%zu (%d/%d chunks)\n",
                 std::chrono::duration<double>(t1 - t0).count(),
                 S, out.size(), keep_chunks, n_chunks);
    std::fflush(stderr);
    return out;
}

} // namespace dflash27b
