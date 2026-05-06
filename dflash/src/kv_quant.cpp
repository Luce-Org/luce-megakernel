// KV-cache quantisation helpers for dflash27b.
//
// Centralises the supported (K, V) ggml_type pair table and environment-variable
// resolution that was previously inlined in qwen35_target_graph.cpp.
//
// Supported pairs mirror fattn.cu with GGML_CUDA_FA_ALL_QUANTS=ON:
//
//   K ∈ {F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0}
//     × V ∈ {F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, TQ3_0}
//
//   K = TQ3_0
//     × V ∈ {F16, BF16, Q4_0, Q8_0, TQ3_0}

#include "kv_quant.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace dflash {

// ─── String <-> type helpers ────────────────────────────────────────────────

static std::string to_lower(const char * s) {
    std::string out;
    for (; *s; ++s) {
        out += static_cast<char>(std::tolower(static_cast<unsigned char>(*s)));
    }
    return out;
}

ggml_type parse_kv_type(const char * s) {
    if (!s) return GGML_TYPE_COUNT;
    const std::string lower = to_lower(s);
    if (lower == "f16")   return GGML_TYPE_F16;
    if (lower == "bf16")  return GGML_TYPE_BF16;
    if (lower == "q4_0")  return GGML_TYPE_Q4_0;
    if (lower == "q4_1")  return GGML_TYPE_Q4_1;
    if (lower == "q5_0")  return GGML_TYPE_Q5_0;
    if (lower == "q5_1")  return GGML_TYPE_Q5_1;
    if (lower == "q8_0")  return GGML_TYPE_Q8_0;
    if (lower == "tq3_0") return GGML_TYPE_TQ3_0;
    return GGML_TYPE_COUNT;
}

const char * kv_type_name(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F16:   return "f16";
        case GGML_TYPE_BF16:  return "bf16";
        case GGML_TYPE_Q4_0:  return "q4_0";
        case GGML_TYPE_Q4_1:  return "q4_1";
        case GGML_TYPE_Q5_0:  return "q5_0";
        case GGML_TYPE_Q5_1:  return "q5_1";
        case GGML_TYPE_Q8_0:  return "q8_0";
        case GGML_TYPE_TQ3_0: return "tq3_0";
        default:              return "?";
    }
}

// ─── Supported pair table ────────────────────────────────────────────────────

// Each entry is a (K type, V type) pair supported by the CUDA fattn kernels
// when GGML_CUDA_FA_ALL_QUANTS=ON.
struct KVPair {
    ggml_type k;
    ggml_type v;
};

// clang-format off
static const KVPair SUPPORTED_PAIRS[] = {
    // K ∈ {F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0} × V ∈ {F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, TQ3_0}
    { GGML_TYPE_F16,   GGML_TYPE_F16   },
    { GGML_TYPE_F16,   GGML_TYPE_BF16  },
    { GGML_TYPE_F16,   GGML_TYPE_Q4_0  },
    { GGML_TYPE_F16,   GGML_TYPE_Q4_1  },
    { GGML_TYPE_F16,   GGML_TYPE_Q5_0  },
    { GGML_TYPE_F16,   GGML_TYPE_Q5_1  },
    { GGML_TYPE_F16,   GGML_TYPE_Q8_0  },
    { GGML_TYPE_F16,   GGML_TYPE_TQ3_0 },

    { GGML_TYPE_BF16,  GGML_TYPE_F16   },
    { GGML_TYPE_BF16,  GGML_TYPE_BF16  },
    { GGML_TYPE_BF16,  GGML_TYPE_Q4_0  },
    { GGML_TYPE_BF16,  GGML_TYPE_Q4_1  },
    { GGML_TYPE_BF16,  GGML_TYPE_Q5_0  },
    { GGML_TYPE_BF16,  GGML_TYPE_Q5_1  },
    { GGML_TYPE_BF16,  GGML_TYPE_Q8_0  },
    { GGML_TYPE_BF16,  GGML_TYPE_TQ3_0 },

    { GGML_TYPE_Q4_0,  GGML_TYPE_F16   },
    { GGML_TYPE_Q4_0,  GGML_TYPE_BF16  },
    { GGML_TYPE_Q4_0,  GGML_TYPE_Q4_0  },
    { GGML_TYPE_Q4_0,  GGML_TYPE_Q4_1  },
    { GGML_TYPE_Q4_0,  GGML_TYPE_Q5_0  },
    { GGML_TYPE_Q4_0,  GGML_TYPE_Q5_1  },
    { GGML_TYPE_Q4_0,  GGML_TYPE_Q8_0  },
    { GGML_TYPE_Q4_0,  GGML_TYPE_TQ3_0 },

    { GGML_TYPE_Q4_1,  GGML_TYPE_F16   },
    { GGML_TYPE_Q4_1,  GGML_TYPE_BF16  },
    { GGML_TYPE_Q4_1,  GGML_TYPE_Q4_0  },
    { GGML_TYPE_Q4_1,  GGML_TYPE_Q4_1  },
    { GGML_TYPE_Q4_1,  GGML_TYPE_Q5_0  },
    { GGML_TYPE_Q4_1,  GGML_TYPE_Q5_1  },
    { GGML_TYPE_Q4_1,  GGML_TYPE_Q8_0  },
    { GGML_TYPE_Q4_1,  GGML_TYPE_TQ3_0 },

    { GGML_TYPE_Q5_0,  GGML_TYPE_F16   },
    { GGML_TYPE_Q5_0,  GGML_TYPE_BF16  },
    { GGML_TYPE_Q5_0,  GGML_TYPE_Q4_0  },
    { GGML_TYPE_Q5_0,  GGML_TYPE_Q4_1  },
    { GGML_TYPE_Q5_0,  GGML_TYPE_Q5_0  },
    { GGML_TYPE_Q5_0,  GGML_TYPE_Q5_1  },
    { GGML_TYPE_Q5_0,  GGML_TYPE_Q8_0  },
    { GGML_TYPE_Q5_0,  GGML_TYPE_TQ3_0 },

    { GGML_TYPE_Q5_1,  GGML_TYPE_F16   },
    { GGML_TYPE_Q5_1,  GGML_TYPE_BF16  },
    { GGML_TYPE_Q5_1,  GGML_TYPE_Q4_0  },
    { GGML_TYPE_Q5_1,  GGML_TYPE_Q4_1  },
    { GGML_TYPE_Q5_1,  GGML_TYPE_Q5_0  },
    { GGML_TYPE_Q5_1,  GGML_TYPE_Q5_1  },
    { GGML_TYPE_Q5_1,  GGML_TYPE_Q8_0  },
    { GGML_TYPE_Q5_1,  GGML_TYPE_TQ3_0 },

    { GGML_TYPE_Q8_0,  GGML_TYPE_F16   },
    { GGML_TYPE_Q8_0,  GGML_TYPE_BF16  },
    { GGML_TYPE_Q8_0,  GGML_TYPE_Q4_0  },
    { GGML_TYPE_Q8_0,  GGML_TYPE_Q4_1  },
    { GGML_TYPE_Q8_0,  GGML_TYPE_Q5_0  },
    { GGML_TYPE_Q8_0,  GGML_TYPE_Q5_1  },
    { GGML_TYPE_Q8_0,  GGML_TYPE_Q8_0  },
    { GGML_TYPE_Q8_0,  GGML_TYPE_TQ3_0 },

    // K = TQ3_0 × V ∈ {F16, BF16, Q4_0, Q8_0, TQ3_0}
    { GGML_TYPE_TQ3_0, GGML_TYPE_F16   },
    { GGML_TYPE_TQ3_0, GGML_TYPE_BF16  },
    { GGML_TYPE_TQ3_0, GGML_TYPE_Q4_0  },
    { GGML_TYPE_TQ3_0, GGML_TYPE_Q8_0  },
    { GGML_TYPE_TQ3_0, GGML_TYPE_TQ3_0 },
};
// clang-format on

static constexpr int N_SUPPORTED_PAIRS =
    static_cast<int>(sizeof(SUPPORTED_PAIRS) / sizeof(SUPPORTED_PAIRS[0]));

bool is_supported_kv_pair(ggml_type k, ggml_type v) {
    for (int i = 0; i < N_SUPPORTED_PAIRS; ++i) {
        if (SUPPORTED_PAIRS[i].k == k && SUPPORTED_PAIRS[i].v == v) {
            return true;
        }
    }
    return false;
}

// ─── Environment-variable resolution ────────────────────────────────────────

void resolve_kv_types(ggml_type & k_out, ggml_type & v_out) {
    ggml_type k = GGML_TYPE_Q4_0;
    ggml_type v = GGML_TYPE_Q4_0;

    // Layer 2: legacy shorthand (last wins, mirrors qwen35_target_graph.cpp:96-108)
    if (const char * s = std::getenv("DFLASH27B_KV_F16")) {
        if (std::atoi(s) != 0) { k = GGML_TYPE_F16;   v = GGML_TYPE_F16;   }
    }
    if (const char * s = std::getenv("DFLASH27B_KV_Q4")) {
        if (std::atoi(s) != 0) { k = GGML_TYPE_Q4_0;  v = GGML_TYPE_Q4_0;  }
    }
    if (const char * s = std::getenv("DFLASH27B_KV_TQ3")) {
        if (std::atoi(s) != 0) { k = GGML_TYPE_TQ3_0; v = GGML_TYPE_TQ3_0; }
    }

    // Layer 1: explicit per-axis override (highest precedence)
    if (const char * s = std::getenv("DFLASH27B_KV_K")) {
        const ggml_type parsed = parse_kv_type(s);
        if (parsed == GGML_TYPE_COUNT) {
            std::fprintf(stderr, "[dflash] Unknown KV K type: \"%s\"\n", s);
            std::abort();
        }
        k = parsed;
    }
    if (const char * s = std::getenv("DFLASH27B_KV_V")) {
        const ggml_type parsed = parse_kv_type(s);
        if (parsed == GGML_TYPE_COUNT) {
            std::fprintf(stderr, "[dflash] Unknown KV V type: \"%s\"\n", s);
            std::abort();
        }
        v = parsed;
    }

    // Validate the resolved (K, V) pair
    if (!is_supported_kv_pair(k, v)) {
        std::fprintf(stderr,
            "[dflash] KV pair (K=%s, V=%s) not supported by fattn-cuda. Supported pairs:\n",
            kv_type_name(k), kv_type_name(v));
        for (int i = 0; i < N_SUPPORTED_PAIRS; ++i) {
            std::fprintf(stderr, "  K=%-6s  V=%s\n",
                kv_type_name(SUPPORTED_PAIRS[i].k),
                kv_type_name(SUPPORTED_PAIRS[i].v));
        }
        std::abort();
    }

    k_out = k;
    v_out = v;
}

}  // namespace dflash
