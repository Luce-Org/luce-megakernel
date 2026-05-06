#pragma once
#include "ggml.h"
#include <string>

namespace dflash {

// Parses a KV-cache element type string (case-insensitive).
// Accepted: "f16", "bf16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "tq3_0".
// Returns GGML_TYPE_COUNT on unknown input (caller should treat as error).
ggml_type parse_kv_type(const char * s);

// Returns the canonical lowercase string for a supported KV ggml_type,
// or "?" for unsupported.
const char * kv_type_name(ggml_type t);

// True iff the (K, V) ggml_type pair is supported by the CUDA flash-attention
// kernels currently compiled in (mirror of fattn.cu type-pair table when
// GGML_CUDA_FA_ALL_QUANTS=ON, which is now forced ON in dflash/CMakeLists.txt).
bool is_supported_kv_pair(ggml_type k, ggml_type v);

// Resolves K and V types from environment variables.
// Precedence (high -> low):
//   1. DFLASH27B_KV_K=<type> / DFLASH27B_KV_V=<type>  (independent override)
//   2. DFLASH27B_KV_F16 / _KV_Q4 / _KV_TQ3            (legacy shorthand, K==V)
//   3. Default: GGML_TYPE_Q4_0 for both (with FWHT K-rotation)
// On invalid input or unsupported (K,V) pair, prints an explanatory message
// and calls std::abort(). Returns the resolved pair via out params.
void resolve_kv_types(ggml_type & k_out, ggml_type & v_out);

}  // namespace dflash
