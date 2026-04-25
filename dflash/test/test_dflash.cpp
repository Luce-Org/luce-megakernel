// DFlash speculative decoding end-to-end test.
//
// Pipeline:
//   1. Load target (Qwen3.5-27B qwen35) + draft (z-lab Qwen3.5-27B-DFlash).
//   2. Prefill: single-token decode over the prompt, capture_layers=true so
//      target_feat gets populated for every prompt pos.
//   3. Decode loop (until max_new):
//      a. Build noise block [last_tok, MASK*15] on CPU via target.tok_embd.
//      b. Draft forward (uses target_feat[0..committed] + noise) → 16 candidates.
//      c. snapshot SSM state. Batched target verify on the 16 draft tokens with
//         causal mask, capture_layers=true.
//      d. Greedy longest-prefix accept + 1 bonus token from target's argmax.
//      e. Restore SSM state. Replay the accepted tokens through target (batched
//         with causal mask, capture_layers=true) so state + target_feat are
//         cleanly advanced only by what was committed.
//      f. Update committed, last_tok.
//
// Usage: test_dflash <target.gguf> <draft.safetensors> <prompt_ids.bin>
//                    <n_gen> <out_ids.bin>

#include "dflash27b.h"
#include "internal.h"
#include "dflash_graph.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>

// Half-precision → f32 widen kernel launchers (src/f16_convert.cu). Used by
// the DDtree rollback (ssm_intermediate slot → cache.ssm_state) and the
// drafter prep path (target_feat → sg.target_hidden_cat). We store the
// per-token intermediate cache in f16 and the target_feat buffer in bf16 to
// halve their memory footprint.
extern "C" void dflash27b_launch_f16_to_f32(const void * src,
                                            void * dst,
                                            size_t n_elems,
                                            cudaStream_t stream);
extern "C" void dflash27b_launch_bf16_to_f32(const void * src,
                                             void * dst,
                                             size_t n_elems,
                                             cudaStream_t stream);

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>

#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#include <io.h>
#ifdef _WIN64
#define ssize_t __int64
#else
#define ssize_t long
#endif
#else
#include <unistd.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

using namespace dflash27b;

// ─── Small utilities ──────────────────────────────────────────────

static std::vector<int32_t> read_int32_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<int32_t> out(sz / sizeof(int32_t));
    f.read((char *)out.data(), sz);
    return out;
}

static bool write_int32_file(const std::string & path, const std::vector<int32_t> & v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write((const char *)v.data(), v.size() * sizeof(int32_t));
    return (bool)f;
}

static int argmax_f32(const float * x, int n) {
    int best = 0;
    float bv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > bv) { bv = x[i]; best = i; }
    return best;
}

// ggml_flash_attn_ext expects kv_len aligned to KQ_MASK_PAD (32) on the
// f16/Q* paths, and to FATTN_KQ_STRIDE (256) on the TurboQuant FA paths.
// The global `g_kq_stride_pad` below is set at init time and applied by
// both build_causal_mask and build_tree_mask so the mask dim matches the
// K/V view length used in build_attn_block.
static constexpr int KQ_MASK_PAD = 32;
static int g_kq_stride_pad = KQ_MASK_PAD;   // overridden to 256 when TBQ KV is active
static int g_max_ctx_override = 0;           // overridden by --max-ctx=N (default 4096)
static int align_up(int x, int a) { return ((x + a - 1) / a) * a; }

// F16 encoding for the two values we use: 0 and -inf.
// 0 in F16 is 0x0000. -inf is 0xFC00.
static constexpr uint16_t F16_ZERO = 0x0000;
static constexpr uint16_t F16_NEG_INF = 0xFC00;

static void build_causal_mask(std::vector<uint16_t> & out,
                              int kv_len, int n_tokens, int kv_start) {
    const int kv_pad = align_up(kv_len, g_kq_stride_pad);
    const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
    out.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    for (int q = 0; q < n_tokens; q++) {
        const int max_k = kv_start + q;
        for (int k = 0; k <= max_k && k < kv_len; k++) {
            out[(size_t)q * kv_pad + k] = F16_ZERO;
        }
    }
}

// ─── DDTree support (ported from liranringel/ddtree/ddtree.py) ────────

// Per-position top-K softmax extraction. Computes log-probabilities (needed
// so that cross-depth prefix comparisons in the best-first heap are valid)
// via a single pass over the vocab that also maintains top-K in a heap and
// computes logsumexp online. Runs on CPU since draft logits are already on
// host after ggml_backend_tensor_get.
//
// Input:  logits [n_positions × vocab] f32
// Output: out_log_probs [n_positions × K] f32, out_token_ids [n_positions × K] i32
//         both sorted by log-probability DESCENDING (rank 0 = argmax).
static void extract_draft_topk(const float * logits,
                               int n_positions, int vocab, int K,
                               float * out_log_probs,
                               int32_t * out_token_ids,
                               float temperature = 1.0f) {
    struct Entry { float logit; int32_t id; };
    auto cmp_greater = [](const Entry & a, const Entry & b) {
        return a.logit > b.logit;
    };

    // Temperature scaling: dividing logits by T<1 sharpens the softmax,
    // widening the gap between top-1 and lower ranks. This compensates for
    // Q4_K_M quantization that flattens the draft's softmax — without it,
    // pure best-first picks shallow bushy trees instead of going deep.
    const float inv_t = 1.0f / std::max(1e-3f, temperature);

    // Parallelize across positions — each i is independent.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_positions; i++) {
        const float * li = logits + (size_t)i * vocab;
        std::vector<Entry> heap;
        heap.reserve(K);

        // Online log-sum-exp with running max. Single pass over the vocab,
        // simultaneously maintaining top-K.
        float running_max     = -INFINITY;
        float running_sum_exp = 0.0f;
        for (int j = 0; j < vocab; j++) {
            const float l = li[j] * inv_t;

            // Online logsumexp
            if (l > running_max) {
                // rescale previous sum to the new max
                if (running_max > -INFINITY) {
                    running_sum_exp = running_sum_exp * std::exp(running_max - l);
                }
                running_sum_exp += 1.0f;
                running_max = l;
            } else {
                running_sum_exp += std::exp(l - running_max);
            }

            // Top-K maintenance
            if ((int)heap.size() < K) {
                heap.push_back({l, (int32_t)j});
                std::push_heap(heap.begin(), heap.end(), cmp_greater);
            } else if (l > heap.front().logit) {
                std::pop_heap(heap.begin(), heap.end(), cmp_greater);
                heap.back() = {l, (int32_t)j};
                std::push_heap(heap.begin(), heap.end(), cmp_greater);
            }
        }
        const float log_z = running_max + std::log(running_sum_exp);

        // Sort the K entries descending (largest logit first) and emit.
        // sort_heap with cmp_greater on a min-heap already produces descending
        // order (cppreference: "sort_heap leaves the range sorted in the same
        // order as sort would with the same comparator" — greater→descending).
        std::sort_heap(heap.begin(), heap.end(), cmp_greater);
        for (int k = 0; k < K; k++) {
            out_log_probs[(size_t)i * K + k] = heap[k].logit - log_z;
            out_token_ids[(size_t)i * K + k] = heap[k].id;
        }
    }
}

// A flat DFS-ordered tree built from the draft's top-K softmax distributions.
// Slot 0 is the tree root (the bonus token from the previous spec round);
// slots 1..n_nodes are the DFS-ordered tree nodes. `parents[i]` gives each
// node's parent index in the same flat array (parents[0] = -1). `depth[i]`
// is the absolute depth within the block-diffusion prediction window, with
// the root at depth 0 and its children at depth 1. `child_maps[i]` maps a
// token_id to the child's flat index, used for the tree walk post-verify.
// `visibility[i][j]` (ancestor-only mask) is true iff j is an ancestor of i
// in the tree (including j == i); used to build the attention mask.
struct DDTree {
    int                         n_nodes = 0;          // excludes root
    std::vector<int32_t>        token_ids;            // size n_nodes
    std::vector<int>            depths;               // size n_nodes (1..L)
    std::vector<int>            parents;              // size n_nodes + 1
    std::vector<std::unordered_map<int32_t, int>> child_maps;  // size n_nodes + 1
    std::vector<uint8_t>        visibility;           // (1 + n_nodes)^2 row-major
};

// Port of build_ddtree_tree() from ddtree.py. Runs a best-first heap over
// prefixes of the per-position top-K distributions, pops until `budget`
// nodes are accumulated. Populates the flat DFS-ordered tree structure.
//
// top_log_probs: [L × K]  the drafter's per-position top-K log-probabilities
// top_token_ids: [L × K]  matching token ids, rank 0 = argmax per position
// L:             max tree depth (e.g. q_len - 1 for a block diffusion block)
// K:             top-K per position (same as used in extract_draft_topk)
// budget:        maximum number of non-root tree nodes
static DDTree build_ddtree(const float * top_log_probs,
                           const int32_t * top_token_ids,
                           int L, int K, int budget,
                           bool chain_seed = true) {
    DDTree tree;
    if (budget <= 0 || L <= 0) {
        tree.parents.push_back(-1);
        tree.child_maps.emplace_back();
        tree.visibility.assign(1, 1);
        return tree;
    }

    // Heap entry:
    //   neg_logw, ranks (encoded as a small vector), parent_index, depth, rank, logw
    // We sort by neg_logw ASCENDING, which is equivalent to logw DESCENDING.
    struct HeapEntry {
        float                neg_logw;
        std::vector<int>     ranks;        // rank tuple used only to prevent duplicate state; not strictly needed
        int                  parent_index; // index in the flat tree of this candidate's parent
        int                  depth;        // 1..L
        int                  rank;         // rank within top-K at depth-1 (0-indexed)
        float                logw;         // actual log-prob sum so far
    };
    struct HeapCmp {
        bool operator()(const HeapEntry & a, const HeapEntry & b) const {
            // std::priority_queue is a max-heap; we want SMALLEST neg_logw at the top
            // so that we pop the highest-probability prefix first.
            return a.neg_logw > b.neg_logw;
        }
    };
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapCmp> heap;

    tree.token_ids.reserve(budget);
    tree.depths.reserve(budget);
    tree.parents.reserve(budget + 1);
    tree.parents.push_back(-1);                 // root
    tree.child_maps.emplace_back();             // root's children

    // Two seeding strategies:
    //   - chain_seed=true: pre-seed full top-1 chain (defensive, guarantees
    //     AL >= chain mode even with flat-softmax draft like Q4_K_M). Compensates
    //     for quantization that shrinks top-1/top-2 logp gap.
    //   - chain_seed=false: paper's pure best-first — heap starts with just
    //     the depth-1 top-1 child of the root. Tree shape emerges from log-prob
    //     ordering. Works only when the draft top-1 is dominant enough.
    if (chain_seed) {
        const int chain_depth = std::min(L, budget);
        float cum_logw = 0.0f;
        int   prev_idx = 0;
        for (int d = 1; d <= chain_depth; d++) {
            const int32_t tok_id = top_token_ids[(size_t)(d - 1) * K + 0];
            cum_logw += top_log_probs[(size_t)(d - 1) * K + 0];

            const int cur_idx = tree.n_nodes + 1;
            tree.token_ids.push_back(tok_id);
            tree.depths.push_back(d);
            tree.parents.push_back(prev_idx);
            tree.child_maps.emplace_back();
            tree.child_maps[prev_idx][tok_id] = cur_idx;
            tree.n_nodes++;

            if (K > 1) {
                const float sibling_logw = cum_logw
                    - top_log_probs[(size_t)(d - 1) * K + 0]
                    + top_log_probs[(size_t)(d - 1) * K + 1];
                heap.push({
                    /*neg_logw*/ -sibling_logw,
                    /*ranks   */ {1},
                    /*parent  */ prev_idx,
                    /*depth   */ d,
                    /*rank    */ 1,
                    /*logw    */ sibling_logw,
                });
            }
            prev_idx = cur_idx;
        }
    } else {
        // Paper-style pure best-first: seed heap with depth-1 top-1 only.
        const float root_logw = top_log_probs[0 * K + 0];
        heap.push({
            /*neg_logw*/ -root_logw,
            /*ranks   */ {0},
            /*parent  */ 0,  // root flat index
            /*depth   */ 1,
            /*rank    */ 0,
            /*logw    */ root_logw,
        });
    }

    while (!heap.empty() && tree.n_nodes < budget) {
        HeapEntry top = heap.top();
        heap.pop();

        const int    depth_minus_1 = top.depth - 1;
        const int    rank          = top.rank;
        const int32_t token_id     = top_token_ids[(size_t)depth_minus_1 * K + rank];

        const int current_index = tree.n_nodes + 1;  // slot in flat tree
        tree.token_ids.push_back(token_id);
        tree.depths.push_back(top.depth);
        tree.parents.push_back(top.parent_index);
        tree.child_maps.emplace_back();
        tree.child_maps[top.parent_index][token_id] = current_index;
        tree.n_nodes++;

        // Push next sibling (same depth, next-best rank at this depth).
        if (rank + 1 < K) {
            const float sibling_logw = top.logw
                - top_log_probs[(size_t)depth_minus_1 * K + rank]
                + top_log_probs[(size_t)depth_minus_1 * K + rank + 1];
            std::vector<int> sibling_ranks = top.ranks;
            sibling_ranks.back() = rank + 1;
            heap.push({
                /*neg_logw*/ -sibling_logw,
                /*ranks   */ std::move(sibling_ranks),
                /*parent  */ top.parent_index,
                /*depth   */ top.depth,
                /*rank    */ rank + 1,
                /*logw    */ sibling_logw,
            });
        }

        // Push first child (next depth, top-1 rank under this node).
        if (top.depth < L) {
            const float child_logw = top.logw
                + top_log_probs[(size_t)top.depth /*new depth_minus_1*/ * K + 0];
            std::vector<int> child_ranks = top.ranks;
            child_ranks.push_back(0);
            heap.push({
                /*neg_logw*/ -child_logw,
                /*ranks   */ std::move(child_ranks),
                /*parent  */ current_index,
                /*depth   */ top.depth + 1,
                /*rank    */ 0,
                /*logw    */ child_logw,
            });
        }
    }

    // Build ancestor-only visibility mask (flat row-major, (1+n)^2).
    const int N = 1 + tree.n_nodes;
    tree.visibility.assign((size_t)N * N, 0);
    tree.visibility[0 * N + 0] = 1;  // root sees itself
    for (int i = 1; i < N; i++) {
        const int p = tree.parents[i];  // immediate parent
        // Inherit the parent's visibility row up to column i-1,
        // then mark self at column i.
        for (int j = 0; j < i; j++) {
            tree.visibility[(size_t)i * N + j] = tree.visibility[(size_t)p * N + j];
        }
        tree.visibility[(size_t)i * N + i] = 1;
    }

    return tree;
}

// Walk the verified tree following the target's argmax (posterior) at each
// node. Returns the list of flat-tree indices that make up the accepted path
// (starting at root), plus the next "bonus" token (target's argmax at the
// deepest accepted node, which didn't match any of that node's children).
static std::vector<int> follow_verified_tree(const DDTree & tree,
                                             const int32_t * posterior,
                                             int & out_next_token) {
    std::vector<int> accepted;
    accepted.reserve(tree.n_nodes + 1);
    accepted.push_back(0);

    int current_index = 0;
    int next_token    = posterior[current_index];
    while (true) {
        const auto & children = tree.child_maps[current_index];
        auto it = children.find(next_token);
        if (it == children.end()) break;
        current_index = it->second;
        accepted.push_back(current_index);
        next_token = posterior[current_index];
    }
    out_next_token = next_token;
    return accepted;
}

// Build an f16 ancestor-only attention mask for tree verify:
//   mask[q=i][k<past_length]          = 0    (past KV cache, attend freely)
//   mask[q=i][k=past_length+j]        = 0 iff j is an ancestor of i in the tree
//                                              (including j == i)
//                                     = -inf otherwise
// Shape matches the ggml flash_attn_ext expectation: [kv_pad, q_pad] f16.
static void build_tree_mask(const DDTree & tree, int past_length,
                            std::vector<uint16_t> & out_mask) {
    const int N      = 1 + tree.n_nodes;
    const int kv_len = past_length + N;
    const int kv_pad = align_up(kv_len, g_kq_stride_pad);
    const int q_pad  = align_up(N,      KQ_MASK_PAD);
    out_mask.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    for (int q = 0; q < N; q++) {
        // Past KV (prompt + previously-committed decode tokens): always visible.
        for (int k = 0; k < past_length; k++) {
            out_mask[(size_t)q * kv_pad + k] = F16_ZERO;
        }
        // Tree region: ancestors-only per the tree.visibility matrix.
        for (int j = 0; j < N; j++) {
            if (tree.visibility[(size_t)q * N + j]) {
                out_mask[(size_t)q * kv_pad + (past_length + j)] = F16_ZERO;
            }
        }
    }
}

// ─── StepGraph — rebuilt per call since kv_len varies ──

struct StepGraph {
    ggml_context *  ctx = nullptr;
    ggml_cgraph *   gf  = nullptr;
    ggml_gallocr_t  alloc = nullptr;

    // Named inputs (look up via ggml_get_tensor by name)
    ggml_tensor *   inp_embed = nullptr;
    ggml_tensor *   positions = nullptr;
    ggml_tensor *   attn_mask = nullptr;     // may be null
    ggml_tensor *   parent_ids = nullptr;    // DDTree tree-mode; null for chain mode
    ggml_tensor *   target_hidden_cat = nullptr;  // draft only
    ggml_tensor *   positions_k = nullptr;        // draft only

    // Output
    ggml_tensor *   logits = nullptr;

    // Per-delta-net-layer captures (verify only). One entry per delta-net layer.
    // Each entry's tensors are graph views on the gated_delta_net result:
    //   ssm_intermediate_states: [S_v, S_v, H_v, n_tokens]  (f32, ~50 MB/layer for n_tokens=16)
    //   conv_input:              [kernel-1+n_tokens, conv_channels, 1]
    // Marked as graph outputs so their data is valid after ggml_backend_graph_compute.
    std::vector<DeltaNetCapture> delta_captures;
};

// Reset the per-call graph state (ctx + graph + tensor handles) but KEEP the
// persistent CUDA buffer in `sg.alloc` alive across steps. When the next
// build_*_step re-walks ggml_gallocr_alloc_graph on the same (or smaller)
// peak shape, gallocr reuses the existing CUDA buffer instead of doing a
// fresh cudaMalloc/cudaFree of multi-GB at every step. That's the single
// biggest per-step cost at long context — see the [timing] breakdown:
// draft_compute 21ms and verify_compute 61ms both include the alloc cycle
// inside ggml_backend_graph_compute when the buffer is fresh.
static void step_graph_free(StepGraph & sg) {
    if (sg.ctx)   { ggml_free(sg.ctx); sg.ctx = nullptr; }
    sg.gf = nullptr;
    sg.inp_embed = sg.positions = sg.attn_mask = nullptr;
    sg.target_hidden_cat = sg.positions_k = nullptr;
    sg.parent_ids = nullptr;
    sg.logits = nullptr;
    sg.delta_captures.clear();
}

// Called at shutdown only. Releases the persistent gallocr + its CUDA
// backing buffer in addition to what step_graph_free already does.
static void step_graph_destroy(StepGraph & sg) {
    if (sg.alloc) { ggml_gallocr_free(sg.alloc); sg.alloc = nullptr; }
    step_graph_free(sg);
}

// Forward declaration used only by the tree-mode build_target_step_tree.
struct DDTree;  // full def above

// Build a target verify graph configured for a DDTree-flattened tree block.
// Key differences vs build_target_step:
//   - n_tokens = 1 + tree.n_nodes (root + flat tree nodes)
//   - with_mask is always true; the caller fills a custom ancestor-only mask
//     rather than the usual causal strip
//   - A parent_ids[n_tokens] i32 input is added and wired into the delta-net
//     kernel so recurrent state is reloaded at DFS branch transitions
//   - capture_layers and capture_delta_intermediate are always on (the spec
//     loop uses per-step SSM states for rollback and target_feat for the
//     next iter's draft)
static bool build_target_step_tree(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int kv_start,
    int n_tokens);   // implemented below after the regular build_target_step

static bool build_target_step(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int kv_start,
    int n_tokens,
    bool with_mask,
    bool capture,
    bool capture_delta_intermediate = false) {
    step_graph_free(sg);

    ggml_init_params ip{};
    // ctx arena holds tensor *descriptors* only (no_alloc = true), so size
    // just needs to cover the struct count. 512 MB is plenty for the target
    // graph even with capture_delta_intermediate enabled (the 48 extra delta
    // captures add ~48 descriptors, nothing).
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = w.n_embd;
    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_name(sg.positions, "positions");
    ggml_set_input(sg.positions);

    if (with_mask) {
        const int kv_len = kv_start + n_tokens;
        const int kv_pad = align_up(kv_len, g_kq_stride_pad);
        const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
        sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
        ggml_set_name(sg.attn_mask, "attn_mask");
        ggml_set_input(sg.attn_mask);
    }

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    QwenGraphInputs gi{};
    gi.inp_embed                  = sg.inp_embed;
    gi.positions                  = sg.positions;
    gi.attn_mask                  = sg.attn_mask;
    gi.n_tokens                   = n_tokens;
    gi.kv_start                   = kv_start;
    gi.capture_layers             = capture;
    gi.capture_delta_intermediate = capture_delta_intermediate;

    QwenGraphOutputs go = build_qwen35_graph(sg.ctx, sg.gf, w, cache, gi);
    if (!go.logits) return false;
    sg.logits = go.logits;
    sg.delta_captures = std::move(go.delta_captures);
    ggml_set_output(sg.logits);
    ggml_build_forward_expand(sg.gf, sg.logits);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

// DDTree tree-verify graph builder. Same shape as build_target_step except:
//   - n_tokens is the flat tree size (1 + tree.n_nodes)
//   - attn_mask is caller-filled (ancestor-only); we build the tensor here
//     but the values come from build_tree_mask() before compute
//   - A fresh parent_ids[n_tokens] i32 input tensor is added and wired into
//     QwenGraphInputs so build_delta_net_block can call ggml_gated_delta_net_tree
//   - capture_layers=true, capture_delta_intermediate=true (spec loop relies
//     on per-step intermediates for rollback)
static bool build_target_step_tree(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int kv_start,
    int n_tokens) {
    step_graph_free(sg);

    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = w.n_embd;
    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_name(sg.positions, "positions");
    ggml_set_input(sg.positions);

    const int kv_len = kv_start + n_tokens;
    const int kv_pad = align_up(kv_len, g_kq_stride_pad);
    const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
    sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
    ggml_set_name(sg.attn_mask, "attn_mask");
    ggml_set_input(sg.attn_mask);

    // parent_ids[n_tokens] i32 — tree-mode DeltaNet input. -1 = reload from
    // pre-block state, k = reload from intermediate[k], t-1 = sequential (hot path).
    sg.parent_ids = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(sg.parent_ids, "parent_ids");
    ggml_set_input(sg.parent_ids);

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    QwenGraphInputs gi{};
    gi.inp_embed                  = sg.inp_embed;
    gi.positions                  = sg.positions;
    gi.attn_mask                  = sg.attn_mask;
    gi.n_tokens                   = n_tokens;
    gi.kv_start                   = kv_start;
    gi.capture_layers             = true;
    gi.capture_delta_intermediate = true;
    gi.parent_ids                 = sg.parent_ids;

    QwenGraphOutputs go = build_qwen35_graph(sg.ctx, sg.gf, w, cache, gi);
    if (!go.logits) return false;
    sg.logits = go.logits;
    sg.delta_captures = std::move(go.delta_captures);
    ggml_set_output(sg.logits);
    ggml_build_forward_expand(sg.gf, sg.logits);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

static bool build_draft_step(
    StepGraph & sg,
    const DraftWeights & dw,
    const TargetWeights & tw,   // for lm_head
    ggml_backend_t backend,
    int ctx_len) {
    step_graph_free(sg);

    ggml_init_params ip{};
    ip.mem_size   = 256 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = tw.n_embd;
    const int q_len  = dw.hparams.block_size;
    const int fc_in  = dw.hparams.n_target_layers * hidden;

    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, q_len, 1);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    sg.target_hidden_cat = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, fc_in, ctx_len, 1);
    ggml_set_name(sg.target_hidden_cat, "target_hidden_cat");
    ggml_set_input(sg.target_hidden_cat);

    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, q_len);
    ggml_set_name(sg.positions, "positions_q");
    ggml_set_input(sg.positions);

    sg.positions_k = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, ctx_len + q_len);
    ggml_set_name(sg.positions_k, "positions_k");
    ggml_set_input(sg.positions_k);

    sg.gf = ggml_new_graph_custom(sg.ctx, 4096, false);

    DraftGraphInputs gi{};
    gi.ctx_len           = ctx_len;
    gi.noise_embed       = sg.inp_embed;
    gi.target_hidden_cat = sg.target_hidden_cat;
    gi.positions_q       = sg.positions;
    gi.positions_k       = sg.positions_k;
    gi.lm_head           = tw.output;     // project through target.output (q6_K)
    DraftGraphOutputs go = build_draft_graph(sg.ctx, dw, gi);
    if (!go.logits) {
        std::fprintf(stderr, "draft graph missing logits (lm_head was null?)\n");
        return false;
    }
    sg.logits = go.logits;
    ggml_set_output(sg.logits);
    ggml_build_forward_expand(sg.gf, sg.logits);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

// ─── Main ─────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <target.gguf> <draft.safetensors> [<prompt_ids.bin> <n_gen> <out_ids.bin>] [--daemon] ...\n", argv[0]);
        return 2;
    }
    // TurboQuant FA kernel requires kv_len aligned to FATTN_KQ_STRIDE=256.
    // Bump the mask stride accordingly so the mask dim matches the kv view.
    if (const char * s = std::getenv("DFLASH27B_KV_TBQ")) {
        if (std::atoi(s) != 0) g_kq_stride_pad = 256;
    }
    if (const char * s = std::getenv("DFLASH27B_KV_TQ3")) {
        if (std::atoi(s) != 0) g_kq_stride_pad = 256;
    }
    const char * target_path = argv[1];
    const char * draft_path  = argv[2];
    const char * prompt_path = (argc >= 6 && argv[3][0] != '-') ? argv[3] : nullptr;
    int          n_gen       = (argc >= 6 && argv[3][0] != '-') ? std::atoi(argv[4]) : 0;
    const char * out_path    = (argc >= 6 && argv[3][0] != '-') ? argv[5] : nullptr;
    // --seq-verify: run the target verify as q_len independent single-token
    // decodes instead of one batched forward with a causal mask. Isolates
    // the correctness-of-batched-verify hypothesis from z-lab issue #57.
    //
    // --fast-rollback: use per-step SSM intermediate-state capture (kernel mod
    // in ggml_gated_delta_net) to roll back state after verify without the
    // replay forward pass. Implicit-bonus variant: commit only accept_n tokens
    // per step, let next iter's draft pick up the "bonus" via last_tok.
    //
    // --ddtree [--ddtree-budget=B]: use DDTree-style tree-structured verify on
    // top of the fast-rollback path. Ported from liranringel/ddtree.py with
    // our tree-aware gated_delta_net kernel handling the DeltaNet/SSM tree
    // recurrence via parent_ids (Qwen3.5 hybrid support beyond the original
    // paper's pure-attention Qwen3 experiments). Default budget = 64.
    bool  seq_verify    = false;
    bool  fast_rollback = false;
    bool  ddtree_mode   = false;
    int   ddtree_budget = 64;
    float ddtree_temp   = 1.0f;   // softmax temperature for top-K extract
    bool  ddtree_chain_seed = true;  // pre-seed full chain (vs paper's pure best-first)
    bool  profile_scaling = false;  // microbench: time target forward at varying N
    int   stream_fd     = -1;     // write each committed token to this fd (int32 LE) as they land
    bool  daemon_mode   = false;
    for (int i = 3; i < argc; i++) {
        if      (std::strcmp(argv[i], "--daemon") == 0)        daemon_mode = true;
        else if (std::strcmp(argv[i], "--seq-verify") == 0)    seq_verify = true;
        else if (std::strcmp(argv[i], "--fast-rollback") == 0) fast_rollback = true;
        else if (std::strcmp(argv[i], "--ddtree") == 0)        { ddtree_mode = true; fast_rollback = true; }
        else if (std::strncmp(argv[i], "--ddtree-budget=", 16) == 0) {
            ddtree_budget = std::atoi(argv[i] + 16);
            if (ddtree_budget <= 0) ddtree_budget = 64;
        }
        else if (std::strncmp(argv[i], "--ddtree-temp=", 14) == 0) {
            ddtree_temp = (float)std::atof(argv[i] + 14);
            if (ddtree_temp <= 0.0f) ddtree_temp = 1.0f;
        }
        else if (std::strcmp(argv[i], "--ddtree-no-chain-seed") == 0) {
            ddtree_chain_seed = false;
        }
        else if (std::strcmp(argv[i], "--profile-scaling") == 0) {
            profile_scaling = true;
        }
        else if (std::strncmp(argv[i], "--stream-fd=", 12) == 0) {
            stream_fd = std::atoi(argv[i] + 12);
        }
        else if (std::strncmp(argv[i], "--max-ctx=", 10) == 0) {
            g_max_ctx_override = std::atoi(argv[i] + 10);
        }
    }

    if (!daemon_mode && (!prompt_path || !out_path)) {
        std::fprintf(stderr, "Missing positional arguments for non-daemon mode.\n");
        return 2;
    }

    // Helper: write a committed token to the stream fd immediately (int32 LE).
    // Caller invokes after every out_all.push_back(tok) when stream_fd >= 0.
    // On Windows stream_fd holds a Win32 HANDLE value (passed via msvcrt.get_osfhandle).
    auto stream_emit = [&](int32_t tok) {
        if (stream_fd < 0) return;
        int32_t v = tok;
#if defined(_WIN32)
        DWORD written;
        WriteFile((HANDLE)(intptr_t)stream_fd, &v, sizeof(v), &written, nullptr);
#else
        ssize_t n = ::write(stream_fd, &v, sizeof(v));
        (void)n;
#endif
    };
    if (fast_rollback && seq_verify && !ddtree_mode) {
        std::fprintf(stderr, "--fast-rollback and --seq-verify are mutually exclusive\n");
        return 2;
    }
    std::printf("[cfg] seq_verify=%d fast_rollback=%d ddtree=%d budget=%d temp=%.2f chain_seed=%d\n",
                (int)seq_verify, (int)fast_rollback, (int)ddtree_mode,
                ddtree_budget, ddtree_temp, (int)ddtree_chain_seed);

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    TargetWeights w;
    if (!load_target_gguf(target_path, backend, w)) {
        std::fprintf(stderr, "target load: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("[target] %s\n", dflash27b_last_error());

    DraftWeights dw;
    if (!load_draft_safetensors(draft_path, backend, dw)) {
        std::fprintf(stderr, "draft load: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("[draft]  loaded\n");

    const int max_ctx = g_max_ctx_override > 0 ? g_max_ctx_override : 4096;
    // Size the ssm_intermediate / conv_input_cache buffers to cover whichever
    // verify mode we'll use. DDTree needs room for 1 + ddtree_budget tree nodes.
    // Profile mode intentionally keeps the intermediate cache tiny (no capture)
    // so we can go up to n_tokens=128 without OOM.
    const int max_verify_tokens = profile_scaling
        ? dw.hparams.block_size
        : (ddtree_mode
            ? std::max<int>(dw.hparams.block_size, ddtree_budget + 1)
            : dw.hparams.block_size);
    TargetCache cache;
    if (!create_target_cache(w, max_ctx, max_verify_tokens, backend, cache)) {
        std::fprintf(stderr, "cache: %s\n", dflash27b_last_error());
        return 1;
    }

    // ── Profile mode: microbench target forward at varying N ───────────
    if (profile_scaling) {
        const int hidden_p = w.n_embd;
        StepGraph psg;
        const int n_values[] = { 1, 4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128 };
        std::printf("[profile] target forward ms at varying N (kv_start=0, no capture)\n");
        std::printf("%6s %10s %10s\n", "N", "total_ms", "ms_per_N");
        for (int n : n_values) {
            if (!build_target_step(psg, w, cache, backend,
                                   /*kv_start=*/0, /*n_tokens=*/n,
                                   /*with_mask=*/true,
                                   /*capture=*/false,
                                   /*capture_delta_intermediate=*/false)) {
                std::fprintf(stderr, "profile build N=%d failed\n", n); return 1;
            }
            // Fake embed input (zeros) + fake positions + fake causal mask.
            std::vector<float> emb(hidden_p * n, 0.0f);
            ggml_backend_tensor_set(psg.inp_embed, emb.data(), 0, sizeof(float) * emb.size());
            std::vector<int32_t> pos4(4 * n);
            for (int i = 0; i < n; i++) {
                pos4[0 * n + i] = i;
                pos4[1 * n + i] = i;
                pos4[2 * n + i] = i;
                pos4[3 * n + i] = 0;
            }
            ggml_backend_tensor_set(psg.positions, pos4.data(), 0, sizeof(int32_t) * 4 * n);
            if (psg.attn_mask) {
                const int kv_pad = (int)psg.attn_mask->ne[0];
                const int q_pad  = (int)psg.attn_mask->ne[1];
                std::vector<uint16_t> mask_buf_p((size_t)kv_pad * q_pad, F16_NEG_INF);
                for (int q = 0; q < n; q++) {
                    for (int k = 0; k <= q; k++) {
                        mask_buf_p[(size_t)q * kv_pad + k] = F16_ZERO;
                    }
                }
                ggml_backend_tensor_set(psg.attn_mask, mask_buf_p.data(), 0,
                                        sizeof(uint16_t) * mask_buf_p.size());
            }
            // Warmup
            ggml_backend_graph_compute(backend, psg.gf);
            // Time 5 runs, take median
            std::vector<double> times;
            for (int rep = 0; rep < 5; rep++) {
                auto t0 = std::chrono::steady_clock::now();
                ggml_backend_graph_compute(backend, psg.gf);
                auto t1 = std::chrono::steady_clock::now();
                times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            std::sort(times.begin(), times.end());
            double median = times[times.size() / 2];
            std::printf("%6d %10.2f %10.3f\n", n, median, median / n);
        }
        free_target_cache(cache);
        free_target_weights(w);
        ggml_backend_free(backend);
        return 0;
    }

    const int q_len  = dw.hparams.block_size;
    const int hidden = w.n_embd;
    const int vocab  = (int)w.output->ne[1];
    const int mask_tok = DFLASH27B_DRAFT_MASK_TOKEN_ID;

    if (daemon_mode) {
        std::printf("[daemon] ready\n");
        std::fflush(stdout);
    }

    StepGraph sg;
    bool daemon_first_iter = true;

    while (true) {
        std::string prompt_file_str;
        if (daemon_mode) {
            std::string line;
            if (!std::getline(std::cin, line)) break;
            char ppath[1024];
            if (std::sscanf(line.c_str(), "%1023s %d", ppath, &n_gen) != 2) continue;
            prompt_file_str = ppath;
            prompt_path = prompt_file_str.c_str();

            // Rebuild cache + step graph between requests so KV / SSM / conv /
            // target_feat ring start fresh. Weights stay resident.
            if (!daemon_first_iter) {
                step_graph_free(sg);
                sg = StepGraph{};
                free_target_cache(cache);
                if (!create_target_cache(w, max_ctx, max_verify_tokens, backend, cache)) {
                    std::fprintf(stderr, "cache realloc: %s\n", dflash27b_last_error());
                    stream_emit(-1);
                    continue;
                }
            }
            daemon_first_iter = false;
        }

        auto prompt = read_int32_file(prompt_path);
        if (prompt.empty()) {
            std::fprintf(stderr, "empty prompt\n");
            if (daemon_mode) { stream_emit(-1); continue; } else return 1;
        }
        std::printf("[prompt] %zu tokens\n", prompt.size());

        if ((int)prompt.size() + n_gen + q_len > max_ctx) {
            std::fprintf(stderr, "prompt (%zu) + gen (%d) + block (%d) = %d exceeds max_ctx (%d)\n",
                         prompt.size(), n_gen, q_len, (int)prompt.size() + n_gen + q_len, max_ctx);
            if (daemon_mode) { stream_emit(-1); continue; } else return 1;
        }

        std::vector<float>   embed_buf(hidden);
        std::vector<int32_t> out_all = prompt;
        int committed = 0;
        int32_t last_tok = -1;

    // ── Prefill: batched decode over prompt. Chunks of up to PREFILL_UBATCH
    // tokens are pushed through a single forward pass with a causal mask,
    // matching llama.cpp's n_ubatch behavior. Bit-equivalent to the old
    // single-token loop because the persistent SSM/conv state + KV cache
    // advance identically across calls. The last-token logits from the final
    // chunk seed the decode loop.
    // Prefill batch size. Default picks a short-prompt-friendly value of 16
    // (matches the DFlash block_size and chain-verify q_len, so per-chunk
    // FA drift is smallest) for prompts ≤ 2048 tokens and bumps to 192 for
    // longer prompts where prefill time dominates. 192 hits the OOM ceiling
    // from the embedded gated_delta_net intermediate-state region — bigger
    // batches would need a kernel-side fix. At 192 we get ~878 tok/s on
    // 13K prompts vs ~336 tok/s at UBATCH=16, with a ~5% AL cost on the
    // following decode. Override via `DFLASH27B_PREFILL_UBATCH=N` env.
    const int prompt_len_auto = (int)prompt.size();
    int prefill_ubatch_env = (prompt_len_auto > 2048) ? 192 : 16;
    if (const char * s = std::getenv("DFLASH27B_PREFILL_UBATCH")) {
        prefill_ubatch_env = std::max(1, std::atoi(s));
    }
    const int PREFILL_UBATCH = prefill_ubatch_env;
    std::printf("[prefill] ubatch=%d\n", PREFILL_UBATCH);
    auto t_pf0 = std::chrono::steady_clock::now();
    std::vector<uint16_t> pf_mask_buf;
    std::vector<float>    pf_embed_buf;
    std::vector<int32_t>  pf_pos_buf;
    std::vector<float>    pf_logits_buf;
    const int prompt_len = (int)prompt.size();
    for (int start = 0; start < prompt_len; start += PREFILL_UBATCH) {
        const int n_tokens = std::min(PREFILL_UBATCH, prompt_len - start);
        const int kv_len   = start + n_tokens;

        // TBQ FA requires kv_len aligned to 256, which means we always need
        // a mask (padding positions must be -inf). For f16/Q-KV the existing
        // n_tokens>1 heuristic stays valid.
        const bool pf_with_mask = (g_kq_stride_pad > KQ_MASK_PAD) || (n_tokens > 1);
        if (!build_target_step(sg, w, cache, backend,
                                /*kv_start=*/start, /*n_tokens=*/n_tokens,
                                /*with_mask=*/pf_with_mask, /*capture=*/true)) {
            std::fprintf(stderr, "prefill build @%d\n", start); return 1;
        }

        pf_embed_buf.assign((size_t)hidden * n_tokens, 0.0f);
        if (!w.embedder.embed(prompt.data() + start, n_tokens, pf_embed_buf.data())) return 1;
        ggml_backend_tensor_set(sg.inp_embed, pf_embed_buf.data(), 0,
                                sizeof(float) * pf_embed_buf.size());

        // M-RoPE 4D text layout: [axis0 × n_tokens, axis1 × n_tokens,
        // axis2 × n_tokens, axis3 × n_tokens]. First 3 axes hold absolute
        // positions, axis 3 is 0 for plain text.
        pf_pos_buf.assign((size_t)4 * n_tokens, 0);
        for (int i = 0; i < n_tokens; i++) {
            const int p = start + i;
            pf_pos_buf[0 * n_tokens + i] = p;
            pf_pos_buf[1 * n_tokens + i] = p;
            pf_pos_buf[2 * n_tokens + i] = p;
            pf_pos_buf[3 * n_tokens + i] = 0;
        }
        ggml_backend_tensor_set(sg.positions, pf_pos_buf.data(), 0,
                                sizeof(int32_t) * pf_pos_buf.size());

        // Causal mask required when n_tokens > 1 OR when the TBQ FA kernel
        // is active (which pads kv_len to 256 and needs -inf on the padding
        // positions even for a single query).
        if (pf_with_mask) {
            build_causal_mask(pf_mask_buf, kv_len, n_tokens, /*kv_start=*/start);
            ggml_backend_tensor_set(sg.attn_mask, pf_mask_buf.data(), 0,
                                    sizeof(uint16_t) * pf_mask_buf.size());
        }

        auto st = ggml_backend_graph_compute(backend, sg.gf);
        if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "prefill compute @%d\n", start); return 1; }

        // Only need the last position's logits to seed decode.
        pf_logits_buf.assign(vocab, 0.0f);
        const size_t last_row_off = (size_t)(n_tokens - 1) * vocab * sizeof(float);
        ggml_backend_tensor_get(sg.logits, pf_logits_buf.data(), last_row_off,
                                sizeof(float) * vocab);
        last_tok = argmax_f32(pf_logits_buf.data(), vocab);
        committed = start + n_tokens;
    }
    auto t_pf1 = std::chrono::steady_clock::now();
    std::printf("[prefill] %d tokens in %.2f s, last_tok=%d\n",
                committed,
                std::chrono::duration<double>(t_pf1 - t_pf0).count(),
                last_tok);

    // ── DFlash decode loop
    int n_draft_steps = 0, n_accept_sum = 0, n_generated = 0;
    std::vector<float>   noise_embed_buf(hidden * q_len);
    std::vector<int32_t> noise_ids(q_len);
    std::vector<int32_t> draft_tok(q_len), target_tok(q_len);
    std::vector<float>   draft_logits_buf((size_t)vocab * q_len);
    // Sized for the max of chain q_len and DDTree flat tree size (budget+1).
    const int verify_max_tokens = std::max(q_len, ddtree_mode ? ddtree_budget + 1 : q_len);
    std::vector<float>   verify_logits_buf((size_t)vocab * verify_max_tokens);
    std::vector<uint16_t> mask_buf;
    std::vector<int32_t> pos_q_buf(q_len), pos_k_buf(max_ctx + q_len);
    std::vector<int32_t> pos4_buf(4 * q_len);

    auto t_gen0 = std::chrono::steady_clock::now();

    // Per-phase timing accumulators (microseconds)
    double tt_draft_build = 0, tt_draft_copy_feat = 0, tt_draft_set = 0,
           tt_draft_compute = 0, tt_draft_logits = 0,
           tt_snap = 0, tt_verify_build = 0, tt_verify_set = 0,
           tt_verify_compute = 0, tt_verify_logits = 0,
           tt_accept = 0, tt_restore = 0,
           tt_replay_build = 0, tt_replay_set = 0, tt_replay_compute = 0,
           tt_replay_logits = 0;
    auto sync_us = [&](){
        ggml_backend_synchronize(backend);
        return std::chrono::steady_clock::now();
    };

    while (n_generated < n_gen) {
        const int need_commit_budget = n_gen - n_generated;

        auto T0 = sync_us();

        // 1) Noise block [last_tok, MASK*15]
        noise_ids[0] = last_tok;
        for (int i = 1; i < q_len; i++) noise_ids[i] = mask_tok;
        if (!w.embedder.embed(noise_ids.data(), q_len, noise_embed_buf.data())) return 1;

        // Draft target-attention window. The draft transformer attends over
        // a slice of the history captured in cache.target_feat; this caps the
        // slice so the draft's [5*hidden × ctx_len] target_hidden_cat tensor
        // stays bounded even at 16K+ target context. Tokens older than the
        // window are invisible to the draft but still in the target's KV
        // cache (the target verify uses the full history).
        constexpr int DRAFT_CTX_MAX = 2048;
        const int draft_ctx   = std::min(committed, DRAFT_CTX_MAX);
        const int draft_start = committed - draft_ctx;

        // 2) Draft forward
        if (!build_draft_step(sg, dw, w, backend, /*ctx_len=*/draft_ctx)) {
            std::fprintf(stderr, "draft build failed\n"); return 1;
        }
        auto T_draft_build = sync_us();
        tt_draft_build += std::chrono::duration<double, std::micro>(T_draft_build - T0).count();

        ggml_backend_tensor_set(sg.inp_embed, noise_embed_buf.data(), 0,
                                sizeof(float) * noise_embed_buf.size());

        // target_hidden_cat: copy the draft-window slice of cache.target_feat
        // (positions draft_start..committed) directly device→device.
        // cache.target_feat is a ring of `target_feat_cap` bf16 slots, so
        // positions map via `pos % cap`. If the draft window straddles the
        // wrap boundary we split the bf16→f32 widen into two kernel calls.
        const size_t fc_in    = (size_t)5 * hidden;
        const int    cap      = cache.target_feat_cap;
        const size_t elt_feat = ggml_element_size(cache.target_feat);
        const int    slot0    = draft_start % cap;
        const int    pre_n    = std::min(draft_ctx, cap - slot0);
        const int    post_n   = draft_ctx - pre_n;

        dflash27b_launch_bf16_to_f32(
            (const char *)cache.target_feat->data + (size_t)slot0 * elt_feat * fc_in,
            sg.target_hidden_cat->data,
            (size_t)pre_n * fc_in,
            nullptr);
        if (post_n > 0) {
            dflash27b_launch_bf16_to_f32(
                (const char *)cache.target_feat->data,
                (char *)sg.target_hidden_cat->data + (size_t)pre_n * fc_in * sizeof(float),
                (size_t)post_n * fc_in,
                nullptr);
        }
        auto T_draft_copy = sync_us();
        tt_draft_copy_feat += std::chrono::duration<double, std::micro>(T_draft_copy - T_draft_build).count();

        for (int i = 0; i < q_len; i++) pos_q_buf[i] = draft_ctx + i;
        for (int i = 0; i < draft_ctx + q_len; i++) pos_k_buf[i] = i;
        ggml_backend_tensor_set(sg.positions,   pos_q_buf.data(), 0, sizeof(int32_t) * q_len);
        ggml_backend_tensor_set(sg.positions_k, pos_k_buf.data(), 0, sizeof(int32_t) * (draft_ctx + q_len));
        auto T_draft_set = sync_us();
        tt_draft_set += std::chrono::duration<double, std::micro>(T_draft_set - T_draft_copy).count();

        auto st = ggml_backend_graph_compute(backend, sg.gf);
        if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "draft compute %d\n", (int)st); return 1; }
        auto T_draft_compute = sync_us();
        tt_draft_compute += std::chrono::duration<double, std::micro>(T_draft_compute - T_draft_set).count();

        ggml_backend_tensor_get(sg.logits, draft_logits_buf.data(), 0,
                                sizeof(float) * vocab * q_len);
        for (int i = 0; i < q_len; i++) {
            draft_tok[i] = argmax_f32(draft_logits_buf.data() + (size_t)i * vocab, vocab);
        }
        // The block-diffusion draft is free to "denoise" position 0 even though
        // the input there is the unmasked last_tok. Pin it back so verify and
        // replay see the correct prefix.
        draft_tok[0] = last_tok;

        // DDTree top-K extraction. Positions 1..q_len-1 of the draft output
        // are the per-position distributions for the block-diffusion tree
        // (position 0 is the root/bonus slot and is fixed to last_tok). Only
        // computed in ddtree_mode to keep the argmax fast path untouched.
        // ddtree_K controls how many candidates per position are available as
        // tree siblings. Budget <= L means pure chain → no siblings needed, so
        // we can skip the O(L*vocab) top-K extract entirely and just fill rank 0
        // from draft_tok. For larger budgets we need real top-K.
        const int ddtree_K = (ddtree_budget > q_len - 1) ? 8 : 1;
        static std::vector<float>   ddtree_top_log_probs; // [L × K]
        static std::vector<int32_t> ddtree_top_token_ids; // [L × K]
        if (ddtree_mode) {
            const int L = q_len - 1;
            if ((int)ddtree_top_log_probs.size() < L * ddtree_K) {
                ddtree_top_log_probs.assign((size_t)L * ddtree_K, 0.0f);
                ddtree_top_token_ids.assign((size_t)L * ddtree_K, 0);
            }
            if (ddtree_K == 1) {
                // Fast path: draft_tok already holds the top-1 per position.
                // Skip extract; log-probs are irrelevant for pure chain build.
                for (int i = 0; i < L; i++) {
                    ddtree_top_log_probs[i] = 0.0f;
                    ddtree_top_token_ids[i] = draft_tok[i + 1];  // +1 to skip slot 0
                }
            } else {
                extract_draft_topk(draft_logits_buf.data() + (size_t)vocab,
                                   L, vocab, ddtree_K,
                                   ddtree_top_log_probs.data(),
                                   ddtree_top_token_ids.data(),
                                   ddtree_temp);
            }
        }
        auto T_draft_logits = sync_us();
        tt_draft_logits += std::chrono::duration<double, std::micro>(T_draft_logits - T_draft_compute).count();

        // 3) Snapshot SSM state (skipped in fast_rollback mode: the patched
        //    gated_delta_net kernel captures per-step intermediate states, so
        //    we don't need a pre-verify snapshot to restore from).
        if (!fast_rollback) {
            snapshot_ssm_state(cache);
        }
        auto T_snap = sync_us();
        tt_snap += std::chrono::duration<double, std::micro>(T_snap - T_draft_logits).count();

        // 4) Target verify on draft tokens.
        //
        // Two paths, toggled by --seq-verify:
        //   - Batched (default): one target forward over q_len tokens with a causal
        //     mask. Fast but suspected of numerical divergence vs stepwise decode
        //     per z-lab issue #57 ("batched greedy verification diverges from
        //     stepwise baseline").
        //   - Sequential: q_len independent single-token decodes. Slow but
        //     bit-equivalent to what the target would produce during plain
        //     autoregressive decode — the ground truth for greedy spec decoding.
        //
        // In both paths we set capture=true so if commit_n == q_len we can skip
        // the replay entirely. For commit_n < q_len, replay overwrites target_feat
        // at [committed..committed+commit_n-1]; positions past that are stale but
        // never read by the next iteration's draft.

        auto T_verify_build = T_snap;
        auto T_verify_set = T_snap;
        auto T_verify_compute = T_snap;

        // ── DDTree path: tree-structured verify + walk + rollback ─────────
        //
        // Structure of one DDTree round (ported from liranringel/ddtree.py):
        //   1. Build tree from draft top-K via best-first heap (Algorithm 1)
        //   2. Flatten tree in DFS order: slot 0 = root (= last_tok), slots
        //      1..n_nodes = tree nodes. Positions = committed + depth.
        //   3. Build an ancestor-only attention mask + parent_ids array.
        //   4. Run target forward via build_target_step_tree (our kernel mod
        //      handles DeltaNet/SSM tree recurrence via parent_ids).
        //   5. Walk the tree from root following target.argmax; the matched
        //      path is the accepted prefix. First unmatched target token
        //      becomes the next round's bonus (implicit via last_tok).
        //   6. Rollback: SSM state ← cache.ssm_intermediate[last_accepted_dfs_idx]
        //      KV cache ← cudaMemcpy the accepted DFS-order slots to slots 0..k-1
        //      conv_state ← use the (depth-1)-th slot of cache.conv_input_cache
        if (ddtree_mode) {
            const int L = q_len - 1;
            DDTree tree = build_ddtree(
                ddtree_top_log_probs.data(),
                ddtree_top_token_ids.data(),
                L, ddtree_K, ddtree_budget,
                ddtree_chain_seed);

            const int N = 1 + tree.n_nodes;  // flat size including root

            if (!build_target_step_tree(sg, w, cache, backend,
                                        /*kv_start=*/committed, /*n_tokens=*/N)) {
                std::fprintf(stderr, "ddtree verify build failed\n"); return 1;
            }
            T_verify_build = sync_us();
            tt_verify_build += std::chrono::duration<double, std::micro>(T_verify_build - T_snap).count();

            // Embeddings: [last_tok, tree.token_ids[0..n_nodes-1]]
            std::vector<int32_t> flat_tokens(N);
            flat_tokens[0] = last_tok;
            for (int i = 0; i < tree.n_nodes; i++) flat_tokens[1 + i] = tree.token_ids[i];

            std::vector<float> tree_embed(hidden * N);
            if (!w.embedder.embed(flat_tokens.data(), N, tree_embed.data())) return 1;
            ggml_backend_tensor_set(sg.inp_embed, tree_embed.data(), 0,
                                    sizeof(float) * hidden * N);

            // M-RoPE axis-major positions: committed + depth_of_node.
            // Slot 0 = root = depth 0 → position `committed`.
            std::vector<int32_t> pos4(4 * N);
            for (int i = 0; i < N; i++) {
                int p = committed + (i == 0 ? 0 : tree.depths[i - 1]);
                pos4[0 * N + i] = p;
                pos4[1 * N + i] = p;
                pos4[2 * N + i] = p;
                pos4[3 * N + i] = 0;
            }
            ggml_backend_tensor_set(sg.positions, pos4.data(), 0, sizeof(int32_t) * 4 * N);

            // Ancestor-only attention mask (f16).
            build_tree_mask(tree, /*past_length=*/committed, mask_buf);
            ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0,
                                    sizeof(uint16_t) * mask_buf.size());

            // parent_ids for tree-mode DeltaNet kernel.
            // Slot 0 (root): -1 (reload initial state — matches kernel's skip
            // at t==0). Slots 1..N-1: the tree's parent index in the flat array.
            std::vector<int32_t> parent_ids(N);
            parent_ids[0] = -1;
            for (int i = 1; i < N; i++) parent_ids[i] = (int32_t)tree.parents[i];
            ggml_backend_tensor_set(sg.parent_ids, parent_ids.data(), 0,
                                    sizeof(int32_t) * N);

            T_verify_set = sync_us();
            tt_verify_set += std::chrono::duration<double, std::micro>(T_verify_set - T_verify_build).count();

            st = ggml_backend_graph_compute(backend, sg.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "ddtree verify compute %d\n", (int)st); return 1;
            }
            T_verify_compute = sync_us();
            tt_verify_compute += std::chrono::duration<double, std::micro>(T_verify_compute - T_verify_set).count();

            // Read the N verify logits, compute posterior argmax per slot.
            ggml_backend_tensor_get(sg.logits, verify_logits_buf.data(), 0,
                                    sizeof(float) * vocab * N);
            std::vector<int32_t> posterior(N);
            for (int i = 0; i < N; i++) {
                posterior[i] = argmax_f32(verify_logits_buf.data() + (size_t)i * vocab, vocab);
            }

            // Walk tree: accepted DFS indices and next bonus token.
            int next_token = -1;
            std::vector<int> accepted = follow_verified_tree(tree, posterior.data(), next_token);
            const int accept_depth = (int)accepted.size();  // includes root

            // Detect when the walk takes a sibling branch (accepted node
            // whose DFS index is OUTSIDE the chain spine [0..L]).
            bool walked_sibling = false;
            for (int x : accepted) {
                if (x > L) { walked_sibling = true; break; }
            }
            if (walked_sibling || n_draft_steps < 2) {
                std::printf("[dbg sib step %d] N=%d accept=%d walked_sib=%d\n",
                            n_draft_steps, N, accept_depth, walked_sibling ? 1 : 0);
                std::printf("  walk:");
                for (int x : accepted) std::printf(" %d", x);
                if (walked_sibling) {
                    std::printf("\n  sibling info:");
                    for (int i = L; i < tree.n_nodes; i++) {
                        std::printf(" [%d:d%d:p%d:%d]",
                            i + 1, tree.depths[i], tree.parents[i + 1], tree.token_ids[i]);
                    }
                }
                std::printf("\n");
            }


            std::printf("[step %d] committed=%d last_tok=%d tree_N=%d accept=%d next=%d\n",
                        n_draft_steps, committed, last_tok, N, accept_depth, next_token);

            // Commit count: matches chain mode's accept_n semantics. The root
            // (= previous iter's last_tok) is "pending" — not yet in out_all —
            // and gets committed here along with each accepted child token.
            // next_token (target's correction at the deepest accepted node)
            // becomes the new last_tok, pending for the next iter.
            int commit_n = accept_depth;  // root + accepted children
            if (commit_n > need_commit_budget) commit_n = need_commit_budget;

            // Push the accepted path's tokens to out_all. The root token is
            // last_tok (the pending token from the previous iter). Each
            // subsequent accepted node contributes its own tree.token_ids
            // entry (dfs_idx - 1 because flat slot 0 = root, slot 1..N-1 =
            // tree.token_ids[0..n_nodes-1]).
            bool hit_eos = false;
            for (int i = 0; i < commit_n; i++) {
                const int dfs_idx = accepted[i];
                const int32_t tok = (dfs_idx == 0)
                    ? last_tok
                    : tree.token_ids[dfs_idx - 1];
                out_all.push_back(tok); stream_emit(tok);
                if (tok == 248045) hit_eos = true;
            }
            last_tok = next_token;

            auto T_accept = sync_us();
            tt_accept += std::chrono::duration<double, std::micro>(T_accept - T_verify_compute).count();

            if (hit_eos) break;

            // Rollback: per-layer DeltaNet SSM and conv state + KV compaction
            // for full-attention layers.
            //
            // SSM: the kernel wrote intermediate[i] for each flat-tree token i
            // (i = DFS index, 0 = root). We want the state AFTER processing
            // all `commit_n` tokens we just committed, i.e. the state after
            // the last committed DFS node = accepted[commit_n - 1]. For the
            // common case commit_n == accept_depth, this is the deepest
            // accepted node.
            const int rollback_dfs = (commit_n > 0)
                ? accepted[commit_n - 1]
                : 0;
            // Fast path detection: pure-chain walk has accepted[i] == i for
            // every i. Used by rollback to skip the parent-chain gather.
            bool walked_sibling_for_rollback = false;
            for (int i = 0; i < commit_n; i++) {
                if (accepted[i] != i) { walked_sibling_for_rollback = true; break; }
            }

            {
                const int n_delta = (int)sg.delta_captures.size();
                cudaStream_t stream = nullptr;
                for (int il = 0; il < n_delta; il++) {
                    const DeltaNetCapture & cap = sg.delta_captures[il];
                    if (!cap.ssm_intermediate_states || !cap.conv_input) {
                        std::fprintf(stderr, "ddtree rollback: missing capture layer %d\n", il);
                        return 1;
                    }
                    // SSM state rollback: source is cache.ssm_intermediate_states
                    // (f16, [S_v, S_v, H_v, max_verify_tokens]) at slot
                    // rollback_dfs. Destination is cache.ssm_state[il] (f32).
                    // Use a tiny CUDA kernel (src/f16_convert.cu) to widen f16
                    // → f32 in a single launch per layer.
                    const size_t ssm_elems =
                        (size_t)cache.ssm_state[il]->ne[0] *
                        (size_t)cache.ssm_state[il]->ne[1] *
                        (size_t)cache.ssm_state[il]->ne[2];
                    const size_t ssm_src_offset =
                        (size_t)rollback_dfs * cap.ssm_intermediate_states->nb[3];
                    const void * ssm_src =
                        (const char *)cap.ssm_intermediate_states->data + ssm_src_offset;
                    dflash27b_launch_f16_to_f32(ssm_src,
                                                cache.ssm_state[il]->data,
                                                ssm_elems,
                                                stream);
                    cudaError_t ce = cudaSuccess;  // launch error checked in the conv block below

                    // Conv rollback: copy the K-1 most recent inputs along
                    // the rolled-back token's ANCESTRY (not DFS order). Two
                    // paths:
                    //   - Pure chain accept (walked_sibling == false): the
                    //     conv window is 3 contiguous slots in conv_input, so
                    //     a single cudaMemcpy2DAsync handles it. Hot path.
                    //   - Sibling accept: scattered slots, fall back to K-1
                    //     individual column copies via parent-chain walk.
                    const int K_conv = 4;
                    const int row_cnt = (int)cap.conv_input->ne[1];
                    const size_t elt = ggml_element_size(cap.conv_input);
                    const size_t dpitch = (K_conv - 1) * elt;
                    const size_t spitch = cap.conv_input->nb[1];
                    if (!walked_sibling_for_rollback) {
                        // Fast path: 3 contiguous slots ending at rollback_dfs.
                        const int conv_off = rollback_dfs + 1;
                        const void * conv_src =
                            (const char *)cap.conv_input->data + (size_t)conv_off * elt;
                        ce = cudaMemcpy2DAsync(cache.conv_state[il]->data, dpitch,
                                               conv_src, spitch,
                                               (K_conv - 1) * elt, row_cnt,
                                               cudaMemcpyDeviceToDevice, stream);
                        if (ce != cudaSuccess) {
                            std::fprintf(stderr, "ddtree conv fast il=%d: %s\n",
                                         il, cudaGetErrorString(ce));
                            return 1;
                        }
                    } else {
                        int virt[K_conv - 1];
                        virt[K_conv - 2] = rollback_dfs;
                        for (int k = K_conv - 3; k >= 0; k--) {
                            const int prev = virt[k + 1];
                            virt[k] = (prev >= 0) ? (int)tree.parents[prev] : (prev - 1);
                        }
                        for (int k = 0; k < K_conv - 1; k++) {
                            const int sx_slot = (K_conv - 1) + virt[k];
                            const void * src_col =
                                (const char *)cap.conv_input->data + (size_t)sx_slot * elt;
                            char * dst_col =
                                (char *)cache.conv_state[il]->data + (size_t)k * elt;
                            ce = cudaMemcpy2DAsync(dst_col, dpitch,
                                                   src_col, spitch,
                                                   elt, row_cnt,
                                                   cudaMemcpyDeviceToDevice, stream);
                            if (ce != cudaSuccess) {
                                std::fprintf(stderr, "ddtree conv col il=%d k=%d: %s\n",
                                             il, k, cudaGetErrorString(ce));
                                return 1;
                            }
                        }
                    }
                }

                // target_feat compaction: written in DFS order during verify
                // (column kv_start+i = dfs slot i's features). Same logic as
                // KV cache: when accepted[d] != d, copy the accepted DFS slot's
                // features to the spine slot at d so next iter's draft reads
                // the right history. Position→slot uses `% target_feat_cap`
                // to account for the ring buffer.
                if (cache.target_feat) {
                    const size_t elt = ggml_element_size(cache.target_feat);
                    const int    fc_in = (int)cache.target_feat->ne[0];  // 5*hidden
                    const size_t col_stride = cache.target_feat->nb[1];
                    const int    tcap = cache.target_feat_cap;
                    for (int d = 1; d < commit_n; d++) {
                        const int src_dfs = accepted[d];
                        if (src_dfs == d) continue;
                        const int    src_slot = (committed + src_dfs) % tcap;
                        const int    dst_slot = (committed + d)       % tcap;
                        const size_t src_off  = (size_t)src_slot * col_stride;
                        const size_t dst_off  = (size_t)dst_slot * col_stride;
                        cudaMemcpyAsync((char *)cache.target_feat->data + dst_off,
                                        (const char *)cache.target_feat->data + src_off,
                                        (size_t)fc_in * elt,
                                        cudaMemcpyDeviceToDevice, stream);
                    }
                }

                // Full-attention KV compaction: the verify wrote K/V at slots
                // [committed..committed+N-1] in DFS tree order (slot 0 = root).
                // For the next iter's verify to see the correct committed
                // prefix, slots [committed..committed+commit_n-1] must hold
                // the K/V of the accepted path's committed tokens. For each
                // committed position d in 0..commit_n-1, the source K/V is at
                // DFS slot accepted[d]. d==0 is always the root (DFS slot 0),
                // trivially aligned. For d>=1, copy if accepted[d] != d.
                const int n_full_attn = (int)cache.attn_k.size();
                for (int d = 0; d < commit_n; d++) {
                    const int src_dfs = accepted[d];
                    const int dst_slot = d;
                    if (src_dfs == dst_slot) continue;  // already aligned
                    for (int l = 0; l < n_full_attn; l++) {
                        // Each slot: head_dim * n_kv floats in f16 per tensor.
                        ggml_tensor * ck = cache.attn_k[l];
                        ggml_tensor * cv = cache.attn_v[l];
                        const size_t slot_bytes = ck->nb[1];  // stride between slots
                        const size_t src_off = (size_t)(committed + src_dfs) * slot_bytes;
                        const size_t dst_off = (size_t)(committed + dst_slot) * slot_bytes;
                        // Per-head-kv layout: shape [head_dim, max_ctx, n_head_kv].
                        // nb[2] is distance between heads; we copy one slot's
                        // slice per head. For simplicity, do a 2D copy across
                        // the head dimension.
                        const int n_kv = (int)ck->ne[2];
                        for (int h = 0; h < n_kv; h++) {
                            const size_t head_src = src_off + (size_t)h * ck->nb[2];
                            const size_t head_dst = dst_off + (size_t)h * ck->nb[2];
                            cudaMemcpyAsync((char *)ck->data + head_dst,
                                            (const char *)ck->data + head_src,
                                            slot_bytes, cudaMemcpyDeviceToDevice, stream);
                            cudaMemcpyAsync((char *)cv->data + head_dst,
                                            (const char *)cv->data + head_src,
                                            slot_bytes, cudaMemcpyDeviceToDevice, stream);
                        }
                    }
                }
                // No explicit sync: stream==nullptr (default stream) serializes
                // these copies before the next iter's draft/verify kernels.
                // The CPU returns immediately and the next iter's CPU work
                // (graph build, embed) can overlap with the GPU compaction.
            }

            committed    += commit_n;
            n_generated  += commit_n;
            n_accept_sum += commit_n;  // for stats
            n_draft_steps++;
            continue;  // skip the rest of the verify/commit logic for this iter
        }

        if (!seq_verify) {
            if (!build_target_step(sg, w, cache, backend,
                                    /*kv_start=*/committed, /*n_tokens=*/q_len,
                                    /*with_mask=*/true, /*capture=*/true,
                                    /*capture_delta_intermediate=*/fast_rollback)) {
                std::fprintf(stderr, "verify build failed\n"); return 1;
            }
            T_verify_build = sync_us();
            tt_verify_build += std::chrono::duration<double, std::micro>(T_verify_build - T_snap).count();

            std::vector<float> verify_embed(hidden * q_len);
            if (!w.embedder.embed(draft_tok.data(), q_len, verify_embed.data())) return 1;
            ggml_backend_tensor_set(sg.inp_embed, verify_embed.data(), 0,
                                    sizeof(float) * verify_embed.size());

            // M-RoPE axis-major layout: [axis0_tok0..axis0_tokN-1, axis1_..., axis2_..., axis3_...].
            // First 3 axes hold the token position; axis 3 is always 0 for text.
            for (int i = 0; i < q_len; i++) {
                int p = committed + i;
                pos4_buf[0 * q_len + i] = p;
                pos4_buf[1 * q_len + i] = p;
                pos4_buf[2 * q_len + i] = p;
                pos4_buf[3 * q_len + i] = 0;
            }
            ggml_backend_tensor_set(sg.positions, pos4_buf.data(), 0, sizeof(int32_t) * 4 * q_len);

            build_causal_mask(mask_buf, committed + q_len, q_len, committed);
            ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0, sizeof(uint16_t) * mask_buf.size());
            T_verify_set = sync_us();
            tt_verify_set += std::chrono::duration<double, std::micro>(T_verify_set - T_verify_build).count();

            st = ggml_backend_graph_compute(backend, sg.gf);
            if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "verify compute %d\n", (int)st); return 1; }
            T_verify_compute = sync_us();
            tt_verify_compute += std::chrono::duration<double, std::micro>(T_verify_compute - T_verify_set).count();

            ggml_backend_tensor_get(sg.logits, verify_logits_buf.data(), 0,
                                    sizeof(float) * vocab * q_len);
            for (int i = 0; i < q_len; i++) {
                target_tok[i] = argmax_f32(verify_logits_buf.data() + (size_t)i * vocab, vocab);
            }
        } else {
            // Sequential verify: q_len independent single-token decodes.
            // Each call writes K/V at slot committed+i and advances SSM by 1.
            // After the loop, target cache state is identical to the batched
            // path's state (both end at committed+q_len). Restore/replay below
            // still apply correctly.
            std::vector<float> single_embed(hidden);
            int32_t p4_single[4];
            for (int i = 0; i < q_len; i++) {
                if (!build_target_step(sg, w, cache, backend,
                                        /*kv_start=*/committed + i, /*n_tokens=*/1,
                                        /*with_mask=*/false, /*capture=*/true)) {
                    std::fprintf(stderr, "seq verify build %d failed\n", i); return 1;
                }
                int32_t t = draft_tok[i];
                if (!w.embedder.embed(&t, 1, single_embed.data())) return 1;
                ggml_backend_tensor_set(sg.inp_embed, single_embed.data(), 0,
                                        sizeof(float) * hidden);
                int p = committed + i;
                p4_single[0] = p; p4_single[1] = p; p4_single[2] = p; p4_single[3] = 0;
                ggml_backend_tensor_set(sg.positions, p4_single, 0, sizeof(int32_t) * 4);

                st = ggml_backend_graph_compute(backend, sg.gf);
                if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "seq verify compute %d at %d\n", (int)st, i); return 1; }

                ggml_backend_tensor_get(sg.logits,
                                        verify_logits_buf.data() + (size_t)i * vocab,
                                        0, sizeof(float) * vocab);
                target_tok[i] = argmax_f32(verify_logits_buf.data() + (size_t)i * vocab, vocab);
            }
            T_verify_compute = sync_us();
            tt_verify_compute += std::chrono::duration<double, std::micro>(T_verify_compute - T_snap).count();
        }
        auto T_verify_logits = sync_us();
        tt_verify_logits += std::chrono::duration<double, std::micro>(T_verify_logits - T_verify_compute).count();

        std::printf("[step %d] committed=%d last_tok=%d\n", n_draft_steps, committed, last_tok);

        // 5) Greedy longest-prefix accept with standard spec-decoding comparison.
        //
        //   - draft_tok[0] should equal last_tok (the correct first token from the
        //     previous forward). Accept it unconditionally.
        //   - target_tok[i] = argmax(logit at position committed+i) = target's
        //     prediction for the token AT position committed+i+1 (given draft_tok[0..i]).
        //   - So the check is: draft_tok[i+1] == target_tok[i], for i=0..q_len-2.
        //   - First mismatch at i=k → accept draft_tok[0..k] (k+1 tokens),
        //     bonus = target_tok[k] (the correct replacement for draft_tok[k+1]).

        int accept_n = 1;  // draft_tok[0] assumed = last_tok
        for (int i = 0; i < q_len - 1; i++) {
            if (draft_tok[i + 1] == target_tok[i]) accept_n++;
            else break;
        }
        // Two commit strategies:
        //   - Legacy (replay path): commit_n = accept_n + 1, the extra is the
        //     "bonus" token (target's correction or target_tok[q_len-1] when
        //     all accepted). Requires a replay forward pass to advance state.
        //   - Fast-rollback path: commit_n = accept_n, no explicit bonus. Use
        //     verify_logits[accept_n-1] as next last_tok; the "bonus" becomes
        //     draft_tok[0] of the next iter and is accepted unconditionally.
        //     Identical output stream, one fewer commit per iter tallied, but
        //     no extra forward pass needed.
        int bonus_tok = -1;
        int commit_n;
        if (fast_rollback) {
            commit_n = accept_n;
        } else {
            if (accept_n < q_len) {
                bonus_tok = target_tok[accept_n - 1];
            }
            commit_n = accept_n + (bonus_tok >= 0 ? 1 : 0);
        }
        std::printf("[step %d] accept_n=%d bonus=%d commit_n=%d\n",
                    n_draft_steps, accept_n, bonus_tok, commit_n);

        // Don't overshoot n_gen
        if (commit_n > need_commit_budget) {
            commit_n = need_commit_budget;
            // If we were going to add the bonus but budget is tight, drop it.
            if (commit_n <= accept_n) bonus_tok = -1;
        }
        auto T_accept = sync_us();
        tt_accept += std::chrono::duration<double, std::micro>(T_accept - T_verify_logits).count();

        // 6) Rollback and commit.
        //
        // Fast-rollback path: no replay. Use the per-step SSM intermediate states
        // captured during verify to roll back DeltaNet state, and slice the conv
        // input tensor for the conv state. Next last_tok comes from verify's
        // logits at position (accept_n - 1) — the target's prediction at position
        // committed+accept_n given the accepted prefix. The implicit bonus
        // becomes the next iter's draft_tok[0].
        //
        // Legacy path: restore SSM state from snapshot and run the replay.
        double t_rollback_us = 0, t_replay_build_us = 0, t_replay_set_us = 0;
        double t_replay_compute_us = 0, t_replay_logits_us = 0;

        if (fast_rollback) {
            auto T_rb0 = sync_us();

            // Rollback SSM + conv state unless we fully accepted (in which case
            // state after processing all q_len tokens is exactly what we want).
            if (commit_n < q_len) {
                const int rollback_idx = commit_n - 1;  // index into per-step intermediates
                // Temporary ctx for view tensors (no data alloc — views inherit
                // data pointers from their already-live sources).
                ggml_init_params tp{};
                tp.mem_size   = 1024 * 1024;
                tp.mem_buffer = nullptr;
                tp.no_alloc   = true;
                ggml_context * tmp_ctx = ggml_init(tp);
                if (!tmp_ctx) { std::fprintf(stderr, "rollback ctx init failed\n"); return 1; }

                const int n_delta = (int)sg.delta_captures.size();
                cudaStream_t stream = nullptr;  // use default stream
                for (int il = 0; il < n_delta; il++) {
                    const DeltaNetCapture & cap = sg.delta_captures[il];
                    if (!cap.ssm_intermediate_states || !cap.conv_input) {
                        std::fprintf(stderr, "rollback: missing capture at layer %d\n", il);
                        return 1;
                    }

                    // ── SSM rollback: copy intermediate[rollback_idx] → cache.ssm_state[il]
                    //
                    // cap.ssm_intermediate_states is the persistent cache buffer
                    // cache.ssm_intermediate[il], shape [S_v, S_v, H_v, q_len].
                    // Stored in f16 (see create_target_cache) to halve memory;
                    // cache.ssm_state[il] is f32. Use the widen kernel to
                    // convert on copy, same as the DDtree rollback path.
                    const size_t ssm_elems =
                        (size_t)cache.ssm_state[il]->ne[0] *
                        (size_t)cache.ssm_state[il]->ne[1] *
                        (size_t)cache.ssm_state[il]->ne[2];
                    const size_t ssm_src_offset =
                        (size_t)rollback_idx * cap.ssm_intermediate_states->nb[3];
                    const void * ssm_src =
                        (const char *)cap.ssm_intermediate_states->data + ssm_src_offset;
                    dflash27b_launch_f16_to_f32(ssm_src,
                                                cache.ssm_state[il]->data,
                                                ssm_elems,
                                                stream);
                    cudaError_t ce = cudaSuccess;

                    // ── Conv rollback: copy conv_input[commit_n..commit_n+K-2, :, :]
                    //    into cache.conv_state[il].
                    //
                    // conv_input shape: [kernel-1 + n_tokens, conv_channels, 1]
                    //   nb[0] = elt, nb[1] = (kernel-1+n_tokens)*elt
                    // conv_state shape: [kernel-1, conv_channels, 1]
                    //   nb[0] = elt, nb[1] = (kernel-1)*elt
                    //
                    // Need cudaMemcpy2D because the source has a larger row stride
                    // (spans kernel-1+n_tokens values along dim 0) than the dest.
                    const int K_conv = 4;                            // qwen3.5 DeltaNet conv kernel
                    const int row_cnt = (int)cap.conv_input->ne[1];  // conv_channels (10240)
                    const size_t elt = ggml_element_size(cap.conv_input);
                    const size_t dpitch = (K_conv - 1) * elt;        // 12 bytes
                    const size_t spitch = cap.conv_input->nb[1];     // (K-1+n_tokens)*elt
                    const size_t width  = (K_conv - 1) * elt;        // copy 3 floats per row
                    const void * conv_src =
                        (const char *)cap.conv_input->data + commit_n * elt;
                    ce = cudaMemcpy2DAsync(cache.conv_state[il]->data, dpitch,
                                           conv_src, spitch,
                                           width, row_cnt,
                                           cudaMemcpyDeviceToDevice, stream);
                    if (ce != cudaSuccess) {
                        std::fprintf(stderr, "cudaMemcpy2D conv rollback il=%d: %s\n",
                                     il, cudaGetErrorString(ce));
                        return 1;
                    }
                }
                cudaStreamSynchronize(stream);

                ggml_free(tmp_ctx);
            }

            // Next last_tok: target's prediction at position committed+accept_n
            // given the accepted prefix.
            //   - commit_n < q_len: verify_logits[accept_n-1] (target_tok[accept_n-1]).
            //   - commit_n == q_len: verify_logits[q_len-1]  (target_tok[q_len-1]).
            // Both already computed as `target_tok[commit_n-1]` during accept.
            last_tok = target_tok[commit_n - 1];

            auto T_rb1 = sync_us();
            t_rollback_us = std::chrono::duration<double, std::micro>(T_rb1 - T_rb0).count();
            tt_restore += t_rollback_us;

            // Commit: push accepted draft tokens to out_all. No bonus — next iter
            // picks it up as last_tok.
            bool hit_eos = false;
            for (int i = 0; i < commit_n; i++) {
                out_all.push_back(draft_tok[i]); stream_emit(draft_tok[i]);
                if (draft_tok[i] == 248045) hit_eos = true;
            }
            if (hit_eos) break;
        } else {
            // ── Legacy replay path ──
            restore_ssm_state(cache);
            auto T_restore = sync_us();
            tt_restore += std::chrono::duration<double, std::micro>(T_restore - T_accept).count();
            std::vector<int32_t> replay_tok(commit_n);
            for (int i = 0; i < commit_n; i++) {
                if (i < accept_n && i < (int)draft_tok.size()) {
                    replay_tok[i] = draft_tok[i];
                } else {
                    replay_tok[i] = bonus_tok;
                }
            }

            bool replay_with_mask = (commit_n > 1);
            if (!build_target_step(sg, w, cache, backend,
                                    committed, commit_n,
                                    replay_with_mask, /*capture=*/true)) {
                std::fprintf(stderr, "replay build failed\n"); return 1;
            }
            auto T_replay_build = sync_us();
            tt_replay_build += std::chrono::duration<double, std::micro>(T_replay_build - T_restore).count();

            std::vector<float> replay_embed(hidden * commit_n);
            if (!w.embedder.embed(replay_tok.data(), commit_n, replay_embed.data())) return 1;
            ggml_backend_tensor_set(sg.inp_embed, replay_embed.data(), 0, sizeof(float) * replay_embed.size());
            std::vector<int32_t> replay_pos(4 * commit_n);
            for (int i = 0; i < commit_n; i++) {
                int p = committed + i;
                replay_pos[0 * commit_n + i] = p;
                replay_pos[1 * commit_n + i] = p;
                replay_pos[2 * commit_n + i] = p;
                replay_pos[3 * commit_n + i] = 0;
            }
            ggml_backend_tensor_set(sg.positions, replay_pos.data(), 0, sizeof(int32_t) * 4 * commit_n);
            if (replay_with_mask) {
                build_causal_mask(mask_buf, committed + commit_n, commit_n, committed);
                ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0, sizeof(uint16_t) * mask_buf.size());
            }
            auto T_replay_set = sync_us();
            tt_replay_set += std::chrono::duration<double, std::micro>(T_replay_set - T_replay_build).count();

            st = ggml_backend_graph_compute(backend, sg.gf);
            if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "replay compute %d\n", (int)st); return 1; }
            auto T_replay_compute = sync_us();
            tt_replay_compute += std::chrono::duration<double, std::micro>(T_replay_compute - T_replay_set).count();

            std::vector<float> last_logits(vocab);
            ggml_backend_tensor_get(sg.logits, last_logits.data(),
                                    sizeof(float) * vocab * (commit_n - 1),
                                    sizeof(float) * vocab);
            last_tok = argmax_f32(last_logits.data(), vocab);
            auto T_replay_logits = sync_us();
            tt_replay_logits += std::chrono::duration<double, std::micro>(T_replay_logits - T_replay_compute).count();

            bool hit_eos = false;
            for (int i = 0; i < commit_n; i++) {
                out_all.push_back(replay_tok[i]); stream_emit(replay_tok[i]);
                if (replay_tok[i] == 248045) hit_eos = true;
            }
            if (hit_eos) break;
        }

        committed    += commit_n;
        n_generated  += commit_n;
        n_accept_sum += accept_n;
        n_draft_steps++;
    }

    auto t_gen1 = std::chrono::steady_clock::now();
    double gen_s = std::chrono::duration<double>(t_gen1 - t_gen0).count();
    double tps = n_generated / std::max(1e-9, gen_s);

    auto avg_ms = [&](double us){ return us / std::max(1, n_draft_steps) / 1000.0; };
    std::printf("\n[timing] per-step averages over %d steps (ms):\n", n_draft_steps);
    std::printf("  draft_build    %.2f\n", avg_ms(tt_draft_build));
    std::printf("  draft_copyfeat %.2f\n", avg_ms(tt_draft_copy_feat));
    std::printf("  draft_set      %.2f\n", avg_ms(tt_draft_set));
    std::printf("  draft_compute  %.2f\n", avg_ms(tt_draft_compute));
    std::printf("  draft_logits   %.2f\n", avg_ms(tt_draft_logits));
    std::printf("  snapshot_ssm   %.2f\n", avg_ms(tt_snap));
    std::printf("  verify_build   %.2f\n", avg_ms(tt_verify_build));
    std::printf("  verify_set     %.2f\n", avg_ms(tt_verify_set));
    std::printf("  verify_compute %.2f\n", avg_ms(tt_verify_compute));
    std::printf("  verify_logits  %.2f\n", avg_ms(tt_verify_logits));
    std::printf("  accept         %.2f\n", avg_ms(tt_accept));
    std::printf("  restore_ssm    %.2f\n", avg_ms(tt_restore));
    std::printf("  replay_build   %.2f\n", avg_ms(tt_replay_build));
    std::printf("  replay_set     %.2f\n", avg_ms(tt_replay_set));
    std::printf("  replay_compute %.2f\n", avg_ms(tt_replay_compute));
    std::printf("  replay_logits  %.2f\n", avg_ms(tt_replay_logits));
    double sum_ms = avg_ms(tt_draft_build + tt_draft_copy_feat + tt_draft_set + tt_draft_compute + tt_draft_logits
                           + tt_snap + tt_verify_build + tt_verify_set + tt_verify_compute + tt_verify_logits
                           + tt_accept + tt_restore + tt_replay_build + tt_replay_set + tt_replay_compute + tt_replay_logits);
    std::printf("  ----- sum     %.2f\n", sum_ms);

    std::printf("\n[dflash] generated %d tokens in %.3f s  ->  %.2f tok/s\n",
                n_generated, gen_s, tps);
    std::printf("[dflash] %d draft steps, accepted=%d/%d (%.1f%% per step), "
                "avg commit/step=%.2f\n",
                n_draft_steps, n_accept_sum, n_draft_steps * q_len,
                (n_draft_steps > 0 ? 100.0 * n_accept_sum / (n_draft_steps * q_len) : 0.0),
                (n_draft_steps > 0 ? (double)n_generated / n_draft_steps : 0.0));
    std::printf("[dflash] output tail: ");
    int tail_start = std::max(0, (int)out_all.size() - 20);
    for (int i = tail_start; i < (int)out_all.size(); i++) std::printf("%d ", out_all[i]);
    std::printf("\n");

    if (daemon_mode) {
        stream_emit(-1);
    } else {
        if (out_path) write_int32_file(out_path, out_all);
        break;
    }

    } // end while(true)

    step_graph_destroy(sg);
    free_target_cache(cache);
    free_draft_weights(dw);
    free_target_weights(w);
    ggml_backend_free(backend);
    return 0;
}
