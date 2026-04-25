// Builds a ggml compute graph for one forward pass of the DFlash draft
// (5-layer non-causal Qwen3-flavored block-diffusion model).
//
// Stateless: no KV cache. Each call takes:
//   - noise_embed         [hidden,   q_len, 1]   bf16   (target.tok_embd on [last_tok, MASK*15])
//   - target_hidden_cat   [5*hidden, ctx_len, 1] bf16   (5 target layers concat along features)
//   - positions_q         [q_len]                i32    values [ctx_len..ctx_len+q_len-1]
//   - positions_k         [ctx_len+q_len]        i32    values [0..ctx_len+q_len-1]
// and returns:
//   - hidden_states       [hidden,   q_len, 1]   bf16   (final RMSNorm; NO lm_head here)
//
// The caller projects `hidden_states` through the TARGET's lm_head separately
// (the draft has no lm_head of its own, it shares the target's).
//
// Semantics match megaqwen3_27b_dflash/reference/dflash_reference.py exactly:
//   - fc @ target_hidden_cat -> rms_norm with hidden_norm -> target_feat
//   - Per layer (non-causal):
//       h_norm = rms_norm(h) * input_layernorm
//       Q  = wq  @ h_norm   -> per-head q_norm
//       K_ctx/V_ctx = wk/wv @ target_feat
//       K_noi/V_noi = wk/wv @ h_norm
//       K = concat[K_ctx, K_noi]  -> per-head k_norm
//       V = concat[V_ctx, V_noi]
//       RoPE(Q, positions_q); RoPE(K, positions_k)    (NEOX style, theta=10M)
//       attn = flash_attn_ext(Q, K, V, mask=null, scale=1/sqrt(head_dim))   non-causal
//       h   += wo @ attn
//       h_norm = rms_norm(h) * post_attention_layernorm
//       h   += w_down @ (silu(w_gate @ h_norm) * (w_up @ h_norm))
//   - h = rms_norm(h) * norm

#include "internal.h"
#include "dflash_graph.h"

#include <cmath>

namespace dflash27b {

DraftGraphOutputs build_draft_graph(
    ggml_context *            ctx,
    const DraftWeights &      w,
    const DraftGraphInputs &  in) {

    const int q_len    = w.hparams.block_size;
    const int ctx_len  = in.ctx_len;
    const int total_k  = ctx_len + q_len;
    const int n_head   = w.hparams.n_head;
    const int n_kv     = w.hparams.n_kv_head;
    const int head_dim = w.hparams.head_dim;
    const float eps    = w.hparams.rms_eps;
    const float rope_base = w.hparams.rope_theta;
    (void)ctx_len;

    const float freq_scale = 1.0f / w.hparams.rope_factor;
    const float ext_factor = (w.hparams.rope_factor > 1.0f) ? 1.0f : 0.0f;
    const float mscale = (w.hparams.rope_factor > 1.0f) ?
        1.0f / (0.1f * std::log(w.hparams.rope_factor) + 1.0f) : 1.0f;

    // ── 1. Feature fusion: target_feat = rms_norm(fc @ target_hidden_cat, hidden_norm)
    //    fc:                [5*hidden, hidden]  (ggml: ne[0]=5*hidden, ne[1]=hidden)
    //    target_hidden_cat: [5*hidden, ctx_len, 1]
    //    Result:            [hidden,   ctx_len, 1]
    ggml_tensor * target_feat = ggml_mul_mat(ctx, w.fc, in.target_hidden_cat);
    target_feat = ggml_rms_norm(ctx, target_feat, eps);
    target_feat = ggml_mul    (ctx, target_feat, w.hidden_norm);
    ggml_set_name(target_feat, "target_feat");

    // ── 2. Decoder layers
    ggml_tensor * h = in.noise_embed;  // [hidden, q_len, 1]

    for (int il = 0; il < w.hparams.n_layer; il++) {
        const DraftLayer & L = w.layers[il];

        // ── 2a. Attention pre-norm
        ggml_tensor * hn = ggml_rms_norm(ctx, h, eps);
        hn = ggml_mul(ctx, hn, L.attn_norm);

        // ── 2b. Q from noise only, then per-head RMSNorm
        //     wq: [hidden, q_dim=4096]
        ggml_tensor * Q = ggml_mul_mat(ctx, L.wq, hn);  // [q_dim, q_len, 1]
        Q = ggml_reshape_3d(ctx, Q, head_dim, n_head, q_len);  // [head_dim, n_head, q_len]
        Q = ggml_rms_norm(ctx, Q, eps);                        // normalize along head_dim
        Q = ggml_mul     (ctx, Q, L.q_norm);                   // broadcast [head_dim]

        // ── 2c. K and V from target_feat AND noise, then concat along sequence
        //     wk, wv: [hidden, kv_dim=1024]
        ggml_tensor * Kctx = ggml_mul_mat(ctx, L.wk, target_feat); // [kv_dim, ctx_len, 1]
        ggml_tensor * Kn   = ggml_mul_mat(ctx, L.wk, hn);          // [kv_dim, q_len,   1]
        ggml_tensor * Vctx = ggml_mul_mat(ctx, L.wv, target_feat);
        ggml_tensor * Vn   = ggml_mul_mat(ctx, L.wv, hn);

        // concat along ne[1] (sequence) — ggml_concat second arg dim=1
        ggml_tensor * K = ggml_concat(ctx, Kctx, Kn, 1);  // [kv_dim, total_k, 1]
        ggml_tensor * V = ggml_concat(ctx, Vctx, Vn, 1);

        // Per-head k_norm
        K = ggml_reshape_3d(ctx, K, head_dim, n_kv, total_k);
        K = ggml_rms_norm(ctx, K, eps);
        K = ggml_mul     (ctx, K, L.k_norm);

        V = ggml_reshape_3d(ctx, V, head_dim, n_kv, total_k);

        // ── 2d. RoPE (NEOX, theta=10M)
        //   Q: positions_q  [q_len]      values [ctx_len..ctx_len+q_len-1]
        //   K: positions_k  [total_k]    values [0..total_k-1]
        Q = ggml_rope_ext(ctx, Q, in.positions_q, nullptr,
                          head_dim, GGML_ROPE_TYPE_NEOX, w.hparams.rope_orig_ctx,
                          rope_base, freq_scale,
                          ext_factor, mscale,
                          w.hparams.rope_beta_fast, w.hparams.rope_beta_slow);
        K = ggml_rope_ext(ctx, K, in.positions_k, nullptr,
                          head_dim, GGML_ROPE_TYPE_NEOX, w.hparams.rope_orig_ctx,
                          rope_base, freq_scale,
                          ext_factor, mscale,
                          w.hparams.rope_beta_fast, w.hparams.rope_beta_slow);

        // ── 2e. Permute into the layout flash_attn_ext wants
        //   q: [n_embd_k=head_dim, n_batch=q_len, n_head,   ne3]
        //   k: [n_embd_k=head_dim, n_kv=total_k, n_head_kv, ne3]
        //   v: [n_embd_v=head_dim, n_kv=total_k, n_head_kv, ne3]  (not transposed)
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);  // [head_dim, q_len,    n_head, 1]
        Q = ggml_cont   (ctx, Q);
        K = ggml_permute(ctx, K, 0, 2, 1, 3);  // [head_dim, total_k,  n_kv,   1]
        K = ggml_cont   (ctx, K);
        V = ggml_permute(ctx, V, 0, 2, 1, 3);  // [head_dim, total_k,  n_kv,   1]
        V = ggml_cont   (ctx, V);

        // ── 2f. Non-causal flash attention; GQA broadcast handled internally.
        const float scale = 1.0f / std::sqrt((float)head_dim);
        ggml_tensor * attn = ggml_flash_attn_ext(ctx, Q, K, V, /*mask=*/nullptr,
                                                 scale, /*max_bias=*/0.0f,
                                                 /*logit_softcap=*/0.0f);
        // attn result: [n_embd_v=head_dim, n_head, n_batch=q_len, 1]
        attn = ggml_reshape_2d(ctx, attn, head_dim * n_head, q_len);
        // attn: [q_dim, q_len]

        // ── 2g. Output projection + residual
        //     wo: [q_dim, hidden]  (ne[0]=q_dim, ne[1]=hidden)
        ggml_tensor * attn_out = ggml_mul_mat(ctx, L.wo, attn);  // [hidden, q_len]
        h = ggml_add(ctx, h, attn_out);

        // ── 2h. FFN pre-norm
        ggml_tensor * hf = ggml_rms_norm(ctx, h, eps);
        hf = ggml_mul(ctx, hf, L.ffn_norm);

        // ── 2i. SwiGLU: down(silu(gate(x)) * up(x))
        //     w_gate, w_up: [hidden, intermediate]
        //     w_down:       [intermediate, hidden]
        ggml_tensor * g  = ggml_mul_mat(ctx, L.w_gate, hf);  // [inter, q_len]
        g = ggml_silu(ctx, g);
        ggml_tensor * u  = ggml_mul_mat(ctx, L.w_up,   hf);  // [inter, q_len]
        ggml_tensor * gu = ggml_mul(ctx, g, u);
        ggml_tensor * ffn_out = ggml_mul_mat(ctx, L.w_down, gu);  // [hidden, q_len]

        h = ggml_add(ctx, h, ffn_out);
    }

    // ── 3. Final norm
    ggml_tensor * out = ggml_rms_norm(ctx, h, eps);
    out = ggml_mul(ctx, out, w.out_norm);
    ggml_set_name(out, "draft_hidden_out");

    DraftGraphOutputs og{};
    og.hidden_states = out;
    og.logits = nullptr;

    // ── 4. Optional: project through target's lm_head to emit vocab logits
    if (in.lm_head) {
        ggml_tensor * logits = ggml_mul_mat(ctx, in.lm_head, out);
        ggml_set_name(logits, "draft_logits");
        og.logits = logits;
    }
    return og;
}

} // namespace dflash27b
