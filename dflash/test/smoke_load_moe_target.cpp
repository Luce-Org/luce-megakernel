// Smoke test for the GGUF target loader with Qwen3.6-35B-A3B MoE model.
// Validates that the loader accepts the "qwen35moe" architecture, reads MoE
// hyperparameters, and wires expert tensors correctly.
//
// Usage: smoke_load_moe_target <path/to/qwen35moe.gguf>

#include "dflash27b.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace dflash27b;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <qwen35moe.gguf>\n", argv[0]);
        return 2;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    TargetWeights w;
    if (!load_target_gguf(argv[1], backend, w)) {
        std::fprintf(stderr, "FAIL: load_target_gguf: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("%s\n", dflash27b_last_error());

    // Validate MoE-specific hyperparameters
    bool ok = true;

    if (w.n_layer != 40) {
        std::fprintf(stderr, "FAIL: n_layer=%d expected 40\n", w.n_layer);
        ok = false;
    }
    if (w.n_embd != 2048) {
        std::fprintf(stderr, "FAIL: n_embd=%d expected 2048\n", w.n_embd);
        ok = false;
    }
    if (w.n_head != 16) {
        std::fprintf(stderr, "FAIL: n_head=%d expected 16\n", w.n_head);
        ok = false;
    }
    if (w.n_head_kv != 2) {
        std::fprintf(stderr, "FAIL: n_head_kv=%d expected 2\n", w.n_head_kv);
        ok = false;
    }
    if (w.n_embd_head_k != 256 || w.n_embd_head_v != 256) {
        std::fprintf(stderr, "FAIL: head_dim k=%d v=%d expected 256/256\n",
            w.n_embd_head_k, w.n_embd_head_v);
        ok = false;
    }
    if (w.full_attention_interval != 4) {
        std::fprintf(stderr, "FAIL: fai=%d expected 4\n", w.full_attention_interval);
        ok = false;
    }
    if (w.ssm_d_inner != 4096) {
        std::fprintf(stderr, "FAIL: ssm_d_inner=%d expected 4096\n", w.ssm_d_inner);
        ok = false;
    }
    if (w.ssm_dt_rank != 32) {
        std::fprintf(stderr, "FAIL: ssm_dt_rank=%d expected 32\n", w.ssm_dt_rank);
        ok = false;
    }
    if (w.ssm_d_state != 128) {
        std::fprintf(stderr, "FAIL: ssm_d_state=%d expected 128\n", w.ssm_d_state);
        ok = false;
    }
    if (w.ssm_n_group != 16) {
        std::fprintf(stderr, "FAIL: ssm_n_group=%d expected 16\n", w.ssm_n_group);
        ok = false;
    }

    // MoE-specific: expert fields
    if (w.n_expert != 256) {
        std::fprintf(stderr, "FAIL: n_expert=%d expected 256\n", w.n_expert);
        ok = false;
    }
    if (w.n_expert_used != 8) {
        std::fprintf(stderr, "FAIL: n_expert_used=%d expected 8\n", w.n_expert_used);
        ok = false;
    }
    if (w.expert_ff_dim != 512) {
        std::fprintf(stderr, "FAIL: expert_ff_dim=%d expected 512\n", w.expert_ff_dim);
        ok = false;
    }
    if (w.shared_ff_dim != 512) {
        std::fprintf(stderr, "FAIL: shared_ff_dim=%d expected 512\n", w.shared_ff_dim);
        ok = false;
    }

    // Count layer types
    int n_attn = 0, n_delta = 0;
    int n_expert_layers = 0;
    for (int il = 0; il < w.n_layer; il++) {
        const auto & L = w.layers[il];
        bool attn = L.wq && L.wk && L.wv && L.wo;
        bool ssm  = L.wqkv && L.wqkv_gate && L.ssm_conv1d;
        if (attn) n_attn++;
        if (ssm)  n_delta++;
        if (L.ffn_gate_inp) n_expert_layers++;
    }

    std::printf("hparams: n_layer=%d n_embd=%d n_head=%d n_head_kv=%d head_dim=%d/%d fai=%d\n",
        w.n_layer, w.n_embd, w.n_head, w.n_head_kv, w.n_embd_head_k, w.n_embd_head_v,
        w.full_attention_interval);
    std::printf("ssm:     conv=%d inner=%d state=%d dt_rank=%d n_group=%d\n",
        w.ssm_d_conv, w.ssm_d_inner, w.ssm_d_state, w.ssm_dt_rank, w.ssm_n_group);
    std::printf("moe:     n_expert=%d n_expert_used=%d expert_ff=%d shared_ff=%d\n",
        w.n_expert, w.n_expert_used, w.expert_ff_dim, w.shared_ff_dim);
    std::printf("layer counts: full_attn=%d delta_net=%d expert_layers=%d\n",
        n_attn, n_delta, n_expert_layers);

    if (n_attn != 10) {
        std::fprintf(stderr, "FAIL: expected 10 full-attn layers, got %d\n", n_attn);
        ok = false;
    }
    if (n_delta != 30) {
        std::fprintf(stderr, "FAIL: expected 30 delta-net layers, got %d\n", n_delta);
        ok = false;
    }
    if (n_expert_layers != 40) {
        std::fprintf(stderr, "FAIL: expected 40 layers with ffn_gate_inp, got %d\n",
            n_expert_layers);
        ok = false;
    }

    // Verify expert tensors exist on layer 0
    {
        const auto & L = w.layers[0];
        if (!L.ffn_gate_inp) {
            std::fprintf(stderr, "FAIL: layer 0 missing ffn_gate_inp\n");
            ok = false;
        }
        if (!L.ffn_up_exps) {
            std::fprintf(stderr, "FAIL: layer 0 missing ffn_up_exps\n");
            ok = false;
        }
        if (!L.ffn_gate_exps) {
            std::fprintf(stderr, "FAIL: layer 0 missing ffn_gate_exps\n");
            ok = false;
        }
        if (!L.ffn_down_exps) {
            std::fprintf(stderr, "FAIL: layer 0 missing ffn_down_exps\n");
            ok = false;
        }
        if (!L.ffn_up_shexp) {
            std::fprintf(stderr, "FAIL: layer 0 missing ffn_up_shexp\n");
            ok = false;
        }
        if (!L.ffn_gate_shexp) {
            std::fprintf(stderr, "FAIL: layer 0 missing ffn_gate_shexp\n");
            ok = false;
        }
        if (!L.ffn_down_shexp) {
            std::fprintf(stderr, "FAIL: layer 0 missing ffn_down_shexp\n");
            ok = false;
        }
        if (!L.ffn_gate_inp_shexp) {
            std::fprintf(stderr, "FAIL: layer 0 missing ffn_gate_inp_shexp\n");
            ok = false;
        }

        if (L.ffn_gate_inp) {
            std::printf("layer 0 ffn_gate_inp: [%lld, %lld] type=%s\n",
                (long long)L.ffn_gate_inp->ne[0], (long long)L.ffn_gate_inp->ne[1],
                ggml_type_name(L.ffn_gate_inp->type));
        }
        if (L.ffn_up_exps) {
            std::printf("layer 0 ffn_up_exps: [%lld, %lld, %lld] type=%s\n",
                (long long)L.ffn_up_exps->ne[0], (long long)L.ffn_up_exps->ne[1],
                (long long)L.ffn_up_exps->ne[2], ggml_type_name(L.ffn_up_exps->type));
        }
    }

    free_target_weights(w);
    ggml_backend_free(backend);

    if (ok) {
        std::printf("OK\n");
        return 0;
    } else {
        return 1;
    }
}
