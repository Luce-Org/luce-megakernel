"""Build a parity-gate reference corpus for the WMMA rewrite.

Runs 50 fixed prompts through prefill + 32 greedy decode steps using the
bf16 megakernel and saves deterministic token IDs to corpus/baseline.json.
"""
import json
import os
import subprocess
import time

import torch
from transformers import AutoTokenizer

from model import (
    Decoder,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    FA_QPROJ_SIZE,
    FA_Q_SIZE,
    FA_KV_SIZE,
    DN_CONV_CHANNELS,
    DN_V_SIZE,
    DN_NUM_HEADS,
    MAX_SEQ_LEN,
    _half_dtype,
)
import qwen35_megakernel_bf16_C

# ---------------------------------------------------------------------------
# Fixed prompt list
# ---------------------------------------------------------------------------
PROMPTS = [
    "The capital of France is",
    "Quantum entanglement allows",
    "def factorial(n):\n    if n <= 1:\n        return 1\n    return",
    "The seven wonders of the ancient world include",
    "In machine learning, a transformer is",
    "The mitochondria is the powerhouse of",
    "Python is a programming language that",
    "The square root of 144 is",
    "Albert Einstein was born in",
    "The Pacific Ocean is the largest",
    "DNA stands for",
    "The speed of light in a vacuum is approximately",
    "Shakespeare's most famous play is",
    "The chemical symbol for gold is",
    "Mount Everest is located in",
    "The Industrial Revolution began in",
    "Photosynthesis is the process by which plants",
    "The Roman Empire fell in",
    "Beethoven was a famous",
    "The Mona Lisa was painted by",
    "Antarctica is covered in",
    "The human heart has",
    "JavaScript was created in",
    "The Great Wall of China was built to",
    "Mars is the fourth planet from",
    "World War II ended in",
    "The pyramids of Giza are located in",
    "Newton's first law of motion states",
    "The currency of Japan is",
    "The Pythagorean theorem states that",
    "The longest river in the world is",
    "Bitcoin was invented by",
    "The Eiffel Tower is in",
    "Mozart composed",
    "The atomic number of carbon is",
    "The continents of the world include",
    "The American Revolution began in",
    "Thomas Edison invented",
    "The deepest ocean trench is the",
    "Charles Darwin proposed",
    "The boiling point of water is",
    "The largest planet in our solar system is",
    "Marie Curie was a pioneer in",
    "The official language of Brazil is",
    "Leonardo da Vinci was a",
    "The Berlin Wall fell in",
    "Carbon dioxide is composed of",
    "The smallest country in the world is",
    "Stephen Hawking was famous for",
    "The currency of the United Kingdom is",
]

N_GEN = 32
S_MAX = 512

# ---------------------------------------------------------------------------
# Decoder + tokenizer setup  (mirrors bench_pp_tg.py lines 26-64)
# ---------------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
dec = Decoder(verbose=True)
_pf = torch.ops.qwen35_megakernel_bf16_C.prefill_bf16

bf16 = dict(dtype=_half_dtype(), device="cuda")
f32  = dict(dtype=torch.float32, device="cuda")
i32  = dict(dtype=torch.int32,   device="cuda")

mx = max(DN_CONV_CHANNELS, FA_QPROJ_SIZE, INTERMEDIATE_SIZE)
bufs = dict(
    hidden           = torch.empty(S_MAX * HIDDEN_SIZE,                        **bf16),
    residual         = torch.empty(S_MAX * HIDDEN_SIZE,                        **bf16),
    normalized       = torch.empty(S_MAX * HIDDEN_SIZE,                        **bf16),
    proj_buf         = torch.empty(S_MAX * mx,                                 **bf16),
    proj_buf2        = torch.empty(S_MAX * mx,                                 **bf16),
    attn_buf         = torch.empty(S_MAX * max(FA_Q_SIZE, FA_KV_SIZE),         **bf16),
    mlp_buf          = torch.empty(S_MAX * INTERMEDIATE_SIZE,                  **bf16),
    dn_out_buf       = torch.empty(S_MAX * DN_V_SIZE,                          **bf16),
    beta_buf         = torch.empty(S_MAX * DN_NUM_HEADS,                       **f32),
    alpha_buf        = torch.empty(S_MAX * DN_NUM_HEADS,                       **f32),
    final_normed     = torch.empty(HIDDEN_SIZE,                                **bf16),
    hidden_bf16_out  = torch.empty(HIDDEN_SIZE,                                **bf16),
    lm_bmv           = torch.empty(1024,                                       **f32),
    lm_bmi           = torch.empty(1024,                                       **i32),
)
bufs.update(dec.alloc_prefill_scratch(S_MAX))


def prefill(ids):
    """Run prefill kernel and return first generated token id."""
    ids_t = torch.tensor(ids, dtype=torch.int32, device="cuda")
    _pf(
        dec._out_token, ids_t,
        dec._embed_weight, dec._layer_weights_packed,
        dec._final_norm_weight, dec._lm_head_weight,
        dec._fa_k_cache, dec._fa_v_cache, dec._dn_states, dec._conv_bufs,
        bufs['hidden'], bufs['residual'], bufs['normalized'],
        bufs['proj_buf'], bufs['proj_buf2'],
        bufs['attn_buf'], bufs['mlp_buf'],
        bufs['dn_out_buf'], bufs['beta_buf'], bufs['alpha_buf'],
        bufs['dn_pre_qkv'],
        bufs['dn_u_scratch'], bufs['dn_w_scratch'], bufs['dn_cs_scratch'],
        dec._fused_fa_qkv, dec._fused_gate_up,
        bufs['final_normed'], bufs['hidden_bf16_out'],
        bufs['lm_bmv'], bufs['lm_bmi'], dec.max_seq_len,
    )
    dec._hidden.copy_(bufs['hidden_bf16_out'])
    dec._position = len(ids)
    return dec._out_token.item()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
git_sha = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=os.path.dirname(os.path.abspath(__file__)),
).decode().strip()

gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Main corpus generation loop
# ---------------------------------------------------------------------------
items = []
t_start = time.perf_counter()

for idx, prompt in enumerate(PROMPTS):
    raw_ids = tok.encode(prompt, add_special_tokens=False)
    ids = raw_ids[:min(len(raw_ids), 480)]   # stay well under S_MAX=512

    dec.reset()
    torch.cuda.synchronize()

    first_tok = prefill(ids)

    generated = [first_tok]
    nid = first_tok
    for _ in range(N_GEN - 1):
        nid = dec.step(nid)
        generated.append(nid)
        if nid == tok.eos_token_id:
            break

    torch.cuda.synchronize()

    items.append({
        "id":                  idx,
        "prompt":              prompt,
        "prompt_token_ids":    ids,
        "generated_token_ids": generated,
    })
    print(f"[{idx+1:02d}/{len(PROMPTS)}] prompt_len={len(ids):3d}  gen_len={len(generated):2d}  first_tok={generated[0]}", flush=True)

elapsed = time.perf_counter() - t_start

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "baseline.json")

corpus = {
    "git_sha":      git_sha,
    "gpu":          gpu_name,
    "torch_version": torch.__version__,
    "n_prompts":    len(items),
    "n_gen":        N_GEN,
    "items":        items,
}

with open(out_path, "w") as fh:
    json.dump(corpus, fh, indent=2)

print(f"\nSaved {len(items)} items → {out_path}  ({elapsed:.1f}s total)", flush=True)
