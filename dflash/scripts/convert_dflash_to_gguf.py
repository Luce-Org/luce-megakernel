#!/usr/bin/env python3
"""
Convert the z-lab DFlash draft (safetensors, bf16) to a GGUF that
llama.cpp can load.

Uses llama.cpp's own gguf-py (deps/llama.cpp/gguf-py) — no hand-rolled
binary writer. The library handles header layout, alignment, BF16
storage, and tensor info offsets correctly.

DFlash draft is a 5-layer Qwen-style transformer with two extra
model-level singletons specific to the spec-decode block-diffusion
algorithm:
  - `fc.weight`           [hidden, 5*hidden]  — fuses 5 captured target
                                                 hidden states into the
                                                 draft's input
  - `hidden_norm.weight`  [hidden]            — RMSNorm applied right after
                                                 the fc projection

These are stored under the `dflash.` prefix so llama.cpp can fetch them
via a custom arch loader without colliding with any upstream tensor
name.

Usage:
  PYTHONPATH=../../dflash27b_ggml/deps/llama.cpp/gguf-py python convert_dflash_to_gguf.py \
    models/draft/model.safetensors \
    qwen3.5-27b-dflash-draft.gguf
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

# Use llama.cpp's own GGUF writer — adds bf16 / metadata / alignment
# correctness without any hand-rolled code.
import gguf

# ──────────────────────────────────────────────────────────────────────
# DFlash draft architecture constants (defaults for 27B; overridden by
# auto-detection from tensor shapes when possible)
# ──────────────────────────────────────────────────────────────────────

ARCH                = "qwen35-dflash-draft"
N_LAYER             = 5
HEAD_DIM            = 128
VOCAB               = 248320
N_TARGET_LAYERS     = 5            # fc projects 5*hidden -> hidden
ROPE_THETA          = 1_000_000.0
RMS_EPS             = 1e-6
MASK_TOKEN_ID       = 248070
BLOCK_SIZE          = 16
CTX_LEN             = 32768


# ──────────────────────────────────────────────────────────────────────
# Tensor name mapping  —  DFlash safetensors -> llama.cpp GGUF
# ──────────────────────────────────────────────────────────────────────

def map_name(name: str) -> str | None:
    if name == "fc.weight":          return "dflash.fc.weight"
    if name == "hidden_norm.weight": return "dflash.hidden_norm.weight"
    if name == "norm.weight":        return "output_norm.weight"
    if name.startswith("layers."):
        parts = name.split(".", 2)
        if len(parts) < 3: return None
        i = int(parts[1])
        rest = parts[2]
        layer_map = {
            "input_layernorm.weight":          f"blk.{i}.attn_norm.weight",
            "post_attention_layernorm.weight": f"blk.{i}.ffn_norm.weight",
            "self_attn.q_proj.weight":         f"blk.{i}.attn_q.weight",
            "self_attn.k_proj.weight":         f"blk.{i}.attn_k.weight",
            "self_attn.v_proj.weight":         f"blk.{i}.attn_v.weight",
            "self_attn.o_proj.weight":         f"blk.{i}.attn_output.weight",
            "self_attn.q_norm.weight":         f"blk.{i}.attn_q_norm.weight",
            "self_attn.k_norm.weight":         f"blk.{i}.attn_k_norm.weight",
            "mlp.gate_proj.weight":            f"blk.{i}.ffn_gate.weight",
            "mlp.up_proj.weight":              f"blk.{i}.ffn_up.weight",
            "mlp.down_proj.weight":            f"blk.{i}.ffn_down.weight",
        }
        return layer_map.get(rest)
    return None


# ──────────────────────────────────────────────────────────────────────
# safetensors reader  —  header parse + raw byte slice
# ──────────────────────────────────────────────────────────────────────

def load_safetensors_header(path: Path):
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode("utf-8")
        return header_size, json.loads(header_json)


def read_tensor_bytes(path: Path, header_size: int, info: dict) -> bytes:
    start, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + header_size + start)
        return f.read(end - start)


def bytes_to_np(raw: bytes, dtype: str, shape: list[int]) -> np.ndarray:
    if dtype == "BF16":
        # Convert BF16 -> F16 on the host. Several ggml-cuda ops (mul,
        # binbcast) only accept F32 / F16 inputs, and llama.cpp's
        # build_norm path multiplies normalised activations by the norm
        # weight tensor. Storing the draft as F16 throughout sidesteps
        # the unsupported BF16 path entirely. Quality impact ~0 for
        # weight tensors (BF16 -> F16 keeps 10/8 mantissa bits anyway
        # after the implicit cast).
        u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
        # bf16 = sign(1) + exp(8) + mantissa(7); reinterpret as f32 by
        # putting it in the high half, then narrow to f16.
        u32 = (u16.astype(np.uint32) << 16)
        f32 = u32.view("<f4").reshape(shape)
        return f32.astype("<f2")
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").reshape(shape)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(shape)
    raise ValueError(f"unsupported safetensors dtype {dtype}")


SAFETENSORS_DTYPE_TO_GGUF = {
    "F32":  gguf.GGMLQuantizationType.F32,
    "F16":  gguf.GGMLQuantizationType.F16,
    # BF16 in safetensors -> we narrow to F16 in bytes_to_np above.
    "BF16": gguf.GGMLQuantizationType.F16,
}


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("safetensors", type=Path)
    ap.add_argument("out_gguf",     type=Path)
    args = ap.parse_args()

    if not args.safetensors.exists():
        print(f"[error] safetensors not found: {args.safetensors}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] reading safetensors header from {args.safetensors}")
    header_size, header = load_safetensors_header(args.safetensors)
    n_entries = sum(1 for k in header if k != "__metadata__")
    print(f"[info]   {n_entries} tensor entries")

    # Auto-detect dimensions from tensor shapes
    fc_shape = header.get("fc.weight", {}).get("shape")
    q_shape = header.get("layers.0.self_attn.q_proj.weight", {}).get("shape")
    k_shape = header.get("layers.0.self_attn.k_proj.weight", {}).get("shape")
    gate_shape = header.get("layers.0.mlp.gate_proj.weight", {}).get("shape")

    if fc_shape:
        # fc.weight: [hidden, n_target_layers * hidden] (PyTorch: [out, in])
        HIDDEN = fc_shape[0]
        detected_n_target = fc_shape[1] // HIDDEN
        print(f"[info] auto-detected hidden={HIDDEN} n_target_layers={detected_n_target}")
    else:
        HIDDEN = 5120  # fallback for 27B
        detected_n_target = N_TARGET_LAYERS
        print("[warn] fc.weight not found, using default HIDDEN=5120")

    if q_shape:
        q_dim = q_shape[0]
        N_HEAD = q_dim // HEAD_DIM
    else:
        N_HEAD = 32

    if k_shape:
        kv_dim = k_shape[0]
        N_HEAD_KV = kv_dim // HEAD_DIM
    else:
        N_HEAD_KV = 8

    if gate_shape:
        INTERMEDIATE = gate_shape[0]
    else:
        INTERMEDIATE = 17408

    # Count actual layers from tensor names
    layer_indices = set()
    for name in header:
        if name.startswith("layers."):
            parts = name.split(".")
            if len(parts) >= 2:
                try:
                    layer_indices.add(int(parts[1]))
                except ValueError:
                    pass
    if layer_indices:
        actual_n_layer = max(layer_indices) + 1
    else:
        actual_n_layer = N_LAYER

    print(f"[info] dimensions: hidden={HIDDEN} n_head={N_HEAD} n_head_kv={N_HEAD_KV} "
          f"ffn={INTERMEDIATE} n_layer={actual_n_layer}")

    writer = gguf.GGUFWriter(args.out_gguf, ARCH)

    # Architecture metadata
    writer.add_string("general.name", "DFlash-Draft")
    writer.add_uint32(f"{ARCH}.context_length",          CTX_LEN)
    writer.add_uint32(f"{ARCH}.embedding_length",        HIDDEN)
    writer.add_uint32(f"{ARCH}.block_count",             actual_n_layer)
    writer.add_uint32(f"{ARCH}.feed_forward_length",     INTERMEDIATE)
    writer.add_uint32(f"{ARCH}.attention.head_count",    N_HEAD)
    writer.add_uint32(f"{ARCH}.attention.head_count_kv", N_HEAD_KV)
    writer.add_uint32(f"{ARCH}.attention.key_length",    HEAD_DIM)
    writer.add_uint32(f"{ARCH}.attention.value_length",  HEAD_DIM)
    writer.add_uint32(f"{ARCH}.vocab_size",              VOCAB)
    writer.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", RMS_EPS)
    writer.add_float32(f"{ARCH}.rope.freq_base",         ROPE_THETA)

    # DFlash-specific hyperparameters
    writer.add_uint32(f"{ARCH}.dflash.n_target_layers", detected_n_target)
    writer.add_uint32(f"{ARCH}.dflash.block_size",      BLOCK_SIZE)
    writer.add_uint32(f"{ARCH}.dflash.mask_token_id",   MASK_TOKEN_ID)

    # Walk + add tensors. Sort: dflash.* singletons first, then output_*,
    # then per-layer in numeric order — keeps the on-disk layout stable.
    pending = []
    for st_name, info in header.items():
        if st_name == "__metadata__":
            continue
        gguf_name = map_name(st_name)
        if gguf_name is None:
            print(f"[warn] skipping unmapped: {st_name}")
            continue
        dtype = SAFETENSORS_DTYPE_TO_GGUF.get(info["dtype"])
        if dtype is None:
            print(f"[error] unsupported dtype {info['dtype']} for {st_name}", file=sys.stderr)
            sys.exit(1)
        pending.append((gguf_name, info["dtype"], info["shape"], info))

    def sort_key(t):
        n = t[0]
        if n.startswith("dflash."):     return (0, n)
        if n.startswith("output_"):     return (1, n)
        if n.startswith("blk."):
            i = int(n.split(".")[1])
            return (2, i, n)
        return (3, n)
    pending.sort(key=sort_key)

    for gguf_name, st_dtype, shape, info in pending:
        raw = read_tensor_bytes(args.safetensors, header_size, info)
        arr = bytes_to_np(raw, st_dtype, shape)
        raw_dtype = SAFETENSORS_DTYPE_TO_GGUF[st_dtype]
        # Norm weights and the dflash hidden_norm singleton must be F32:
        # the ggml-cuda mul path that build_norm emits asserts on
        # src1's element size alignment (binbcast.cu nb10 % sizeof) and
        # the F32 path is the safest cross-quant fallback.
        is_norm = (
            gguf_name.endswith("_norm.weight") or
            gguf_name == "output_norm.weight" or
            gguf_name == "dflash.hidden_norm.weight"
        )
        if is_norm:
            arr = arr.astype("<f4")
            raw_dtype = gguf.GGMLQuantizationType.F32
        writer.add_tensor(gguf_name, arr, raw_dtype=raw_dtype)
        print(f"[tensor] {gguf_name:50s} {st_dtype:4s}->{raw_dtype.name:4s} {tuple(shape)}")

    print(f"[info] writing {args.out_gguf}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[done] wrote {args.out_gguf}")


if __name__ == "__main__":
    main()
