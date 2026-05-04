"""Qwen3.5-0.8B Blackwell NVFP4 decode API.

Companion to model.py. The upstream bf16 decoder stays in model.py, unmodified
from the RTX 3090 reference build; this module is the Blackwell (sm_120 /
sm_121a DGX Spark) NVFP4 path, dispatching through the decode_nvfp4 and
prefill_megakernel_nvfp4 torch ops added alongside kernel_gb10_nvfp4.cu and
prefill_megakernel.cu.

Only importable on Blackwell-class GPUs; only supports backend='nvfp4'.
"""

import os
import struct
import torch

if not torch.cuda.is_available():
    raise RuntimeError("model_nvfp4 requires CUDA")
_cap_major, _cap_minor = torch.cuda.get_device_capability()
if _cap_major < 12:
    raise RuntimeError(
        f"model_nvfp4 requires Blackwell (sm_120/sm_121a); detected "
        f"sm_{_cap_major}{_cap_minor}. Use megakernel.model for sm_86 and earlier."
    )


NUM_LAYERS = 24
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3584
VOCAB_SIZE = 248320
MAX_SEQ_LEN = 2048

FA_NUM_Q_HEADS = 8
FA_NUM_KV_HEADS = 2
FA_HEAD_DIM = 256
FA_Q_SIZE = FA_NUM_Q_HEADS * FA_HEAD_DIM
FA_QPROJ_SIZE = FA_Q_SIZE * 2
FA_KV_SIZE = FA_NUM_KV_HEADS * FA_HEAD_DIM

DN_NUM_HEADS = 16
DN_KEY_DIM = 128
DN_VALUE_DIM = 128
DN_QK_SIZE = DN_NUM_HEADS * DN_KEY_DIM
DN_V_SIZE = DN_NUM_HEADS * DN_VALUE_DIM
DN_CONV_CHANNELS = DN_QK_SIZE * 2 + DN_V_SIZE
DN_BETA_ALPHA_SIZE = DN_NUM_HEADS * 2
DN_CONV_KERNEL = 4
DN_PROJ_FUSED_SIZE = DN_CONV_CHANNELS + DN_V_SIZE + DN_BETA_ALPHA_SIZE
DN_PROJ_FUSED_PADDED_SIZE = ((DN_PROJ_FUSED_SIZE + 127) // 128) * 128
PREFILL_PROJ_FUSED_SIZE = max(DN_PROJ_FUSED_PADDED_SIZE, FA_QPROJ_SIZE + 2 * FA_KV_SIZE, INTERMEDIATE_SIZE * 2)
PREFILL_PROJ_SCRATCH_SIZE = max(DN_CONV_CHANNELS, FA_QPROJ_SIZE, INTERMEDIATE_SIZE)
NVFP4_TC_ROWS_PER_TILE = 128
NVFP4_TC_COLS_PER_TILE = 4
NVFP4_TC_BLOCK_K = 16
NVFP4_TC_K_PER_TILE = NVFP4_TC_COLS_PER_TILE * NVFP4_TC_BLOCK_K

LAYER_TYPE = [0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1]
NVFP4_GROUP_SIZE = 32
NVFP4_LM_GROUP_SIZE = 16
LM_HEAD_TENSORCORE_N = 16
LM_HEAD_TENSORCORE_PACKED_BYTES = LM_HEAD_TENSORCORE_N * (HIDDEN_SIZE // 2)
LM_HEAD_TENSORCORE_SCALE_BYTES = ((LM_HEAD_TENSORCORE_N + 127) // 128) * (HIDDEN_SIZE // 64) * 512

_decode = None
_decode_nvfp4 = None
_decode_many_nvfp4 = None
_prefill_bf16 = None
_prefill_bf16_nvfp4_lm = None
_prefill_megakernel_nvfp4 = None
_quantize_nvfp4_out = None
_quantize_nvfp4_lm_out = None


def _load_op():
    global _decode, _decode_nvfp4, _decode_many_nvfp4
    global _prefill_bf16, _prefill_bf16_nvfp4_lm, _prefill_megakernel_nvfp4
    global _quantize_nvfp4_out, _quantize_nvfp4_lm_out
    if _decode is None:
        import qwen35_megakernel_bf16_C
        ops = torch.ops.qwen35_megakernel_bf16_C
        _decode = ops.decode
        _decode_nvfp4 = ops.decode_nvfp4
        _decode_many_nvfp4 = ops.decode_many_nvfp4
        _prefill_bf16 = ops.prefill_bf16
        try:
            _prefill_bf16_nvfp4_lm = ops.prefill_bf16_nvfp4_lm
        except AttributeError:
            _prefill_bf16_nvfp4_lm = None
        _prefill_megakernel_nvfp4 = ops.prefill_megakernel_nvfp4
        _quantize_nvfp4_out = ops.quantize_nvfp4_out
        _quantize_nvfp4_lm_out = ops.quantize_nvfp4_lm_out


def _resolve_backend(backend):
    if backend in (None, "auto", "nvfp4"):
        return "nvfp4"
    raise ValueError(
        f"model_nvfp4 only supports backend='nvfp4' (got {backend!r}); "
        "use megakernel.model for bf16."
    )


def _resolve_prefill_mode():
    # 'hybrid' (default) = bf16 body + NVFP4 LM head via prefill_bf16_nvfp4_lm
    #   from prefill_bw.cu (Blackwell-only, anonymous-namespaced so it does
    #   not collide with upstream prefill.cu).
    # 'raw' = single-dispatch persistent prefill_megakernel_nvfp4 from
    #   prefill_megakernel.cu.
    mode = os.environ.get("MEGAKERNEL_PREFILL_MODE", "hybrid")
    if mode not in ("hybrid", "raw"):
        raise ValueError(
            f"MEGAKERNEL_PREFILL_MODE={mode!r} is not supported; expected 'hybrid' or 'raw'."
        )
    return mode


def _resolve_prefill_graph():
    value = os.environ.get("MEGAKERNEL_PREFILL_GRAPH", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def _resolve_prefill_tc():
    value = os.environ.get("MEGAKERNEL_PREFILL_TC", "0").strip().lower()
    return value not in ("0", "false", "no", "off")


def _quantize_matrix_nvfp4(weight, group_size):
    _load_op()
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"expected bfloat16 weight, got {weight.dtype}")
    if weight.dim() != 2:
        raise ValueError(f"expected 2D weight, got shape {tuple(weight.shape)}")

    rows, cols = weight.shape
    if cols % 2 != 0 or cols % group_size != 0:
        raise ValueError(f"in_dim {cols} must be divisible by 2 and group_size {group_size}")

    packed = torch.empty((rows, cols // 2), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((rows, cols // group_size), dtype=torch.float16, device=weight.device)
    _quantize_nvfp4_out(packed, scales, weight.contiguous(), group_size)
    return {"packed": packed, "scales": scales}


def _quantize_matrix_nvfp4_lm(weight):
    _load_op()
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"expected bfloat16 weight, got {weight.dtype}")
    if weight.dim() != 2:
        raise ValueError(f"expected 2D weight, got shape {tuple(weight.shape)}")

    rows, cols = weight.shape
    if rows % 128 != 0:
        raise ValueError(f"lm_head out_dim {rows} must be divisible by 128")
    if cols % 64 != 0:
        raise ValueError(f"lm_head in_dim {cols} must be divisible by 64")

    scale_tiles = cols // 64
    packed = torch.empty((rows, cols // 2), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((rows // 128) * scale_tiles * 512, dtype=torch.uint8, device=weight.device)
    _quantize_nvfp4_lm_out(packed, scales, weight.contiguous())
    return {"packed": packed, "scales": scales}


def _quantize_matrix_nvfp4_tc(weight, padded_rows=None):
    _load_op()
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"expected bfloat16 weight, got {weight.dtype}")
    if weight.dim() != 2:
        raise ValueError(f"expected 2D weight, got shape {tuple(weight.shape)}")

    rows, cols = weight.shape
    if cols % NVFP4_TC_K_PER_TILE != 0:
        raise ValueError(f"in_dim {cols} must be divisible by {NVFP4_TC_K_PER_TILE}")

    if padded_rows is None:
        padded_rows = ((rows + NVFP4_TC_ROWS_PER_TILE - 1) // NVFP4_TC_ROWS_PER_TILE) * NVFP4_TC_ROWS_PER_TILE
    if padded_rows < rows:
        raise ValueError(f"padded_rows {padded_rows} must be >= rows {rows}")

    if padded_rows != rows:
        padded = torch.zeros((padded_rows, cols), dtype=weight.dtype, device=weight.device)
        padded[:rows].copy_(weight)
        source = padded
    else:
        source = weight.contiguous()

    packed = torch.empty((padded_rows, cols // 2), dtype=torch.uint8, device=weight.device)
    scale_tiles = cols // NVFP4_TC_K_PER_TILE
    scales = torch.zeros(
        ((padded_rows + NVFP4_TC_ROWS_PER_TILE - 1) // NVFP4_TC_ROWS_PER_TILE) * scale_tiles * 512,
        dtype=torch.uint8,
        device=weight.device,
    )
    _quantize_nvfp4_lm_out(packed, scales, source)
    return {"packed": packed, "scales": scales, "rows": rows, "padded_rows": padded_rows}


def _attach_prefill_fused_weights(weights):
    if "prefill_fused_layer_data" in weights:
        return

    fused_layer_data = []
    for ld in weights["layer_data"]:
        ptrs = ld["ptrs"]
        if ld["type"] == 1:
            proj_weight = torch.cat([ptrs[1], ptrs[2], ptrs[3]], dim=0).contiguous()
            gate_up_weight = torch.cat([ptrs[8], ptrs[9]], dim=0).contiguous()
        else:
            proj_weight = torch.cat([ptrs[1], ptrs[2], ptrs[3], ptrs[4]], dim=0).contiguous()
            gate_up_weight = torch.cat([ptrs[11], ptrs[12]], dim=0).contiguous()
        fused_layer_data.append({
            "proj_weight": proj_weight,
            "gate_up_weight": gate_up_weight,
        })

    weights["prefill_fused_layer_data"] = fused_layer_data


def _attach_prefill_nvfp4_weights(weights, verbose=True):
    _attach_prefill_fused_weights(weights)

    fused_layer_data = weights["prefill_fused_layer_data"]
    if fused_layer_data and "proj_weight_packed" in fused_layer_data[0]:
        return

    if verbose:
        print("Quantizing prompt fused weights to NVFP4 tensor-core format...")

    packed_bytes = 0
    scale_bytes = 0
    for i, ld in enumerate(weights["layer_data"]):
        fused = fused_layer_data[i]
        proj_padded_rows = DN_PROJ_FUSED_PADDED_SIZE if ld["type"] == 0 else fused["proj_weight"].shape[0]
        proj_q = _quantize_matrix_nvfp4_tc(fused["proj_weight"], padded_rows=proj_padded_rows)
        gate_up_q = _quantize_matrix_nvfp4_tc(fused["gate_up_weight"])
        fused["proj_weight_packed"] = proj_q["packed"]
        fused["proj_weight_scales"] = proj_q["scales"]
        fused["gate_up_weight_packed"] = gate_up_q["packed"]
        fused["gate_up_weight_scales"] = gate_up_q["scales"]
        packed_bytes += proj_q["packed"].numel() + gate_up_q["packed"].numel()
        scale_bytes += proj_q["scales"].numel() + gate_up_q["scales"].numel()

    if verbose:
        print(
            f"NVFP4 prompt fused weights: {packed_bytes/1e6:.0f} MB packed + "
            f"{scale_bytes/1e6:.0f} MB scales ({(packed_bytes + scale_bytes)/1e6:.0f} MB total)"
        )


def _attach_nvfp4_weights(weights, group_size=NVFP4_GROUP_SIZE, verbose=True):
    if (
        "nvfp4" in weights
        and weights["nvfp4"]["group_size"] == group_size
        and weights["nvfp4"].get("lm_group_size") == NVFP4_LM_GROUP_SIZE
    ):
        return weights

    if verbose:
        print(f"Quantizing decode hot weights to NVFP4 (group_size={group_size})...")

    layer_data_nvfp4 = []
    packed_bytes = 0
    scale_bytes = 0

    for ld in weights["layer_data"]:
        if ld["type"] == 1:
            q_proj = _quantize_matrix_nvfp4(ld["ptrs"][1], group_size)
            k_proj = _quantize_matrix_nvfp4(ld["ptrs"][2], group_size)
            v_proj = _quantize_matrix_nvfp4(ld["ptrs"][3], group_size)
            o_proj = _quantize_matrix_nvfp4(ld["ptrs"][6], group_size)
            gate_proj = _quantize_matrix_nvfp4(ld["ptrs"][8], group_size)
            up_proj = _quantize_matrix_nvfp4(ld["ptrs"][9], group_size)
            down_proj = _quantize_matrix_nvfp4(ld["ptrs"][10], group_size)
            qptrs = [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
            layer_data_nvfp4.append({
                "type": 1,
                "ptrs": [
                    ld["ptrs"][0],
                    q_proj["packed"], q_proj["scales"],
                    k_proj["packed"], k_proj["scales"],
                    v_proj["packed"], v_proj["scales"],
                    ld["ptrs"][4], ld["ptrs"][5],
                    o_proj["packed"], o_proj["scales"],
                    ld["ptrs"][7],
                    gate_proj["packed"], gate_proj["scales"],
                    up_proj["packed"], up_proj["scales"],
                    down_proj["packed"], down_proj["scales"],
                ],
                "quantized": qptrs,
            })
        else:
            qkv_proj = _quantize_matrix_nvfp4(ld["ptrs"][1], group_size)
            z_proj = _quantize_matrix_nvfp4(ld["ptrs"][2], group_size)
            out_proj = _quantize_matrix_nvfp4(ld["ptrs"][9], group_size)
            gate_proj = _quantize_matrix_nvfp4(ld["ptrs"][11], group_size)
            up_proj = _quantize_matrix_nvfp4(ld["ptrs"][12], group_size)
            down_proj = _quantize_matrix_nvfp4(ld["ptrs"][13], group_size)
            qptrs = [qkv_proj, z_proj, out_proj, gate_proj, up_proj, down_proj]
            layer_data_nvfp4.append({
                "type": 0,
                "ptrs": [
                    ld["ptrs"][0],
                    qkv_proj["packed"], qkv_proj["scales"],
                    z_proj["packed"], z_proj["scales"],
                    ld["ptrs"][3], ld["ptrs"][4], ld["ptrs"][5], ld["ptrs"][6], ld["ptrs"][7], ld["ptrs"][8],
                    out_proj["packed"], out_proj["scales"],
                    ld["ptrs"][10],
                    gate_proj["packed"], gate_proj["scales"],
                    up_proj["packed"], up_proj["scales"],
                    down_proj["packed"], down_proj["scales"],
                ],
                "quantized": qptrs,
            })

        for q in qptrs:
            packed_bytes += q["packed"].numel() * q["packed"].element_size()
            scale_bytes += q["scales"].numel() * q["scales"].element_size()

    lm_head_nvfp4 = _quantize_matrix_nvfp4_lm(weights["lm_head_weight"])
    packed_bytes += lm_head_nvfp4["packed"].numel() * lm_head_nvfp4["packed"].element_size()
    scale_bytes += lm_head_nvfp4["scales"].numel() * lm_head_nvfp4["scales"].element_size()

    weights["nvfp4"] = {
        "group_size": group_size,
        "lm_group_size": NVFP4_LM_GROUP_SIZE,
        "layer_data": layer_data_nvfp4,
        "lm_head_weight_packed": lm_head_nvfp4["packed"],
        "lm_head_scales": lm_head_nvfp4["scales"],
    }

    if verbose:
        total_mb = (packed_bytes + scale_bytes) / 1e6
        print(
            f"NVFP4 decode weights: {packed_bytes/1e6:.0f} MB packed + "
            f"{scale_bytes/1e6:.0f} MB scales ({total_mb:.0f} MB total)"
        )

    return weights


def load_weights(
    model_name="Qwen/Qwen3.5-0.8B",
    verbose=True,
    backend="bf16",
    nvfp4_group_size=NVFP4_GROUP_SIZE,
):
    """Load Qwen3.5-0.8B weights and optional GB10 NVFP4 decode weights."""
    if not verbose:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_backend = _resolve_backend(backend)

    if verbose:
        print(f"Loading {model_name} (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    state = model.state_dict()

    layer_data = []
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}."
        lt = LAYER_TYPE[i]

        if lt == 1:
            # Full Attention: 11 pointers (all bf16)
            layer_data.append({
                "type": 1,
                "ptrs": [
                    state[p + "input_layernorm.weight"].contiguous(),
                    state[p + "self_attn.q_proj.weight"].contiguous(),
                    state[p + "self_attn.k_proj.weight"].contiguous(),
                    state[p + "self_attn.v_proj.weight"].contiguous(),
                    state[p + "self_attn.q_norm.weight"].contiguous(),
                    state[p + "self_attn.k_norm.weight"].contiguous(),
                    state[p + "self_attn.o_proj.weight"].contiguous(),
                    state[p + "post_attention_layernorm.weight"].contiguous(),
                    state[p + "mlp.gate_proj.weight"].contiguous(),
                    state[p + "mlp.up_proj.weight"].contiguous(),
                    state[p + "mlp.down_proj.weight"].contiguous(),
                ]
            })
        else:
            # DeltaNet: 14 pointers (all bf16)
            layer_data.append({
                "type": 0,
                "ptrs": [
                    state[p + "input_layernorm.weight"].contiguous(),
                    state[p + "linear_attn.in_proj_qkv.weight"].contiguous(),
                    state[p + "linear_attn.in_proj_z.weight"].contiguous(),
                    state[p + "linear_attn.in_proj_b.weight"].contiguous(),
                    state[p + "linear_attn.in_proj_a.weight"].contiguous(),
                    state[p + "linear_attn.conv1d.weight"].contiguous(),
                    state[p + "linear_attn.A_log"].contiguous(),
                    state[p + "linear_attn.dt_bias"].contiguous(),
                    state[p + "linear_attn.norm.weight"].contiguous(),
                    state[p + "linear_attn.out_proj.weight"].contiguous(),
                    state[p + "post_attention_layernorm.weight"].contiguous(),
                    state[p + "mlp.gate_proj.weight"].contiguous(),
                    state[p + "mlp.up_proj.weight"].contiguous(),
                    state[p + "mlp.down_proj.weight"].contiguous(),
                ]
            })

    embed_weight = state["model.embed_tokens.weight"].contiguous()
    final_norm_weight = state["model.norm.weight"].contiguous()
    lm_head = state.get("lm_head.weight", embed_weight).contiguous()

    weights = {
        "embed_weight": embed_weight,
        "final_norm_weight": final_norm_weight,
        "lm_head_weight": lm_head,
        "layer_data": layer_data,
    }

    del model
    torch.cuda.empty_cache()

    if verbose:
        total = sum(sum(t.numel() for t in ld["ptrs"]) for ld in layer_data) + lm_head.numel()
        print(f"BF16 weights: {total/1e6:.1f}M params ({total*2/1e6:.0f} MB)")

    _attach_prefill_fused_weights(weights)

    if resolved_backend == "nvfp4":
        _attach_nvfp4_weights(weights, group_size=nvfp4_group_size, verbose=verbose)
        if _resolve_prefill_tc():
            _attach_prefill_nvfp4_weights(weights, verbose=verbose)

    return weights, tokenizer


def _pack_layer_weights(layer_data):
    """Pack layer weights into device blob matching LayerWeights struct."""
    ptr_size = 8
    max_ptrs = 14
    header_size = 16
    struct_size = header_size + max_ptrs * ptr_size  # 128

    buf = bytearray(NUM_LAYERS * struct_size)
    for i in range(NUM_LAYERS):
        ld = layer_data[i]
        offset = i * struct_size
        struct.pack_into("iiii", buf, offset, ld["type"], 0, 0, 0)
        for j, tensor in enumerate(ld["ptrs"]):
            struct.pack_into("Q", buf, offset + header_size + j * ptr_size, tensor.data_ptr())
        for j in range(len(ld["ptrs"]), max_ptrs):
            struct.pack_into("Q", buf, offset + header_size + j * ptr_size, 0)

    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


def _pack_layer_weights_nvfp4(layer_data, group_size):
    """Pack layer weights into device blob matching LayerWeightsNVFP4 struct."""
    ptr_size = 8
    max_ptrs = 24
    header_size = 16
    struct_size = header_size + max_ptrs * ptr_size

    buf = bytearray(NUM_LAYERS * struct_size)
    for i in range(NUM_LAYERS):
        ld = layer_data[i]
        offset = i * struct_size
        struct.pack_into("iiii", buf, offset, ld["type"], group_size, 0, 0)
        for j, tensor in enumerate(ld["ptrs"]):
            struct.pack_into("Q", buf, offset + header_size + j * ptr_size, tensor.data_ptr())
        for j in range(len(ld["ptrs"]), max_ptrs):
            struct.pack_into("Q", buf, offset + header_size + j * ptr_size, 0)

    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


def _pack_prefill_fused_layer_weights(fused_layer_data):
    ptr_size = 8
    fields = 6
    struct_size = ptr_size * fields

    buf = bytearray(NUM_LAYERS * struct_size)
    for i in range(NUM_LAYERS):
        ld = fused_layer_data[i]
        offset = i * struct_size
        proj_weight_packed = ld.get("proj_weight_packed")
        proj_weight_scales = ld.get("proj_weight_scales")
        gate_up_weight_packed = ld.get("gate_up_weight_packed")
        gate_up_weight_scales = ld.get("gate_up_weight_scales")
        struct.pack_into("Q", buf, offset, ld["proj_weight"].data_ptr())
        struct.pack_into("Q", buf, offset + ptr_size, ld["gate_up_weight"].data_ptr())
        struct.pack_into("Q", buf, offset + 2 * ptr_size, proj_weight_packed.data_ptr() if proj_weight_packed is not None else 0)
        struct.pack_into("Q", buf, offset + 3 * ptr_size, proj_weight_scales.data_ptr() if proj_weight_scales is not None else 0)
        struct.pack_into("Q", buf, offset + 4 * ptr_size, gate_up_weight_packed.data_ptr() if gate_up_weight_packed is not None else 0)
        struct.pack_into("Q", buf, offset + 5 * ptr_size, gate_up_weight_scales.data_ptr() if gate_up_weight_scales is not None else 0)

    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


class Decoder:
    """Stateful decoder for Qwen3.5-0.8B megakernel backends."""

    def __init__(
        self,
        weights=None,
        tokenizer=None,
        model_name="Qwen/Qwen3.5-0.8B",
        backend="auto",
        nvfp4_group_size=NVFP4_GROUP_SIZE,
        verbose=True,
    ):
        _load_op()
        self.backend = _resolve_backend(backend)
        self.backend_label = "NVFP4 decode" if self.backend == "nvfp4" else "BF16"
        self._nvfp4_group_size = nvfp4_group_size
        self._prefill_mode = _resolve_prefill_mode()
        self._prefill_graph_enabled = (
            self.backend == "nvfp4"
            and self._prefill_mode == "hybrid"
            and _resolve_prefill_graph()
        )

        if weights is None:
            weights, tokenizer = load_weights(
                model_name,
                verbose=verbose,
                backend=self.backend,
                nvfp4_group_size=nvfp4_group_size,
            )
        elif self.backend == "nvfp4":
            _attach_nvfp4_weights(weights, group_size=nvfp4_group_size, verbose=verbose)
            if _resolve_prefill_tc():
                _attach_prefill_nvfp4_weights(weights, verbose=verbose)
        _attach_prefill_fused_weights(weights)
        self.tokenizer = tokenizer
        self._position = 0
        self._weights = weights
        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_data"])
        self._prefill_fused_weights_packed = _pack_prefill_fused_layer_weights(
            weights["prefill_fused_layer_data"]
        )
        self._layer_weights_packed_nvfp4 = None
        self._lm_head_weight_packed = None
        self._lm_head_scales = None
        if self.backend == "nvfp4":
            nvfp4 = weights["nvfp4"]
            self._layer_weights_packed_nvfp4 = _pack_layer_weights_nvfp4(
                nvfp4["layer_data"], nvfp4["group_size"])
            self._lm_head_weight_packed = nvfp4["lm_head_weight_packed"]
            self._lm_head_scales = nvfp4["lm_head_scales"]

        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        f16 = dict(dtype=torch.float16, device="cuda")
        f32 = dict(dtype=torch.float32, device="cuda")
        i32 = dict(dtype=torch.int32, device="cuda")
        u32 = dict(dtype=torch.uint32, device="cuda")

        n_fa = sum(1 for t in LAYER_TYPE if t == 1)
        self._fa_k_cache = torch.zeros(n_fa, FA_NUM_KV_HEADS, MAX_SEQ_LEN, FA_HEAD_DIM, **bf16)
        self._fa_v_cache = torch.zeros_like(self._fa_k_cache)

        n_dn = sum(1 for t in LAYER_TYPE if t == 0)
        self._dn_states = torch.zeros(n_dn, DN_NUM_HEADS, DN_KEY_DIM, DN_VALUE_DIM, **f32)
        self._conv_bufs = torch.zeros(n_dn, DN_CONV_CHANNELS, DN_CONV_KERNEL, **f32)

        self._hidden = torch.empty(HIDDEN_SIZE, **bf16)
        max_scratch = max(FA_QPROJ_SIZE, DN_CONV_CHANNELS, HIDDEN_SIZE * 8 + INTERMEDIATE_SIZE)
        self._activations = torch.empty(max_scratch, **f32)
        self._residual = torch.empty(HIDDEN_SIZE, **bf16)
        self._qkv_scratch = torch.empty(max(FA_QPROJ_SIZE, DN_CONV_CHANNELS), **f32)
        self._kv_scratch = torch.empty(FA_KV_SIZE * 2, **f32)
        self._attn_out = torch.empty(max(FA_Q_SIZE, DN_V_SIZE), **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._z_scratch = torch.empty(DN_V_SIZE, **f32)
        self._beta_scratch = torch.empty(DN_NUM_HEADS, **f32)
        self._alpha_scratch = torch.empty(DN_NUM_HEADS, **f32)
        self._normalized = torch.empty(HIDDEN_SIZE, **f32)
        self._lm_hidden_bf16 = torch.empty((LM_HEAD_TENSORCORE_N, HIDDEN_SIZE), **bf16)
        self._lm_hidden_packed = torch.empty(LM_HEAD_TENSORCORE_PACKED_BYTES, dtype=torch.uint8, device="cuda")
        self._lm_hidden_scales = torch.empty(LM_HEAD_TENSORCORE_SCALE_BYTES, dtype=torch.uint8, device="cuda")
        self._lm_logits_f16 = torch.empty((LM_HEAD_TENSORCORE_N, VOCAB_SIZE), **f16)

        self._barrier_counter = torch.zeros(1, **u32)
        self._barrier_generation = torch.zeros(1, **u32)
        self._block_max_vals = torch.empty(1024, **f32)
        self._block_max_idxs = torch.empty(1024, **i32)
        self._lm_sync_counter = torch.zeros(1, **u32)
        self._out_token = torch.empty(1, **i32)
        self._prefill_buffers = None
        self._prefill_buffer_tokens = 0
        self._prefill_graph_cache = {}

    def _ensure_prefill_buffers(self, max_tokens: int):
        if self._prefill_buffers is not None and self._prefill_buffer_tokens >= max_tokens:
            return self._prefill_buffers

        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        f32 = dict(dtype=torch.float32, device="cuda")
        i32 = dict(dtype=torch.int32, device="cuda")
        self._prefill_buffers = dict(
            hidden=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
            residual=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
            normalized=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
            proj_buf=torch.empty(max_tokens * PREFILL_PROJ_FUSED_SIZE, **bf16),
            proj_buf2=torch.empty(max_tokens * PREFILL_PROJ_SCRATCH_SIZE, **bf16),
            proj_buf_half=torch.empty(max_tokens * PREFILL_PROJ_FUSED_SIZE, dtype=torch.float16, device="cuda"),
            proj_act_packed=torch.empty(max_tokens * (HIDDEN_SIZE // 2), dtype=torch.uint8, device="cuda"),
            proj_act_scales=torch.empty(
                ((max_tokens + NVFP4_TC_ROWS_PER_TILE - 1) // NVFP4_TC_ROWS_PER_TILE)
                * (HIDDEN_SIZE // NVFP4_TC_K_PER_TILE)
                * 512,
                dtype=torch.uint8,
                device="cuda",
            ),
            attn_buf=torch.empty(max_tokens * max(FA_Q_SIZE, FA_KV_SIZE), **bf16),
            mlp_buf=torch.empty(max_tokens * INTERMEDIATE_SIZE, **bf16),
            dn_out_buf=torch.empty(max_tokens * DN_V_SIZE, **bf16),
            beta_buf=torch.empty(max_tokens * DN_NUM_HEADS, **f32),
            alpha_buf=torch.empty(max_tokens * DN_NUM_HEADS, **f32),
            final_normed=torch.empty(HIDDEN_SIZE, **bf16),
            hidden_bf16_out=torch.empty(HIDDEN_SIZE, **bf16),
            lm_bmv=torch.empty(1024, **f32),
            lm_bmi=torch.empty(1024, **i32),
        )
        self._prefill_buffer_tokens = max_tokens
        return self._prefill_buffers

    def _reset_runtime_state(self):
        self._position = 0
        self._fa_k_cache.zero_()
        self._fa_v_cache.zero_()
        self._dn_states.zero_()
        self._conv_bufs.zero_()

    def _run_prefill_bf16_nvfp4_lm(self, token_ids: torch.Tensor, buffers):
        _prefill_bf16_nvfp4_lm(
            self._out_token,
            token_ids,
            self._embed_weight,
            self._layer_weights_packed,
            self._prefill_fused_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._lm_head_weight_packed,
            self._lm_head_scales,
            self._fa_k_cache,
            self._fa_v_cache,
            self._dn_states,
            self._conv_bufs,
            buffers["hidden"],
            buffers["residual"],
            buffers["normalized"],
            buffers["proj_buf"],
            buffers["proj_buf2"],
            buffers["proj_buf_half"],
            buffers["proj_act_packed"],
            buffers["proj_act_scales"],
            buffers["attn_buf"],
            buffers["mlp_buf"],
            buffers["dn_out_buf"],
            buffers["beta_buf"],
            buffers["alpha_buf"],
            buffers["final_normed"],
            buffers["hidden_bf16_out"],
            buffers["lm_bmv"],
            buffers["lm_bmi"],
            self._lm_hidden_bf16,
            self._lm_hidden_packed,
            self._lm_hidden_scales,
            self._lm_logits_f16,
        )
        self._hidden.copy_(buffers["hidden_bf16_out"])

    def _build_prefill_graph(self, prompt_len: int):
        buffers = self._ensure_prefill_buffers(prompt_len)
        static_ids = torch.empty(prompt_len, dtype=torch.int32, device="cuda")
        warmup_stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()

        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            self._reset_runtime_state()
            static_ids.zero_()
            self._run_prefill_bf16_nvfp4_lm(static_ids, buffers)
        warmup_stream.synchronize()
        torch.cuda.current_stream().wait_stream(warmup_stream)

        with torch.cuda.graph(graph):
            self._reset_runtime_state()
            self._run_prefill_bf16_nvfp4_lm(static_ids, buffers)

        state = {"graph": graph, "token_ids": static_ids, "buffers": buffers}
        self._prefill_graph_cache[prompt_len] = state
        return state

    def _prefill_graph_state(self, prompt_len: int):
        state = self._prefill_graph_cache.get(prompt_len)
        if state is None:
            state = self._build_prefill_graph(prompt_len)
        return state

    def step(self, token_id: int) -> int:
        """Decode one token. Returns next token id."""
        if self.backend == "nvfp4":
            _decode_nvfp4(
                self._out_token, token_id,
                self._embed_weight, self._layer_weights_packed_nvfp4,
                self._final_norm_weight, self._lm_head_weight_packed, self._lm_head_scales,
                self._lm_hidden_bf16, self._lm_hidden_packed, self._lm_hidden_scales, self._lm_logits_f16,
                self._fa_k_cache, self._fa_v_cache,
                self._dn_states, self._conv_bufs,
                self._hidden, self._activations, self._residual,
                self._qkv_scratch, self._kv_scratch, self._attn_out,
                self._mlp_inter, self._z_scratch, self._beta_scratch,
                self._alpha_scratch, self._normalized,
                self._barrier_counter, self._barrier_generation,
                self._block_max_vals, self._block_max_idxs,
                self._lm_sync_counter,
                self._position, MAX_SEQ_LEN, self._nvfp4_group_size,
            )
        else:
            _decode(
                self._out_token, token_id,
                self._embed_weight, self._layer_weights_packed,
                self._final_norm_weight, self._lm_head_weight,
                self._fa_k_cache, self._fa_v_cache,
                self._dn_states, self._conv_bufs,
                self._hidden, self._activations, self._residual,
                self._qkv_scratch, self._kv_scratch, self._attn_out,
                self._mlp_inter, self._z_scratch, self._beta_scratch,
                self._alpha_scratch, self._normalized,
                self._barrier_counter, self._barrier_generation,
                self._block_max_vals, self._block_max_idxs,
                self._lm_sync_counter,
                self._position, MAX_SEQ_LEN,
            )
        self._position += 1
        return self._out_token.item()

    def step_many(self, token_id: int, num_steps: int) -> torch.Tensor:
        """Decode multiple NVFP4 steps without per-token host/device synchronization."""
        if self.backend != "nvfp4":
            raise RuntimeError("step_many is only available for the NVFP4 backend")
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if num_steps == 0:
            return torch.empty(0, dtype=torch.int32, device="cuda")

        output_tokens = torch.empty(num_steps, dtype=torch.int32, device="cuda")
        _decode_many_nvfp4(
            output_tokens, self._out_token, token_id,
            self._embed_weight, self._layer_weights_packed_nvfp4,
            self._final_norm_weight, self._lm_head_weight_packed, self._lm_head_scales,
            self._lm_hidden_bf16, self._lm_hidden_packed, self._lm_hidden_scales, self._lm_logits_f16,
            self._fa_k_cache, self._fa_v_cache,
            self._dn_states, self._conv_bufs,
            self._hidden, self._activations, self._residual,
            self._qkv_scratch, self._kv_scratch, self._attn_out,
            self._mlp_inter, self._z_scratch, self._beta_scratch,
            self._alpha_scratch, self._normalized,
            self._barrier_counter, self._barrier_generation,
            self._block_max_vals, self._block_max_idxs,
            self._lm_sync_counter,
            self._position, MAX_SEQ_LEN, self._nvfp4_group_size,
        )
        self._position += num_steps
        return output_tokens

    def prefill_tokens(self, token_ids: torch.Tensor) -> int:
        """Run prompt prefill and return the first generated token id."""
        if token_ids.device.type != "cuda" or token_ids.dtype != torch.int32 or token_ids.dim() != 1:
            raise TypeError("token_ids must be a 1D CUDA int32 tensor")
        prompt_len = int(token_ids.numel())
        if self.backend == "nvfp4" and self._prefill_mode == "raw":
            self.reset()
            _prefill_megakernel_nvfp4(
                self._out_token,
                token_ids,
                self._embed_weight,
                self._layer_weights_packed_nvfp4,
                self._final_norm_weight,
                self._lm_head_weight_packed,
                self._lm_head_scales,
                self._lm_hidden_bf16,
                self._lm_hidden_packed,
                self._lm_hidden_scales,
                self._lm_logits_f16,
                self._fa_k_cache,
                self._fa_v_cache,
                self._dn_states,
                self._conv_bufs,
                self._hidden,
                self._activations,
                self._residual,
                self._qkv_scratch,
                self._kv_scratch,
                self._attn_out,
                self._mlp_inter,
                self._z_scratch,
                self._beta_scratch,
                self._alpha_scratch,
                self._normalized,
                self._barrier_counter,
                self._barrier_generation,
                self._block_max_vals,
                self._block_max_idxs,
                self._lm_sync_counter,
                MAX_SEQ_LEN,
                self._nvfp4_group_size,
            )
        elif self.backend == "nvfp4":
            if self._prefill_graph_enabled:
                state = self._prefill_graph_state(prompt_len)
                state["token_ids"].copy_(token_ids)
                state["graph"].replay()
            else:
                self.reset()
                buffers = self._ensure_prefill_buffers(prompt_len)
                self._run_prefill_bf16_nvfp4_lm(token_ids, buffers)
        else:
            raise RuntimeError("prefill_tokens megakernel path is only implemented for the NVFP4 backend")
        self._position = prompt_len
        return self._out_token.item()

    def reset(self):
        self._reset_runtime_state()

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        self.reset()
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        for tid in ids[:-1]:
            self.step(tid)
        out = []
        next_id = ids[-1]
        eos = self.tokenizer.eos_token_id
        for _ in range(max_tokens):
            next_id = self.step(next_id)
            if next_id == eos:
                break
            out.append(next_id)
        return self.tokenizer.decode(out, skip_special_tokens=True)
