"""Weight loading and decode API for Qwen3.5-0.8B bf16 megakernel."""

import struct
import torch

NUM_LAYERS = 24
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3584
VOCAB_SIZE = 248320
MAX_SEQ_LEN = 2048


def _half_dtype():
    """Return the 16-bit dtype matching the compiled kernel: bf16 on SM>=80, fp16 otherwise."""
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            return torch.float16
    return torch.bfloat16

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
DN_CONV_KERNEL = 4

LAYER_TYPE = [0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1]

_decode = None
_prefill_bf16 = None
_prefill_bf16_qtail = None
_max_safe_decode_blocks = None
_set_decode_blocks = None


def _load_op():
    global _decode, _prefill_bf16, _prefill_bf16_qtail, _max_safe_decode_blocks, _set_decode_blocks
    if _decode is None:
        import qwen35_megakernel_bf16_C
        _decode = torch.ops.qwen35_megakernel_bf16_C.decode
        _prefill_bf16 = torch.ops.qwen35_megakernel_bf16_C.prefill_bf16
        _prefill_bf16_qtail = torch.ops.qwen35_megakernel_bf16_C.prefill_bf16_qtail
        _max_safe_decode_blocks = torch.ops.qwen35_megakernel_bf16_C.max_safe_decode_blocks
        _set_decode_blocks = torch.ops.qwen35_megakernel_bf16_C.set_decode_blocks


def max_safe_decode_blocks() -> int:
    """Return the resident-block ceiling for the current CUDA device."""
    _load_op()
    return int(_max_safe_decode_blocks())


def set_decode_blocks(blocks: int):
    """Override decode blocks, clamped by the CUDA resident-block ceiling."""
    if blocks < 0:
        raise ValueError("blocks must be non-negative")
    _load_op()
    _set_decode_blocks(int(blocks))


def load_weights(model_name="Qwen/Qwen3.5-0.8B", verbose=True):
    """Load Qwen3.5-0.8B weights (bf16 on SM>=80, fp16 on SM<80)."""
    if not verbose:
        import os
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    hdtype = _half_dtype()
    dtype_name = "bf16" if hdtype == torch.bfloat16 else "fp16"

    if verbose:
        print(f"Loading {model_name} ({dtype_name})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=hdtype, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    state = model.state_dict()

    def _w(key):
        """Get weight tensor, ensuring it's in the target half dtype."""
        return state[key].to(hdtype).contiguous()

    layer_data = []
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}."
        lt = LAYER_TYPE[i]

        if lt == 1:
            # Full Attention: 11 pointers
            layer_data.append({
                "type": 1,
                "ptrs": [
                    _w(p + "input_layernorm.weight"),
                    _w(p + "self_attn.q_proj.weight"),
                    _w(p + "self_attn.k_proj.weight"),
                    _w(p + "self_attn.v_proj.weight"),
                    _w(p + "self_attn.q_norm.weight"),
                    _w(p + "self_attn.k_norm.weight"),
                    _w(p + "self_attn.o_proj.weight"),
                    _w(p + "post_attention_layernorm.weight"),
                    _w(p + "mlp.gate_proj.weight"),
                    _w(p + "mlp.up_proj.weight"),
                    _w(p + "mlp.down_proj.weight"),
                ]
            })
        else:
            # DeltaNet: 14 pointers
            layer_data.append({
                "type": 0,
                "ptrs": [
                    _w(p + "input_layernorm.weight"),
                    _w(p + "linear_attn.in_proj_qkv.weight"),
                    _w(p + "linear_attn.in_proj_z.weight"),
                    _w(p + "linear_attn.in_proj_b.weight"),
                    _w(p + "linear_attn.in_proj_a.weight"),
                    _w(p + "linear_attn.conv1d.weight"),
                    _w(p + "linear_attn.A_log"),
                    _w(p + "linear_attn.dt_bias"),
                    _w(p + "linear_attn.norm.weight"),
                    _w(p + "linear_attn.out_proj.weight"),
                    _w(p + "post_attention_layernorm.weight"),
                    _w(p + "mlp.gate_proj.weight"),
                    _w(p + "mlp.up_proj.weight"),
                    _w(p + "mlp.down_proj.weight"),
                ]
            })

    embed_weight = _w("model.embed_tokens.weight")
    final_norm_weight = _w("model.norm.weight")
    lm_head = state.get("lm_head.weight")
    if lm_head is None:
        lm_head = embed_weight
    else:
        lm_head = lm_head.to(hdtype).contiguous()

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
        print(f"{dtype_name.upper()} weights: {total/1e6:.1f}M params ({total*2/1e6:.0f} MB)")

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


class Decoder:
    """Stateful decoder for Qwen3.5-0.8B bf16 megakernel."""

    def __init__(self, weights=None, tokenizer=None,
                 model_name="Qwen/Qwen3.5-0.8B", verbose=True,
                 max_seq_len=MAX_SEQ_LEN, repetition_penalty=1.0,
                 decode_blocks=None):
        _load_op()
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")
        if decode_blocks is not None and decode_blocks < 0:
            raise ValueError("decode_blocks must be non-negative")

        if weights is None:
            weights, tokenizer = load_weights(model_name, verbose=verbose)
        self.tokenizer = tokenizer
        self._position = 0
        self.max_seq_len = int(max_seq_len)
        self.repetition_penalty = float(repetition_penalty)
        self._weights = weights
        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_data"])
        _set_decode_blocks(0 if decode_blocks is None else int(decode_blocks))

        bf16 = dict(dtype=_half_dtype(), device="cuda")
        f32 = dict(dtype=torch.float32, device="cuda")
        i32 = dict(dtype=torch.int32, device="cuda")
        u32 = dict(dtype=torch.uint32, device="cuda")

        n_fa = sum(1 for t in LAYER_TYPE if t == 1)
        self._fa_k_cache = torch.zeros(n_fa, FA_NUM_KV_HEADS, self.max_seq_len, FA_HEAD_DIM, **bf16)
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

        self._barrier_counter = torch.zeros(1, **u32)
        self._barrier_generation = torch.zeros(1, **u32)
        self._block_max_vals = torch.empty(1024, **f32)
        self._block_max_idxs = torch.empty(1024, **i32)
        self._lm_sync_counter = torch.zeros(1, **u32)
        self._seen_token_mask = torch.zeros(VOCAB_SIZE, **f32)
        self._out_token = torch.empty(1, **i32)

        # Pre-pack fused weights for the chunk-parallel prefill kernel:
        # one cuBLAS GEMM per layer instead of three (FA QKV) / two (MLP gate+up).
        layer_data = weights["layer_data"]
        fa_qkv_list = []
        for li in range(NUM_LAYERS):
            ld = layer_data[li]
            if ld['type'] == 1:
                q = ld['ptrs'][1]; k = ld['ptrs'][2]; v = ld['ptrs'][3]
                fa_qkv_list.append(torch.cat([q, k, v], dim=0))
        self._fused_fa_qkv = torch.stack(fa_qkv_list, dim=0).contiguous()
        gate_up_list = []
        for li in range(NUM_LAYERS):
            ld = layer_data[li]
            if ld['type'] == 0:
                g = ld['ptrs'][11]; u = ld['ptrs'][12]
            else:
                g = ld['ptrs'][8]; u = ld['ptrs'][9]
            gate_up_list.append(torch.cat([g, u], dim=0))
        self._fused_gate_up = torch.stack(gate_up_list, dim=0).contiguous()
        self._fa_q_tail = torch.empty(n_fa, 8, FA_NUM_Q_HEADS, FA_HEAD_DIM, **bf16)
        self._prefill_buffers = None
        self._prefill_buffer_tokens = 0

    def alloc_prefill_scratch(self, S: int):
        """Allocate per-prefill scratch buffers for the chunk-parallel kernel.
        Buffers depend on S (sequence length); call once per distinct S."""
        f32 = dict(dtype=torch.float32, device="cuda")
        S_pad = ((S + 31) // 32) * 32
        return dict(
            dn_pre_qkv=torch.empty(S * DN_CONV_CHANNELS, **f32),
            dn_u_scratch=torch.empty(S_pad * DN_NUM_HEADS * 128, **f32),
            dn_w_scratch=torch.empty(S_pad * DN_NUM_HEADS * 128, **f32),
            dn_cs_scratch=torch.empty(S_pad * DN_NUM_HEADS, **f32),
        )

    def _ensure_prefill_buffers(self, max_tokens: int):
        if self._prefill_buffers is not None and self._prefill_buffer_tokens >= max_tokens:
            return self._prefill_buffers

        bf16 = dict(dtype=_half_dtype(), device="cuda")
        f32 = dict(dtype=torch.float32, device="cuda")
        i32 = dict(dtype=torch.int32, device="cuda")
        mx = max(DN_CONV_CHANNELS, FA_QPROJ_SIZE, INTERMEDIATE_SIZE)
        buffers = dict(
            hidden=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
            residual=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
            normalized=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
            proj_buf=torch.empty(max_tokens * mx, **bf16),
            proj_buf2=torch.empty(max_tokens * mx, **bf16),
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
        buffers.update(self.alloc_prefill_scratch(max_tokens))
        self._prefill_buffers = buffers
        self._prefill_buffer_tokens = max_tokens
        return buffers

    def prefill_tokens(self, token_ids) -> int:
        """Run prompt prefill and return the first generated token id."""
        if isinstance(token_ids, torch.Tensor):
            ids_t = token_ids.to(device="cuda", dtype=torch.int32).contiguous()
            prompt_len = int(ids_t.numel())
        else:
            prompt_len = len(token_ids)
            ids_t = torch.tensor(token_ids, dtype=torch.int32, device="cuda")
        if prompt_len <= 0:
            raise ValueError("prefill_tokens requires at least one token")
        if prompt_len > self.max_seq_len:
            raise ValueError(f"prompt length {prompt_len} exceeds max_seq_len={self.max_seq_len}")

        self.reset()
        b = self._ensure_prefill_buffers(prompt_len)
        q_tail_len = min(self._fa_q_tail.shape[1], prompt_len)
        _prefill_bf16_qtail(
            self._out_token, ids_t,
            self._embed_weight, self._layer_weights_packed,
            self._final_norm_weight, self._lm_head_weight,
            self._fa_k_cache, self._fa_v_cache, self._dn_states, self._conv_bufs,
            b["hidden"], b["residual"], b["normalized"],
            b["proj_buf"], b["proj_buf2"], b["attn_buf"], b["mlp_buf"],
            b["dn_out_buf"], b["beta_buf"], b["alpha_buf"],
            b["dn_pre_qkv"],
            b["dn_u_scratch"], b["dn_w_scratch"], b["dn_cs_scratch"],
            self._fused_fa_qkv, self._fused_gate_up,
            b["final_normed"], b["hidden_bf16_out"],
            b["lm_bmv"], b["lm_bmi"], self._fa_q_tail,
            self.max_seq_len, q_tail_len)
        self._hidden.copy_(b["hidden_bf16_out"])
        self._position = prompt_len
        return self._out_token.item()

    def step(self, token_id: int) -> int:
        """Decode one token. Returns next token id."""
        if self._position >= self.max_seq_len:
            raise ValueError(f"position {self._position} exceeds max_seq_len={self.max_seq_len}")
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
            self._seen_token_mask, self.repetition_penalty,
            self._position, self.max_seq_len,
        )
        self._position += 1
        return self._out_token.item()

    def reset(self):
        self._position = 0
        self._fa_k_cache.zero_()
        self._fa_v_cache.zero_()
        self._dn_states.zero_()
        self._conv_bufs.zero_()
        self._seen_token_mask.zero_()

    def generate(self, prompt: str, max_tokens: int = 100, pflash=None,
                 pflash_threshold: int = 0) -> str:
        if pflash is not None:
            probe_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            if len(probe_ids) >= pflash_threshold:
                prompt = pflash.compress_text(prompt)
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if not ids:
            return ""
        out = []
        next_id = self.prefill_tokens(ids)
        eos = self.tokenizer.eos_token_id
        for _ in range(max_tokens):
            if next_id == eos:
                break
            out.append(next_id)
            next_id = self.step(next_id)
        return self.tokenizer.decode(out, skip_special_tokens=True)
