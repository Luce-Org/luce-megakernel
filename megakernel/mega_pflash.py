"""Qwen3.5 megakernel-native PFlash proof.

This is the first monolithic-PFlash staging point: scoring runs from the
megakernel prefill's resident full-attention Q/K state instead of a separate
Qwen3-0.6B sidecar process. The scorer is intentionally PyTorch for now so the
selection behavior can be validated before fusing it into CUDA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model import FA_HEAD_DIM, FA_NUM_KV_HEADS, FA_NUM_Q_HEADS, Decoder


@dataclass(frozen=True)
class MegaPFlashResult:
    input_tokens: int
    output_tokens: int
    selected_chunks: int
    chunk_size: int
    keep_ratio: float
    token_ids: list[int]
    text: str


class MegaPFlashCompressor:
    """Prompt compressor using Qwen3.5 megakernel prefill attention state."""

    def __init__(
        self,
        decoder: Decoder,
        *,
        keep_ratio: float = 0.05,
        lookahead: int = 8,
        chunk_size: int = 32,
        smooth_kernel: int = 13,
        preserve_edges: bool = True,
        reduction: str = "mean",
        cuda_score: bool = True,
    ):
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0.0, 1.0]")
        if lookahead <= 0:
            raise ValueError("lookahead must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if smooth_kernel <= 0:
            raise ValueError("smooth_kernel must be positive")
        if reduction not in {"mean", "max"}:
            raise ValueError("reduction must be 'mean' or 'max'")
        self.decoder = decoder
        self.tokenizer = decoder.tokenizer
        self.keep_ratio = float(keep_ratio)
        self.lookahead = int(lookahead)
        self.chunk_size = int(chunk_size)
        self.smooth_kernel = int(smooth_kernel)
        self.preserve_edges = bool(preserve_edges)
        self.reduction = reduction
        self.cuda_score = bool(cuda_score)

    def compress_text(self, text: str) -> str:
        return self.compress(text).text

    @torch.no_grad()
    def compress(self, text: str) -> MegaPFlashResult:
        ids = self.tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = [int(t) for t in ids]
        compressed = self.compress_ids(ids)
        return MegaPFlashResult(
            input_tokens=len(ids),
            output_tokens=len(compressed),
            selected_chunks=math.ceil(len(compressed) / self.chunk_size) if compressed else 0,
            chunk_size=self.chunk_size,
            keep_ratio=self.keep_ratio,
            token_ids=compressed,
            text=self.tokenizer.decode(compressed, skip_special_tokens=True),
        )

    @torch.no_grad()
    def compress_ids(self, ids: list[int]) -> list[int]:
        if not ids:
            return []
        if self.keep_ratio >= 1.0 or len(ids) <= self.chunk_size:
            return list(ids)

        self.decoder.prefill_tokens(ids)
        torch.cuda.synchronize()

        if self.cuda_score and hasattr(torch.ops.qwen35_megakernel_bf16_C, "mega_pflash_score"):
            chunk_scores = self._score_chunks_cuda(len(ids))
            selected = self._select_from_chunk_scores(chunk_scores, len(ids))
        else:
            scores = self._score_tokens(len(ids))
            scores = self._smooth(scores)
            selected = self._select_chunks(scores, len(ids))
        return [ids[i] for i in selected]

    def _score_chunks_cuda(self, seq_len: int) -> torch.Tensor:
        tail_len = min(self.lookahead, self.decoder._fa_q_tail.shape[1], seq_len)
        n_chunks = math.ceil(seq_len / self.chunk_size)
        total_rows = self.decoder._fa_q_tail.shape[0] * FA_NUM_Q_HEADS * tail_len
        logit_scratch = torch.empty(total_rows, seq_len, device="cuda", dtype=torch.float32)
        token_scores = torch.empty(seq_len, device="cuda", dtype=torch.float32)
        chunk_scores = torch.empty(n_chunks, device="cuda", dtype=torch.float32)
        reduction_mode = 1 if self.reduction == "max" else 0
        torch.ops.qwen35_megakernel_bf16_C.mega_pflash_score(
            self.decoder._fa_q_tail,
            self.decoder._fa_k_cache,
            logit_scratch,
            token_scores,
            chunk_scores,
            seq_len,
            self.decoder.max_seq_len,
            tail_len,
            self.chunk_size,
            self.smooth_kernel,
            reduction_mode,
        )
        return chunk_scores

    def _score_tokens(self, seq_len: int) -> torch.Tensor:
        tail_len = min(self.lookahead, self.decoder._fa_q_tail.shape[1], seq_len)
        q_tail = self.decoder._fa_q_tail[:, :tail_len].float()
        k_cache = self.decoder._fa_k_cache[:, :, :seq_len].float()

        scores = torch.zeros(seq_len, device=k_cache.device, dtype=torch.float32)
        if self.reduction == "mean":
            count = 0
        scale = 1.0 / math.sqrt(FA_HEAD_DIM)
        gqa = FA_NUM_Q_HEADS // FA_NUM_KV_HEADS
        positions = torch.arange(seq_len, device=k_cache.device)

        for layer_idx in range(q_tail.shape[0]):
            for q_head in range(FA_NUM_Q_HEADS):
                kv_head = q_head // gqa
                q = q_tail[layer_idx, :, q_head, :]
                k = k_cache[layer_idx, kv_head, :, :]
                logits = torch.matmul(q, k.T) * scale

                for tail_idx in range(tail_len):
                    pos = seq_len - tail_len + tail_idx
                    logits[tail_idx, positions > pos] = -torch.inf

                probs = torch.softmax(logits, dim=-1)
                head_scores = probs.max(dim=0).values
                if self.reduction == "max":
                    scores = torch.maximum(scores, head_scores)
                else:
                    scores += head_scores
                    count += 1

        if self.reduction == "mean" and count:
            scores /= count
        return scores

    def _smooth(self, scores: torch.Tensor) -> torch.Tensor:
        kernel = min(self.smooth_kernel, scores.numel())
        if kernel <= 1:
            return scores
        if kernel % 2 == 0:
            kernel -= 1
        if kernel <= 1:
            return scores
        x = scores.view(1, 1, -1)
        return F.avg_pool1d(x, kernel_size=kernel, stride=1, padding=kernel // 2).view(-1)

    def _select_chunks(self, scores: torch.Tensor, seq_len: int) -> list[int]:
        n_chunks = math.ceil(seq_len / self.chunk_size)
        keep_tokens = max(1, int(math.ceil(seq_len * self.keep_ratio)))
        keep_chunks = max(1, min(n_chunks, int(math.ceil(keep_tokens / self.chunk_size))))

        pad = n_chunks * self.chunk_size - seq_len
        if pad:
            scores = F.pad(scores, (0, pad), value=-torch.inf)
        chunk_scores = scores.view(n_chunks, self.chunk_size).amax(dim=1)

        forced = set()
        if self.preserve_edges:
            forced.add(0)
            forced.add(n_chunks - 1)

        remaining = max(0, keep_chunks - len(forced))
        selected = set(forced)
        if remaining:
            ranked = torch.topk(chunk_scores, k=min(n_chunks, remaining + len(forced))).indices.tolist()
            for idx in ranked:
                selected.add(int(idx))
                if len(selected) >= keep_chunks:
                    break

        out: list[int] = []
        for chunk_idx in sorted(selected):
            start = chunk_idx * self.chunk_size
            end = min(seq_len, start + self.chunk_size)
            out.extend(range(start, end))
        return out

    def _select_from_chunk_scores(self, chunk_scores: torch.Tensor, seq_len: int) -> list[int]:
        n_chunks = int(chunk_scores.numel())
        keep_tokens = max(1, int(math.ceil(seq_len * self.keep_ratio)))
        keep_chunks = max(1, min(n_chunks, int(math.ceil(keep_tokens / self.chunk_size))))

        forced = set()
        if self.preserve_edges:
            forced.add(0)
            forced.add(n_chunks - 1)

        selected = set(forced)
        remaining = max(0, keep_chunks - len(selected))
        if remaining:
            ranked = torch.topk(chunk_scores, k=min(n_chunks, remaining + len(forced))).indices.tolist()
            for idx in ranked:
                selected.add(int(idx))
                if len(selected) >= keep_chunks:
                    break

        out: list[int] = []
        for chunk_idx in sorted(selected):
            start = chunk_idx * self.chunk_size
            end = min(seq_len, start + self.chunk_size)
            out.extend(range(start, end))
        return out
