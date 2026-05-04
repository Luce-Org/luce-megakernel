"""Prompt compression helpers for long-context prefill experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable

from .platform import detect_accelerator


@dataclass(frozen=True)
class ChunkScore:
    start: int
    end: int
    score: float


@dataclass(frozen=True)
class CompressionResult:
    original_tokens: int
    compressed_tokens: int
    compressed_prompt: str
    query_text: str
    kept_chunks: list[ChunkScore]

    @property
    def keep_ratio(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens

    @property
    def compression_ratio(self) -> float:
        if self.compressed_tokens == 0:
            return float("inf")
        return self.original_tokens / self.compressed_tokens


def build_middle_chunk_ranges(
    total_tokens: int,
    head_tokens: int,
    tail_tokens: int,
    chunk_tokens: int,
) -> list[tuple[int, int]]:
    if total_tokens <= 0 or chunk_tokens <= 0:
        return []

    head_end = min(max(head_tokens, 0), total_tokens)
    tail_start = max(head_end, total_tokens - max(tail_tokens, 0))
    ranges: list[tuple[int, int]] = []

    cur = head_end
    while cur < tail_start:
        end = min(cur + chunk_tokens, tail_start)
        ranges.append((cur, end))
        cur = end
    return ranges


def merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((start, end) for start, end in ranges if end > start)
    if not ordered:
        return []

    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def select_scored_ranges(
    ranges: list[tuple[int, int]],
    scores: list[float],
    token_budget: int,
    min_chunks: int,
) -> list[ChunkScore]:
    if len(ranges) != len(scores):
        raise ValueError("ranges and scores must have identical lengths")
    if not ranges:
        return []

    ranked = sorted(
        enumerate(zip(ranges, scores)),
        key=lambda item: (item[1][1], -item[0]),
        reverse=True,
    )

    chosen: set[int] = set()
    used = 0
    need = min(max(min_chunks, 0), len(ranges))

    for idx, ((start, end), _score) in ranked:
        span_tokens = end - start
        must_take = len(chosen) < need
        fits = used + span_tokens <= token_budget
        if must_take or fits or not chosen:
            chosen.add(idx)
            used += span_tokens
        if len(chosen) >= need and used >= token_budget:
            break

    if not chosen:
        start, end = ranges[0]
        return [ChunkScore(start=start, end=end, score=scores[0])]

    return [
        ChunkScore(start=ranges[idx][0], end=ranges[idx][1], score=scores[idx])
        for idx in sorted(chosen, key=lambda i: ranges[i][0])
    ]


def select_top_chunk_scores(
    ranges: list[tuple[int, int]],
    scores: list[float],
    keep_ratio: float,
    always_keep_last_chunks: int,
) -> list[ChunkScore]:
    if len(ranges) != len(scores):
        raise ValueError("ranges and scores must have identical lengths")
    if not ranges:
        return []

    n_keep = max(1, int(math.ceil(len(ranges) * keep_ratio)))
    chosen: set[int] = set()
    if always_keep_last_chunks > 0:
        tail_start = max(0, len(ranges) - always_keep_last_chunks)
        chosen.update(range(tail_start, len(ranges)))

    ranked = sorted(range(len(ranges)), key=lambda idx: (scores[idx], -ranges[idx][0]), reverse=True)
    for idx in ranked:
        chosen.add(idx)
        if len(chosen) >= n_keep:
            break

    return [
        ChunkScore(start=ranges[idx][0], end=ranges[idx][1], score=scores[idx])
        for idx in sorted(chosen, key=lambda i: ranges[i][0])
    ]


def extract_query_anchors(query_text: str) -> list[str]:
    query_lower = query_text.lower()

    targeted_patterns = [
        r"for\s+([a-z0-9_-]{4,})\s+mentioned",
        r"for\s+([a-z0-9_-]{4,})\s+in\s+the\s+provided\s+text",
        r"value\s+([0-9]{4,})",
    ]
    targeted: list[str] = []
    for pattern in targeted_patterns:
        for match in re.finditer(pattern, query_lower):
            val = match.group(1)
            if val not in targeted:
                targeted.append(val)
    if targeted:
        return targeted

    hyphenated = [tok for tok in re.findall(r"[A-Za-z0-9]+-[A-Za-z0-9_-]+", query_lower) if len(tok) >= 4]
    if hyphenated:
        out: list[str] = []
        for tok in hyphenated:
            if tok not in out:
                out.append(tok)
        return out

    stop = {
        "what", "which", "when", "where", "who", "why", "how", "are", "is", "the", "for", "and",
        "all", "special", "magic", "number", "numbers", "mentioned", "provided", "text", "within",
        "following", "question", "answer", "based", "only", "above", "given", "from",
        "grass", "green", "sky", "blue", "sun", "yellow", "here", "there", "back", "again", "go",
    }
    tokens = re.findall(r"[A-Za-z0-9_-]{4,}", query_lower)
    anchors: list[str] = []
    for tok in tokens:
        if tok in stop:
            continue
        if tok not in anchors:
            anchors.append(tok)
    return anchors


def avg_pool_scores(values: list[float], kernel: int) -> list[float]:
    if kernel <= 1 or len(values) <= 1:
        return list(values)

    half = kernel // 2
    out: list[float] = []
    for idx in range(len(values)):
        lo = max(0, idx - half)
        hi = min(len(values), idx + half + 1)
        window = values[lo:hi]
        out.append(sum(window) / len(window))
    return out


class PromptCompressor:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        tokenizer_name: str | None = None,
        *,
        device: str | None = None,
        head_tokens: int = 0,
        tail_tokens: int = 0,
        query_tail_tokens: int = 256,
        chunk_tokens: int = 32,
        min_chunks: int = 0,
        tail_keep_chunks: int = 1,
        anchor_context_chunks: int = 1,
        n_lookahead: int = 8,
        pool_kernel: int = 13,
        batch_size: int = 8,
        max_length: int = 1024,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.head_tokens = head_tokens
        self.tail_tokens = tail_tokens
        self.query_tail_tokens = query_tail_tokens
        self.chunk_tokens = chunk_tokens
        self.min_chunks = min_chunks
        self.tail_keep_chunks = tail_keep_chunks
        self.anchor_context_chunks = anchor_context_chunks
        self.n_lookahead = n_lookahead
        self.pool_kernel = pool_kernel
        self.batch_size = batch_size
        self.max_length = max_length
        self.trust_remote_code = trust_remote_code
        self._tokenizer = None
        self._model = None
        self._torch = None

        accel = detect_accelerator()
        self.device = device or accel.torch_device

    def compress(self, prompt: str, keep_ratio: float = 0.05) -> CompressionResult:
        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in the range (0, 1]")

        tokenizer = self._ensure_tokenizer()
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        total = len(token_ids)

        if total == 0:
            return CompressionResult(0, 0, "", "", [])

        query_text = self._extract_query_text(prompt, token_ids)
        ranges = build_middle_chunk_ranges(total, self.head_tokens, self.tail_tokens, self.chunk_tokens)
        if not ranges:
            return CompressionResult(total, total, prompt, query_text, [])

        chunk_texts = [tokenizer.decode(token_ids[start:end], skip_special_tokens=False) for start, end in ranges]
        scores = self._score_texts(query_text, chunk_texts)
        selected = select_top_chunk_scores(
            ranges,
            scores,
            keep_ratio=keep_ratio,
            always_keep_last_chunks=self.tail_keep_chunks,
        )

        anchors = extract_query_anchors(query_text)
        if anchors and self.anchor_context_chunks > 0:
            anchor_hits: set[int] = set()
            for idx, text in enumerate(chunk_texts):
                text_l = text.lower()
                if any(anchor in text_l for anchor in anchors):
                    lo = max(0, idx - self.anchor_context_chunks)
                    hi = min(len(chunk_texts), idx + self.anchor_context_chunks + 1)
                    anchor_hits.update(range(lo, hi))
            if anchor_hits:
                chosen = {(chunk.start, chunk.end): chunk for chunk in selected}
                for idx in sorted(anchor_hits):
                    start, end = ranges[idx]
                    chosen[(start, end)] = ChunkScore(start=start, end=end, score=scores[idx])
                selected = sorted(chosen.values(), key=lambda chunk: chunk.start)

        merged = merge_ranges((chunk.start, chunk.end) for chunk in selected)
        compressed_ids: list[int] = []
        for start, end in merged:
            compressed_ids.extend(token_ids[start:end])

        compressed_prompt = tokenizer.decode(compressed_ids, skip_special_tokens=False)
        return CompressionResult(
            original_tokens=total,
            compressed_tokens=len(compressed_ids),
            compressed_prompt=compressed_prompt,
            query_text=query_text,
            kept_chunks=selected,
        )

    def _extract_query_text(self, prompt: str, token_ids: list[int]) -> str:
        for marker in ("\nQuestion:", "\nQUESTION:", "\nUser:", "\nUSER:"):
            pos = prompt.rfind(marker)
            if pos != -1:
                return prompt[pos:].strip()

        tokenizer = self._ensure_tokenizer()
        tail_ids = token_ids[-min(self.query_tail_tokens, len(token_ids)):]
        return tokenizer.decode(tail_ids, skip_special_tokens=False).strip()

    def _score_texts(self, query_text: str, chunk_texts: list[str]) -> list[float]:
        if not chunk_texts:
            return []
        query_tokens = self._embed_query_tokens(query_text)
        return self._score_chunk_texts(query_tokens, chunk_texts)

    def _score_chunk_texts(self, query_tokens, chunk_texts: list[str]) -> list[float]:
        torch = self._ensure_torch()
        tokenizer = self._ensure_tokenizer()
        model = self._ensure_model()

        scores: list[float] = []
        for start in range(0, len(chunk_texts), self.batch_size):
            batch_texts = chunk_texts[start:start + self.batch_size]
            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.inference_mode():
                outputs = model(**batch)

            hidden = self._extract_hidden(outputs)
            hidden = torch.nn.functional.normalize(hidden.float(), dim=-1)
            attn_mask = batch["attention_mask"]
            sims = torch.matmul(hidden, query_tokens.transpose(0, 1))
            token_scores = sims.max(dim=-1).values

            for row_idx in range(token_scores.shape[0]):
                valid_len = int(attn_mask[row_idx].sum().item())
                if valid_len <= 0:
                    scores.append(float("-inf"))
                    continue
                values = token_scores[row_idx, :valid_len].detach().cpu().tolist()
                values = avg_pool_scores(values, self.pool_kernel)
                scores.append(float(sum(values) / len(values)))

        return scores

    def _embed_query_tokens(self, query_text: str):
        torch = self._ensure_torch()
        tokenizer = self._ensure_tokenizer()
        model = self._ensure_model()

        batch = tokenizer(
            [query_text],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.inference_mode():
            outputs = model(**batch)

        hidden = self._extract_hidden(outputs)[0]
        valid_len = int(batch["attention_mask"][0].sum().item())
        hidden = hidden[:valid_len]
        if valid_len > self.n_lookahead:
            hidden = hidden[-self.n_lookahead:]
        return torch.nn.functional.normalize(hidden.float(), dim=-1)

    def _embed_texts(self, texts: list[str]):
        torch = self._ensure_torch()
        tokenizer = self._ensure_tokenizer()
        model = self._ensure_model()

        embeddings = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]
            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.inference_mode():
                outputs = model(**batch)
            hidden = self._extract_hidden(outputs)

            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
            embeddings.append(pooled.cpu())

        return torch.cat(embeddings, dim=0)

    @staticmethod
    def _extract_hidden(outputs):
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states:
                hidden = hidden_states[-1]
            elif isinstance(outputs, tuple) and outputs:
                hidden = outputs[0]
        if hidden is None:
            raise RuntimeError("model did not return hidden states")
        return hidden

    def _ensure_torch(self):
        if self._torch is None:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    "PromptCompressor requires PyTorch. Install a ROCm or CPU wheel before using the Strix Halo path."
                ) from exc
            self._torch = torch
        return self._torch

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                trust_remote_code=self.trust_remote_code,
            )
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def _ensure_model(self):
        if self._model is None:
            torch = self._ensure_torch()
            from transformers import AutoModel

            dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
            self._model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                dtype=dtype,
            )
            self._model.to(self.device)
            self._model.eval()
        return self._model
