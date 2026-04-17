"""Shared utilities for ONNX-based MOSS-TTS-Nano inference.

Contains sampling helpers, audio code normalization, and a lightweight
SentencePiece tokenizer wrapper — all independent of the inference engine.
"""
from __future__ import annotations

import numpy as np
import sentencepiece as spm


# ---------------------------------------------------------------------------
# SentencePiece tokenizer wrapper
# ---------------------------------------------------------------------------
class SPTokenizer:
    """Thin wrapper around SentencePiece for encode/decode."""

    def __init__(self, model_path: str) -> None:
        self.model_path = str(model_path)
        self.processor = spm.SentencePieceProcessor()
        self.processor.Load(self.model_path)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(self.processor.encode(str(text), out_type=int))

    def decode(self, token_ids) -> str:
        return str(self.processor.decode(list(token_ids)))


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------
def softmax_numpy(scores: np.ndarray) -> np.ndarray:
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / exp_scores.sum()


def sample_top_k_top_p(logits: np.ndarray, temperature: float, top_k: int, top_p: float) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))

    scores = logits / temperature

    if top_k > 0:
        top_k = min(top_k, len(scores))
        kth = np.partition(scores, -top_k)[-top_k]
        scores[scores < kth] = -np.inf

    if 0.0 < top_p < 1.0:
        sorted_idx = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_idx]
        probs_sorted = softmax_numpy(sorted_scores)
        cumsum = np.cumsum(probs_sorted)
        cutoff = np.searchsorted(cumsum, top_p) + 1
        sorted_scores[cutoff:] = -np.inf
        scores = np.full_like(scores, -np.inf)
        scores[sorted_idx] = sorted_scores

    probs = softmax_numpy(scores)
    return int(np.random.choice(np.arange(probs.size), p=probs))


def apply_repetition_penalty(logits: np.ndarray, previous_token_ids, repetition_penalty: float) -> np.ndarray:
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    if repetition_penalty == 1.0 or previous_token_ids is None or len(previous_token_ids) == 0:
        return logits

    scores = logits.copy()
    unique_ids = np.unique(previous_token_ids)
    unique_ids = unique_ids[(unique_ids >= 0) & (unique_ids < len(scores))]
    for token_id in unique_ids:
        if scores[token_id] < 0:
            scores[token_id] *= repetition_penalty
        else:
            scores[token_id] /= repetition_penalty
    return scores


# ---------------------------------------------------------------------------
# Audio code normalization
# ---------------------------------------------------------------------------
def normalize_audio_codes(audio_codes, n_vq: int) -> np.ndarray:
    """Normalize audio codes to shape [T, n_vq] regardless of input layout."""
    tensor = np.asarray(audio_codes)
    if tensor.ndim == 1:
        tensor = tensor[:, np.newaxis]

    if tensor.ndim == 3:
        if tensor.shape[1] == 1 and tensor.shape[0] >= n_vq:
            tensor = tensor[:n_vq, 0, :].transpose(1, 0)
        elif tensor.shape[0] == 1:
            tensor = tensor[0]
        elif tensor.shape[1] == n_vq:
            tensor = tensor.transpose(0, 2, 1)[0]
        elif tensor.shape[-1] == n_vq:
            tensor = tensor[0]
        else:
            raise ValueError(f"Unable to normalize audio codes with shape {tuple(tensor.shape)}")

    if tensor.ndim != 2:
        raise ValueError(f"Expected audio codes with 2 dims after normalization, got {tuple(tensor.shape)}")

    if tensor.shape[-1] != n_vq and tensor.shape[0] == n_vq:
        tensor = tensor.transpose(1, 0)
    elif tensor.shape[-1] != n_vq and tensor.shape[0] > n_vq:
        tensor = tensor[:n_vq].transpose(1, 0)
    elif tensor.shape[-1] > n_vq:
        tensor = tensor[:, :n_vq]

    if tensor.shape[-1] != n_vq:
        raise ValueError(f"Expected normalized audio codes with trailing dim {n_vq}, got {tuple(tensor.shape)}")
    return tensor.astype(np.int64, copy=False)


def mask_unused_audio_channels(audio_codes: np.ndarray, nq: int, audio_pad_token_id: int) -> np.ndarray:
    """Zero-out audio channels beyond nq with the pad token."""
    tensor = np.asarray(audio_codes, dtype=np.int64)
    if tensor.ndim != 2:
        raise ValueError(f"Expected prompt audio codes with shape [T, n_vq], got {tuple(tensor.shape)}")
    if nq < tensor.shape[-1]:
        tensor = tensor.copy()
        tensor[:, nq:] = int(audio_pad_token_id)
    return tensor
