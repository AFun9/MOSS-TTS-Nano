"""Shared trace-safe helpers used by every wrapper.

Two things live here because they are duplicated by definition (prefill,
decode_step, local_decoder, local_cached_step all need them):

1. `build_multi_channel_inputs_embeds(lm, input_ids_long)`
   - sums text-channel embedding + 16 audio-channel embeddings
   - skips Python-level shape asserts and out-of-range guards (those bake
     constants into the ONNX graph during tracing)

2. `patch_attention(lm, also_local=False)`
   - replaces every `_eager_attention` with `_trace_safe_eager_attention`
     that uses explicit `permute(...)` to defeat the
     `transpose().transpose()` fusion bug
   - when `also_local=True`, also patches `lm.local_transformer.h`
"""
from __future__ import annotations

import torch
from torch import nn


def _trace_safe_rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    """Math: y[..., 2k] = -x[..., 2k+1] ; y[..., 2k+1] = x[..., 2k]

    Uses stack+flatten so onnx exporter emits Concat+Reshape with shapes
    derived from Shape op (no baked dims) — safe for dynamic batch/seq/heads.
    """
    even = hidden_states[..., 0::2]
    odd = hidden_states[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(start_dim=-2)


def _trace_safe_apply_rotary_pos_emb(hidden_states, cos, sin):
    return hidden_states * cos + _trace_safe_rotate_half(hidden_states) * sin


def _trace_safe_eager_attention(self, query, key, value, attention_mask):
    """Drop-in replacement for MossTTSNanoGPT2Attention._eager_attention that
    survives torch.onnx tracing. Inputs (after wrapper) are layout [B, T, H, D].

    The original impl uses `key.transpose(1,2).transpose(-1,-2)` which the
    onnx exporter incorrectly fuses into a single perm=[0,2,1,3] (instead of
    correct [0,2,3,1]), breaking attention. We use explicit `permute(...)`.
    """
    q = query.permute(0, 2, 1, 3).contiguous()  # [B, H, T, D]
    k = key.permute(0, 2, 1, 3).contiguous()
    v = value.permute(0, 2, 1, 3).contiguous()

    scale = 1.0
    if self.scale_attn_weights:
        scale /= self.head_dim ** 0.5
    if self.scale_attn_by_inverse_layer_idx:
        scale /= float(self.layer_idx + 1)

    k_t = k.permute(0, 1, 3, 2).contiguous()  # [B, H, D, T]
    scores = torch.matmul(q, k_t) * scale

    causal_mask = self._causal_attention_mask(
        attention_mask=attention_mask,
        query_length=q.shape[-2],
        key_length=k.shape[-2],
        device=q.device,
    )
    scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    return out.permute(0, 2, 1, 3).contiguous()  # [B, T, H, D]


def _patch_blocks(blocks):
    for blk in blocks:
        attn = blk.attn
        attn.attn_implementation = "eager"
        attn._eager_attention = _trace_safe_eager_attention.__get__(attn, type(attn))


def patch_attention(lm, *, also_local: bool = False, patch_rope: bool = False) -> None:
    """Force every block to use trace-safe eager attention.

    Args:
        also_local: when True, also patch `lm.local_transformer.h` (needed by
            local_decoder / local_cached_step exporters).
        patch_rope: only needed if your toolchain miscompiles the original
            `rotate_half`. Keep False unless m1 verify shows a numerical drift.
    """
    if patch_rope:
        import sys
        target_mods = [
            m for name, m in list(sys.modules.items())
            if m is not None and name.endswith("gpt2_decoder")
        ]
        for m in target_mods:
            if hasattr(m, "apply_rotary_pos_emb"):
                m.apply_rotary_pos_emb = _trace_safe_apply_rotary_pos_emb
            if hasattr(m, "rotate_half"):
                m.rotate_half = _trace_safe_rotate_half

    _patch_blocks(lm.transformer.h)
    lm.transformer.attn_implementation = "eager"

    if also_local and hasattr(lm, "local_transformer"):
        _patch_blocks(lm.local_transformer.h)
        lm.local_transformer.attn_implementation = "eager"


def build_multi_channel_inputs_embeds(lm, input_ids_long: torch.Tensor) -> torch.Tensor:
    """Trace-safe variant of MossTTSNanoForCausalLM._build_inputs_embeds.

    Shape: input_ids_long is [..., n_vq+1] long. Channel 0 is text, channels
    1..n_vq are audio codes (or audio_pad if not active at this row).

    Output: [..., hidden_size]
    """
    n_vq = int(lm.config.n_vq)
    audio_pad = int(lm.config.audio_pad_token_id)

    text_ids = input_ids_long[..., 0]
    embeds = lm.transformer.wte(text_ids)
    for channel_index in range(n_vq):
        channel_ids = input_ids_long[..., channel_index + 1]
        valid_mask = channel_ids.ne(audio_pad)
        safe_ids = channel_ids.masked_fill(~valid_mask, 0)
        audio_embeds = lm.audio_embeddings[channel_index](safe_ids)
        audio_embeds = audio_embeds * valid_mask.unsqueeze(-1).to(audio_embeds.dtype)
        embeds = embeds + audio_embeds
    return embeds
