"""Prefill wrapper - reproduces official `moss_tts_prefill.onnx` IO protocol.

Inputs:
    input_ids       INT32 [batch, prefill_seq, 17]
    attention_mask  INT32 [batch, prefill_seq]

Outputs (25 = 1 + 12*2):
    global_hidden            FLOAT [batch, prefill_seq, 768]
    present_key_i / present_value_i  FLOAT [batch, prefill_seq, 12, 64]   for i in 0..11

KV layout note: HF GPT2 returns past_key_values as list of (key, value) per
layer with shape [B, num_heads, seq, head_dim]. Our fork
(modeling_moss_tts_nano.py) already returns [B, seq, num_heads, head_dim]
matching official ONNX cache ordering, so no transpose is needed.

Trace-safe attention monkey-patch lives in `_common.patch_attention`. See
that file for the rationale around defeating the transpose-fusion bug.
"""
from __future__ import annotations
from typing import List

import torch
from torch import nn

# Re-export so callers don't have to know the new layout.
from ._common import (  # noqa: F401
    patch_attention,
    build_multi_channel_inputs_embeds,
    _trace_safe_eager_attention,
)


class PrefillWrapper(nn.Module):
    """Wraps MossTTSNanoForCausalLM into the official prefill IO contract."""

    def __init__(self, lm) -> None:
        super().__init__()
        self.lm = lm
        self.num_layers = int(lm.config.gpt2_config.n_layer)
        self.n_vq = int(lm.config.n_vq)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        input_ids_long = input_ids.to(torch.long)
        attention_mask_long = attention_mask.to(torch.long)

        inputs_embeds = build_multi_channel_inputs_embeds(self.lm, input_ids_long)
        out = self.lm.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_long,
            use_cache=True,
            return_dict=True,
        )
        hidden = out.last_hidden_state  # [B, T, hidden]

        outputs: List[torch.Tensor] = [hidden]
        for layer_idx in range(self.num_layers):
            k, v = out.past_key_values[layer_idx]
            outputs.append(k.contiguous())
            outputs.append(v.contiguous())
        return tuple(outputs)


def io_names(num_layers: int) -> tuple[list[str], list[str], dict[str, dict[int, str]]]:
    """Return (input_names, output_names, dynamic_axes) matching official meta."""
    input_names = ["input_ids", "attention_mask"]
    output_names = ["global_hidden"]
    for i in range(num_layers):
        output_names.append(f"present_key_{i}")
        output_names.append(f"present_value_{i}")
    dynamic_axes: dict[str, dict[int, str]] = {
        "input_ids": {0: "batch", 1: "prefill_seq"},
        "attention_mask": {0: "batch", 1: "prefill_seq"},
        "global_hidden": {0: "batch", 1: "prefill_seq"},
    }
    for i in range(num_layers):
        dynamic_axes[f"present_key_{i}"] = {0: "batch", 1: "prefill_seq"}
        dynamic_axes[f"present_value_{i}"] = {0: "batch", 1: "prefill_seq"}
    return input_names, output_names, dynamic_axes
