"""Decode-step wrapper - reproduces official `moss_tts_decode_step.onnx`.

IO contract (matches official):
    Inputs:
        input_ids            INT32  [batch, step_seq, 17]
        past_valid_lengths   INT32  [batch]
        past_key_i           FLOAT  [batch, past_seq, 12, 64]   for i in 0..L-1
        past_value_i         FLOAT  [batch, past_seq, 12, 64]   for i in 0..L-1

    Outputs:
        global_hidden        FLOAT  [batch, step_seq, 768]
        present_key_i        FLOAT  [batch, total_seq, 12, 64]
        present_value_i      FLOAT  [batch, total_seq, 12, 64]
        (total_seq = past_seq + step_seq)

The wrapper builds an HF-compatible attention_mask of length past_seq+step_seq
from `past_valid_lengths`, so the rest of the GPT2 transformer (with rotary
positions and KV append) sees the same shapes as during prefill — only the
prefix carries cache.
"""
from __future__ import annotations
from typing import List

import torch
from torch import nn

from ._common import build_multi_channel_inputs_embeds


class DecodeStepWrapper(nn.Module):
    def __init__(self, lm) -> None:
        super().__init__()
        self.lm = lm
        cfg = lm.config
        self.num_layers = int(cfg.gpt2_config.n_layer)
        self.n_vq = int(cfg.n_vq)

    def forward(
        self,
        input_ids: torch.Tensor,             # [B, step, 17] int32
        past_valid_lengths: torch.Tensor,    # [B] int32
        *past_kv: torch.Tensor,              # 2L tensors, each [B, past_seq, 12, 64]
    ):
        input_ids_long = input_ids.to(torch.long)
        B = input_ids.shape[0]
        step_seq = input_ids.shape[1]
        past_seq = past_kv[0].shape[1]

        # mask[b, t] = 1 if (t < past_valid_lengths[b]) OR (t >= past_seq)
        positions = torch.arange(past_seq, device=input_ids.device).unsqueeze(0)  # [1, past_seq]
        valid_lens = past_valid_lengths.to(torch.int64).unsqueeze(-1)             # [B, 1]
        past_mask = positions < valid_lens                                        # [B, past_seq]
        cur_mask = torch.ones(B, step_seq, dtype=torch.bool, device=input_ids.device)
        attention_mask = torch.cat([past_mask, cur_mask], dim=1).to(torch.int64)

        inputs_embeds = build_multi_channel_inputs_embeds(self.lm, input_ids_long)

        past_key_values = tuple(
            (past_kv[2 * i], past_kv[2 * i + 1]) for i in range(self.num_layers)
        )

        out = self.lm.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        hidden = out.last_hidden_state  # [B, step_seq, 768]

        outputs: List[torch.Tensor] = [hidden]
        for layer_idx in range(self.num_layers):
            k, v = out.past_key_values[layer_idx]
            outputs.append(k.contiguous())
            outputs.append(v.contiguous())
        return tuple(outputs)


def io_names(num_layers: int):
    in_names = ["input_ids", "past_valid_lengths"]
    for i in range(num_layers):
        in_names.append(f"past_key_{i}")
        in_names.append(f"past_value_{i}")
    out_names = ["global_hidden"]
    for i in range(num_layers):
        out_names.append(f"present_key_{i}")
        out_names.append(f"present_value_{i}")
    dyn_axes = {
        "input_ids": {0: "batch", 1: "step_seq"},
        "past_valid_lengths": {0: "batch"},
        "global_hidden": {0: "batch", 1: "step_seq"},
    }
    for i in range(num_layers):
        dyn_axes[f"past_key_{i}"] = {0: "batch", 1: "past_seq"}
        dyn_axes[f"past_value_{i}"] = {0: "batch", 1: "past_seq"}
        dyn_axes[f"present_key_{i}"] = {0: "batch", 1: "total_seq"}
        dyn_axes[f"present_value_{i}"] = {0: "batch", 1: "total_seq"}
    return in_names, out_names, dyn_axes
