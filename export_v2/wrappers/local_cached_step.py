"""Local-cached-step wrapper - reproduces official `moss_tts_local_cached_step.onnx`.

IO contract (matches official):
    Inputs:
        global_hidden          FLOAT [batch, 768]
        text_token_id          INT32 [batch]
        audio_token_id         INT32 [batch]
        channel_index          INT32 [batch]   (0..15)
        step_type              INT32 [batch]   (0=global, 1=text, 2=audio)
        past_valid_lengths     INT32 [batch]
        local_past_key_i       FLOAT [batch, local_past_seq, 12, 64]   for i in 0..L-1
        local_past_value_i     FLOAT [batch, local_past_seq, 12, 64]

    Outputs:
        text_logits            FLOAT [batch, vocab]
        audio_logits           FLOAT [batch, 16, 1024]
        local_present_key_i    FLOAT [batch, local_total_seq, 12, 64]
        local_present_value_i  FLOAT [batch, local_total_seq, 12, 64]

Per-step input embedding selection (matches official):
    if step_type == 0: input_emb = global_hidden            (frame's first step)
    if step_type == 1: input_emb = wte(text_token_id)        (assistant text step)
    if step_type == 2: input_emb = audio_emb_at_channel      (audio channel step)
where audio_emb_at_channel = sum_k 1[channel_index == k] * audio_embeddings[k](audio_token_id)
(implemented as a chain of Where over Equal — fully trace-safe).

ALL lm_heads are evaluated unconditionally (text + 16 audio), so the caller
indexes into audio_logits[:, channel_index] for the current channel.
"""
from __future__ import annotations
from typing import List

import torch
from torch import nn


class LocalCachedStepWrapper(nn.Module):
    def __init__(self, lm) -> None:
        super().__init__()
        self.lm = lm
        cfg = lm.config
        self.n_vq = int(cfg.n_vq)
        self.n_local_layers = int(cfg.local_transformer_layers)
        self.hidden_size = int(cfg.gpt2_config.hidden_size)
        self.vocab_size = int(lm.text_lm_head.out_features)

    def _select_audio_emb(
        self, audio_token_id: torch.Tensor, channel_index: torch.Tensor
    ) -> torch.Tensor:
        """Chain Where over Equal to pick audio_embeddings[channel_index](audio_token_id).

        audio_token_id : [B] int64
        channel_index  : [B] int64
        returns        : [B, 768]
        """
        B = audio_token_id.shape[0]
        local_dtype = self.lm.local_transformer.ln_f.weight.dtype
        # safe fallback: zeros if no Equal matches (should never happen at runtime)
        out = torch.zeros(B, self.hidden_size, dtype=local_dtype, device=audio_token_id.device)
        for k in range(self.n_vq):
            mask = (channel_index == k).unsqueeze(-1)            # [B, 1] bool
            emb = self.lm.audio_embeddings[k](audio_token_id).to(dtype=local_dtype)
            out = torch.where(mask, emb, out)
        return out

    def forward(
        self,
        global_hidden: torch.Tensor,            # [B, 768] float
        text_token_id: torch.Tensor,            # [B] int32
        audio_token_id: torch.Tensor,           # [B] int32
        channel_index: torch.Tensor,            # [B] int32
        step_type: torch.Tensor,                # [B] int32
        past_valid_lengths: torch.Tensor,       # [B] int32
        *past_kv: torch.Tensor,                 # 2L tensors  [B, past_seq, 12, 64]
    ):
        device = global_hidden.device
        B = global_hidden.shape[0]
        local_dtype = self.lm.local_transformer.ln_f.weight.dtype

        text_id_long = text_token_id.to(torch.long)
        audio_id_long = audio_token_id.to(torch.long)
        channel_long = channel_index.to(torch.long)
        step_long = step_type.to(torch.long)

        text_emb = self.lm.transformer.wte(text_id_long).to(dtype=local_dtype)   # [B, 768]
        audio_emb = self._select_audio_emb(audio_id_long, channel_long)          # [B, 768]
        global_e = global_hidden.to(dtype=local_dtype)                            # [B, 768]

        is_global = (step_long == 0).unsqueeze(-1)                               # [B, 1]
        is_text = (step_long == 1).unsqueeze(-1)
        # is_audio is the default (step_long == 2)
        emb = torch.where(is_global, global_e,
              torch.where(is_text, text_emb, audio_emb))                         # [B, 768]
        inputs_embeds = emb.unsqueeze(1)                                         # [B, 1, 768]

        past_seq = past_kv[0].shape[1]
        positions = torch.arange(past_seq, device=device).unsqueeze(0)           # [1, past_seq]
        valid_lens = past_valid_lengths.to(torch.int64).unsqueeze(-1)            # [B, 1]
        past_mask = positions < valid_lens                                       # [B, past_seq]
        cur_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        attention_mask = torch.cat([past_mask, cur_mask], dim=1).to(torch.int64)

        past_key_values = tuple(
            (past_kv[2 * i], past_kv[2 * i + 1]) for i in range(self.n_local_layers)
        )

        out = self.lm.local_transformer(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        local_hidden = out.last_hidden_state[:, -1, :]                           # [B, 768]

        text_logits = self.lm.text_lm_head(local_hidden)                          # [B, vocab]

        audio_logits_list: List[torch.Tensor] = []
        for k in range(self.n_vq):
            audio_logits_list.append(self.lm.audio_lm_heads[k](local_hidden))    # [B, 1024]
        audio_logits = torch.stack(audio_logits_list, dim=1)                      # [B, 16, 1024]

        outputs: List[torch.Tensor] = [text_logits, audio_logits]
        for i in range(self.n_local_layers):
            k_t, v_t = out.past_key_values[i]
            outputs.append(k_t.contiguous())
            outputs.append(v_t.contiguous())
        return tuple(outputs)


def io_names(n_local_layers: int):
    in_names = [
        "global_hidden", "text_token_id", "audio_token_id",
        "channel_index", "step_type", "past_valid_lengths",
    ]
    for i in range(n_local_layers):
        in_names.append(f"local_past_key_{i}")
        in_names.append(f"local_past_value_{i}")
    out_names = ["text_logits", "audio_logits"]
    for i in range(n_local_layers):
        out_names.append(f"local_present_key_{i}")
        out_names.append(f"local_present_value_{i}")
    return in_names, out_names
