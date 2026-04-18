"""Local-decoder wrapper - reproduces official `moss_tts_local_decoder.onnx`.

IO contract (matches official):
    Inputs:
        global_hidden            FLOAT [batch, 768]
        text_token_id            INT32 [batch]
        audio_prefix_token_ids   INT32 [batch, 15]   (channels 0..14)

    Outputs:
        text_logits              FLOAT [batch, 16384]
        audio_logits             FLOAT [batch, 16, 1024]

This is a single-shot prefill of the local transformer over a 17-position
sequence:
    pos 0  : global_hidden  (an already-projected hidden vector)
    pos 1  : text_token_emb = transformer.wte(text_token_id)
    pos 2+k: audio_embeddings[k](audio_prefix_token_ids[:, k])  for k in 0..14

The local transformer runs causally with use_cache=False (HF-style growing
context). Heads:
    text_logits        = text_lm_head(local_hidden[:, 0, :])
    audio_logits[:, k] = audio_lm_heads[k](local_hidden[:, 1 + k, :])  k=0..15

Note: official audio_logits has 16 channels; channel k uses local_hidden at
position 1+k. So audio_logits[:, 0] only needs the text token embedding,
audio_logits[:, 1] needs text + audio[0], ..., audio_logits[:, 15] needs
text + audio[0..14] — exactly the official IO contract.
"""
from __future__ import annotations
from typing import List

import torch
from torch import nn


class LocalDecoderWrapper(nn.Module):
    def __init__(self, lm) -> None:
        super().__init__()
        self.lm = lm
        cfg = lm.config
        self.n_vq = int(cfg.n_vq)
        self.hidden_size = int(cfg.gpt2_config.hidden_size)
        # text_lm_head is tied to transformer.wte → vocab = wte.num_embeddings
        self.vocab_size = int(lm.text_lm_head.out_features)

    def forward(
        self,
        global_hidden: torch.Tensor,            # [B, 768]
        text_token_id: torch.Tensor,            # [B]   int64
        audio_prefix_token_ids: torch.Tensor,   # [B, 15] int64
    ):
        B = global_hidden.shape[0]
        device = global_hidden.device

        local_dtype = self.lm.local_transformer.ln_f.weight.dtype

        text_emb = self.lm.transformer.wte(text_token_id.to(torch.long)).to(dtype=local_dtype)
        audio_embeds: List[torch.Tensor] = []
        for k in range(self.n_vq - 1):
            channel_ids = audio_prefix_token_ids[:, k].to(torch.long)
            emb = self.lm.audio_embeddings[k](channel_ids).to(dtype=local_dtype)
            audio_embeds.append(emb)

        # Build sequence: [global_hidden, text_emb, audio_emb_0..audio_emb_14]
        seq = [global_hidden.to(dtype=local_dtype), text_emb] + audio_embeds
        local_inputs_embeds = torch.stack(seq, dim=1)  # [B, 17, 768]

        attn = torch.ones(B, local_inputs_embeds.shape[1], dtype=torch.int64, device=device)
        out = self.lm.local_transformer(
            input_ids=None,
            inputs_embeds=local_inputs_embeds,
            attention_mask=attn,
            use_cache=False,
            return_dict=True,
        )
        local_hidden = out.last_hidden_state  # [B, 17, 768]

        # text logits from position 0
        text_logits = self.lm.text_lm_head(local_hidden[:, 0, :])  # [B, vocab]

        # audio logits per channel from position 1+k
        audio_logits_list: List[torch.Tensor] = []
        for k in range(self.n_vq):
            h_k = local_hidden[:, 1 + k, :]                          # [B, 768]
            logit = self.lm.audio_lm_heads[k](h_k)                   # [B, codebook_size]
            audio_logits_list.append(logit)
        audio_logits = torch.stack(audio_logits_list, dim=1)         # [B, 16, codebook]

        return text_logits, audio_logits


def io_names() -> tuple[list[str], list[str]]:
    return (
        ["global_hidden", "text_token_id", "audio_prefix_token_ids"],
        ["text_logits", "audio_logits"],
    )
