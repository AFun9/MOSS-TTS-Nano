"""Export MOSS-TTS-Nano to an ONNX bundle.

Produces ``onnx_export/`` containing:

    audio_encoder.onnx              Audio tokenizer encoder (used for the
                                    voice-clone prompt path)
    audio_decoder.onnx              Audio tokenizer decoder, exported with
                                    explicit KV-cache I/O (one frame per call)
    audio_decoder_state_spec.json   KV-cache layout descriptor
    global_transformer.onnx         12-layer GPT2 + token embeddings
                                    (last-token hidden state)
    local_decoder_text.onnx         1-layer local GPT2 + 2-row candidate wte
    local_decoder_audio.onnx        1-layer local GPT2 + embedded audio heads
                                    (hybrid input mode)
    tokenizer.model                 SentencePiece tokenizer
    config.json                     Inference parameters
    manifest.json                   Bundle metadata + file sizes

INT8 versions (``*_int8.onnx``) are produced via dynamic quantization with
codex-validated optimal settings (per_channel=False, reduce_range=False).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

NQ = 16
LOCAL_MAX_LEN = NQ + 1  # 17: fixed sequence length for local decoder ONNX
NUM_LAYERS = 12
NUM_HEADS = 12
HEAD_DIM = 64
HIDDEN = 768
OPSET = 17

# Audio tokenizer's `create_sin_embedding` is needed by the audio decoder
# wrapper. Make the tokenizer's source dir importable.
_AUDIO_TOKENIZER_ROOT = Path(__file__).resolve().parent / "MOSS-Audio-Tokenizer-Nano"
if _AUDIO_TOKENIZER_ROOT.is_dir() and str(_AUDIO_TOKENIZER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUDIO_TOKENIZER_ROOT))


# ---------------------------------------------------------------------------
# Wrapper 1: Audio Encoder
# ---------------------------------------------------------------------------
class AudioEncoderWrapper(nn.Module):
    """Wraps Audio Tokenizer encode path: encoder stack + RLFQ quantizer."""

    def __init__(self, audio_tokenizer):
        super().__init__()
        self.encoder = audio_tokenizer.encoder
        self.quantizer = audio_tokenizer.quantizer
        self.downsample_rate = audio_tokenizer.downsample_rate
        self.number_channels = audio_tokenizer.number_channels
        self.enable_channel_interleave = audio_tokenizer.enable_channel_interleave

    def forward(self, waveform: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform:      float32 [2, T] stereo 48kHz
            input_lengths: int64 [1] — pre-computed interleaved+padded length
                           (must be a tensor input so ONNX treats it as dynamic)
        Returns:
            audio_codes: int64 [16, 1, S]
        """
        x = waveform.unsqueeze(0)  # [1, 2, T]

        pad_rem = x.shape[-1] % self.downsample_rate
        if pad_rem != 0:
            x = F.pad(x, (0, self.downsample_rate - pad_rem))

        if self.number_channels > 1 and self.enable_channel_interleave:
            x = x.transpose(1, 2).contiguous().view(x.shape[0], 1, -1)

        h, h_len = x, input_lengths
        for enc_module in self.encoder:
            h, h_len = enc_module(h, h_len)

        z = self.quantizer.input_proj(h).float()
        batch_size, _, max_time = z.shape
        mask = torch.arange(max_time, device=z.device).expand(batch_size, max_time) < h_len.unsqueeze(1)

        residual = z.clone()
        all_indices = []
        for i, quant in enumerate(self.quantizer.quantizers):
            if i >= self.quantizer.num_quantizers:
                break
            masked_residual = residual * mask.unsqueeze(1)
            z_e = quant.in_proj(masked_residual)
            encodings = z_e.transpose(1, 2).reshape(-1, z_e.shape[1])
            codebook = quant.codebook.weight.float()
            encodings = F.normalize(encodings)
            codebook_n = F.normalize(codebook)
            dist = (
                encodings.pow(2).sum(1, keepdim=True)
                - 2 * encodings @ codebook_n.t()
                + codebook_n.pow(2).sum(1, keepdim=True).t()
            )
            indices = (-dist).max(1)[1]
            indices = indices.reshape(z_e.size(0), -1)

            z_q = F.embedding(indices, codebook).transpose(1, 2)
            z_q = quant.out_proj(z_q).float()

            update_mask = mask.unsqueeze(1)
            residual = residual - z_q * update_mask
            all_indices.append(indices)

        audio_codes = torch.stack(all_indices)  # [16, 1, S]
        return audio_codes


# ---------------------------------------------------------------------------
# Wrapper 2: Global Transformer (Embedding + 12-layer GPT2 w/ KV cache)
# ---------------------------------------------------------------------------
class GlobalTransformerWrapper(nn.Module):
    """Embedding layer + GPT2 global transformer with explicit KV cache I/O."""

    def __init__(self, tts_model, nq: int = NQ):
        super().__init__()
        self.transformer = tts_model.transformer
        self.text_embedding = tts_model.transformer.wte
        self.audio_embeddings = nn.ModuleList(
            list(tts_model.audio_embeddings[:nq])
        )
        self.audio_pad_token_id = tts_model.config.audio_pad_token_id
        self.nq = nq
        self.n_layers = len(self.transformer.h)

    def _build_inputs_embeds(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        text_ids = input_ids[..., 0]
        embeds = self.text_embedding(text_ids)
        for ch, emb_layer in enumerate(self.audio_embeddings):
            ch_ids = input_ids[..., ch + 1]
            valid = ch_ids.ne(self.audio_pad_token_id)
            safe = ch_ids.masked_fill(~valid, 0)
            a_emb = emb_layer(safe)
            embeds = embeds + a_emb * valid.unsqueeze(-1)
        return embeds

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        *past_kv_flat: torch.Tensor,
    ) -> tuple:
        """
        Args:
            input_ids:      [1, T, nq+1]
            attention_mask: [1, total_T]
            position_ids:   [1, T]
            past_kv_flat:   2*n_layers tensors, each [1, past_T, num_heads, head_dim]
        Returns:
            hidden_state, present_key_0, present_value_0, ..., present_key_N, present_value_N
        """
        past_key_values = tuple(
            (past_kv_flat[2 * i], past_kv_flat[2 * i + 1])
            for i in range(self.n_layers)
        )

        inputs_embeds = self._build_inputs_embeds(input_ids)

        outputs = self.transformer(
            input_ids=None,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cu_seqlens=None,
            num_sequences=None,
        )

        hidden = outputs.last_hidden_state[:, -1:, :]
        presents = outputs.past_key_values

        result = [hidden]
        for layer_kv in presents:
            result.append(layer_kv[0])
            result.append(layer_kv[1])
        return tuple(result)


# ---------------------------------------------------------------------------
# Wrapper 3: Local Decoder — Shared base for Text and Audio heads
# ---------------------------------------------------------------------------
class _LocalDecoderBase(nn.Module):
    """Shared single-layer transformer logic for both local decoders."""

    def __init__(self, tts_model):
        super().__init__()
        lt = tts_model.local_transformer
        block = lt.h[0]
        attn = block.attn
        self.ln_1 = block.ln_1
        self.ln_2 = block.ln_2
        self.ln_f = lt.ln_f
        self.c_attn = attn.c_attn
        self.c_proj = attn.c_proj
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.hidden_size = attn.embed_dim
        if attn.rotary_emb is not None:
            self.register_buffer("rotary_inv_freq", attn.rotary_emb.inv_freq.clone())
        else:
            self.rotary_inv_freq = None
        self.mlp_fc_in = block.mlp.fc_in
        self.mlp_fc_out = block.mlp.fc_out
        self.mlp_act = block.mlp.act

    def _local_forward(
        self,
        h: torch.FloatTensor,
        position_id: torch.LongTensor,
        past_key: torch.FloatTensor,
        past_value: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run single-layer attention + MLP, return (hidden [1,H], key, value)."""
        H = self.hidden_size
        normed = self.ln_1(h)
        qkv = self.c_attn(normed)
        q, k, v = qkv.split(H, dim=-1)
        q = q.view(1, 1, self.num_heads, self.head_dim)
        k = k.view(1, 1, self.num_heads, self.head_dim)
        v = v.view(1, 1, self.num_heads, self.head_dim)
        if self.rotary_inv_freq is not None:
            half_dim = self.rotary_inv_freq.shape[0]
            freqs = torch.einsum("bs,d->bsd", position_id.float(), self.rotary_inv_freq)
            freqs_full = freqs.unsqueeze(-1).expand(-1, -1, -1, 2).reshape(1, 1, half_dim * 2)
            cos = freqs_full.cos().unsqueeze(2).to(dtype=q.dtype)
            sin = freqs_full.sin().unsqueeze(2).to(dtype=q.dtype)
            even_q, odd_q = q[..., ::2], q[..., 1::2]
            q = q * cos + torch.stack((-odd_q, even_q), dim=-1).reshape_as(q) * sin
            even_k, odd_k = k[..., ::2], k[..., 1::2]
            k = k * cos + torch.stack((-odd_k, even_k), dim=-1).reshape_as(k) * sin
        k = torch.cat([past_key, k], dim=1)
        v = torch.cat([past_value, v], dim=1)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(q_t, k_t.transpose(-1, -2)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v_t)
        attn_out = attn_out.transpose(1, 2).reshape(1, 1, H)
        attn_out = self.c_proj(attn_out)
        h = h + attn_out
        mlp_out = self.mlp_fc_in(self.ln_2(h))
        mlp_out = self.mlp_act(mlp_out)
        mlp_out = self.mlp_fc_out(mlp_out)
        h = h + mlp_out
        h = self.ln_f(h)
        hidden = h.squeeze(1)  # [1, H]
        return hidden, k, v


class LocalDecoderTextWrapper(_LocalDecoderBase):
    """1-layer local transformer for text token prediction (1x per frame).

    Only embeds the 2 candidate token rows from wte (audio_slot, audio_end),
    not the full 16384-row table. Outputs:
      - candidate_logits [1, 2]: logits for [audio_slot, audio_end]
      - candidate_embeds [2, H]: embeddings for both candidates (caller picks
        one after sampling and passes it to the audio model)
    """

    def __init__(self, tts_model):
        super().__init__(tts_model)
        cfg = tts_model.config
        slot_id = cfg.audio_assistant_slot_token_id
        end_id = cfg.audio_end_token_id
        full_wte = tts_model.transformer.wte.weight.detach()
        self.candidate_wte = nn.Parameter(
            full_wte[[slot_id, end_id]].clone(), requires_grad=False,
        )  # [2, H] — row 0 = audio_slot, row 1 = audio_end

    def forward(
        self,
        input_embed: torch.FloatTensor,   # [1, 1, H]
        position_id: torch.LongTensor,     # [1, 1]
        past_key: torch.FloatTensor,       # [1, past_T, num_heads, head_dim]
        past_value: torch.FloatTensor,
    ) -> tuple:
        """Returns (candidate_logits [1, 2], candidate_embeds [2, H],
                    present_key, present_value)."""
        hidden, k, v = self._local_forward(input_embed, position_id, past_key, past_value)
        logits = torch.mm(hidden, self.candidate_wte.t())  # [1, 2]
        return logits, self.candidate_wte, k, v


class LocalDecoderAudioWrapper(_LocalDecoderBase):
    """1-layer local transformer for audio channel prediction (16x per frame).

    Hybrid input resolution:
    - use_input_embed=1: uses externally provided input_embed
    - use_input_embed=0: performs internal Gather from audio_lm_heads
    """

    def __init__(self, tts_model, nq: int = NQ):
        super().__init__(tts_model)
        audio_weights = torch.stack([
            tts_model.audio_embeddings[i].weight.detach().clone()
            for i in range(nq)
        ])  # [nq, audio_V, H]
        self.audio_lm_heads = nn.Parameter(audio_weights, requires_grad=False)

    def forward(
        self,
        input_embed: torch.FloatTensor,       # [1, 1, H] — external embed
        position_id: torch.LongTensor,        # [1, 1]
        past_key: torch.FloatTensor,          # [1, past_T, ...]
        past_value: torch.FloatTensor,
        head_id: torch.LongTensor,            # [1] — 1..16 audio channel
        use_input_embed: torch.LongTensor,    # [1] — 1=use input_embed, 0=internal lookup
        prev_audio_ch: torch.LongTensor,      # [1] — 0-indexed audio ch for lookup
        prev_token_id: torch.LongTensor,      # [1] — token id for internal lookup
    ) -> tuple:
        """Returns (audio_logits [1, audio_V], present_key, present_value)."""
        looked_up = self.audio_lm_heads[prev_audio_ch[0], prev_token_id[0]]
        ext = input_embed.squeeze(0).squeeze(0)
        h = torch.where(use_input_embed.bool(), ext, looked_up)
        h = h.unsqueeze(0).unsqueeze(0)  # [1, 1, H]

        hidden, k, v = self._local_forward(h, position_id, past_key, past_value)

        audio_idx = (head_id[0] - 1).clamp(min=0)
        audio_w = self.audio_lm_heads[audio_idx]
        audio_logits = torch.mm(hidden, audio_w.t())  # [1, audio_V]
        return audio_logits, k, v


# ---------------------------------------------------------------------------
# Export Helpers
# ---------------------------------------------------------------------------
def set_attn_implementation(model, impl: str = "sdpa"):
    """Recursively switch all attention layers to the given implementation."""
    for module in model.modules():
        if hasattr(module, "attn_implementation"):
            module.attn_implementation = impl


def export_audio_encoder(audio_tokenizer, output_dir: Path, device: torch.device):
    log.info("Exporting audio_encoder.onnx ...")
    wrapper = AudioEncoderWrapper(audio_tokenizer).to(device).eval()
    set_attn_implementation(wrapper, "sdpa")

    T = 48000  # 1 second of audio
    dummy_waveform = torch.randn(2, T, device=device, dtype=torch.float32)

    # Compute the actual interleaved+padded length for the dummy input
    ds = wrapper.downsample_rate
    interleaved_len = T * (wrapper.number_channels if wrapper.enable_channel_interleave else 1)
    pad_rem = interleaved_len % ds
    if pad_rem != 0:
        interleaved_len += ds - pad_rem
    dummy_input_lengths = torch.tensor([interleaved_len], dtype=torch.long, device=device)

    onnx_path = output_dir / "audio_encoder.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_waveform, dummy_input_lengths),
        str(onnx_path),
        opset_version=OPSET,
        input_names=["waveform", "input_lengths"],
        output_names=["audio_codes"],
        dynamic_axes={
            "waveform": {1: "waveform_length"},
            "audio_codes": {2: "token_length"},
        },
    )
    log.info("  -> %s (%.1f MB)", onnx_path, onnx_path.stat().st_size / 1e6)
    return onnx_path


def _patch_gpt2_for_onnx(module: nn.Module):
    """Monkey-patch GPT2 internals for clean ONNX tracing.

    Fixes two classes of ONNX export bugs:
    1. repeat_interleave in RoPE → expand+reshape (avoids constant-folding corruption)
    2. Python max() in _causal_attention_mask → tensor clamp (avoids frozen offset
       that breaks KV-cache decode steps)
    """
    import types
    import importlib

    gpt2_mod = importlib.import_module(
        "transformers_modules.MOSS_hyphen_TTS_hyphen_Nano_hyphen_100M.gpt2_decoder"
    )

    # --- Fix 1: RoPE repeat_interleave ---
    def _safe_rotary_forward(self, position_ids, *, device, dtype):
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        pos = position_ids.to(device=device, dtype=self.inv_freq.dtype)
        freqs = pos.unsqueeze(-1) * self.inv_freq.unsqueeze(0)  # [B, S, half]
        c = freqs.cos()
        s = freqs.sin()
        half = c.shape[-1]
        cos = c.unsqueeze(-1).expand(-1, -1, -1, 2).reshape(
            c.shape[0], c.shape[1], half * 2
        ).unsqueeze(2).to(dtype=dtype)
        sin = s.unsqueeze(-1).expand(-1, -1, -1, 2).reshape(
            s.shape[0], s.shape[1], half * 2
        ).unsqueeze(2).to(dtype=dtype)
        return cos, sin

    def _safe_rotate_half(hidden_states):
        even = hidden_states[..., ::2]
        odd = hidden_states[..., 1::2]
        neg_odd = -odd
        interleaved = torch.stack([neg_odd, even], dim=-1)
        return interleaved.reshape(
            interleaved.shape[:-2] + (interleaved.shape[-2] * 2,)
        )

    # --- Fix 2: causal mask offset uses tensor ops instead of Python max() ---
    def _safe_causal_attention_mask(self, attention_mask, query_length, key_length, device):
        query_positions = torch.arange(query_length, device=device, dtype=torch.long)
        key_positions = torch.arange(key_length, device=device, dtype=torch.long)
        offset = torch.clamp(key_positions[-1:] - query_positions[-1:], min=0)
        query_positions = query_positions + offset
        causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        causal = causal.unsqueeze(0).unsqueeze(0)
        if attention_mask is None:
            return causal
        key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
        return causal & key_mask

    for m in module.modules():
        cls_name = type(m).__name__
        if cls_name == "MossTTSNanoGPT2RotaryEmbedding":
            m.forward = types.MethodType(_safe_rotary_forward, m)
        if cls_name == "MossTTSNanoGPT2Attention":
            m._causal_attention_mask = types.MethodType(
                _safe_causal_attention_mask, m
            )

    gpt2_mod.rotate_half = _safe_rotate_half

    return gpt2_mod


def export_global_transformer(tts_model, output_dir: Path, device: torch.device):
    log.info("Exporting global_transformer.onnx ...")
    wrapper = GlobalTransformerWrapper(tts_model, nq=NQ).to(device).eval()
    set_attn_implementation(wrapper, "eager")
    _patch_gpt2_for_onnx(wrapper)

    n_layers = wrapper.n_layers
    T = 10
    dummy_input_ids = torch.zeros(1, T, NQ + 1, device=device, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, T, device=device, dtype=torch.bool)
    dummy_position_ids = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0)

    past_kv_flat = []
    for _ in range(n_layers):
        past_kv_flat.append(torch.zeros(1, 0, NUM_HEADS, HEAD_DIM, device=device))
        past_kv_flat.append(torch.zeros(1, 0, NUM_HEADS, HEAD_DIM, device=device))

    input_names = ["input_ids", "attention_mask", "position_ids"]
    output_names = ["hidden_state"]
    dynamic_axes = {
        "input_ids": {1: "seq_len"},
        "attention_mask": {1: "total_seq_len"},
        "position_ids": {1: "seq_len"},
        "hidden_state": {1: "seq_len"},
    }
    for i in range(n_layers):
        input_names.append(f"past_key_{i}")
        input_names.append(f"past_value_{i}")
        output_names.append(f"present_key_{i}")
        output_names.append(f"present_value_{i}")
        dynamic_axes[f"past_key_{i}"] = {1: "past_seq_len"}
        dynamic_axes[f"past_value_{i}"] = {1: "past_seq_len"}
        dynamic_axes[f"present_key_{i}"] = {1: "total_seq_len"}
        dynamic_axes[f"present_value_{i}"] = {1: "total_seq_len"}

    args = (dummy_input_ids, dummy_attention_mask, dummy_position_ids, *past_kv_flat)

    onnx_path = output_dir / "global_transformer.onnx"
    torch.onnx.export(
        wrapper,
        args,
        str(onnx_path),
        opset_version=OPSET,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    log.info("  -> %s (%.1f MB)", onnx_path, onnx_path.stat().st_size / 1e6)
    return onnx_path


def export_local_decoder_text(tts_model, output_dir: Path, device: torch.device):
    log.info("Exporting local_decoder_text.onnx (wte embedded, 1x/frame) ...")
    wrapper = LocalDecoderTextWrapper(tts_model).to(device).eval()

    dummy_embed = torch.randn(1, 1, HIDDEN, device=device)
    dummy_pos = torch.zeros(1, 1, device=device, dtype=torch.long)
    dummy_past_k = torch.zeros(1, 0, NUM_HEADS, HEAD_DIM, device=device)
    dummy_past_v = torch.zeros(1, 0, NUM_HEADS, HEAD_DIM, device=device)

    onnx_path = output_dir / "local_decoder_text.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_embed, dummy_pos, dummy_past_k, dummy_past_v),
        str(onnx_path),
        opset_version=OPSET,
        input_names=["input_embed", "position_id", "past_key", "past_value"],
        output_names=["candidate_logits", "candidate_embeds", "present_key", "present_value"],
        dynamic_axes={
            "past_key": {1: "past_seq_len"},
            "past_value": {1: "past_seq_len"},
            "present_key": {1: "total_seq_len"},
            "present_value": {1: "total_seq_len"},
        },
    )
    log.info("  -> %s (%.1f MB)", onnx_path, onnx_path.stat().st_size / 1e6)
    return onnx_path


def export_local_decoder_audio(tts_model, output_dir: Path, device: torch.device):
    log.info("Exporting local_decoder_audio.onnx (audio heads embedded, no wte_lookup) ...")
    wrapper = LocalDecoderAudioWrapper(tts_model, nq=NQ).to(device).eval()

    dummy_embed = torch.randn(1, 1, HIDDEN, device=device)
    dummy_pos = torch.zeros(1, 1, device=device, dtype=torch.long)
    dummy_past_k = torch.zeros(1, 0, NUM_HEADS, HEAD_DIM, device=device)
    dummy_past_v = torch.zeros(1, 0, NUM_HEADS, HEAD_DIM, device=device)
    dummy_head_id = torch.tensor([1], device=device, dtype=torch.long)
    dummy_use_ext = torch.tensor([1], device=device, dtype=torch.long)
    dummy_prev_ch = torch.tensor([0], device=device, dtype=torch.long)
    dummy_prev_tok = torch.tensor([0], device=device, dtype=torch.long)

    onnx_path = output_dir / "local_decoder_audio.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_embed, dummy_pos, dummy_past_k, dummy_past_v,
         dummy_head_id, dummy_use_ext, dummy_prev_ch, dummy_prev_tok),
        str(onnx_path),
        opset_version=OPSET,
        input_names=["input_embed", "position_id", "past_key", "past_value",
                     "head_id", "use_input_embed", "prev_audio_ch", "prev_token_id"],
        output_names=["audio_logits", "present_key", "present_value"],
        dynamic_axes={
            "past_key": {1: "past_seq_len"},
            "past_value": {1: "past_seq_len"},
            "present_key": {1: "total_seq_len"},
            "present_value": {1: "total_seq_len"},
        },
    )
    log.info("  -> %s (%.1f MB)", onnx_path, onnx_path.stat().st_size / 1e6)
    return onnx_path


# ---------------------------------------------------------------------------
# Wrapper 4: Audio Decoder (one frame per call, explicit KV-cache I/O)
# ---------------------------------------------------------------------------
# The audio tokenizer is a causal model whose decoder is naturally autoregressive
# over time. We export it with the KV cache exposed as ONNX inputs/outputs so a
# single ORT call processes one code frame in O(1) given the previous state.

@dataclass
class AttentionStateSpec:
    decoder_module_index: int
    layer_index: int
    context: int
    num_heads: int
    head_dim: int

    @property
    def prefix(self) -> str:
        return f"decoder_{self.decoder_module_index}_layer_{self.layer_index}"


@dataclass
class TransformerStateSpec:
    decoder_module_index: int
    num_layers: int

    @property
    def prefix(self) -> str:
        return f"decoder_{self.decoder_module_index}"


def _collect_state_specs(model) -> tuple[list[TransformerStateSpec], list[AttentionStateSpec]]:
    transformer_specs: list[TransformerStateSpec] = []
    attention_specs: list[AttentionStateSpec] = []
    for decoder_module_index, decoder_module in enumerate(model.decoder):
        transformer = getattr(decoder_module, "transformer", None)
        if transformer is None:
            continue
        transformer_specs.append(
            TransformerStateSpec(
                decoder_module_index=decoder_module_index,
                num_layers=len(transformer.layers),
            )
        )
        for layer_index, layer in enumerate(transformer.layers):
            self_attn = layer.self_attn
            attention_specs.append(
                AttentionStateSpec(
                    decoder_module_index=decoder_module_index,
                    layer_index=layer_index,
                    context=int(self_attn.context),
                    num_heads=int(self_attn.num_heads),
                    head_dim=int(self_attn.embed_dim // self_attn.num_heads),
                )
            )
    return transformer_specs, attention_specs


class AudioDecoderWrapper(nn.Module):
    """Audio-decoder wrapper for ONNX export with explicit KV-cache I/O.

    Forward:
        codes  : int64 [num_quantizers, batch=1, frames=1]
        states : KV cache + per-stage offset tensors, in canonical order
    Returns:
        audio              : float32 [batch=1, channels, samples]
        audio_lengths      : int64 [batch=1]
        next_state_tensors : same canonical order as inputs
    """

    def __init__(self, audio_tokenizer, *, num_quantizers: int = NQ) -> None:
        super().__init__()
        self.quantizer = audio_tokenizer.quantizer
        self.decoder = audio_tokenizer.decoder
        self.number_channels = int(audio_tokenizer.number_channels)
        self.enable_channel_interleave = bool(audio_tokenizer.enable_channel_interleave)
        self.sampling_rate = int(audio_tokenizer.sampling_rate)
        self.downsample_rate = int(audio_tokenizer.downsample_rate)
        self.num_quantizers = int(num_quantizers)

        self.transformer_specs, self.attention_specs = _collect_state_specs(audio_tokenizer)

    def _attention_step(self, self_attn, x, *,
                        cached_keys, cached_values, cached_positions, offset):
        batch_size, chunk_length, _ = x.shape
        q, k_cur, v_cur = self_attn._project_qkv(x)
        if self_attn.rope is not None:
            q, k_cur = self_attn.rope(q, k_cur, offset, time_before_heads=False)
        pos_q = offset.view(-1, 1) + torch.arange(chunk_length, device=x.device, dtype=torch.long).view(1, -1)
        k_all = torch.cat([cached_keys, k_cur], dim=2)
        v_all = torch.cat([cached_values, v_cur], dim=2)
        pos_k = torch.cat([cached_positions, pos_q], dim=1)
        attn_bias = self_attn._build_streaming_sdpa_bias(pos_q, pos_k)
        out = F.scaled_dot_product_attention(q, k_all, v_all, attn_bias, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(batch_size, chunk_length, self_attn.embed_dim)
        out = self_attn.out_proj(out)

        context = int(self_attn.context)
        new_cached_keys = k_all[:, :, -context:, :].contiguous()
        new_cached_values = v_all[:, :, -context:, :].contiguous()
        new_cached_positions = pos_k[:, -context:].contiguous()
        new_offset = offset + chunk_length
        return out, new_cached_keys, new_cached_values, new_cached_positions, new_offset

    def _projected_transformer_step(self, module, x, input_lengths, *,
                                     transformer_offset, layer_states):
        from modeling_moss_audio_tokenizer import create_sin_embedding

        x = module.input_proj(x.transpose(1, 2))
        transformer = module.transformer
        if transformer.positional_embedding in {"sin", "sin_rope"}:
            positions = torch.arange(x.shape[1], device=x.device).view(1, -1) + transformer_offset.view(-1, 1)
            x = x + transformer.positional_scale * create_sin_embedding(
                positions, x.shape[-1], max_period=transformer.max_period, dtype=x.dtype,
            )

        next_layer_states = []
        for layer_index, layer in enumerate(transformer.layers):
            cached_keys, cached_values, cached_positions, offset = layer_states[layer_index]
            residual = x
            normed = layer.norm1(x)
            attn_out, new_ck, new_cv, new_cp, new_off = self._attention_step(
                layer.self_attn, normed,
                cached_keys=cached_keys, cached_values=cached_values,
                cached_positions=cached_positions, offset=offset,
            )
            x = residual.to(attn_out) + layer.layer_scale_1(attn_out)
            residual = x
            normed = layer.norm2(x)
            x = residual.to(normed) + layer.layer_scale_2(layer.ffn(normed))
            next_layer_states.append((new_ck, new_cv, new_cp, new_off))

        next_transformer_offset = transformer_offset + x.shape[1]
        x = module.output_proj(x).transpose(1, 2)
        return x, input_lengths, next_transformer_offset, next_layer_states

    def _restore_channels(self, output_values, output_lengths):
        if self.number_channels == 1 or not self.enable_channel_interleave:
            return output_values.float(), output_lengths
        output_values = (
            output_values.squeeze(1).contiguous()
            .view(output_values.shape[0], -1, self.number_channels)
            .transpose(1, 2).contiguous().float()
        )
        output_lengths = torch.div(output_lengths, self.number_channels, rounding_mode="floor")
        return output_values, output_lengths

    def forward(self, codes: torch.Tensor, *state_inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if codes.dim() != 3:
            raise ValueError(f"Expected codes shape [nq, 1, 1], got {tuple(codes.shape)}")
        device = codes.device
        batch_size = codes.shape[1]
        if batch_size != 1:
            raise ValueError(f"Only batch_size=1 supported, got {batch_size}")

        input_lengths = torch.ones((batch_size,), device=device, dtype=torch.long)
        x = self.quantizer.decode_codes(codes[: self.num_quantizers]).float()

        attention_modules: dict[int, list[AttentionStateSpec]] = {}
        for spec in self.attention_specs:
            attention_modules.setdefault(spec.decoder_module_index, []).append(spec)

        state_iter = iter(state_inputs)
        output_state_tensors: list[torch.Tensor] = []
        for decoder_module_index, decoder_module in enumerate(self.decoder):
            if getattr(decoder_module, "transformer", None) is None:
                x, input_lengths = decoder_module(x, input_lengths)
                continue
            transformer_offset = next(state_iter)
            layer_states = []
            for _ in attention_modules[decoder_module_index]:
                ck = next(state_iter); cv = next(state_iter)
                cp = next(state_iter); off = next(state_iter)
                layer_states.append((ck, cv, cp, off))

            x, input_lengths, next_off, new_layer_states = self._projected_transformer_step(
                decoder_module, x, input_lengths,
                transformer_offset=transformer_offset, layer_states=layer_states,
            )
            output_state_tensors.append(next_off)
            for new_ck, new_cv, new_cp, new_off in new_layer_states:
                output_state_tensors.extend([new_ck, new_cv, new_cp, new_off])

        audio, audio_lengths = self._restore_channels(x, input_lengths)
        return (audio, audio_lengths, *output_state_tensors)


def _build_audio_decoder_io(transformer_specs, attention_specs):
    """Return (input_names, dummy_inputs, output_names) in canonical order."""
    input_names: list[str] = ["codes"]
    inputs: list[torch.Tensor] = [torch.zeros((NQ, 1, 1), dtype=torch.long)]
    output_names: list[str] = ["audio", "audio_lengths"]

    grouped: dict[int, list[AttentionStateSpec]] = {}
    for spec in attention_specs:
        grouped.setdefault(spec.decoder_module_index, []).append(spec)

    for spec in transformer_specs:
        input_names.append(f"{spec.prefix}_transformer_offset")
        inputs.append(torch.zeros((1,), dtype=torch.long))
        output_names.append(f"new_{spec.prefix}_transformer_offset")
        for attn_spec in grouped[spec.decoder_module_index]:
            p = attn_spec.prefix
            input_names.extend([f"{p}_cached_keys", f"{p}_cached_values",
                                 f"{p}_cached_positions", f"{p}_offset"])
            inputs.extend([
                torch.zeros((1, attn_spec.num_heads, attn_spec.context, attn_spec.head_dim), dtype=torch.float32),
                torch.zeros((1, attn_spec.num_heads, attn_spec.context, attn_spec.head_dim), dtype=torch.float32),
                torch.full((1, attn_spec.context), -1, dtype=torch.long),
                torch.zeros((1,), dtype=torch.long),
            ])
            output_names.extend([f"new_{p}_cached_keys", f"new_{p}_cached_values",
                                  f"new_{p}_cached_positions", f"new_{p}_offset"])
    return input_names, inputs, output_names


def export_audio_decoder(audio_tokenizer, output_dir: Path,
                          device: torch.device) -> dict[str, Any]:
    """Export the audio decoder ONNX + state-spec sidecar.

    Reuses the already-loaded ``audio_tokenizer`` and writes both
    ``audio_decoder.onnx`` and ``audio_decoder_state_spec.json`` into
    ``output_dir``.
    """
    log.info("Exporting audio_decoder.onnx ...")
    wrapper = AudioDecoderWrapper(audio_tokenizer, num_quantizers=NQ).to(device).eval()
    set_attn_implementation(wrapper, "sdpa")

    input_names, inputs, output_names = _build_audio_decoder_io(
        wrapper.transformer_specs, wrapper.attention_specs,
    )
    inputs = [t.to(device) for t in inputs]
    log.info("  state spec: %d transformer stages, %d attention layers",
             len(wrapper.transformer_specs), len(wrapper.attention_specs))

    fp32_path = output_dir / "audio_decoder.onnx"
    torch.onnx.export(
        wrapper, tuple(inputs), str(fp32_path),
        opset_version=OPSET,
        input_names=input_names, output_names=output_names,
        dynamic_axes=None, do_constant_folding=True,
    )
    log.info("  -> %s (%.1f MB)", fp32_path, fp32_path.stat().st_size / 1e6)

    state_spec = {
        "version": "v1",
        "model_path_fp32": fp32_path.name,
        "model_path_int8": "audio_decoder_int8.onnx",
        "num_quantizers": int(wrapper.num_quantizers),
        "batch_size": 1,
        "frames_per_call": 1,
        "sample_rate": int(wrapper.sampling_rate),
        "downsample_rate": int(wrapper.downsample_rate),
        "input_names": input_names,
        "output_names": output_names,
        "transformer_specs": [spec.__dict__ for spec in wrapper.transformer_specs],
        "attention_specs": [spec.__dict__ for spec in wrapper.attention_specs],
    }
    spec_path = output_dir / "audio_decoder_state_spec.json"
    spec_path.write_text(json.dumps(state_spec, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("  -> %s", spec_path.name)
    return state_spec


def _verify_audio_decoder(audio_tokenizer, fp32_path: Path, state_spec: dict,
                           device: torch.device) -> float:
    """Smoke-test ORT vs PyTorch wrapper on a single random frame."""
    import onnxruntime as ort

    wrapper = AudioDecoderWrapper(audio_tokenizer, num_quantizers=NQ).to(device).eval()
    set_attn_implementation(wrapper, "sdpa")

    rng = np.random.default_rng(1234)
    codebook_size = int(audio_tokenizer.quantizer.codebook_size)
    feed = {"codes": rng.integers(0, codebook_size, size=(NQ, 1, 1), dtype=np.int64)}
    grouped: dict[int, list[dict]] = {}
    for s in state_spec["attention_specs"]:
        grouped.setdefault(int(s["decoder_module_index"]), []).append(s)
    for ts in state_spec["transformer_specs"]:
        di = int(ts["decoder_module_index"])
        feed[f"decoder_{di}_transformer_offset"] = np.zeros((1,), dtype=np.int64)
        for a in grouped[di]:
            p = f"decoder_{a['decoder_module_index']}_layer_{a['layer_index']}"
            nh, ctx, hd = int(a["num_heads"]), int(a["context"]), int(a["head_dim"])
            feed[f"{p}_cached_keys"] = np.zeros((1, nh, ctx, hd), dtype=np.float32)
            feed[f"{p}_cached_values"] = np.zeros((1, nh, ctx, hd), dtype=np.float32)
            feed[f"{p}_cached_positions"] = np.full((1, ctx), -1, dtype=np.int64)
            feed[f"{p}_offset"] = np.zeros((1,), dtype=np.int64)

    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    ort_audio = sess.run(None, feed)[0]
    with torch.no_grad():
        torch_inputs = [torch.from_numpy(feed[name]).to(device) for name in state_spec["input_names"]]
        torch_audio = wrapper(*torch_inputs)[0].detach().cpu().numpy()
    diff = float(np.max(np.abs(ort_audio.astype(np.float32) - torch_audio)))
    status = "PASS" if diff < 1e-3 else "FAIL"
    log.info("Verifying audio_decoder ... max_diff=%.6e (%s)", diff, status)
    return diff


def optimize_model_graphs(output_dir: Path):
    """Apply graph-level optimizations: operator fusion, constant folding."""
    log.info("Running graph optimization ...")

    # GPT2-specific fusions for global transformer
    from onnxruntime.transformers import optimizer
    gt_path = str(output_dir / "global_transformer.onnx")
    opt = optimizer.optimize_model(
        gt_path, model_type="gpt2",
        num_heads=NUM_HEADS, hidden_size=HIDDEN, opt_level=2,
    )
    stats = opt.get_fused_operator_statistics()
    fused = {k: v for k, v in stats.items() if v > 0}
    opt.save_model_to_file(gt_path)
    log.info("  global_transformer fusions: %s", fused)

    # Generic ORT optimization for other models (pre-bake fused ops)
    import onnxruntime as ort
    for name in ["local_decoder_text", "local_decoder_audio", "audio_encoder"]:
        src = output_dir / f"{name}.onnx"
        if not src.exists():
            continue
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        dst = str(output_dir / f"{name}_opt.onnx")
        opts.optimized_model_filepath = dst
        ort.InferenceSession(str(src), opts, providers=["CPUExecutionProvider"])
        if Path(dst).exists():
            Path(dst).replace(src)
            log.info("  %s optimized", name)


def export_tokenizer_and_config(tts_model, tts_checkpoint: str, output_dir: Path, nq: int = NQ):
    log.info("Exporting tokenizer and config ...")
    src_dir = Path(tts_checkpoint)
    tokenizer_src = src_dir / "tokenizer.model"
    if tokenizer_src.exists():
        shutil.copy2(tokenizer_src, output_dir / "tokenizer.model")
        log.info("  -> tokenizer.model copied")
    else:
        log.warning("  tokenizer.model not found at %s", tokenizer_src)

    cfg = tts_model.config
    export_config = {
        "nq": nq,
        "n_vq": cfg.n_vq,
        "hidden_size": cfg.gpt2_config.hidden_size,
        "num_layers": cfg.gpt2_config.n_layer,
        "num_heads": cfg.gpt2_config.num_attention_heads,
        "head_dim": cfg.gpt2_config.hidden_size // cfg.gpt2_config.num_attention_heads,
        "vocab_size": cfg.gpt2_config.vocab_size,
        "audio_codebook_size": cfg.audio_codebook_sizes[0],
        "audio_pad_token_id": cfg.audio_pad_token_id,
        "pad_token_id": cfg.pad_token_id,
        "im_start_token_id": cfg.im_start_token_id,
        "im_end_token_id": cfg.im_end_token_id,
        "audio_start_token_id": cfg.audio_start_token_id,
        "audio_end_token_id": cfg.audio_end_token_id,
        "audio_user_slot_token_id": cfg.audio_user_slot_token_id,
        "audio_assistant_slot_token_id": cfg.audio_assistant_slot_token_id,
        "audio_tokenizer_sample_rate": cfg.audio_tokenizer_sample_rate,
        "audio_tokenizer_downsample_rate": 3840,
        "audio_tokenizer_num_channels": 2,
        "sampling_defaults": {
            "text_temperature": 1.0,
            "text_top_p": 1.0,
            "text_top_k": 50,
            "audio_temperature": 0.8,
            "audio_top_p": 0.95,
            "audio_top_k": 25,
            "audio_repetition_penalty": 1.2,
        },
    }
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(export_config, indent=2, ensure_ascii=False))
    log.info("  -> config.json")


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------
def verify_module(name: str, onnx_path: Path, wrapper: nn.Module,
                  dummy_inputs: tuple, output_count: int, device: torch.device):
    """Compare ONNX Runtime output against PyTorch output."""
    import onnxruntime as ort

    log.info("Verifying %s ...", name)
    with torch.no_grad():
        pt_outputs = wrapper(*dummy_inputs)
    if not isinstance(pt_outputs, tuple):
        pt_outputs = (pt_outputs,)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {}
    for inp, meta in zip(dummy_inputs, sess.get_inputs()):
        val = inp.detach().cpu().numpy() if isinstance(inp, torch.Tensor) else inp
        ort_inputs[meta.name] = val

    ort_outputs = sess.run(None, ort_inputs)

    max_diff = 0.0
    for idx in range(min(len(pt_outputs), len(ort_outputs), output_count)):
        pt_val = pt_outputs[idx].detach().cpu().float().numpy() if isinstance(pt_outputs[idx], torch.Tensor) else pt_outputs[idx]
        ort_val = ort_outputs[idx]
        if pt_val.shape != ort_val.shape:
            log.warning("  output[%d] shape mismatch: PT %s vs ORT %s", idx, pt_val.shape, ort_val.shape)
            continue
        diff = float(np.max(np.abs(pt_val - ort_val.astype(np.float32))))
        max_diff = max(max_diff, diff)

    status = "PASS" if max_diff < 1e-3 else "FAIL"
    log.info("  %s: max_diff=%.6f (%s)", name, max_diff, status)
    return max_diff


def _make_local_kv(past_T: int, device: torch.device):
    """Create past key/value pair for local decoder verification."""
    if past_T > 0:
        pk = torch.randn(1, past_T, NUM_HEADS, HEAD_DIM, device=device)
        pv = torch.randn(1, past_T, NUM_HEADS, HEAD_DIM, device=device)
    else:
        pk = torch.zeros(1, 0, NUM_HEADS, HEAD_DIM, device=device)
        pv = pk
    return pk, pv


def _compute_interleaved_length(raw_len: int, ds: int, n_ch: int) -> int:
    """Compute padded interleaved length for audio encoder verification."""
    il = raw_len * n_ch
    pad_r = il % ds
    if pad_r != 0:
        il += ds - pad_r
    return il


# ---------------------------------------------------------------------------
# Quantization & manifest helpers
# ---------------------------------------------------------------------------
# Settings validated as smallest *and* fastest by codex review for these graphs:
# per-tensor INT8 weights, full op coverage, no reduced range.
QUANT_KWARGS = dict(per_channel=False, reduce_range=False)

# All exportable models in the bundle (without ``_int8`` suffix).
BUNDLE_MODELS = [
    "audio_encoder",
    "global_transformer",
    "local_decoder_text",
    "local_decoder_audio",
    "audio_decoder",
]


def _quantize_models(output_dir: Path, model_names: list[str]) -> None:
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType

    log.info("=" * 60)
    log.info("Running INT8 dynamic quantization (%s) ...", QUANT_KWARGS)
    for mname in model_names:
        fp32 = output_dir / f"{mname}.onnx"
        int8 = output_dir / f"{mname}_int8.onnx"
        if not fp32.exists():
            log.warning("  FP32 missing: %s (skip)", fp32.name)
            continue
        log.info("  %s.onnx", mname)
        quantize_dynamic(
            str(fp32), str(int8),
            weight_type=QuantType.QInt8,
            extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
            **QUANT_KWARGS,
        )
        a = fp32.stat().st_size / 1e6
        b = int8.stat().st_size / 1e6
        log.info("    %.1f MB -> %.1f MB (%.0f%% reduction)", a, b, (1 - b / a) * 100)


def _write_manifest(output_dir: Path) -> None:
    """Emit manifest.json describing the bundle so consumers can introspect it."""
    entries = []
    for f in sorted(output_dir.iterdir()):
        if not f.is_file() or f.name == "manifest.json":
            continue
        entries.append({
            "name": f.name,
            "size_bytes": f.stat().st_size,
            "size_mb": round(f.stat().st_size / 1e6, 3),
        })
    manifest = {
        "schema": "moss-tts-nano-onnx-bundle/v2",
        "quantization": {"weight_type": "QInt8", **QUANT_KWARGS},
        "files": entries,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote manifest -> %s (%d files)", path, len(entries))


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------
def _verify_audio_encoder(audio_tokenizer, enc_path: Path, nq: int, device: torch.device):
    enc_wrapper = AudioEncoderWrapper(audio_tokenizer).to(device).eval()
    set_attn_implementation(enc_wrapper, "sdpa")
    ds = enc_wrapper.downsample_rate
    n_ch = enc_wrapper.number_channels if enc_wrapper.enable_channel_interleave else 1
    for test_dur in (48000, 48000 * 4, 48000 * 8):
        il = _compute_interleaved_length(test_dur, ds, n_ch)
        verify_module(
            f"audio_encoder({test_dur // 48000}s)", enc_path, enc_wrapper,
            (torch.randn(2, test_dur, device=device),
             torch.tensor([il], dtype=torch.long, device=device)),
            1, device,
        )


def _verify_global_transformer(tts_model, gt_path: Path, nq: int, device: torch.device):
    gt_wrapper = GlobalTransformerWrapper(tts_model, nq=nq).to(device).eval()
    set_attn_implementation(gt_wrapper, "eager")
    _patch_gpt2_for_onnx(gt_wrapper)
    n_layers = gt_wrapper.n_layers
    T_test = 8
    gt_past = tuple(torch.zeros(1, 0, NUM_HEADS, HEAD_DIM, device=device) for _ in range(2 * n_layers))
    verify_module(
        "global_transformer", gt_path, gt_wrapper,
        (torch.randint(0, 100, (1, T_test, nq + 1), device=device, dtype=torch.long),
         torch.ones(1, T_test, device=device, dtype=torch.bool),
         torch.arange(T_test, device=device, dtype=torch.long).unsqueeze(0),
         *gt_past),
        1, device,
    )


def _verify_local_decoders(tts_model, ld_text_path: Path, ld_audio_path: Path,
                            nq: int, device: torch.device):
    ld_text_wrapper = LocalDecoderTextWrapper(tts_model).to(device).eval()
    for past_T, desc in [(0, "empty KV"), (LOCAL_MAX_LEN - 1, "full KV")]:
        pk, pv = _make_local_kv(past_T, device)
        verify_module(
            f"local_decoder_text({desc})", ld_text_path, ld_text_wrapper,
            (torch.randn(1, 1, HIDDEN, device=device),
             torch.tensor([[past_T]], device=device, dtype=torch.long),
             pk, pv),
            2, device,
        )

    ld_audio_wrapper = LocalDecoderAudioWrapper(tts_model, nq=nq).to(device).eval()
    cases = [
        (1, 1, 1, 0, 0,    "audio ch0, external embed"),
        (2, 2, 0, 0, 500,  "audio ch1, lookup audio_embed_0[500]"),
        (5, 8, 0, 2, 512,  "audio ch7, lookup audio_embed_2[512]"),
    ]
    for past_T, hid, use_ext, pch, ptok, desc in cases:
        pk, pv = _make_local_kv(past_T, device)
        verify_module(
            f"local_decoder_audio({desc})", ld_audio_path, ld_audio_wrapper,
            (torch.randn(1, 1, HIDDEN, device=device),
             torch.tensor([[past_T]], device=device, dtype=torch.long),
             pk, pv,
             torch.tensor([hid], device=device, dtype=torch.long),
             torch.tensor([use_ext], device=device, dtype=torch.long),
             torch.tensor([pch], device=device, dtype=torch.long),
             torch.tensor([ptok], device=device, dtype=torch.long)),
            1, device,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Export MOSS-TTS-Nano to ONNX."
    )
    parser.add_argument("--tts-checkpoint", default="./MOSS-TTS-Nano-100M")
    parser.add_argument("--audio-tokenizer-checkpoint", default="./MOSS-Audio-Tokenizer-Nano")
    parser.add_argument("--output-dir", default="./onnx_export")
    parser.add_argument("--nq", type=int, default=NQ)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--skip-quantize", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nq = args.nq

    log.info("Loading TTS model from %s ...", args.tts_checkpoint)
    tts_model = AutoModelForCausalLM.from_pretrained(
        args.tts_checkpoint, trust_remote_code=True
    )
    tts_model._set_attention_implementation("sdpa")
    set_attn_implementation(tts_model, "sdpa")
    tts_model.to(device=device, dtype=torch.float32).eval()

    log.info("Loading Audio Tokenizer from %s ...", args.audio_tokenizer_checkpoint)
    audio_tokenizer = AutoModel.from_pretrained(
        args.audio_tokenizer_checkpoint, trust_remote_code=True,
        local_files_only=True, force_download=True,
    )
    audio_tokenizer.set_attention_implementation("sdpa")
    set_attn_implementation(audio_tokenizer, "sdpa")
    audio_tokenizer.to(device=device, dtype=torch.float32).eval()

    # --- FP32 ONNX export ---
    with torch.no_grad():
        enc_path = export_audio_encoder(audio_tokenizer, output_dir, device)
        gt_path = export_global_transformer(tts_model, output_dir, device)
        ld_text_path = export_local_decoder_text(tts_model, output_dir, device)
        ld_audio_path = export_local_decoder_audio(tts_model, output_dir, device)
        decoder_state_spec = export_audio_decoder(audio_tokenizer, output_dir, device)

    export_tokenizer_and_config(tts_model, args.tts_checkpoint, output_dir, nq=nq)
    optimize_model_graphs(output_dir)

    # --- Verification (ORT vs PyTorch) ---
    if not args.skip_verify:
        log.info("=" * 60)
        log.info("Running ONNX verification ...")
        with torch.no_grad():
            _verify_audio_encoder(audio_tokenizer, enc_path, nq, device)
            _verify_global_transformer(tts_model, gt_path, nq, device)
            _verify_local_decoders(tts_model, ld_text_path, ld_audio_path, nq, device)
            _verify_audio_decoder(
                audio_tokenizer,
                output_dir / "audio_decoder.onnx",
                decoder_state_spec, device,
            )

    # --- INT8 dynamic quantization (codex-validated optimal config) ---
    if not args.skip_quantize:
        _quantize_models(output_dir, BUNDLE_MODELS)

    # --- Manifest ---
    _write_manifest(output_dir)

    log.info("=" * 60)
    log.info("Export complete -> %s", output_dir)
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            log.info("  %s (%.1f MB)", f.name, f.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
