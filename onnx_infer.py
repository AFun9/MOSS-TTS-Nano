#!/usr/bin/env python3
"""ONNX inference for MOSS-TTS-Nano.

Yields waveform chunks as the autoregressive loop produces code frames,
so the first audio chunk is available before the full sequence finishes.

Usage:
    python onnx_infer.py \
        --onnx-dir onnx_export --precision int8 \
        --text "你好，这是一个推理测试。" \
        --output output.wav
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from onnx_tts_utils import (
    SPTokenizer,
    apply_repetition_penalty,
    mask_unused_audio_channels,
    normalize_audio_codes,
    sample_top_k_top_p,
)
from text_normalization_pipeline import prepare_tts_request_texts

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
log = logging.getLogger(__name__)

MIN_THREADS = 1
MAX_THREADS = 8
FIXED_INTER_OP_THREADS = 1
DEFAULT_THREADS = 2
DEFAULT_FRAME_SAMPLES = 3840
DEFAULT_MAX_NEW_FRAMES = 300
DEFAULT_FRAMES_PER_CHAR = 2.5
DEFAULT_MIN_FRAMES = 10
DEFAULT_MAX_FRAMES = 80
SINGLE_PASS_TARGET_FRAMES = 30
MIN_RMS_THRESHOLD = 0.02

USER_ROLE_PREFIX = "user\n"
USER_TEMPLATE_REFERENCE_PREFIX = "<user_inst>\n- Reference(s):\n"
USER_TEMPLATE_AFTER_REFERENCE = (
    "\n- Instruction:\nNone\n"
    "- Tokens:\nNone\n"
    "- Quality:\nNone\n"
    "- Sound Event:\nNone\n"
    "- Ambient Sound:\nNone\n"
    "- Language:\nNone\n"
    "- Text:\n"
)
USER_TEMPLATE_SUFFIX = "\n</user_inst>"
ASSISTANT_TURN_PREFIX = "\n"
ASSISTANT_ROLE_PREFIX = "assistant\n"


class PromptBuilder:
    """Builds model input prompts from tokenizer + config."""

    def __init__(self, tokenizer: SPTokenizer, config: dict) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self._n_vq = int(config["n_vq"])
        self._audio_pad = int(config["audio_pad_token_id"])
        enc = lambda t: list(tokenizer.encode(t, add_special_tokens=False))
        self._user_prefix = [int(config["im_start_token_id"])] + enc(USER_ROLE_PREFIX) + enc(USER_TEMPLATE_REFERENCE_PREFIX)
        self._after_ref = enc(USER_TEMPLATE_AFTER_REFERENCE)
        self._assistant_prefix = (
            enc(USER_TEMPLATE_SUFFIX) + [int(config["im_end_token_id"])]
            + enc(ASSISTANT_TURN_PREFIX) + [int(config["im_start_token_id"])]
            + enc(ASSISTANT_ROLE_PREFIX)
        )
        self._none_ids = enc("None")

    def _text_rows(self, ids):
        rows = np.full((len(ids), self._n_vq + 1), self._audio_pad, dtype=np.int64)
        for i, tok in enumerate(ids):
            rows[i, 0] = tok
        return rows

    def _audio_prefix_rows(self, codes, slot_token_id):
        T = codes.shape[0]
        rows = np.full((T, self._n_vq + 1), self._audio_pad, dtype=np.int64)
        rows[:, 0] = slot_token_id
        rows[:, 1:] = codes
        return rows

    def _to_prompt(self, sections):
        all_rows = np.concatenate(sections, axis=0)
        input_ids = all_rows[np.newaxis, :, :]
        attention_mask = np.ones((1, input_ids.shape[1]), dtype=np.bool_)
        return input_ids, attention_mask

    def build_continuation_prompt(self, text_token_ids, prompt_audio_codes=None):
        prompt_ids = self._user_prefix + self._none_ids + self._after_ref + list(text_token_ids) + self._assistant_prefix
        sections = [self._text_rows(prompt_ids), self._text_rows([int(self.config["audio_start_token_id"])])]
        if prompt_audio_codes is not None:
            sections.append(self._audio_prefix_rows(prompt_audio_codes, slot_token_id=int(self.config["audio_assistant_slot_token_id"])))
        return self._to_prompt(sections)

    def build_voice_clone_prompt(self, text_token_ids, prompt_audio_codes):
        prefix_ids = self._user_prefix + [int(self.config["audio_start_token_id"])]
        suffix_ids = (
            [int(self.config["audio_end_token_id"])]
            + self._after_ref + list(text_token_ids) + self._assistant_prefix
            + [int(self.config["audio_start_token_id"])]
        )
        sections = [
            self._text_rows(prefix_ids),
            self._audio_prefix_rows(prompt_audio_codes, slot_token_id=int(self.config["audio_user_slot_token_id"])),
            self._text_rows(suffix_ids),
        ]
        return self._to_prompt(sections)


class AudioDecoder:
    """ONNX audio decoder runner with managed KV-cache state.

    Each call to :meth:`step` consumes one ``[nq, 1, 1]`` int64 code frame
    and returns ``[channels, downsample_rate]`` float32 audio. Cache tensors
    are kept inside this object so callers only pass the new frame.
    """

    def __init__(self, model_path: Path, state_spec_path: Path, sess_options) -> None:
        import onnxruntime as ort
        self.model_path = Path(model_path)
        self.state_spec = json.loads(Path(state_spec_path).read_text(encoding="utf-8"))
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = list(self.state_spec["input_names"])
        self.output_names = list(self.state_spec["output_names"])
        self.num_quantizers = int(self.state_spec["num_quantizers"])
        self.sample_rate = int(self.state_spec["sample_rate"])
        self.downsample_rate = int(self.state_spec["downsample_rate"])
        self._zero_state = self._build_zero_state()
        self.state: dict[str, np.ndarray] = {k: v.copy() for k, v in self._zero_state.items()}

    def _build_zero_state(self) -> dict[str, np.ndarray]:
        state: dict[str, np.ndarray] = {}
        grouped: dict[int, list[dict]] = {}
        for spec in self.state_spec["attention_specs"]:
            grouped.setdefault(int(spec["decoder_module_index"]), []).append(spec)
        for ts in self.state_spec["transformer_specs"]:
            di = int(ts["decoder_module_index"])
            state[f"decoder_{di}_transformer_offset"] = np.zeros((1,), dtype=np.int64)
            for attn in grouped[di]:
                p = f"decoder_{attn['decoder_module_index']}_layer_{attn['layer_index']}"
                nh, ctx, hd = int(attn["num_heads"]), int(attn["context"]), int(attn["head_dim"])
                state[f"{p}_cached_keys"] = np.zeros((1, nh, ctx, hd), dtype=np.float32)
                state[f"{p}_cached_values"] = np.zeros((1, nh, ctx, hd), dtype=np.float32)
                state[f"{p}_cached_positions"] = np.full((1, ctx), -1, dtype=np.int64)
                state[f"{p}_offset"] = np.zeros((1,), dtype=np.int64)
        return state

    def reset(self) -> None:
        """Wipe state so the decoder is ready for a new utterance."""
        self.state = {k: v.copy() for k, v in self._zero_state.items()}

    def step(self, frame_codes: np.ndarray) -> np.ndarray:
        """Decode one frame.

        Args:
            frame_codes: int64 array shaped ``[nq, 1, 1]`` (or broadcastable).

        Returns:
            ``[channels, downsample_rate]`` float32 audio chunk.
        """
        codes = np.asarray(frame_codes, dtype=np.int64)
        if codes.ndim == 1:
            codes = codes[:, np.newaxis, np.newaxis]
        elif codes.ndim != 3:
            raise ValueError(f"Expected codes [nq,1,1] or [nq], got {codes.shape}")
        feed = dict(self.state)
        feed["codes"] = codes[: self.num_quantizers]
        outputs = self.session.run(None, feed)
        out_map = {name: value for name, value in zip(self.output_names, outputs)}
        # Update state from "new_*" outputs
        self.state = {name[4:]: out_map[name] for name in self.output_names[2:]}
        return np.asarray(out_map["audio"], dtype=np.float32)[0]


class OnnxTTSEngine:
    """End-to-end ONNX TTS engine.

    Loads ``audio_encoder``, ``global_transformer``, ``local_decoder_text``,
    ``local_decoder_audio`` for autoregressive code generation, and
    ``audio_decoder`` (+ state spec) for waveform decoding.
    """

    def __init__(
        self,
        onnx_dir: str,
        precision: str = "auto",
        threads: int = DEFAULT_THREADS,
    ):
        import onnxruntime as ort

        onnx_dir_path = Path(onnx_dir).expanduser().resolve()
        suffix, resolved_precision = self._resolve_precision(onnx_dir_path, precision)

        self.onnx_dir = str(onnx_dir_path)
        self.precision = resolved_precision
        self.threads = int(threads)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = self.threads
        sess_options.inter_op_num_threads = FIXED_INTER_OP_THREADS
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        def load_session(name, allow_fp32_fallback=False):
            path = onnx_dir_path / f"{name}{suffix}.onnx"
            if not path.is_file():
                raise FileNotFoundError(f"Missing: {path}")
            log.info("  Loading %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
            try:
                return ort.InferenceSession(str(path), sess_options=sess_options, providers=["CPUExecutionProvider"])
            except Exception as e:
                if allow_fp32_fallback and suffix:
                    fp32_path = onnx_dir_path / f"{name}.onnx"
                    if fp32_path.is_file():
                        log.warning("  INT8 %s failed (%s), falling back to FP32", name, e)
                        return ort.InferenceSession(str(fp32_path), sess_options=sess_options, providers=["CPUExecutionProvider"])
                raise

        log.info("Loading ONNX sessions from %s (precision=%s) ...", onnx_dir_path, resolved_precision)
        self.audio_encoder = load_session("audio_encoder", allow_fp32_fallback=True)
        self.global_transformer = load_session("global_transformer")
        self.local_decoder_text = load_session("local_decoder_text")
        self.local_decoder_audio = load_session("local_decoder_audio")

        base = "audio_decoder"
        int8_path = onnx_dir_path / f"{base}_int8.onnx"
        fp32_path = onnx_dir_path / f"{base}.onnx"
        spec_path = onnx_dir_path / f"{base}_state_spec.json"
        if not spec_path.is_file():
            raise FileNotFoundError(
                f"Audio decoder state spec missing: {spec_path}. "
                "Run export_onnx.py to produce the bundle."
            )
        preferred = int8_path if (resolved_precision == "int8" and int8_path.is_file()) else fp32_path
        if not preferred.is_file():
            preferred = fp32_path if fp32_path.is_file() else int8_path
        if not preferred.is_file():
            raise FileNotFoundError(f"No audio decoder ONNX in {onnx_dir_path}")
        log.info("  Loading %s (%.1f MB)", preferred.name, preferred.stat().st_size / 1e6)
        self.audio_decoder = AudioDecoder(preferred, spec_path, sess_options)

        self.config = json.loads((onnx_dir_path / "config.json").read_text(encoding="utf-8"))
        self.tokenizer = SPTokenizer(str(onnx_dir_path / "tokenizer.model"))
        self.prompt_builder = PromptBuilder(self.tokenizer, self.config)

        self.nq = int(self.config.get("nq", self.config.get("n_vq", 16)))
        self.hidden_size = int(self.config["hidden_size"])
        self.num_layers = int(self.config["num_layers"])
        self.num_heads = int(self.config["num_heads"])
        self.head_dim = int(self.config["head_dim"])
        self._gt_kv_names = [inp.name for inp in self.global_transformer.get_inputs() if inp.name.startswith("past_")]
        self._empty_gt_kv = np.zeros((1, 0, self.num_heads, self.head_dim), dtype=np.float32)

        log.info("Engine ready (nq=%d, hidden=%d)", self.nq, self.hidden_size)

    @staticmethod
    def _resolve_precision(onnx_dir_path, precision):
        # The accumulating audio_decoder is intentionally excluded: it is no
        # longer required by the engine and the bundle may omit it.
        base_names = ["audio_encoder", "global_transformer",
                      "local_decoder_text", "local_decoder_audio"]

        def has_all(suffix):
            return all((onnx_dir_path / f"{name}{suffix}.onnx").is_file() for name in base_names)

        if precision == "int8":
            if not has_all("_int8"):
                raise FileNotFoundError(f"INT8 models incomplete in {onnx_dir_path}")
            return "_int8", "int8"
        if precision == "fp32":
            if not has_all(""):
                raise FileNotFoundError(f"FP32 models incomplete in {onnx_dir_path}")
            return "", "fp32"
        if has_all("_int8"):
            return "_int8", "int8"
        if has_all(""):
            return "", "fp32"
        raise FileNotFoundError(f"No complete ONNX export in {onnx_dir_path}")

    def encode_audio(self, waveform):
        ds = int(self.config.get("audio_tokenizer_downsample_rate", DEFAULT_FRAME_SAMPLES))
        n_ch = int(self.config.get("audio_tokenizer_num_channels", 2))
        raw_len = int(waveform.shape[-1])
        input_length = raw_len * n_ch
        pad_rem = input_length % ds
        if pad_rem != 0:
            input_length += ds - pad_rem
        result = self.audio_encoder.run(None, {
            "waveform": np.asarray(waveform, dtype=np.float32),
            "input_lengths": np.array([input_length], dtype=np.int64),
        })
        return result[0]

    def _encode_prompt_audio(self, prompt_audio_path):
        cfg = self.config
        waveform, sample_rate = sf.read(prompt_audio_path, dtype="float32", always_2d=True)
        waveform = waveform.T
        target_sr = int(cfg["audio_tokenizer_sample_rate"])
        target_ch = int(cfg["audio_tokenizer_num_channels"])
        if sample_rate != target_sr:
            import resampy
            waveform = resampy.resample(waveform, sample_rate, target_sr, axis=-1)
        if waveform.shape[0] < target_ch:
            waveform = np.repeat(waveform, target_ch, axis=0)[:target_ch]
        elif waveform.shape[0] > target_ch:
            waveform = waveform[:target_ch]
        audio_codes = self.encode_audio(waveform)
        nq = self.nq
        audio_codes = normalize_audio_codes(audio_codes, nq)
        audio_codes = mask_unused_audio_channels(audio_codes, nq, int(cfg["audio_pad_token_id"]))
        log.info("Encoded prompt audio: %s", audio_codes.shape)
        return audio_codes

    def _run_global_transformer(self, current_ids, mask, position_ids, past_kv):
        feed = {
            "input_ids": np.asarray(current_ids, dtype=np.int64),
            "attention_mask": np.asarray(mask, dtype=np.bool_),
            "position_ids": np.asarray(position_ids, dtype=np.int64),
        }
        if past_kv is None:
            for name in self._gt_kv_names:
                feed[name] = self._empty_gt_kv
        else:
            for name, arr in zip(self._gt_kv_names, past_kv):
                feed[name] = arr
        outputs = self.global_transformer.run(None, feed)
        hidden = outputs[0]
        new_kv = outputs[1:]
        return hidden, new_kv

    def _run_local_decoder_text(self, hidden, pos_id, local_key, local_value):
        outputs = self.local_decoder_text.run(None, {
            "input_embed": np.asarray(hidden, dtype=np.float32),
            "position_id": np.asarray(pos_id, dtype=np.int64),
            "past_key": np.asarray(local_key, dtype=np.float32),
            "past_value": np.asarray(local_value, dtype=np.float32),
        })
        return outputs[0], outputs[1], outputs[2], outputs[3]

    def _run_local_decoder_audio(self, embed, pos_id, local_key, local_value,
                                  head_id, use_external, prev_channel, prev_token):
        outputs = self.local_decoder_audio.run(None, {
            "input_embed": np.asarray(embed, dtype=np.float32),
            "position_id": np.asarray(pos_id, dtype=np.int64),
            "past_key": np.asarray(local_key, dtype=np.float32),
            "past_value": np.asarray(local_value, dtype=np.float32),
            "head_id": np.asarray(head_id, dtype=np.int64),
            "use_input_embed": np.asarray(use_external, dtype=np.int64),
            "prev_audio_ch": np.asarray(prev_channel, dtype=np.int64),
            "prev_token_id": np.asarray(prev_token, dtype=np.int64),
        })
        return outputs[0], outputs[1], outputs[2]

    def _autoregressive_generate(
        self,
        input_ids: np.ndarray,
        max_new_frames: int,
        do_sample: bool,
        text_temperature: float,
        text_top_p: float,
        text_top_k: int,
        audio_temperature: float,
        audio_top_p: float,
        audio_top_k: int,
        audio_repetition_penalty: float,
    ) -> np.ndarray:
        """Generate audio token frames. Returns history_buf[:n_generated]."""
        cfg = self.config
        nq = self.nq
        audio_pad = int(cfg["audio_pad_token_id"])
        audio_slot = int(cfg["audio_assistant_slot_token_id"])
        empty_local_kv = np.zeros((1, 0, self.num_heads, self.head_dim), dtype=np.float32)

        past_kv = None
        current_ids = input_ids
        seq_len = int(current_ids.shape[1])
        position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]
        max_total_len = seq_len + max_new_frames
        mask_buf = np.ones((1, max_total_len), dtype=np.bool_)
        mask_len = seq_len

        history_buf = np.zeros((max_new_frames, nq), dtype=np.int64)
        n_generated = 0

        local_pos_ids = [np.array([[i]], dtype=np.int64) for i in range(nq + 1)]
        head_ids = [np.array([i], dtype=np.int64) for i in range(nq + 1)]
        next_row = np.full((1, 1, nq + 1), audio_pad, dtype=np.int64)

        use_ext = np.array([1], dtype=np.int64)
        use_lookup = np.array([0], dtype=np.int64)
        dummy_embed = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
        dummy_ch = np.array([0], dtype=np.int64)
        dummy_tok = np.array([0], dtype=np.int64)

        started = time.time()
        for step in range(max_new_frames):
            hidden, new_kv = self._run_global_transformer(
                current_ids, mask_buf[:, :mask_len], position_ids, past_kv,
            )

            local_key = empty_local_kv.copy()
            local_value = empty_local_kv.copy()
            candidate_logits, candidate_embeds, local_key, local_value = self._run_local_decoder_text(
                hidden, local_pos_ids[0], local_key, local_value,
            )
            candidate_logits = candidate_logits[0]
            if do_sample:
                sampled_idx = sample_top_k_top_p(candidate_logits, text_temperature, text_top_k, text_top_p)
            else:
                sampled_idx = int(np.argmax(candidate_logits))
            if sampled_idx == 1:
                break

            text_embed = candidate_embeds[0:1][np.newaxis, :, :]
            frame_tokens = np.full(nq, audio_pad, dtype=np.int64)
            audio_history = history_buf[:n_generated] if n_generated > 0 else None

            for ch in range(nq):
                if ch == 0:
                    ch_logits, local_key, local_value = self._run_local_decoder_audio(
                        text_embed, local_pos_ids[ch + 1], local_key, local_value,
                        head_ids[ch + 1], use_ext, dummy_ch, dummy_tok,
                    )
                else:
                    ch_logits, local_key, local_value = self._run_local_decoder_audio(
                        dummy_embed, local_pos_ids[ch + 1], local_key, local_value,
                        head_ids[ch + 1], use_lookup, np.array([ch - 1], dtype=np.int64),
                        np.array([frame_tokens[ch - 1]], dtype=np.int64),
                    )
                ch_logits = ch_logits[0]
                previous_ids = audio_history[:, ch] if audio_history is not None else None
                ch_logits = apply_repetition_penalty(ch_logits, previous_ids, audio_repetition_penalty)
                if do_sample:
                    tok = sample_top_k_top_p(ch_logits, audio_temperature, audio_top_k, audio_top_p)
                else:
                    tok = int(np.argmax(ch_logits))
                frame_tokens[ch] = tok

            history_buf[n_generated] = frame_tokens
            n_generated += 1

            next_row[0, 0, 0] = audio_slot
            next_row[0, 0, 1:nq + 1] = frame_tokens
            past_kv = new_kv
            current_ids = next_row
            position_ids = np.array([[mask_len]], dtype=np.int64)
            mask_len += 1

            if (step + 1) % 50 == 0:
                elapsed = time.time() - started
                log.info("  step %d/%d (%.1f frames/sec)", step + 1, max_new_frames, (step + 1) / elapsed)

        elapsed = time.time() - started
        fps = n_generated / elapsed if elapsed > 0 else 0
        log.info("Generation done: %d frames in %.1fs (%.1f frames/sec)", n_generated, elapsed, fps)
        return history_buf[:n_generated]

    def generate(
        self,
        text: str,
        prompt_audio_path: Optional[str] = None,
        mode: str = "continuation",
        max_new_frames: int = DEFAULT_MAX_NEW_FRAMES,
        do_sample: bool = True,
        text_temperature: float = 1.0,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: float = 0.8,
        audio_top_p: float = 0.95,
        audio_top_k: int = 25,
        audio_repetition_penalty: float = 1.2,
    ):
        """Generate audio for ``text``, yielding one chunk per code frame.

        Yields ``(waveform_chunk, frame_index, is_final)``. First-chunk
        latency is ~60-90 ms on CPU INT8 (one AR step + one decoder call).
        """
        cfg = self.config
        nq = self.nq
        audio_pad = int(cfg["audio_pad_token_id"])
        audio_slot = int(cfg["audio_assistant_slot_token_id"])
        text_ids = self.tokenizer.encode(text)

        prompt_audio_codes = self._encode_prompt_audio(prompt_audio_path) if prompt_audio_path else None
        resolved_mode = str(mode or "continuation").strip().lower() or "continuation"
        if prompt_audio_codes is not None and resolved_mode == "continuation":
            resolved_mode = "voice_clone"
        if resolved_mode == "voice_clone" and prompt_audio_codes is None:
            raise ValueError("voice_clone mode requires --prompt-audio-path.")

        if resolved_mode == "voice_clone":
            input_ids, _ = self.prompt_builder.build_voice_clone_prompt(text_ids, prompt_audio_codes)
        else:
            input_ids, _ = self.prompt_builder.build_continuation_prompt(text_ids, prompt_audio_codes)

        log.info("Inference mode: %s", resolved_mode)
        log.info("Prompt shape: input_ids=%s", input_ids.shape)

        # --- Setup KV-cached autoregressive loop ---
        empty_local_kv = np.zeros((1, 0, self.num_heads, self.head_dim), dtype=np.float32)
        past_kv = None
        current_ids = input_ids
        seq_len = int(current_ids.shape[1])
        position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]
        max_total_len = seq_len + max_new_frames
        mask_buf = np.ones((1, max_total_len), dtype=np.bool_)
        mask_len = seq_len

        history_buf = np.zeros((max_new_frames, nq), dtype=np.int64)
        n_generated = 0

        local_pos_ids = [np.array([[i]], dtype=np.int64) for i in range(nq + 1)]
        head_ids = [np.array([i], dtype=np.int64) for i in range(nq + 1)]
        next_row = np.full((1, 1, nq + 1), audio_pad, dtype=np.int64)
        use_ext = np.array([1], dtype=np.int64)
        use_lookup = np.array([0], dtype=np.int64)
        dummy_embed = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
        dummy_ch = np.array([0], dtype=np.int64)
        dummy_tok = np.array([0], dtype=np.int64)

        self.audio_decoder.reset()
        gen_started = time.time()

        for step in range(max_new_frames):
            hidden, new_kv = self._run_global_transformer(
                current_ids, mask_buf[:, :mask_len], position_ids, past_kv,
            )

            local_key = empty_local_kv.copy()
            local_value = empty_local_kv.copy()
            cand_logits, cand_embeds, local_key, local_value = self._run_local_decoder_text(
                hidden, local_pos_ids[0], local_key, local_value,
            )
            cand_logits = cand_logits[0]
            if do_sample:
                sampled_idx = sample_top_k_top_p(cand_logits, text_temperature, text_top_k, text_top_p)
            else:
                sampled_idx = int(np.argmax(cand_logits))

            # EOS: flush remaining codes and exit
            if sampled_idx == 1:
                log.info("EOS at step %d (n_generated=%d)", step, n_generated)
                break

            text_embed = cand_embeds[0:1][np.newaxis, :, :]
            frame_tokens = np.full(nq, audio_pad, dtype=np.int64)
            audio_history = history_buf[:n_generated] if n_generated > 0 else None

            for ch in range(nq):
                if ch == 0:
                    ch_logits, local_key, local_value = self._run_local_decoder_audio(
                        text_embed, local_pos_ids[ch + 1], local_key, local_value,
                        head_ids[ch + 1], use_ext, dummy_ch, dummy_tok,
                    )
                else:
                    ch_logits, local_key, local_value = self._run_local_decoder_audio(
                        dummy_embed, local_pos_ids[ch + 1], local_key, local_value,
                        head_ids[ch + 1], use_lookup, np.array([ch - 1], dtype=np.int64),
                        np.array([frame_tokens[ch - 1]], dtype=np.int64),
                    )
                ch_logits = ch_logits[0]
                previous_ids = audio_history[:, ch] if audio_history is not None else None
                ch_logits = apply_repetition_penalty(ch_logits, previous_ids, audio_repetition_penalty)
                if do_sample:
                    tok = sample_top_k_top_p(ch_logits, audio_temperature, audio_top_k, audio_top_p)
                else:
                    tok = int(np.argmax(ch_logits))
                frame_tokens[ch] = tok

            history_buf[n_generated] = frame_tokens
            n_generated += 1

            next_row[0, 0, 0] = audio_slot
            next_row[0, 0, 1:nq + 1] = frame_tokens
            past_kv = new_kv
            current_ids = next_row
            position_ids = np.array([[mask_len]], dtype=np.int64)
            mask_len += 1

            chunk = self.audio_decoder.step(frame_tokens)
            if chunk.shape[-1] > 0:
                yield chunk, n_generated - 1, False

        elapsed = time.time() - gen_started
        fps = n_generated / elapsed if elapsed > 0 else 0
        log.info("Done: %d frames in %.2fs (%.1f frames/sec)", n_generated, elapsed, fps)
        yield np.zeros((2, 0), dtype=np.float32), n_generated, True


def generate_to_file(
    engine: OnnxTTSEngine,
    text: str,      
    output_path: str,
    prompt_audio_path: Optional[str] = None,
    mode: str = "continuation",
    max_new_frames: int = DEFAULT_MAX_NEW_FRAMES,
):
    """Run generation and save concatenated audio to file."""
    sample_rate = int(engine.config["audio_tokenizer_sample_rate"])
    audio_chunks = []
    first_audio_time = None
    total_code_frames = 0
    started = time.time()

    for chunk, frame_idx, is_final in engine.generate(
        text=text,
        prompt_audio_path=prompt_audio_path,
        mode=mode,
        max_new_frames=max_new_frames,
    ):
        if is_final:
            total_code_frames = frame_idx
            break
        if first_audio_time is None:
            first_audio_time = time.time()
            log.info("First audio chunk at %.3fs", first_audio_time - started)
        audio_chunks.append(chunk)

    if not audio_chunks:
        log.warning("No audio generated!")
        return None

    full_waveform = np.concatenate(audio_chunks, axis=-1)
    audio_seconds = full_waveform.shape[-1] / sample_rate
    wall_seconds = time.time() - started
    first_chunk_latency = (first_audio_time - started) if first_audio_time else 0
    resolved_mode = "voice_clone" if prompt_audio_path else "continuation"

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), full_waveform.T, sample_rate)

    rtf = wall_seconds / audio_seconds if audio_seconds > 0 else float("inf")
    log.info("Wrote %s (%.2fs audio, %d code frames, %d chunks)", out_path, audio_seconds, total_code_frames, len(audio_chunks))
    log.info("RTF=%.3f | wall=%.2fs | first_chunk=%.3fs", rtf, wall_seconds, first_chunk_latency)

    return {
        "audio_path": str(out_path.resolve()),
        "sample_rate": sample_rate,
        "total_frames": total_code_frames,
        "decode_chunks": len(audio_chunks),
        "audio_seconds": audio_seconds,
        "wall_seconds": wall_seconds,
        "first_chunk_latency": first_chunk_latency,
        "rtf": rtf,
        "mode": resolved_mode,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="ONNX inference for MOSS-TTS-Nano")
    parser.add_argument("--onnx-dir", default="onnx_export", help="ONNX model directory")
    parser.add_argument("--precision", default="int8", choices=["auto", "int8", "fp32"])
    parser.add_argument("--output", required=True, help="Output wav path")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--prompt-audio-path", default=None, help="Reference audio for voice cloning")
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Text to synthesize")
    text_group.add_argument("--text-file", help="UTF-8 text file")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_NEW_FRAMES)
    return parser.parse_args()


def main():
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
    args = parse_args()
    text = args.text if args.text else Path(args.text_file).read_text(encoding="utf-8")

    prepared = prepare_tts_request_texts(
        text=text,
        prompt_text="",
        voice="",
        enable_wetext=False,
        enable_normalize_tts_text=True,
        text_normalizer_manager=None,
    )
    normalized_text = str(prepared["text"]).strip() or text
    log.info("Text normalization: method=%s chars=%d",
             prepared.get("normalization_method", "unknown"), len(normalized_text))

    engine = OnnxTTSEngine(
        onnx_dir=args.onnx_dir,
        precision=args.precision,
        threads=args.threads,
    )

    mode = "voice_clone" if args.prompt_audio_path else "continuation"
    summary = generate_to_file(
        engine=engine,
        text=normalized_text,
        output_path=args.output,
        prompt_audio_path=args.prompt_audio_path,
        mode=mode,
        max_new_frames=args.max_frames,
    )

    if summary:
        print("\nConfig:")
        print(f"  onnx_dir: {Path(args.onnx_dir).resolve()}")
        print(f"  precision: {engine.precision}")
        print(f"  threads: {engine.threads}")
        print("Result:")
        print(f"  output: {summary['audio_path']}")
        print(f"  mode: {summary['mode']}")
        print(f"  code_frames: {summary['total_frames']}")
        print(f"  decode_chunks: {summary['decode_chunks']}")
        print(f"  audio_seconds: {summary['audio_seconds']:.2f}")
        print(f"  wall_seconds: {summary['wall_seconds']:.2f}")
        print(f"  first_chunk_latency: {summary['first_chunk_latency']:.3f}")
        print(f"  rtf: {summary['rtf']:.3f}")


if __name__ == "__main__":
    main()
