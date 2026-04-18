#!/usr/bin/env python3
"""Dump deterministic per-step inference trace for the Kotlin port.

Runs ``OnnxTTSEngine``'s 4 code-generation sessions (audio_encoder,
global_transformer, local_decoder_text, local_decoder_audio) in
``do_sample=False`` mode (pure argmax, no RNG) and records, for every
autoregressive step:

  * fingerprint of ``global_transformer`` hidden output
  * fingerprint + argmax of ``local_decoder_text`` candidate logits
  * for each of the 16 audio codebooks: fingerprint + argmax of
    ``local_decoder_audio`` logits  (after repetition penalty)
  * the 16 sampled codebook tokens for this frame

Output is a single JSON file consumed by
``android/app/src/androidTest/.../InferenceLoopTraceTest.kt`` to assert
byte-level equivalence at every layer of the inference pipeline.

The audio_decoder (PCM streaming) is **not** exercised here; that
belongs to M2.6.

Run:

    python tools/dump_inference_trace.py \
        --bundle-dir onnx_export \
        --out android/app/src/androidTest/assets/inference_traces/trace.json \
        --max-steps 50
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from onnx_infer import OnnxTTSEngine  # noqa: E402
from onnx_tts_utils import apply_repetition_penalty  # noqa: E402


# ---------------------------------------------------------------------------
# Fingerprint helpers — int + float variants. Kept compact so the JSON
# stays well under the apk asset budget (target: trace.json <= 200 KB).
# ---------------------------------------------------------------------------
def fp_int(arr: np.ndarray) -> dict:
    flat = np.asarray(arr).flatten().astype(np.int64)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "length": int(flat.size),
        "first_16": flat[:16].tolist(),
        "last_16": flat[-16:].tolist(),
        "sum": int(flat.sum()),
        "sha256": hashlib.sha256(flat.tobytes()).hexdigest(),
    }


def fp_float_lite(arr: np.ndarray) -> dict:
    """Compact float fingerprint — sha256 over raw bytes is the byte-level
    contract; sum / abs_sum are sanity checks that survive print rounding.
    No first_16/last_16 to keep the JSON budget under control."""
    a = np.ascontiguousarray(arr, dtype=np.float32)
    flat = a.flatten()
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "length": int(flat.size),
        "sum": float(flat.sum()),
        "abs_sum": float(np.abs(flat).sum()),
        "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
    }


def fp_float_full(arr: np.ndarray) -> dict:
    """Full fingerprint — used for step 0 (the prefill step) where mismatch
    is most likely and the Kotlin side needs head/tail samples to locate
    the divergence layer-by-layer."""
    a = np.ascontiguousarray(arr, dtype=np.float32)
    flat = a.flatten()
    base = fp_float_lite(a)
    base["first_16"] = [float(x) for x in flat[:16]]
    base["last_16"] = [float(x) for x in flat[-16:]]
    return base


def fp_float(arr: np.ndarray, full: bool = False) -> dict:
    return fp_float_full(arr) if full else fp_float_lite(arr)


# ---------------------------------------------------------------------------
# Synthetic prompt audio codes — same shape contract as
# dump_prompt_fixtures.synth_codes so traces can reference fixture seeds.
# ---------------------------------------------------------------------------
def synth_codes(num_frames: int, n_vq: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(low=0, high=1024, size=(num_frames, n_vq), dtype=np.int64)


# ---------------------------------------------------------------------------
# Trace scenarios — small, byte-equal-checkable, cover the three modes
# that exercise distinct branches of InferenceLoop. Keep ``max_steps``
# small (50) and reuse seeds across script invocations so the JSON is
# reproducible for git diffing.
# ---------------------------------------------------------------------------
SCENARIOS = [
    {
        "id": "F1_zh_continuation",
        "mode": "continuation",
        "text": "今天天气真好",
        "audio_prompt": None,
    },
    {
        "id": "F2_en_continuation",
        "mode": "continuation",
        "text": "The quick brown fox",
        "audio_prompt": None,
    },
    {
        "id": "F3_zh_voice_clone",
        "mode": "voice_clone",
        "text": "你好",
        # 24-frame synthetic audio prompt with seed=11 — same recipe as
        # dump_prompt_fixtures.CONTINUATION_WITH_AUDIO[0].
        "audio_prompt": {"frames": 24, "seed": 11},
    },
]


# ---------------------------------------------------------------------------
# Deterministic generation that mirrors OnnxTTSEngine.generate but
# replaces sampling with argmax and inserts trace probes.
# ---------------------------------------------------------------------------
def trace_one_scenario(
    engine: OnnxTTSEngine,
    text: str,
    mode: str,
    audio_prompt_codes: np.ndarray | None,
    max_steps: int,
    audio_repetition_penalty: float,
) -> dict:
    cfg = engine.config
    nq = engine.nq
    audio_pad = int(cfg["audio_pad_token_id"])
    audio_slot = int(cfg["audio_assistant_slot_token_id"])
    text_ids = engine.tokenizer.encode(text)

    if mode == "voice_clone":
        if audio_prompt_codes is None:
            raise ValueError("voice_clone scenario needs audio_prompt_codes")
        input_ids, _ = engine.prompt_builder.build_voice_clone_prompt(
            text_ids, audio_prompt_codes
        )
    else:
        input_ids, _ = engine.prompt_builder.build_continuation_prompt(
            text_ids, audio_prompt_codes
        )

    # Replicate engine.generate's loop bookkeeping verbatim so the trace
    # we produce here is exactly what InferenceLoop must reproduce on
    # Android — no off-by-one in mask_len, position_ids, etc.
    empty_local_kv = np.zeros(
        (1, 0, engine.num_heads, engine.head_dim), dtype=np.float32
    )
    past_kv = None
    current_ids = input_ids
    seq_len = int(current_ids.shape[1])
    position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]
    max_total_len = seq_len + max_steps
    mask_buf = np.ones((1, max_total_len), dtype=np.bool_)
    mask_len = seq_len

    history_buf = np.zeros((max_steps, nq), dtype=np.int64)
    n_generated = 0

    local_pos_ids = [np.array([[i]], dtype=np.int64) for i in range(nq + 1)]
    head_ids = [np.array([i], dtype=np.int64) for i in range(nq + 1)]
    next_row = np.full((1, 1, nq + 1), audio_pad, dtype=np.int64)
    use_ext = np.array([1], dtype=np.int64)
    use_lookup = np.array([0], dtype=np.int64)
    dummy_embed = np.zeros((1, 1, engine.hidden_size), dtype=np.float32)
    dummy_ch = np.array([0], dtype=np.int64)
    dummy_tok = np.array([0], dtype=np.int64)

    steps_trace: list[dict] = []
    eos_step: int | None = None

    import time
    t_started = time.time()
    for step in range(max_steps):
        # Step 0 is the prefill step (T_in == seq_len); it is by far the
        # most likely place to discover a divergence and also the most
        # informative to debug, so we keep first_16/last_16 there only.
        full_fp = (step == 0)

        hidden, new_kv = engine._run_global_transformer(
            current_ids, mask_buf[:, :mask_len], position_ids, past_kv,
        )

        local_key = empty_local_kv.copy()
        local_value = empty_local_kv.copy()
        cand_logits, cand_embeds, local_key, local_value = (
            engine._run_local_decoder_text(
                hidden, local_pos_ids[0], local_key, local_value,
            )
        )
        cand_logits_1d = cand_logits[0]
        text_argmax = int(np.argmax(cand_logits_1d))

        # EOS check (id 1 in this model). Record & break — no codes for
        # this step. InferenceLoop must do the same.
        if text_argmax == 1:
            eos_step = step
            steps_trace.append({
                "step": step,
                "gt_hidden_fp": fp_float(hidden, full=full_fp),
                "text_logits_fp": fp_float(cand_logits_1d, full=full_fp),
                "text_argmax": text_argmax,
                "is_eos": True,
                "audio_logits_fp_per_ch": [],
                "frame_tokens": [],
            })
            break

        text_embed = cand_embeds[0:1][np.newaxis, :, :]
        frame_tokens = np.full(nq, audio_pad, dtype=np.int64)
        audio_history = history_buf[:n_generated] if n_generated > 0 else None

        per_ch_fp: list[dict] = []
        for ch in range(nq):
            if ch == 0:
                ch_logits, local_key, local_value = (
                    engine._run_local_decoder_audio(
                        text_embed, local_pos_ids[ch + 1], local_key, local_value,
                        head_ids[ch + 1], use_ext, dummy_ch, dummy_tok,
                    )
                )
            else:
                ch_logits, local_key, local_value = (
                    engine._run_local_decoder_audio(
                        dummy_embed, local_pos_ids[ch + 1], local_key, local_value,
                        head_ids[ch + 1], use_lookup,
                        np.array([ch - 1], dtype=np.int64),
                        np.array([frame_tokens[ch - 1]], dtype=np.int64),
                    )
                )
            ch_logits_1d = ch_logits[0]
            previous_ids = audio_history[:, ch] if audio_history is not None else None
            ch_logits_1d = apply_repetition_penalty(
                ch_logits_1d, previous_ids, audio_repetition_penalty,
            )
            tok = int(np.argmax(ch_logits_1d))
            frame_tokens[ch] = tok
            per_ch_fp.append({
                "ch": ch,
                "logits_fp": fp_float(ch_logits_1d, full=full_fp),
                "argmax": tok,
            })

        steps_trace.append({
            "step": step,
            "gt_hidden_fp": fp_float(hidden, full=full_fp),
            "text_logits_fp": fp_float(cand_logits_1d, full=full_fp),
            "text_argmax": text_argmax,
            "is_eos": False,
            "audio_logits_fp_per_ch": per_ch_fp,
            "frame_tokens": frame_tokens.tolist(),
        })

        history_buf[n_generated] = frame_tokens
        n_generated += 1

        next_row[0, 0, 0] = audio_slot
        next_row[0, 0, 1:nq + 1] = frame_tokens
        past_kv = new_kv
        current_ids = next_row
        position_ids = np.array([[mask_len]], dtype=np.int64)
        mask_len += 1

        elapsed = time.time() - t_started
        rate = (step + 1) / elapsed if elapsed > 0 else 0
        print(
            f"    step {step + 1:>3}/{max_steps}  "
            f"text_argmax={text_argmax:>5}  "
            f"frame[0..3]={frame_tokens[:4].tolist()}  "
            f"({rate:.2f} step/s)",
            flush=True,
        )

    final_codes = history_buf[:n_generated].astype(np.int64)
    return {
        "text": text,
        "mode": mode,
        "text_token_ids": list(text_ids),
        "audio_repetition_penalty": audio_repetition_penalty,
        "max_steps": max_steps,
        "input_ids_fp": fp_int(input_ids),
        "n_generated": int(n_generated),
        "eos_step": eos_step,
        "final_codes": final_codes.tolist(),
        "final_codes_fp": fp_int(final_codes),
        "steps": steps_trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", default="onnx_export")
    parser.add_argument(
        "--out",
        default="android/app/src/androidTest/assets/inference_traces/trace.json",
    )
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--audio-repetition-penalty",
        type=float,
        default=1.2,
        help="Match OnnxTTSEngine.generate default; non-1.0 covers the "
             "repetition-penalty branch in InferenceLoop.",
    )
    args = parser.parse_args()

    bundle = Path(args.bundle_dir).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading engine from {bundle} (precision=int8) ...", flush=True)
    engine = OnnxTTSEngine(
        onnx_dir=str(bundle), precision="int8", threads=args.threads,
    )

    nq = engine.nq
    output = {
        "schema": "moss-tts-nano/inference-trace/v1",
        "bundle_dir": str(bundle),
        "precision": engine.precision,
        "n_vq": nq,
        "hidden_size": engine.hidden_size,
        "audio_pad_token_id": int(engine.config["audio_pad_token_id"]),
        "audio_assistant_slot_token_id": int(
            engine.config["audio_assistant_slot_token_id"]
        ),
        "max_steps": args.max_steps,
        "audio_repetition_penalty": args.audio_repetition_penalty,
        "scenarios": [],
    }

    for scenario in SCENARIOS:
        print(f"  scenario {scenario['id']} ({scenario['mode']}) ...", flush=True)
        audio_prompt_codes: np.ndarray | None = None
        ap = scenario.get("audio_prompt")
        if ap is not None:
            audio_prompt_codes = synth_codes(
                num_frames=int(ap["frames"]), n_vq=nq, seed=int(ap["seed"]),
            )
        trace = trace_one_scenario(
            engine=engine,
            text=str(scenario["text"]),
            mode=str(scenario["mode"]),
            audio_prompt_codes=audio_prompt_codes,
            max_steps=int(args.max_steps),
            audio_repetition_penalty=float(args.audio_repetition_penalty),
        )
        trace["id"] = scenario["id"]
        if ap is not None:
            trace["audio_prompt"] = {
                "frames": int(ap["frames"]),
                "seed": int(ap["seed"]),
                "codes": audio_prompt_codes.tolist(),
            }
        output["scenarios"].append(trace)

    out_path.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024.0
    print(
        f"Wrote {out_path} ({size_kb:.1f} KiB, "
        f"{len(output['scenarios'])} scenarios)",
        flush=True,
    )


if __name__ == "__main__":
    main()
