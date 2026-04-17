#!/usr/bin/env python3
"""Dump `PromptBuilder` outputs as golden fixtures for the Kotlin port.

The Python `PromptBuilder.build_continuation_prompt(...)` and
`build_voice_clone_prompt(...)` produce the [1, T, n_vq+1] int64 tensor
that gets fed to the global transformer. We stash a fingerprint of that
tensor (shape + dtype + head/tail slice + sum + sha256) plus a flat
projection of the leading text-row column for every fixture, so the
Kotlin port can fail loudly on the first byte-level divergence instead
of waiting until generated audio sounds wrong on a phone.

Run:

    python tools/dump_prompt_fixtures.py \
        --bundle-dir onnx_export \
        --out android/app/src/test/resources/prompt_fixtures.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from onnx_infer import PromptBuilder  # noqa: E402
from onnx_tts_utils import SPTokenizer  # noqa: E402


def fingerprint(arr: np.ndarray) -> dict:
    """Compact, byte-equal-checkable summary of an int tensor."""
    flat = arr.flatten().astype(np.int64)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "length": int(flat.size),
        "first_16": flat[:16].tolist(),
        "last_16": flat[-16:].tolist(),
        "sum": int(flat.sum()),
        "sha256": hashlib.sha256(flat.tobytes()).hexdigest(),
    }


# Synthetic prompt-audio codes: deterministic, [T, n_vq] int64.
# We don't need real codec output — the PromptBuilder layout is the only
# thing under test; the audio_pad / slot-id wiring is what can break.
def synth_codes(num_frames: int, n_vq: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(low=0, high=1024, size=(num_frames, n_vq), dtype=np.int64)


# Texts cover the same script bands as the tokenizer golden test, plus
# the empty-string and a long-prompt case that PromptBuilder bolts onto
# its big static prefix.
CONTINUATION_TEXTS = [
    "",
    "你好",
    "Hello world",
    "こんにちは、世界",
    "안녕하세요 세계",
    "Привет, мир!",
    "¿Cómo estás hoy?",
    "中英 mixed: hello 世界 12345.",
    "<|im_start|>user\nhi<|im_end|>",  # special-token literal in user text
    # Long realistic prompt close to the 300-char app cap.
    "在过去的二十年里，人工智能技术经历了从规则系统到统计学习再到深度神经网络的"
    "三次范式跃迁，每一次都重塑了我们对'机器是否能够思考'这一古老问题的理解。",
]

# Voice-clone needs prompt audio; we pair every text with a synthetic
# (frames, seed) so the fixture is reproducible bit-for-bit.
VOICE_CLONE_FIXTURES = [
    ("你好", 32, 1),
    ("Hello world", 64, 2),
    ("こんにちは", 16, 3),
    ("Сегодня хорошая погода.", 48, 4),
    # Empty target text + audio prompt is a degenerate-but-legal case.
    ("", 8, 5),
]

# Continuation can also accept prompt audio (e.g. when the user wants
# the model to keep speaking in the same voice as a reference clip).
CONTINUATION_WITH_AUDIO = [
    ("你好世界", 24, 11),
    ("Hello", 64, 12),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", default=str(REPO / "onnx_export"))
    ap.add_argument(
        "--out",
        default=str(REPO / "android" / "app" / "src" / "test" / "resources" / "prompt_fixtures.json"),
    )
    args = ap.parse_args()

    bundle = Path(args.bundle_dir)
    cfg = json.load(open(bundle / "config.json"))
    tk = SPTokenizer(str(bundle / "tokenizer.model"))
    pb = PromptBuilder(tk, cfg)
    n_vq = int(cfg["n_vq"])

    out = []

    # 1. Continuation, no audio prompt. The most common path.
    for text in CONTINUATION_TEXTS:
        ids = list(tk.encode(text, add_special_tokens=False))
        input_ids, attention_mask = pb.build_continuation_prompt(ids)
        out.append({
            "mode": "continuation",
            "text": text,
            "text_ids_len": len(ids),
            "has_audio_prompt": False,
            "input_ids": fingerprint(input_ids),
            "attention_mask": fingerprint(attention_mask.astype(np.int64)),
        })

    # 2. Continuation, *with* an audio prompt slot.
    for text, frames, seed in CONTINUATION_WITH_AUDIO:
        ids = list(tk.encode(text, add_special_tokens=False))
        codes = synth_codes(frames, n_vq, seed=seed)
        input_ids, attention_mask = pb.build_continuation_prompt(ids, prompt_audio_codes=codes)
        out.append({
            "mode": "continuation",
            "text": text,
            "text_ids_len": len(ids),
            "has_audio_prompt": True,
            "audio_prompt": {
                "frames": frames,
                "seed": seed,
                "n_vq": n_vq,
                "sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
            },
            "input_ids": fingerprint(input_ids),
            "attention_mask": fingerprint(attention_mask.astype(np.int64)),
        })

    # 3. Voice clone. Audio prompt is required.
    for text, frames, seed in VOICE_CLONE_FIXTURES:
        ids = list(tk.encode(text, add_special_tokens=False))
        codes = synth_codes(frames, n_vq, seed=seed)
        input_ids, attention_mask = pb.build_voice_clone_prompt(ids, prompt_audio_codes=codes)
        out.append({
            "mode": "voice_clone",
            "text": text,
            "text_ids_len": len(ids),
            "has_audio_prompt": True,
            "audio_prompt": {
                "frames": frames,
                "seed": seed,
                "n_vq": n_vq,
                "sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
            },
            "input_ids": fingerprint(input_ids),
            "attention_mask": fingerprint(attention_mask.astype(np.int64)),
        })

    # Also record the relevant slice of config so the Kotlin test does
    # not have to re-derive token IDs from a different code path.
    bundle_meta = {
        "n_vq": n_vq,
        "im_start_token_id": int(cfg["im_start_token_id"]),
        "im_end_token_id": int(cfg["im_end_token_id"]),
        "audio_start_token_id": int(cfg["audio_start_token_id"]),
        "audio_end_token_id": int(cfg["audio_end_token_id"]),
        "audio_pad_token_id": int(cfg["audio_pad_token_id"]),
        "audio_user_slot_token_id": int(cfg["audio_user_slot_token_id"]),
        "audio_assistant_slot_token_id": int(cfg["audio_assistant_slot_token_id"]),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"meta": bundle_meta, "fixtures": out},
        open(out_path, "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )
    print(f"wrote {out_path.relative_to(REPO)}  ({out_path.stat().st_size / 1024:.1f} KiB · {len(out)} fixtures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
