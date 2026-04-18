"""dump_inference_trace_v2 - byte-equal Python <-> Kotlin InferenceLoop trace.

Produces an `inference_traces_v2/trace.json` consumed by the Kotlin
on-device test `InferenceLoopTraceTest`. Both sides:
    - load the SAME 4 TTS ONNX graphs (`prefill` / `decode_step` /
      `local_cached_step`) from `export_v2/release/`,
    - feed the SAME `input_ids` (taken from the deterministic text_samples
      + builtin_voices in `reference/browser_poc_manifest.json`),
    - run argmax-only sampling (temperature=0, no repetition penalty),
      so the only legitimate source of divergence is ORT's CPU EP itself
      (which is bit-deterministic for a fixed graph + thread setting +
      EP).

Each scenario records:
    - text_id + voice_idx + the SHA256/length of the prompt input_ids
    - frame-by-frame: 16-channel argmax codes and the text-head argmax
    - n_generated and eos_frame
    - SHA256 of the full [n_generated, 16] code grid (last-line-of-defence)

Tiny on purpose (`max_frames=8`) so the round-trip on a phone takes
<2 minutes and the JSON stays small.

Re-generate with:
    python tools/dump_inference_trace_v2.py
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "export_v2" / "scripts"))
import demo_generate  # noqa: E402

RELEASE = ROOT / "export_v2" / "release"
REFERENCE_MANIFEST = ROOT / "export_v2" / "reference" / "browser_poc_manifest.json"
OUT_DIR = ROOT / "android" / "app" / "src" / "androidTest" / "assets" / "inference_traces_v2"
OUT = OUT_DIR / "trace.json"

N_VQ = demo_generate.N_VQ
NUM_GLOBAL_LAYERS = demo_generate.NUM_GLOBAL_LAYERS
NUM_LOCAL_LAYERS = demo_generate.NUM_LOCAL_LAYERS


def sha256_int_grid(arr: np.ndarray) -> str:
    """SHA256 of an int grid as little-endian int64 bytes (matches Kotlin)."""
    return hashlib.sha256(arr.astype(np.int64).tobytes()).hexdigest()


def make_sessions():
    s_pre = demo_generate.make_session(RELEASE / "moss_tts_prefill.onnx")
    s_dec = demo_generate.make_session(RELEASE / "moss_tts_decode_step.onnx")
    s_loc = demo_generate.make_session(RELEASE / "moss_tts_local_cached_step.onnx")
    return s_pre, s_dec, s_loc


def run_argmax(
    s_pre, s_dec, s_loc, manifest,
    text_ids: list[int], prompt_codes: np.ndarray, max_frames: int,
):
    cfg = manifest["tts_config"]
    audio_assistant_slot = cfg["audio_assistant_slot_token_id"]
    audio_end = cfg["audio_end_token_id"]

    input_ids = demo_generate.build_input_ids(manifest, prompt_codes, text_ids)
    prompt_seq_len = int(input_ids.shape[1])
    prompt_sha = hashlib.sha256(input_ids.astype(np.int64).tobytes()).hexdigest()

    global_hidden_full, global_kv = demo_generate.run_prefill(s_pre, input_ids)
    last_hidden = global_hidden_full[:, -1, :]

    steps = []
    audio_codes_per_frame: list[np.ndarray] = []
    eos_frame = None

    for frame_idx in range(max_frames):
        local_kv = demo_generate.empty_kv_cache(NUM_LOCAL_LAYERS)
        text_logits, _al, local_kv = demo_generate.run_local_step(
            s_loc, last_hidden, 0, 0, 0, step_type=0, past_kv=local_kv,
        )
        cand_ids = np.array([audio_assistant_slot, audio_end], dtype=np.int64)
        cand_logits = text_logits[0, cand_ids]
        sampled_idx = int(np.argmax(cand_logits))
        next_text_token = int(cand_ids[sampled_idx])

        if next_text_token == audio_end:
            eos_frame = frame_idx
            steps.append({
                "frame": frame_idx,
                "text_argmax_id": next_text_token,
                "is_eos": True,
                "frame_codes": [],
            })
            break

        frame_codes = np.empty(N_VQ, dtype=np.int64)
        for k in range(N_VQ):
            if k == 0:
                _tl, audio_logits, local_kv = demo_generate.run_local_step(
                    s_loc, last_hidden, next_text_token, 0, 0,
                    step_type=1, past_kv=local_kv,
                )
            else:
                _tl, audio_logits, local_kv = demo_generate.run_local_step(
                    s_loc, last_hidden, 0, int(frame_codes[k - 1]), k - 1,
                    step_type=2, past_kv=local_kv,
                )
            tok = int(np.argmax(audio_logits[0, k]))
            frame_codes[k] = tok

        audio_codes_per_frame.append(frame_codes)
        steps.append({
            "frame": frame_idx,
            "text_argmax_id": next_text_token,
            "is_eos": False,
            "frame_codes": [int(x) for x in frame_codes.tolist()],
        })

        next_row = np.empty((1, 1, N_VQ + 1), dtype=np.int32)
        next_row[0, 0, 0] = audio_assistant_slot
        next_row[0, 0, 1:] = frame_codes.astype(np.int32)
        last_hidden_full, global_kv = demo_generate.run_decode_step(s_dec, next_row, list(global_kv))
        last_hidden = last_hidden_full[:, -1, :]

    n_gen = len(audio_codes_per_frame)
    if n_gen > 0:
        codes_grid = np.stack(audio_codes_per_frame, axis=0)
        final_sha = sha256_int_grid(codes_grid)
    else:
        final_sha = sha256_int_grid(np.empty((0, N_VQ), dtype=np.int64))

    return {
        "prompt_seq_len": prompt_seq_len,
        "prompt_input_ids_sha256": prompt_sha,
        "n_generated": n_gen,
        "eos_frame": eos_frame,
        "final_codes_sha256": final_sha,
        "steps": steps,
    }


def build_scenarios(s_pre, s_dec, s_loc, ref_manifest, max_frames: int):
    """Three small scenarios spanning text languages and voices."""
    plan = [
        ("zh_voice0", "zh_browser_poc", 0),
        ("en_voice0", "en_browser_poc", 0),
        ("zh_voice5", "zh_browser_poc", 5),
    ]
    scenarios = []
    for sid, text_id, voice_idx in plan:
        sample = next(s for s in ref_manifest["text_samples"] if s["id"] == text_id)
        voice = ref_manifest["builtin_voices"][voice_idx]
        prompt_codes = np.asarray(voice["prompt_audio_codes"], dtype=np.int64)
        text_ids = list(sample["text_token_ids"])
        print(f"[scn] {sid:>12}  text='{sample['text'][:24]}…'  "
              f"voice='{voice.get('display_name', voice.get('voice'))}'  "
              f"prompt_frames={prompt_codes.shape[0]}  text_tokens={len(text_ids)}")
        result = run_argmax(s_pre, s_dec, s_loc, ref_manifest,
                            text_ids, prompt_codes, max_frames)
        print(f"        prompt_seq={result['prompt_seq_len']}  "
              f"n_gen={result['n_generated']}  "
              f"eos={result['eos_frame']}  "
              f"final_sha={result['final_codes_sha256'][:12]}")
        scenarios.append({
            "id": sid,
            "text_id": text_id,
            "voice_idx": voice_idx,
            "text_token_ids": text_ids,
            "prompt_audio_codes": {
                "frames": int(prompt_codes.shape[0]),
                "n_vq": int(prompt_codes.shape[1]),
                "codes": [int(x) for x in prompt_codes.flatten().tolist()],
            },
            **result,
        })
    return scenarios


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-frames", type=int, default=8,
                    help="how many frames to argmax-decode per scenario (default 8)")
    args = ap.parse_args()

    ref_manifest = json.loads(REFERENCE_MANIFEST.read_text())
    s_pre, s_dec, s_loc = make_sessions()
    try:
        scenarios = build_scenarios(s_pre, s_dec, s_loc, ref_manifest, args.max_frames)
    finally:
        del s_pre, s_dec, s_loc

    payload = {
        "schema": "moss-tts-nano-inference-trace-v2/1",
        "release_version": json.loads((RELEASE / "manifest.json").read_text()).get("release_version"),
        "n_vq": N_VQ,
        "max_frames": args.max_frames,
        "sampling": {
            "mode": "argmax",
            "text_temperature": 0.0,
            "audio_temperature": 0.0,
            "audio_repetition_penalty": 1.0,
        },
        "scenarios": scenarios,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    size_kb = OUT.stat().st_size / 1024
    print(f"\nwrote {OUT.relative_to(ROOT)}  ({len(scenarios)} scenarios, {size_kb:.1f} KiB)")


if __name__ == "__main__":
    main()
