"""demo_generate - end-to-end TTS generation using the release/ ONNX bundle.

This is the canonical inference demo. It loads the final flat bundle
(release/), runs either:
    * a builtin voice from the official manifest (--voice N), OR
    * a fresh voice cloned from any wav file (--clone PATH),
through prefill + decode_step + local_cached_step + audio decoder, and
writes a WAV.

Pipeline:
    [optional] audio tokenizer encode -> audio_codes [F, 16]   (clone only)
    prefill                     -> kv cache + last hidden  (global transformer)
    decode_step                 -> next-frame global hidden + grow kv cache
    local_cached_step           -> 17 sub-steps per frame, Python sampling
    audio tokenizer decode      -> audio_codes [F, 16] -> wav PCM

Bundle source:
    Defaults to export_v2/release/ (the make_release.py output).
    Override with --bundle DIR or $TTS_BUNDLE env var.

Output: WAV files in export_v2/outputs/.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "export_v2" / "reference" / "browser_poc_manifest.json"

# Default bundle is the final flat release/ produced by make_release.py.
DEFAULT_BUNDLE = ROOT / "export_v2" / "release"
OUT_DIR = ROOT / "export_v2" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_VQ = 16
NUM_GLOBAL_LAYERS = 12
NUM_LOCAL_LAYERS = 1
N_HEADS = 12
HEAD_DIM = 64


def make_session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Use real CPU count, not provider count (was a bug capping us to 1-2 threads).
    so.intra_op_num_threads = max(1, (os.cpu_count() or 4))
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def load_manifest():
    with open(MANIFEST) as f:
        return json.load(f)


CODEC_SR = 48000      # codec encoder expects 48 kHz stereo
CODEC_CHANNELS = 2


def _read_wav_for_codec(wav_path: Path) -> np.ndarray:
    """Read a wav (any sr/channels), return float32 [2, T] @ 48 kHz in [-1, 1].

    Uses linear interpolation for resampling (sufficient for codec encoder;
    the encoder is robust to mild resample artifacts since it does its own
    spectrogram-style feature extraction).
    """
    import soundfile as sf
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)  # [T, C]
    data = data.T  # -> [C, T]
    n_in = data.shape[1]

    if data.shape[0] == 1:
        data = np.repeat(data, CODEC_CHANNELS, axis=0)
    elif data.shape[0] > CODEC_CHANNELS:
        data = data[:CODEC_CHANNELS]
    elif data.shape[0] < CODEC_CHANNELS:
        pad = CODEC_CHANNELS - data.shape[0]
        data = np.concatenate([data, np.tile(data[-1:], (pad, 1))], axis=0)

    if sr != CODEC_SR:
        n_out = int(round(n_in * CODEC_SR / sr))
        x_in = np.linspace(0.0, 1.0, n_in, dtype=np.float64)
        x_out = np.linspace(0.0, 1.0, n_out, dtype=np.float64)
        out = np.empty((CODEC_CHANNELS, n_out), dtype=np.float32)
        for c in range(CODEC_CHANNELS):
            out[c] = np.interp(x_out, x_in, data[c]).astype(np.float32)
        data = out
        print(f"[clone] resampled {sr} Hz -> {CODEC_SR} Hz  ({n_in} -> {n_out} samples)")

    peak = float(np.max(np.abs(data)))
    if peak > 1.0:
        data = data / peak
    return data


def clone_voice_codes(wav_path: Path, encoder_sess: ort.InferenceSession,
                      max_seconds: float = 20.0) -> np.ndarray:
    """Encode reference wav -> audio_codes [F, 16] using the codec encoder.

    Caps reference length to `max_seconds` (longer clips just bloat prefill;
    20 s of 48 kHz audio gives ~250 codec frames, plenty for voice timbre).
    """
    pcm = _read_wav_for_codec(wav_path)
    max_n = int(max_seconds * CODEC_SR)
    if pcm.shape[1] > max_n:
        print(f"[clone] truncating {pcm.shape[1]/CODEC_SR:.1f}s -> {max_seconds:.1f}s")
        pcm = pcm[:, :max_n]

    waveform = pcm[None, ...]  # [1, 2, T]
    input_lengths = np.array([pcm.shape[1]], dtype=np.int32)
    print(f"[clone] encoder feed: waveform={waveform.shape} ({waveform.shape[2]/CODEC_SR:.2f}s)")

    out = encoder_sess.run(["audio_codes", "audio_code_lengths"],
                           {"waveform": waveform, "input_lengths": input_lengths})
    codes_full = out[0][0]               # [F_padded, 16]  int32
    code_len = int(out[1][0])
    codes = codes_full[:code_len].astype(np.int64)
    print(f"[clone] encoded -> {codes.shape[0]} codec frames")
    return codes


def build_input_ids(manifest: dict, prompt_audio_codes: np.ndarray,
                    text_token_ids: list[int]) -> np.ndarray:
    """Build the input_ids[1, T, 17] sequence matching the official voice_clone
    layout (replicates _build_audio_prefix_rows + voice_clone branch in
    modeling_moss_tts_nano.py).

    Layout (rows are time-major):
        user_prompt_prefix_token_ids                 [text rows]
        [audio_start_token_id]                       [single text row]
        prompt_audio_codes                           [F rows, col[0]=user_slot, col[1:]=codes]
        [audio_end_token_id]                         [single text row]
        user_prompt_after_reference_token_ids        [text rows]
        text_token_ids (target text)                 [text rows]
        assistant_prompt_prefix_token_ids            [text rows]
        [audio_start_token_id]                       [single text row]
    """
    cfg = manifest["tts_config"]
    n_vq = cfg["n_vq"]
    audio_pad = cfg["audio_pad_token_id"]
    audio_user_slot = cfg["audio_user_slot_token_id"]
    audio_start = cfg["audio_start_token_id"]
    audio_end = cfg["audio_end_token_id"]

    F = prompt_audio_codes.shape[0]
    pre = manifest["prompt_templates"]
    user_prefix = pre["user_prompt_prefix_token_ids"]
    user_after = pre["user_prompt_after_reference_token_ids"]
    asst_prefix = pre["assistant_prompt_prefix_token_ids"]

    rows = []

    def text_row(tid):
        r = [audio_pad] * (n_vq + 1)
        r[0] = int(tid)
        return r

    for tid in user_prefix:
        rows.append(text_row(tid))
    rows.append(text_row(audio_start))                     # audio_start before audio prefix
    for f in range(F):                                      # audio prefix rows
        r = [int(audio_user_slot)] + prompt_audio_codes[f].tolist()
        rows.append(r)
    rows.append(text_row(audio_end))                        # audio_end after audio prefix
    for tid in user_after:
        rows.append(text_row(tid))
    for tid in text_token_ids:
        rows.append(text_row(tid))
    for tid in asst_prefix:
        rows.append(text_row(tid))
    rows.append(text_row(audio_start))                      # final audio_start kicks off generation

    arr = np.asarray([rows], dtype=np.int32)  # [1, T, 17]
    return arr


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def sample_token(
    logits: np.ndarray,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_seen: np.ndarray | None = None,
    repetition_penalty: float = 1.0,
    rng: np.random.Generator | None = None,
) -> int:
    rng = rng or np.random.default_rng()
    logits = logits.astype(np.float32).copy()  # [V]

    if repetition_seen is not None and repetition_penalty != 1.0:
        seen_idx = np.nonzero(repetition_seen)[0]
        if seen_idx.size > 0:
            v = logits[seen_idx]
            logits[seen_idx] = np.where(v < 0, v * repetition_penalty, v / repetition_penalty)

    logits = logits / max(temperature, 1e-5)

    if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
        thr = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < thr, -np.inf, logits)

    if top_p is not None and 0.0 < top_p < 1.0:
        order = np.argsort(-logits)
        sorted_logits = logits[order]
        sorted_probs = softmax(sorted_logits)
        cum = np.cumsum(sorted_probs)
        # remove tail beyond top_p (keep at least 1)
        remove = cum > top_p
        remove[1:] = remove[:-1].copy()
        remove[0] = False
        sorted_logits = np.where(remove, -np.inf, sorted_logits)
        new_logits = np.full_like(logits, -np.inf)
        new_logits[order] = sorted_logits
        logits = new_logits

    probs = softmax(logits)
    if not np.isfinite(probs).all() or probs.sum() <= 0:
        return int(np.argmax(logits))
    return int(rng.choice(logits.shape[-1], p=probs))


def empty_kv_cache(num_layers: int, batch: int = 1) -> list[np.ndarray]:
    """Return 2L empty [B, 0, H, D] tensors as starting past_kv."""
    out = []
    for _ in range(num_layers):
        out.append(np.zeros((batch, 0, N_HEADS, HEAD_DIM), dtype=np.float32))
        out.append(np.zeros((batch, 0, N_HEADS, HEAD_DIM), dtype=np.float32))
    return out


def run_prefill(sess: ort.InferenceSession, input_ids: np.ndarray):
    attention_mask = np.ones(input_ids.shape[:2], dtype=np.int32)
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(out_names, {"input_ids": input_ids, "attention_mask": attention_mask})
    hidden = outs[0]                 # [1, T, 768]
    kv = outs[1:]                    # 2L tensors [1, T, 12, 64]
    return hidden, kv


def run_decode_step(sess: ort.InferenceSession, next_row: np.ndarray, past_kv: list[np.ndarray]):
    past_seq = past_kv[0].shape[1]
    feeds = {
        "input_ids": next_row,                                          # [1, 1, 17]
        "past_valid_lengths": np.array([past_seq], dtype=np.int32),
    }
    for i in range(NUM_GLOBAL_LAYERS):
        feeds[f"past_key_{i}"] = past_kv[2 * i]
        feeds[f"past_value_{i}"] = past_kv[2 * i + 1]
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(out_names, feeds)
    hidden = outs[0]                 # [1, 1, 768]
    kv = outs[1:]
    return hidden, kv


def run_local_step(
    sess: ort.InferenceSession,
    global_hidden: np.ndarray,       # [1, 768]
    text_token_id: int,
    audio_token_id: int,
    channel_index: int,
    step_type: int,
    past_kv: list[np.ndarray],
):
    past_seq = past_kv[0].shape[1]
    feeds = {
        "global_hidden": global_hidden,
        "text_token_id": np.array([text_token_id], dtype=np.int32),
        "audio_token_id": np.array([audio_token_id], dtype=np.int32),
        "channel_index": np.array([channel_index], dtype=np.int32),
        "step_type": np.array([step_type], dtype=np.int32),
        "past_valid_lengths": np.array([past_seq], dtype=np.int32),
    }
    for i in range(NUM_LOCAL_LAYERS):
        feeds[f"local_past_key_{i}"] = past_kv[2 * i]
        feeds[f"local_past_value_{i}"] = past_kv[2 * i + 1]
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(out_names, feeds)
    text_logits = outs[0]            # [1, 16384]
    audio_logits = outs[1]           # [1, 16, 1024]
    kv = outs[2:]
    return text_logits, audio_logits, kv


def generate(
    text_id: str,
    max_new_frames: int,
    seed: int = 0,
    do_sample: bool = True,
    bundle_dir: Path = DEFAULT_BUNDLE,
    voice_idx: int | None = None,
    clone_wav: Path | None = None,
):
    if (voice_idx is None) == (clone_wav is None):
        raise ValueError("specify exactly one of voice_idx (builtin) or clone_wav (clone)")

    print(f"[bundle] {bundle_dir}")
    s_pre = make_session(bundle_dir / "moss_tts_prefill.onnx")
    s_dec = make_session(bundle_dir / "moss_tts_decode_step.onnx")
    s_loc = make_session(bundle_dir / "moss_tts_local_cached_step.onnx")
    audio_decoder = bundle_dir / "moss_audio_tokenizer_decode_full.onnx"

    manifest = load_manifest()
    cfg = manifest["tts_config"]
    audio_pad = cfg["audio_pad_token_id"]
    pad = cfg["pad_token_id"]
    audio_assistant_slot = cfg["audio_assistant_slot_token_id"]
    audio_end = cfg["audio_end_token_id"]
    gen_d = manifest["generation_defaults"]

    sample = next(s for s in manifest["text_samples"] if s["id"] == text_id)
    print(f"[text] '{sample['text'][:60]}...'  ({len(sample['text_token_ids'])} tokens)")

    if clone_wav is not None:
        print(f"[clone] reference wav: {clone_wav}")
        encoder_sess = make_session(bundle_dir / "moss_audio_tokenizer_encode.onnx")
        prompt_codes = clone_voice_codes(clone_wav, encoder_sess)
        del encoder_sess
    else:
        voice = manifest["builtin_voices"][voice_idx]
        prompt_codes = np.asarray(voice["prompt_audio_codes"], dtype=np.int64)
        print(f"[builtin] voice='{voice.get('display_name', voice.get('voice'))}' "
              f"prompt_frames={prompt_codes.shape[0]}")

    input_ids = build_input_ids(manifest, prompt_codes, sample["text_token_ids"])
    print(f"[seq] prefill_seq = {input_ids.shape[1]}")

    rng = np.random.default_rng(seed)

    print("[run] prefill ...")
    global_hidden_full, global_kv = run_prefill(s_pre, input_ids)
    last_hidden = global_hidden_full[:, -1, :]   # [1, 768]

    audio_codes_per_frame: list[np.ndarray] = []  # each [16] int
    repetition_seen = np.zeros((N_VQ, 1024), dtype=np.bool_)

    for frame_idx in range(max_new_frames):
        # --- local 17-step loop ---
        local_kv = empty_kv_cache(NUM_LOCAL_LAYERS)

        # step 0: text decision (input = global_hidden)
        text_logits, _audio_logits, local_kv = run_local_step(
            s_loc, last_hidden, 0, 0, 0, step_type=0, past_kv=local_kv,
        )
        # text token must be one of {assistant_slot, end}
        cand_ids = np.array([audio_assistant_slot, audio_end], dtype=np.int64)
        cand_logits = text_logits[0, cand_ids]
        if do_sample:
            sampled_idx = sample_token(
                cand_logits,
                temperature=float(gen_d["text_temperature"]),
                top_k=int(gen_d["text_top_k"]),
                top_p=float(gen_d["text_top_p"]),
                rng=rng,
            )
        else:
            sampled_idx = int(np.argmax(cand_logits))
        next_text_token = int(cand_ids[sampled_idx])

        if next_text_token == audio_end:
            print(f"[frame {frame_idx}] audio_end token sampled - stopping")
            break

        # step 1..16: 16 audio channels
        frame_codes = np.empty(N_VQ, dtype=np.int64)
        cur_text_for_step = next_text_token  # used for step_type=1
        cur_audio = 0
        cur_channel = 0
        for k in range(N_VQ):
            if k == 0:
                # step_type=1: input = wte(next_text_token)
                _tl, audio_logits, local_kv = run_local_step(
                    s_loc, last_hidden, cur_text_for_step, 0, 0,
                    step_type=1, past_kv=local_kv,
                )
            else:
                # step_type=2: input = audio_embeddings[k-1](sampled_audio[k-1])
                _tl, audio_logits, local_kv = run_local_step(
                    s_loc, last_hidden, 0, int(frame_codes[k - 1]), k - 1,
                    step_type=2, past_kv=local_kv,
                )
            ch_logits = audio_logits[0, k]   # [1024]
            if do_sample:
                tok = sample_token(
                    ch_logits,
                    temperature=float(gen_d["audio_temperature"]),
                    top_k=int(gen_d["audio_top_k"]),
                    top_p=float(gen_d["audio_top_p"]),
                    repetition_seen=repetition_seen[k],
                    repetition_penalty=float(gen_d.get("audio_repetition_penalty", 1.0)),
                    rng=rng,
                )
            else:
                tok = int(np.argmax(ch_logits))
            frame_codes[k] = tok
            repetition_seen[k, tok] = True

        audio_codes_per_frame.append(frame_codes)

        if (frame_idx + 1) % 25 == 0:
            print(f"[frame {frame_idx + 1}/{max_new_frames}] codes[:5]={frame_codes[:5].tolist()}")

        # --- global decode_step: build next_row from frame_codes, append, advance ---
        # Match _build_generation_row: col[0]=audio_assistant_slot, col[1:]=audio_token_ids
        next_row = np.empty((1, 1, N_VQ + 1), dtype=np.int32)
        next_row[0, 0, 0] = audio_assistant_slot
        next_row[0, 0, 1:] = frame_codes.astype(np.int32)
        last_hidden_full, global_kv = run_decode_step(s_dec, next_row, list(global_kv))
        last_hidden = last_hidden_full[:, -1, :]

    if not audio_codes_per_frame:
        raise RuntimeError("no audio codes generated")

    audio_codes = np.stack(audio_codes_per_frame, axis=0).astype(np.int32)  # [F, 16]
    print(f"[done] generated {audio_codes.shape[0]} frames")

    # ---- decode to wav via codec decoder from the same bundle ----
    print(f"[load] {audio_decoder.name}")
    s_codec = make_session(audio_decoder)
    feeds = {
        "audio_codes": audio_codes[None, ...],   # [1, F, 16]
        "audio_code_lengths": np.array([audio_codes.shape[0]], dtype=np.int32),
    }
    out = s_codec.run(["audio", "audio_lengths"], feeds)
    pcm = out[0]                  # [1, channels, T]
    L = int(out[1][0])
    pcm = pcm[0, :, :L]            # [channels, T]
    return pcm, audio_codes


def save_wav(path: Path, pcm: np.ndarray, sr: int = 48000):
    import wave
    pcm = np.clip(pcm, -1.0, 1.0)
    pcm_int = (pcm * 32767.0).astype(np.int16)  # [channels, T]
    if pcm_int.shape[0] > 1:
        # interleave channels
        interleaved = pcm_int.T.reshape(-1).astype(np.int16)
        channels = pcm_int.shape[0]
    else:
        interleaved = pcm_int[0]
        channels = 1
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(interleaved.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", default=os.environ.get("TTS_BUNDLE", str(DEFAULT_BUNDLE)),
                    help=f"path to release bundle dir (default: {DEFAULT_BUNDLE.relative_to(ROOT)})")
    voice_grp = ap.add_mutually_exclusive_group()
    voice_grp.add_argument("--voice", type=int, default=None,
                           help="builtin_voices index 0..17 (default 0 if no --clone)")
    voice_grp.add_argument("--clone", type=str, default=None, metavar="WAV",
                           help="path to a reference wav; clones its timbre via codec encoder")
    ap.add_argument("--text", default="zh_browser_poc",
                    choices=["zh_browser_poc", "en_browser_poc"])
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-sample", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bundle_dir = Path(args.bundle)
    if not bundle_dir.exists():
        raise SystemExit(f"bundle dir not found: {bundle_dir}\n"
                         f"Run `python export_v2/scripts/make_release.py` first.")

    voice_idx = None
    clone_wav = None
    if args.clone:
        clone_wav = Path(args.clone)
        if not clone_wav.exists():
            raise SystemExit(f"clone wav not found: {clone_wav}")
        voice_tag = f"clone-{clone_wav.stem}"
    else:
        voice_idx = args.voice if args.voice is not None else 0
        voice_tag = f"voice{voice_idx}"

    pcm, codes = generate(args.text, args.frames, args.seed,
                          do_sample=not args.no_sample, bundle_dir=bundle_dir,
                          voice_idx=voice_idx, clone_wav=clone_wav)

    out = Path(args.out) if args.out else OUT_DIR / f"{args.text}_{voice_tag}_seed{args.seed}.wav"
    save_wav(out, pcm)
    print(f"[wav] saved {out}  shape={pcm.shape}  duration={pcm.shape[-1]/48000:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
