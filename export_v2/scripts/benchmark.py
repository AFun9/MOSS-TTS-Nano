"""benchmark - micro-benchmark a TTS-Nano release bundle.

Default target: export_v2/release/  (the make_release.py output).
Override with positional args to compare multiple bundles side-by-side.

Per bundle measures:
    - bundle disk size
    - cold session-init time
    - prefill (warm, prefill_seq=195)
    - decode_step (median of 30, past_seq=200)
    - local_cached_step (median of 30, past_seq=8)
    - estimated per-frame latency (= decode + 17 * local)

Usage:
    python benchmark.py                          # release/ only
    python benchmark.py release _build/tts_int8_inlined  # multi-bundle
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "export_v2" / "outputs"
DEFAULT_BUNDLES = ["release"]

N_GLOBAL_LAYERS = 12
N_LOCAL_LAYERS = 1
N_HEADS = 12
HEAD_DIM = 64
ROW_W = 17


def bundle_size(bundle_dir: Path) -> int:
    if not bundle_dir.exists():
        return 0
    return sum(p.stat().st_size for p in bundle_dir.iterdir() if p.is_file())


def make_sess(p: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = max(1, (os.cpu_count() or 4))
    return ort.InferenceSession(str(p), sess_options=so, providers=["CPUExecutionProvider"])


def time_session_init(p: Path, n: int = 3) -> float:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        _ = make_sess(p)
        times.append(time.perf_counter() - t0)
    return min(times)


def time_run(sess: ort.InferenceSession, feeds: dict, n_warm: int = 3, n_iter: int = 30) -> float:
    out_names = [o.name for o in sess.get_outputs()]
    for _ in range(n_warm):
        sess.run(out_names, feeds)
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        sess.run(out_names, feeds)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def bench_bundle(name: str) -> dict:
    bundle_dir = ROOT / "export_v2" / name
    print(f"\n=== {name} ===")
    if not bundle_dir.exists():
        print(f"  (not found)")
        return {"present": False}

    size_b = bundle_size(bundle_dir)
    print(f"  size: {size_b/1024/1024:.1f} MB")

    rng = np.random.default_rng(0)
    res = {"present": True, "size_mb": round(size_b/1024/1024, 1)}

    # prefill
    p_prefill = bundle_dir / "moss_tts_prefill.onnx"
    init_t = time_session_init(p_prefill)
    res["prefill_init_ms"] = round(init_t * 1000, 1)
    sess = make_sess(p_prefill)
    feeds_p = {
        "input_ids": rng.integers(0, 100, (1, 195, ROW_W), dtype=np.int32),
        "attention_mask": np.ones((1, 195), dtype=np.int32),
    }
    t_p = time_run(sess, feeds_p, n_warm=2, n_iter=5)
    res["prefill_ms"] = round(t_p * 1000, 1)
    print(f"  prefill (init={init_t*1000:.0f}ms): {t_p*1000:.1f} ms (seq=195, median of 5)")

    # decode_step
    p_dec = bundle_dir / "moss_tts_decode_step.onnx"
    init_t = time_session_init(p_dec)
    res["decode_init_ms"] = round(init_t * 1000, 1)
    sess = make_sess(p_dec)
    past = 200
    feeds_d = {
        "input_ids": rng.integers(0, 100, (1, 1, ROW_W), dtype=np.int32),
        "past_valid_lengths": np.array([past], dtype=np.int32),
    }
    for i in range(N_GLOBAL_LAYERS):
        feeds_d[f"past_key_{i}"] = rng.standard_normal((1, past, N_HEADS, HEAD_DIM), dtype=np.float32)
        feeds_d[f"past_value_{i}"] = rng.standard_normal((1, past, N_HEADS, HEAD_DIM), dtype=np.float32)
    t_d = time_run(sess, feeds_d, n_warm=3, n_iter=30)
    res["decode_step_ms"] = round(t_d * 1000, 2)
    print(f"  decode_step (init={init_t*1000:.0f}ms): {t_d*1000:.2f} ms (past=200, median of 30)")

    # local_cached_step
    p_loc = bundle_dir / "moss_tts_local_cached_step.onnx"
    init_t = time_session_init(p_loc)
    res["local_init_ms"] = round(init_t * 1000, 1)
    sess = make_sess(p_loc)
    past = 8
    feeds_l = {
        "global_hidden": rng.standard_normal((1, 768), dtype=np.float32),
        "text_token_id": np.array([10], dtype=np.int32),
        "audio_token_id": np.array([5], dtype=np.int32),
        "channel_index": np.array([2], dtype=np.int32),
        "step_type": np.array([2], dtype=np.int32),
        "past_valid_lengths": np.array([past], dtype=np.int32),
    }
    for i in range(N_LOCAL_LAYERS):
        feeds_l[f"local_past_key_{i}"] = rng.standard_normal((1, past, N_HEADS, HEAD_DIM), dtype=np.float32)
        feeds_l[f"local_past_value_{i}"] = rng.standard_normal((1, past, N_HEADS, HEAD_DIM), dtype=np.float32)
    t_l = time_run(sess, feeds_l, n_warm=3, n_iter=30)
    res["local_step_ms"] = round(t_l * 1000, 2)
    print(f"  local_cached_step (init={init_t*1000:.0f}ms): {t_l*1000:.2f} ms (past=8, median of 30)")

    # 1 frame = 1 decode + 17 local
    res["per_frame_est_ms"] = round(t_d * 1000 + 17 * t_l * 1000, 1)
    print(f"  est per-frame (1 decode + 17 local): {res['per_frame_est_ms']} ms")
    print(f"  realtime factor @ 12.5fps (80ms/frame): {res['per_frame_est_ms']/80:.2f}x")
    return res


def main():
    bundles = sys.argv[1:] or DEFAULT_BUNDLES
    report = {}
    for b in bundles:
        report[b] = bench_bundle(b)

    print()
    print("=" * 84)
    print(f"{'bundle':30s}  {'size_mb':>8s}  {'prefill_ms':>10s}  {'decode_ms':>9s}  {'local_ms':>8s}  {'frame_ms':>8s}")
    print("-" * 84)
    for b in bundles:
        r = report[b]
        if not r.get("present"):
            print(f"{b:30s}  (not found)")
            continue
        print(f"{b:30s}  {r['size_mb']:>8.1f}  {r['prefill_ms']:>10.1f}  "
              f"{r['decode_step_ms']:>9.2f}  {r['local_step_ms']:>8.2f}  {r['per_frame_est_ms']:>8.1f}")
    print("=" * 84)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "benchmark.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
