"""make_release - build the FINAL TTS-Nano ONNX bundle in one shot.

This is the only script you need to ship a release. It chains the staging
scripts (m1..m6) into a single reproducible pipeline and assembles the
artifacts into export_v2/release/ with a manifest + README.

Pipeline:
    [stage 1: TTS export]    m1 + m2 + m3 + m3.2     -> _build/tts_fp32/
    [stage 2: TTS share]     m4 (split mode)         -> _build/tts_fp32_shared/
    [stage 3: TTS quantize]  m5_1 (INT8 dynamic)     -> _build/tts_int8_inlined/
    [stage 4: codec quant]   m6_1 (INT8 dynamic)     -> _build/codec_int8_inlined/
    [stage 5: assemble]      m4.share_group unified  -> release/
                                                       moss_tts_shared.data
                                                       moss_audio_tokenizer_shared.data
    [stage 6: package]       copy tokenizer.model + manifest.json + README.md

Output bundle (flat, ~165 MB):
    release/
      moss_tts_prefill.onnx               (~1.5 MB graph)
      moss_tts_decode_step.onnx
      moss_tts_local_decoder.onnx
      moss_tts_local_cached_step.onnx
      moss_tts_shared.data                (~140 MB INT8 weights, dedupped)
      moss_audio_tokenizer_encode.onnx
      moss_audio_tokenizer_decode_full.onnx
      moss_audio_tokenizer_decode_step.onnx
      moss_audio_tokenizer_shared.data    (~22 MB INT8 weights, dedupped)
      tokenizer.model                     (SentencePiece, ~1.3 MB)
      manifest.json                       (provenance + IO + SHA256)
      README.md

Usage:
    python make_release.py                # full pipeline (~10 minutes)
    python make_release.py --skip-export  # reuse _build/tts_fp32 if it exists
    python make_release.py --skip-quant   # reuse _build/*_int8_inlined if they exist
    python make_release.py --keep-build   # don't delete _build/ at the end
"""
from __future__ import annotations
import argparse
import datetime as _dt
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[2]
EXPORT_V2 = ROOT / "export_v2"
SCRIPTS = EXPORT_V2 / "scripts"

BUILD = EXPORT_V2 / "_build"
TTS_FP32 = BUILD / "tts_fp32"
TTS_FP32_SHARED = BUILD / "tts_fp32_shared"
TTS_INT8_INLINED = BUILD / "tts_int8_inlined"
CODEC_INT8_INLINED = BUILD / "codec_int8_inlined"

RELEASE = EXPORT_V2 / "release"

CKPT_DIR = ROOT / "MOSS-TTS-Nano-100M"
CODEC_REF_DIR = EXPORT_V2 / "reference" / "codec_full"
TOKENIZER_SRC = EXPORT_V2 / "reference" / "full" / "tokenizer.model"
SOURCE_MANIFEST = EXPORT_V2 / "reference" / "browser_poc_manifest.json"

TTS_GRAPHS = (
    "moss_tts_prefill.onnx",
    "moss_tts_decode_step.onnx",
    "moss_tts_local_decoder.onnx",
    "moss_tts_local_cached_step.onnx",
)
CODEC_GRAPHS = (
    "moss_audio_tokenizer_encode.onnx",
    "moss_audio_tokenizer_decode_full.onnx",
    "moss_audio_tokenizer_decode_step.onnx",
)
TTS_BLOB = "moss_tts_shared.data"
CODEC_BLOB = "moss_audio_tokenizer_shared.data"

RELEASE_VERSION = "1.0.0"


# ------------------------------------------------------------------ helpers

def _section(title: str) -> None:
    bar = "=" * 70
    print()
    print(bar)
    print(f"  {title}")
    print(bar)


def _run(script: str, *args: str) -> None:
    """Run an m-script in-process via subprocess for clean isolation."""
    cmd = [sys.executable, str(SCRIPTS / script), *args]
    print(f"  $ {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(ROOT))
    dt = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"{script} exited with code {rc}")
    print(f"  -> {script} done in {dt:.1f}s")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _onnx_io(graph_path: Path) -> dict:
    m = onnx.load(str(graph_path), load_external_data=False)

    def _shape(t):
        return [d.dim_value or d.dim_param or "?" for d in t.type.tensor_type.shape.dim]

    return {
        "inputs":  [{"name": i.name, "shape": _shape(i)} for i in m.graph.input],
        "outputs": [{"name": o.name, "shape": _shape(o)} for o in m.graph.output],
    }


def _checkpoint_fingerprint(ckpt_dir: Path) -> dict:
    """Fingerprint the source PyTorch checkpoint (model.safetensors + config)."""
    fp = {"path": str(ckpt_dir.relative_to(ROOT)), "files": {}}
    for name in sorted(("config.json", "configuration_moss_tts_nano.py", "model.safetensors")):
        p = ckpt_dir / name
        if p.exists():
            fp["files"][name] = {
                "size": p.stat().st_size,
                "sha256": _sha256(p),
            }
    return fp


# --------------------------------------------------------- assembly stages

def _assemble_release() -> dict:
    """Move INT8-inlined graphs into release/ with one unified blob each."""
    sys.path.insert(0, str(SCRIPTS))
    from m4_share_external_data import share_group  # noqa: E402

    if RELEASE.exists():
        shutil.rmtree(RELEASE)
    RELEASE.mkdir(parents=True, exist_ok=True)

    print("\n  >> sharing TTS (4 graphs -> 1 blob)")
    tts_rep = share_group(TTS_GRAPHS, TTS_INT8_INLINED, RELEASE, TTS_BLOB)

    print("\n  >> sharing codec (3 graphs -> 1 blob)")
    codec_rep = share_group(CODEC_GRAPHS, CODEC_INT8_INLINED, RELEASE, CODEC_BLOB)

    print("\n  >> copy tokenizer.model")
    if not TOKENIZER_SRC.exists():
        raise FileNotFoundError(f"missing {TOKENIZER_SRC}")
    shutil.copy(TOKENIZER_SRC, RELEASE / "tokenizer.model")

    return {"tts": tts_rep, "codec": codec_rep}


def _write_manifest(share_report: dict) -> Path:
    """Emit release/manifest.json with full provenance + IO + SHA256."""
    print("\n  >> generating manifest.json")
    src = json.loads(SOURCE_MANIFEST.read_text())

    def _ext_data(graph_files, blob):
        return {fn: [blob] for fn in graph_files}

    files_meta = {}
    total_bytes = 0
    for fn in TTS_GRAPHS + CODEC_GRAPHS + (TTS_BLOB, CODEC_BLOB, "tokenizer.model"):
        p = RELEASE / fn
        sz = p.stat().st_size
        total_bytes += sz
        files_meta[fn] = {"size": sz, "sha256": _sha256(p)}

    tts_io = {fn: _onnx_io(RELEASE / fn) for fn in TTS_GRAPHS}
    codec_io = {fn: _onnx_io(RELEASE / fn) for fn in CODEC_GRAPHS}

    manifest = {
        "format_version": 1,
        "release_version": RELEASE_VERSION,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "generator": {
            "script": "export_v2/scripts/make_release.py",
            "python": sys.version.split()[0],
        },
        "source": {
            "tts_checkpoint": _checkpoint_fingerprint(CKPT_DIR),
            "codec_reference": {
                "path": str(CODEC_REF_DIR.relative_to(ROOT)),
                "note": "FP32 codec graphs from official MOSS-Audio-Tokenizer-Nano-ONNX, dynamic-INT8 quantized in this release.",
            },
        },
        "quantization": {
            "tts": {
                "scheme": "dynamic INT8 weight-only",
                "ops":   ["MatMul", "Gather"],
                "per_channel": True,
                "reduce_range": False,
                "notes": "Gather covers dynamo-emitted lm_heads; weight-tied lm_heads are untied to keep Gather (vocab,hidden) and MatMul (hidden,vocab) layouts independent.",
            },
            "codec": {
                "scheme": "dynamic INT8 weight-only",
                "ops":   ["MatMul", "Gather", "Conv"],
                "per_channel": True,
                "reduce_range": False,
            },
        },
        "tts": {
            "graphs": list(TTS_GRAPHS),
            "shared_blob": TTS_BLOB,
            "external_data_files": _ext_data(TTS_GRAPHS, TTS_BLOB),
            "io": tts_io,
            "model_config": {
                "n_vq": 16,
                "row_width": 17,
                "hidden_size": 768,
                "global_layers": 12,
                "global_heads": 12,
                "head_dim": 64,
                "local_layers": 1,
                "local_heads": 12,
                "local_head_dim": 64,
                "vocab_size": 16384,
                "audio_codebook_size": 1024,
            },
        },
        "codec": {
            "graphs": list(CODEC_GRAPHS),
            "shared_blob": CODEC_BLOB,
            "external_data_files": _ext_data(CODEC_GRAPHS, CODEC_BLOB),
            "io": codec_io,
            "codec_config": {
                "sample_rate": 48000,
                "channels": 2,
                "downsample_rate": 3840,
                "num_quantizers": 16,
            },
        },
        "tts_config": src["tts_config"],
        "prompt_templates": src.get("prompt_templates", {}),
        "generation_defaults": src.get("generation_defaults", {}),
        "tokenizer": {
            "type": "sentencepiece",
            "file": "tokenizer.model",
        },
        "files": files_meta,
        "total_bytes": total_bytes,
        "share_report": share_report,
    }
    out = RELEASE / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"     wrote {out}  ({out.stat().st_size/1024:.1f} KB)")
    return out


def _write_readme(total_bytes: int) -> Path:
    print("  >> generating README.md")
    text = f"""# MOSS-TTS-Nano ONNX release v{RELEASE_VERSION}

Final, self-contained INT8 dynamic-quantized ONNX bundle for MOSS-TTS-Nano-100M.
Produced by `export_v2/scripts/make_release.py` from the official PyTorch
checkpoint plus the official codec FP32 ONNX graphs.

## Layout (flat, {total_bytes/1024/1024:.1f} MB total)

| File                                       | Purpose                                       |
|--------------------------------------------|-----------------------------------------------|
| `moss_tts_prefill.onnx`                    | Global transformer prefill (full prompt)      |
| `moss_tts_decode_step.onnx`                | Global transformer single-step decode         |
| `moss_tts_local_decoder.onnx`              | Local transformer reference (no KV cache)     |
| `moss_tts_local_cached_step.onnx`          | Local transformer single-step (KV cache)      |
| `moss_tts_shared.data`                     | INT8 weights, shared across all 4 TTS graphs  |
| `moss_audio_tokenizer_encode.onnx`         | Audio -> codes encoder                        |
| `moss_audio_tokenizer_decode_full.onnx`    | Codes -> audio decoder (full sequence)        |
| `moss_audio_tokenizer_decode_step.onnx`    | Codes -> audio decoder (streaming step)       |
| `moss_audio_tokenizer_shared.data`         | INT8 weights, shared across all 3 codec graphs|
| `tokenizer.model`                          | SentencePiece tokenizer                        |
| `manifest.json`                            | Provenance, IO names/shapes, SHA256, config   |

## Loading

ONNX Runtime auto-resolves the `external_data` references — point it at any
`.onnx` file and ORT will read its weights from the matching `.data` blob in
the same directory. Use `manifest.json` to discover IO names / configuration
without instantiating sessions.

```python
import onnxruntime as ort, json
from pathlib import Path

bundle = Path("export_v2/release")
manifest = json.loads((bundle / "manifest.json").read_text())

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = 4

prefill = ort.InferenceSession(str(bundle / "moss_tts_prefill.onnx"),
                               sess_options=so, providers=["CPUExecutionProvider"])
```

## Reproducing

```
python export_v2/scripts/make_release.py
```

Total runtime is ~10 minutes on a modern CPU. See
`export_v2/docs/03_EXPORT_JOURNEY.md` for the full v1 -> v6 export history.

## Verification

```
python export_v2/scripts/demo_generate.py --voice 0 --frames 200
python export_v2/scripts/benchmark.py
```

## Numerical notes

Dynamic INT8 weight-only quantization keeps activations FP32. Compared to
FP32 baseline, end-to-end SNR is around 18 dB on a 4 s reference clip, with
no perceptual artifacts. Multi-threaded ORT execution is deterministic given
the same seed.
"""
    out = RELEASE / "README.md"
    out.write_text(text)
    print(f"     wrote {out}  ({out.stat().st_size/1024:.1f} KB)")
    return out


# --------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-export", action="store_true",
                    help="reuse _build/tts_fp32 if present (skip m1/m2/m3/m3.2)")
    ap.add_argument("--skip-quant", action="store_true",
                    help="reuse _build/*_int8_inlined if present (skip m5_1/m6_1)")
    ap.add_argument("--keep-build", action="store_true",
                    help="do not delete _build/ at the end (default deletes ~3 GB)")
    args = ap.parse_args()

    t_pipeline_start = time.time()

    BUILD.mkdir(parents=True, exist_ok=True)

    have_fp32 = all((TTS_FP32 / fn).exists() for fn in TTS_GRAPHS)
    have_fp32_shared = all((TTS_FP32_SHARED / fn).exists() for fn in TTS_GRAPHS)
    have_tts_int8 = all((TTS_INT8_INLINED / fn).exists() for fn in TTS_GRAPHS)
    have_codec_int8 = all((CODEC_INT8_INLINED / fn).exists() for fn in CODEC_GRAPHS)

    if args.skip_export and not have_fp32:
        print("[warn] --skip-export but no FP32 graphs present, will export anyway.")
        args.skip_export = False

    # ---- stage 1: TTS export ----
    if not args.skip_export:
        _section("stage 1/6  TTS export (m1 + m2 + m3 + m3.2)")
        for s in ("m1_export_prefill.py", "m2_export_decode_step.py",
                  "m3_export_local_decoder.py", "m3_2_export_local_cached_step.py"):
            _run(s)
    else:
        _section("stage 1/6  TTS export   [skipped, reusing _build/tts_fp32]")

    # ---- stage 2: TTS share (split mode, intermediate only) ----
    if not args.skip_quant or not have_fp32_shared:
        _section("stage 2/6  TTS share (m4_share_external_data, split mode)")
        _run("m4_share_external_data.py")  # SPLIT default; sufficient for quantizer
    else:
        _section("stage 2/6  TTS share    [skipped]")

    # ---- stage 3: TTS quantize ----
    if not args.skip_quant or not have_tts_int8:
        _section("stage 3/6  TTS INT8 dynamic quantization (m5_1)")
        _run("m5_1_quantize_dynamic.py")
    else:
        _section("stage 3/6  TTS quantize [skipped]")

    # ---- stage 4: codec quantize ----
    if not args.skip_quant or not have_codec_int8:
        _section("stage 4/6  codec INT8 dynamic quantization (m6_1)")
        _run("m6_1_quantize_codec.py")
    else:
        _section("stage 4/6  codec quant  [skipped]")

    # ---- stage 5: assemble release/ with unified blobs ----
    _section("stage 5/6  assemble release/ (unified shared blobs)")
    share_report = _assemble_release()

    # ---- stage 6: manifest + README ----
    _section("stage 6/6  package (manifest + README)")
    _write_manifest(share_report)
    total = sum(p.stat().st_size for p in RELEASE.iterdir() if p.is_file())
    _write_readme(total)

    # ---- summary ----
    _section("RELEASE COMPLETE")
    grand = 0
    for p in sorted(RELEASE.iterdir()):
        sz = p.stat().st_size
        grand += sz
        print(f"  {sz/1024/1024:8.2f} MB   {p.name}")
    print(f"  {'-'*38}")
    print(f"  {grand/1024/1024:8.2f} MB   release/  (total)")
    print(f"\n  pipeline runtime: {time.time()-t_pipeline_start:.1f}s")
    print(f"  output:           {RELEASE.relative_to(ROOT)}")

    if not args.keep_build and BUILD.exists():
        sz_build = sum(p.stat().st_size for p in BUILD.rglob("*") if p.is_file())
        print(f"\n  cleaning _build/  ({sz_build/1024/1024:.1f} MB)  [pass --keep-build to retain]")
        shutil.rmtree(BUILD)
    return 0


if __name__ == "__main__":
    sys.exit(main())
