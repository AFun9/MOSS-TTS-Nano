"""M4 - dedup initializers across ONNX graphs and remap to shared external data files.

Two modes:
  * SPLIT (mimics official):
        moss_tts_global_shared.data   <-  prefill + decode_step
        moss_tts_local_shared.data    <-  local_decoder + local_cached_step
    Use this when you need bit-equal layout vs `OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX`.

  * UNIFIED (most compact, used by make_release.py for the final bundle):
        moss_tts_shared.data          <-  all 4 graphs
    Saves ~24 MB extra over SPLIT in INT8 mode (lm_heads + ln weights are
    byte-equal across global/local graphs because dynamic quantization is
    deterministic and they're all referenced by Gather only).

For each group/all graphs:
    1. Load every graph.
    2. Walk every initializer; compute sha256 of (raw_bytes, dtype, shape).
    3. The first time we see a content-hash, append its bytes to the shared blob
       and record (offset, length).
    4. Rewrite each initializer's TensorProto to use ExternalDataInfo pointing
       to the shared file with the recorded offset/length.
    5. Save the rewritten ONNX (graph only, no inline weights).

CLI:
    python m4_share_external_data.py                  # SPLIT mode, _build/tts_fp32 -> _build/tts_fp32_shared
    python m4_share_external_data.py --unified        # UNIFIED mode (single shared blob)
    python m4_share_external_data.py --src DIR --dst DIR [--unified]
"""
from __future__ import annotations
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import onnx
from onnx import TensorProto
from onnx.external_data_helper import (
    load_external_data_for_model,
    set_external_data,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = ROOT / "export_v2" / "_build" / "tts_fp32"
DEFAULT_DST = ROOT / "export_v2" / "_build" / "tts_fp32_shared"

GLOBAL_GROUP = ("moss_tts_prefill.onnx", "moss_tts_decode_step.onnx")
LOCAL_GROUP = ("moss_tts_local_decoder.onnx", "moss_tts_local_cached_step.onnx")
ALL_GRAPHS = GLOBAL_GROUP + LOCAL_GROUP
# add local_fixed_sampled_frame.onnx to LOCAL_GROUP later when M3.3 is exported

GLOBAL_BLOB = "moss_tts_global_shared.data"
LOCAL_BLOB = "moss_tts_local_shared.data"
UNIFIED_BLOB = "moss_tts_shared.data"

# Tensors smaller than this stay inline in the .onnx file. ONNX shape inference
# (and ORT model load) cannot resolve external storage for tiny constants like
# Unsqueeze/Reshape axes, so we must keep them embedded.
INLINE_BYTES_THRESHOLD = 1024


def tensor_raw_bytes(t: TensorProto) -> bytes:
    """Return raw bytes for a TensorProto, materializing from external storage
    if needed."""
    if t.HasField("raw_data") and len(t.raw_data) > 0:
        return t.raw_data
    if t.data_location == TensorProto.EXTERNAL:
        # load_external_data_for_model() must have already been called
        # so raw_data is materialized; re-check
        raise RuntimeError(f"initializer '{t.name}' still external after load")
    # fall back: pack typed fields back into raw bytes
    import numpy as np
    from onnx.numpy_helper import to_array
    arr = to_array(t)
    return arr.tobytes()


def content_hash(t: TensorProto) -> str:
    h = hashlib.sha256()
    h.update(int(t.data_type).to_bytes(4, "little"))
    h.update(b"|")
    for d in t.dims:
        h.update(int(d).to_bytes(8, "little"))
    h.update(b"|")
    h.update(tensor_raw_bytes(t))
    return h.hexdigest()


def share_group(
    graph_files: Tuple[str, ...],
    src_dir: Path,
    dst_dir: Path,
    blob_name: str,
) -> Dict[str, Dict[str, int]]:
    """Dedup initializers across the listed ONNX files into one shared blob.

    Returns:
        report  ->  { graph_filename: { 'n_init': int, 'n_unique_added': int,
                                        'inline_size_bytes': int } }
    """
    print(f"\n=== sharing group -> {blob_name} ===")
    print(f"   graphs: {list(graph_files)}")

    blob_path = dst_dir / blob_name
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_fp = open(blob_path, "wb")
    blob_offset = 0
    hash_to_loc: Dict[str, Tuple[int, int]] = {}     # hash -> (offset, length)

    report: Dict[str, Dict[str, int]] = {}

    for graph_filename in graph_files:
        src_path = src_dir / graph_filename
        dst_path = dst_dir / graph_filename
        print(f"\n--- {graph_filename} ---")
        print(f"  load {src_path}")
        m = onnx.load(str(src_path), load_external_data=False)
        load_external_data_for_model(m, str(src_dir))

        n_init = len(m.graph.initializer)
        inline_bytes = 0
        n_unique_added = 0

        n_kept_inline = 0
        for t in m.graph.initializer:
            raw = tensor_raw_bytes(t)
            inline_bytes += len(raw)

            if len(raw) < INLINE_BYTES_THRESHOLD:
                # keep inline as raw_data (also covers Constant/axes tensors
                # that ORT needs at load-time for shape inference)
                t.raw_data = raw
                for f in ("float_data", "int32_data", "string_data", "int64_data",
                          "double_data", "uint64_data"):
                    t.ClearField(f)
                t.data_location = TensorProto.DEFAULT
                del t.external_data[:]
                n_kept_inline += 1
                continue

            ch = content_hash(t)
            if ch not in hash_to_loc:
                hash_to_loc[ch] = (blob_offset, len(raw))
                blob_fp.write(raw)
                blob_offset += len(raw)
                n_unique_added += 1
            offset, length = hash_to_loc[ch]

            t.raw_data = raw  # set_external_data needs raw_data to exist
            for f in ("float_data", "int32_data", "string_data", "int64_data",
                      "double_data", "uint64_data"):
                t.ClearField(f)
            del t.external_data[:]
            set_external_data(t, location=blob_name, offset=offset, length=length)
            t.ClearField("raw_data")
            t.data_location = TensorProto.EXTERNAL

        report[graph_filename] = {
            "n_init": n_init,
            "n_unique_added": n_unique_added,
            "n_kept_inline": n_kept_inline,
            "inline_size_bytes": inline_bytes,
        }
        print(f"  initializers: {n_init}  newly-unique-added: {n_unique_added}  "
              f"kept-inline (<{INLINE_BYTES_THRESHOLD}B): {n_kept_inline}")
        print(f"  total inline bytes (this graph): {inline_bytes/1024/1024:.1f} MB")

        # Save graph WITHOUT writing external data alongside (we already wrote
        # it to the shared blob).
        onnx.save_model(
            m,
            str(dst_path),
            save_as_external_data=False,
        )
        print(f"  wrote   {dst_path} ({dst_path.stat().st_size/1024:.1f} KB)")

    blob_fp.close()
    blob_size = blob_path.stat().st_size
    print(f"\n>> shared blob: {blob_path} ({blob_size/1024/1024:.1f} MB)")
    inline_total = sum(r["inline_size_bytes"] for r in report.values())
    print(f"   inline-total across graphs: {inline_total/1024/1024:.1f} MB")
    print(f"   dedup ratio: {blob_size/inline_total*100:.1f}% "
          f"(saved {(inline_total-blob_size)/1024/1024:.1f} MB)")

    return report


def run_share(src_dir: Path, dst_dir: Path, unified: bool) -> dict:
    """Top-level driver. Returns a report dict written to share_report.json."""
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    full_report: Dict[str, dict] = {"mode": "unified" if unified else "split"}

    if unified:
        rep = share_group(ALL_GRAPHS, src_dir, dst_dir, UNIFIED_BLOB)
        full_report["unified"] = {"graphs": rep, "blob": UNIFIED_BLOB}
    else:
        rep_g = share_group(GLOBAL_GROUP, src_dir, dst_dir, GLOBAL_BLOB)
        full_report["global"] = {"graphs": rep_g, "blob": GLOBAL_BLOB}
        rep_l = share_group(LOCAL_GROUP, src_dir, dst_dir, LOCAL_BLOB)
        full_report["local"] = {"graphs": rep_l, "blob": LOCAL_BLOB}

    total_after = sum(p.stat().st_size for p in dst_dir.iterdir() if p.is_file())
    total_before = 0
    for fn in ALL_GRAPHS:
        gp = src_dir / fn
        if gp.exists():
            total_before += gp.stat().st_size
        for sidecar in src_dir.glob(f"{fn}.*"):
            total_before += sidecar.stat().st_size
    # Also count any standalone shared blobs already living in src_dir.
    for blob_name in (GLOBAL_BLOB, LOCAL_BLOB, UNIFIED_BLOB):
        bp = src_dir / blob_name
        if bp.exists():
            total_before += bp.stat().st_size

    print()
    print("=" * 60)
    print(f"FINAL  (mode={full_report['mode']})")
    print("=" * 60)
    print(f"before ({src_dir.name}):   {total_before/1024/1024:8.1f} MB")
    print(f"after  ({dst_dir.name}):   {total_after/1024/1024:8.1f} MB")
    if total_before > 0:
        print(f"saved:                              {(total_before-total_after)/1024/1024:8.1f} MB"
              f"  ({(1 - total_after/total_before)*100:.1f}%)")
    (dst_dir / "share_report.json").write_text(json.dumps(full_report, indent=2))
    print()
    print(f"=== contents of {dst_dir} ===")
    for p in sorted(dst_dir.iterdir()):
        print(f"  {p.stat().st_size/1024/1024:8.2f} MB   {p.name}")
    return full_report


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=str(DEFAULT_SRC),
                    help=f"source bundle dir (default: {DEFAULT_SRC.relative_to(ROOT)})")
    ap.add_argument("--dst", default=str(DEFAULT_DST),
                    help=f"output bundle dir (default: {DEFAULT_DST.relative_to(ROOT)})")
    ap.add_argument("--unified", action="store_true",
                    help="dedup all 4 graphs into ONE shared blob (smaller; not bit-equal to official layout)")
    args = ap.parse_args()
    run_share(Path(args.src), Path(args.dst), unified=args.unified)
    return 0


if __name__ == "__main__":
    sys.exit(main())
