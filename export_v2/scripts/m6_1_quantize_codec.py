"""M6.1 - INT8 dynamic quantization for the audio codec graphs.

Source: export_v2/reference/codec_full/
    moss_audio_tokenizer_encode.onnx       (+ encode.data, 43 MB FP32)
    moss_audio_tokenizer_decode_full.onnx  (+ decode_shared.data, 43 MB FP32)
    moss_audio_tokenizer_decode_step.onnx  (shares decode_shared.data)

Output: export_v2/_build/codec_int8_inlined/
    All weights inlined as INT8. Downstream: make_release.py re-shares all
    3 codec graphs into a single moss_audio_tokenizer_shared.data blob.

Recon notes:
    - 0 weight-tied initializers (no Gather/MatMul share); no untie needed.
    - Op composition: MatMul/Gemm dominant (80-96 per graph), Conv/ConvT
      secondary (17-32). Quantize all of them + Gather (RVQ codebook lookup).
    - decode_full and decode_step are 178/178 byte-equal in their large
      initializers => full backbone dedup possible after quantization.
"""
from __future__ import annotations
import shutil
import sys
import time
from pathlib import Path

import onnx
from onnx.external_data_helper import load_external_data_for_model
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "export_v2" / "reference" / "codec_full"
TMP_FP32 = ROOT / "export_v2" / "_build" / "codec_int8_inlined" / "_tmp_inlined_fp32"
DST = ROOT / "export_v2" / "_build" / "codec_int8_inlined"

GRAPHS = [
    "moss_audio_tokenizer_encode.onnx",
    "moss_audio_tokenizer_decode_full.onnx",
    "moss_audio_tokenizer_decode_step.onnx",
]

# Codec is convolutional + transformer hybrid. Cover all weight-bearing ops.
OP_TYPES = ["MatMul", "Gather", "Conv"]


def materialize(src_path: Path, dst_path: Path) -> None:
    """Inline all weights from shared external data, clear value_info."""
    print(f"  [materialize] {src_path.name}")
    m = onnx.load(str(src_path), load_external_data=False)
    load_external_data_for_model(m, str(src_path.parent))
    from onnx import TensorProto
    from onnx.numpy_helper import to_array
    for t in m.graph.initializer:
        if t.data_location == TensorProto.EXTERNAL or not t.HasField("raw_data"):
            arr = to_array(t)
            t.ClearField("raw_data")
            for f in ("float_data", "int32_data", "string_data", "int64_data",
                      "double_data", "uint64_data"):
                t.ClearField(f)
            t.raw_data = arr.tobytes()
            del t.external_data[:]
            t.data_location = TensorProto.DEFAULT
    del m.graph.value_info[:]
    onnx.save_model(m, str(dst_path), save_as_external_data=False)


def main():
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True, exist_ok=True)
    TMP_FP32.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: materialize codec graphs to standalone FP32 ===")
    for g in GRAPHS:
        materialize(SRC / g, TMP_FP32 / g)

    print()
    print(f"=== Step 2: dynamic INT8 quantization ({', '.join(OP_TYPES)}) ===")
    for g in GRAPHS:
        src_p = TMP_FP32 / g
        dst_p = DST / g
        print(f"  [quantize] {g}")
        t0 = time.time()
        quantize_dynamic(
            model_input=str(src_p),
            model_output=str(dst_p),
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=False,
            op_types_to_quantize=OP_TYPES,
            extra_options={"MatMulConstBOnly": True},
        )
        # Strip stale value_info from quant output (same fix as m5_1).
        m = onnx.load(str(dst_p), load_external_data=False)
        del m.graph.value_info[:]
        onnx.save_model(m, str(dst_p), save_as_external_data=False)
        dt = time.time() - t0
        print(f"    -> {dst_p.name}  {dst_p.stat().st_size/1024/1024:.2f} MB  ({dt:.1f}s)")

    print()
    print("=== Step 3: cleanup tmp ===")
    shutil.rmtree(TMP_FP32)

    print()
    print("=== final dst contents ===")
    grand = 0
    for p in sorted(DST.iterdir()):
        sz = p.stat().st_size
        grand += sz
        print(f"  {sz/1024/1024:8.2f} MB   {p.name}")
    print(f"  ----------")
    print(f"  {grand/1024/1024:8.2f} MB   TOTAL (codec INT8-dyn bundle)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
