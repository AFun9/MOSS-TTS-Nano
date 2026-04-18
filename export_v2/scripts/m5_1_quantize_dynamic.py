"""M5.1 - INT8 dynamic quantization (weight-only) for the 4 ONNX graphs.

Source: export_v2/_build/tts_fp32_shared/  (FP32, shared external data)
Output: export_v2/_build/tts_int8_inlined/  (INT8, fully inlined)

Downstream: make_release.py re-shares the inlined INT8 weights via
m4_share_external_data.share_group(unified=True) into release/.

For dynamic quantization the activations stay FP32 at runtime; only the
constant weights of MatMul/Gather are stored as INT8 with per-channel scales,
and the kernel transparently dequantizes them at execution time. No
calibration data is required, and accuracy regression is usually small for
LLM-style transformers.

We must first materialize the shared external data into each graph (the
dynamic quantizer cannot follow our shared blob), then quantize. The
downstream make_release.py then re-shares the resulting INT8 weights into
the final release/ bundle.
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
SRC = ROOT / "export_v2" / "_build" / "tts_fp32_shared"
TMP_FP32 = ROOT / "export_v2" / "_build" / "tts_int8_inlined" / "_tmp_inlined_fp32"
DST = ROOT / "export_v2" / "_build" / "tts_int8_inlined"

GRAPHS = [
    "moss_tts_prefill.onnx",
    "moss_tts_decode_step.onnx",
    "moss_tts_local_decoder.onnx",
    "moss_tts_local_cached_step.onnx",
]

# Quantize MatMul + Gather. Gather is critical: dynamo-export emits all
# lm_heads (text + 16 audio = ~99 MB FP32) as `Gather(weight, ids)` rather
# than `MatMul(hidden, weight^T)`. Without it those weights stay FP32.
OP_TYPES = ["MatMul", "Gather"]


def _untie_weight_tied_lm_heads(model) -> int:
    """Duplicate any initializer that is referenced by BOTH Gather (axis=0) and
    MatMul-style ops in the same graph. ORT quantize_dynamic prefers MatMul's
    per-channel transpose layout (hidden, vocab) which then breaks Gather
    output shape (Gather expects (vocab, hidden)).

    For each tied weight `W` we keep `W` for Gather (vocab, hidden) and create
    a separate `W__matmul` initializer (identical bytes) referenced by the
    MatMul/Gemm ops. After quantization the two copies become independent INT8
    blocks (one untransposed for Gather, one transposed for MatMul) and dedup
    cannot merge them — but the savings vs FP32 are still ~75% per copy.
    """
    init_by_name = {t.name: t for t in model.graph.initializer}
    # collect users
    gather_users = {}    # name -> list of nodes
    matmul_users = {}
    for n in model.graph.node:
        if n.op_type == "Gather" and n.input and n.input[0] in init_by_name:
            gather_users.setdefault(n.input[0], []).append(n)
        elif n.op_type in ("MatMul", "Gemm") and len(n.input) >= 2:
            for slot, inp in enumerate(n.input):
                if inp in init_by_name:
                    matmul_users.setdefault(inp, []).append((n, slot))

    n_untied = 0
    for w_name in list(gather_users.keys()):
        if w_name not in matmul_users:
            continue
        # duplicate
        src_t = init_by_name[w_name]
        new_name = w_name + "__matmul"
        clone = onnx.TensorProto()
        clone.CopyFrom(src_t)
        clone.name = new_name
        model.graph.initializer.append(clone)
        for node, slot in matmul_users[w_name]:
            node.input[slot] = new_name
        n_untied += 1
        print(f"    untied weight-tied lm_head: {w_name}  shape={list(src_t.dims)}")
    return n_untied


def materialize(src_path: Path, dst_path: Path) -> None:
    """Read ONNX with shared external data and re-save with ALL weights inlined.

    Inline (raw_data) is required because onnx.shape_inference.infer_shapes_path
    (called by onnxruntime's quantizer) cannot resolve external constants for
    shape propagation, leading to spurious InferenceError.

    Also untie weight-tied lm_heads so Gather and MatMul reference distinct
    initializers (see `_untie_weight_tied_lm_heads`).
    """
    print(f"  [materialize] {src_path.name} -> {dst_path}")
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
    n = _untie_weight_tied_lm_heads(m)
    if n:
        print(f"    untied {n} weight-tied initializer(s)")
    del m.graph.value_info[:]
    onnx.save_model(m, str(dst_path), save_as_external_data=False)


def main():
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True, exist_ok=True)
    TMP_FP32.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: materialize shared FP32 graphs to standalone FP32 ===")
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
        # Post-process: dynamo-export emits buggy value_info for Gather outputs
        # (e.g. 'embedding*' declared as (1,768) but actual is (1,vocab)). ORT
        # would try to merge declared with inferred and fail. Strip ALL
        # value_info so ORT re-infers fresh from initializer shapes after the
        # DequantizeLinear inserts.
        m = onnx.load(str(dst_p), load_external_data=False)
        del m.graph.value_info[:]
        onnx.save_model(m, str(dst_p), save_as_external_data=False)
        dt = time.time() - t0
        print(f"    -> wrote {dst_p.name}  {dst_p.stat().st_size/1024/1024:.2f} MB  ({dt:.1f}s)")

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
    print(f"  {grand/1024/1024:8.2f} MB   TOTAL (INT8-dyn bundle)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
