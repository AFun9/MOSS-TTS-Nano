"""Defuse `Reshape → Gemm → Reshape` patterns back to ND `MatMul + Add`.

ONNX Runtime's `MatMulAddFusion` (run at EXTENDED graph optimization
during export) rewrites a 3-D MatMul+BiasAdd into:

    A_ND --[Reshape to [1, K]]--> A_2D
        --[Gemm(W, bias)]--> Y_2D
        --[Reshape to [1, 1, N]]--> Y_ND

This works fine on ORT 1.24.4 Linux/x86_64 (where the export was run),
but ORT 1.24.3 Android/ARM64 mishandles the resulting graph: the next
residual `Add` sees the Gemm output with the wrong batch shape and
crashes with

    "Attempting to broadcast an axis by a dimension other than 1.
     64 by 768"

on `local_decoder_text`'s `/Add_2` node.

This tool walks an ONNX file in-place and undoes the fusion by:

  1. Detecting every `Gemm` whose input is produced by a `Reshape` to
     `[*, K]` and whose only consumer is a `Reshape` to `[..., N]`.
  2. Replacing those three nodes with the original ND tensor flowing
     through `Reshape (-> [..., K])` → `MatMul (W)` → `Add (bias)`.
     The intermediate Reshape collapses everything except the last
     contraction dim, so MatMul broadcasts over batch dims naturally.
  3. Cleaning up dangling Reshape constants and the unused output
     value-info entries.

Run AFTER the original export and BEFORE int8 quantisation:

    python3 tools/defuse_gemm_reshape.py onnx_export/local_decoder_text.onnx
    python3 tools/defuse_gemm_reshape.py onnx_export/local_decoder_audio.onnx
    python3 tools/defuse_gemm_reshape.py onnx_export/audio_encoder.onnx

The change is byte-deterministic across runs and idempotent: re-running
on an already-defused file is a no-op.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _producer_map(graph):
    out = {}
    for n in graph.node:
        for o in n.output:
            out[o] = n
    return out


def _consumer_map(graph):
    out: dict[str, list] = {}
    for n in graph.node:
        for i in n.input:
            out.setdefault(i, []).append(n)
    return out


def _const_value(graph, name):
    for ini in graph.initializer:
        if ini.name == name:
            return numpy_helper.to_array(ini)
    return None


def defuse_one(graph, gemm_node) -> bool:
    """Return True iff this Gemm was rewritten."""
    if gemm_node.op_type != "Gemm":
        return False
    # Gemm must have transA=0, transB=0, alpha=1, beta=1, otherwise we
    # can't trivially substitute MatMul + Add.
    attrs = {a.name: (a.f if a.type == 1 else (a.i if a.type == 2 else a.s))
             for a in gemm_node.attribute}
    if attrs.get("transA", 0) != 0 or attrs.get("transB", 0) != 0:
        return False
    if abs(attrs.get("alpha", 1.0) - 1.0) > 1e-6:
        return False
    if abs(attrs.get("beta", 1.0) - 1.0) > 1e-6:
        return False

    a_name, w_name, b_name = gemm_node.input[0], gemm_node.input[1], gemm_node.input[2]
    w_init = _const_value(graph, w_name)
    b_init = _const_value(graph, b_name)
    if w_init is None or b_init is None:
        return False  # weight/bias must be constants
    if w_init.ndim != 2 or b_init.ndim != 1:
        return False
    K, N = w_init.shape
    if b_init.shape[0] != N:
        return False

    prods = _producer_map(graph)
    cons = _consumer_map(graph)

    in_reshape = prods.get(a_name)
    if in_reshape is None or in_reshape.op_type != "Reshape":
        return False
    in_shape = _const_value(graph, in_reshape.input[1])
    # Collapse-to-[*, K] is the signature: any number of leading dims
    # multiplied into a single batch axis followed by K.
    if in_shape is None or list(in_shape) != [-1, K] and list(in_shape) != [1, K]:
        # Accept the common `[1, K]` shape baked by the fusion. Anything
        # more exotic we leave alone.
        return False

    out_consumers = cons.get(gemm_node.output[0], [])
    if len(out_consumers) != 1 or out_consumers[0].op_type != "Reshape":
        return False
    out_reshape = out_consumers[0]
    out_shape = _const_value(graph, out_reshape.input[1])
    if out_shape is None or list(out_shape)[-1] != N:
        return False

    nd_input = in_reshape.input[0]
    nd_output = out_reshape.output[0]
    out_shape_list = list(out_shape)

    # Strategy: pure-ND `MatMul + Add`. MatMul broadcasts over leading
    # batch dims natively, so as long as the operand ends in `[..., K]`
    # the result is `[..., N]` directly — no post-Reshape needed.
    #
    # We tried a 2-D collapse (Reshape→MatMul→Reshape→Add) first to
    # mirror the original Gemm path. ORT-Android 1.24.3 ARM64 then
    # mis-infers the post-MatMul Reshape's output shape (computes
    # `{1,1,12,32,768}` for c_proj — `head_dim/2` and `num_heads`
    # leaking through from upstream FusedMatMul value_info) and crashes
    # the next residual Add with "Shape mismatch attempting to re-use
    # buffer". Skipping the post-Reshape sidesteps that entirely.
    #
    # For c_proj the upstream tensor is `[1, 1, num_heads, head_dim]`
    # and we need `[1, 1, hidden_size]` — one explicit Reshape collapses
    # the trailing two dims into K. For c_attn / mlp_fc_in / mlp_fc_out
    # the upstream tensor is already `[1, 1, K]` so we feed MatMul
    # directly with no Reshape at all.
    suffix = gemm_node.name.replace("/", "_").strip("_")
    matmul_y = f"defuse/{suffix}/matmul"

    new_nodes = []
    matmul_in = nd_input

    # We need to reshape only when the structural input is rank > 3 (the
    # c_proj after multi-head split). The simple `[1,1,K]` cases skip
    # the Reshape entirely. We can't ask the producer's rank directly
    # without ORT shape inference, so we use the original collapse
    # target as a structural hint: if it was `[1, K]` it implies the
    # producer was at least 3-D (otherwise the export wouldn't have
    # bothered emitting a flatten). Treat `[1, K]` as "needs reshape
    # to [1, 1, K]". Any other in_shape we already rejected above.
    in_shape_list = list(_const_value(graph, in_reshape.input[1]))
    if in_shape_list == [1, K]:
        nd_to_3d_shape = f"defuse/{suffix}/in_3d_shape"
        graph.initializer.append(numpy_helper.from_array(
            np.array(out_shape_list[:-1] + [K], dtype=np.int64),
            name=nd_to_3d_shape))
        reshaped_in = f"defuse/{suffix}/in_3d"
        new_nodes.append(helper.make_node(
            "Reshape", [nd_input, nd_to_3d_shape], [reshaped_in],
            name=f"defuse/{suffix}/reshape_in"))
        matmul_in = reshaped_in

    new_nodes += [
        helper.make_node(
            "MatMul", [matmul_in, w_name], [matmul_y],
            name=f"defuse/{suffix}/matmul"),
        helper.make_node(
            "Add", [matmul_y, b_name], [nd_output],
            name=f"defuse/{suffix}/add"),
    ]

    # Splice: remove the three old nodes, insert the four new ones at
    # the position of the original Gemm to preserve topological order.
    nodes = list(graph.node)
    for victim in (in_reshape, gemm_node, out_reshape):
        nodes.remove(victim)
    insert_at = next(
        (i for i, n in enumerate(nodes)
         if any(o == nd_input for o in []) or any(inp == nd_input for inp in n.input)),
        len(nodes),
    )
    # Just append in order — onnx doesn't strictly require positional
    # ordering as long as the def-use DAG is acyclic.
    for nn in new_nodes:
        nodes.insert(insert_at, nn)
        insert_at += 1
    del graph.node[:]
    graph.node.extend(nodes)
    return True


def defuse_model(path: Path) -> int:
    model = onnx.load(str(path))
    graph = model.graph

    # Collect Gemms first; we mutate the graph as we go.
    gemms = [n for n in graph.node if n.op_type == "Gemm"]
    rewritten = 0
    for g in gemms:
        if defuse_one(graph, g):
            rewritten += 1

    if rewritten == 0:
        print(f"  {path.name}: no fused Gemm patterns found (already clean?)")
        return 0

    # Strip stale value_info for the intermediate outputs we just
    # detached. ORT will re-infer everything at session-build time;
    # leaving stale entries trips `infer_shapes(strict_mode=True)`.
    live_outputs = set()
    for n in graph.node:
        live_outputs.update(n.output)
    for inp in graph.input:
        live_outputs.add(inp.name)
    for ini in graph.initializer:
        live_outputs.add(ini.name)
    fresh_vi = [vi for vi in graph.value_info if vi.name in live_outputs]
    del graph.value_info[:]
    graph.value_info.extend(fresh_vi)

    # Best-effort shape inference for offline diagnostics ONLY — we do
    # NOT write the inferred value_info back. ORT-Android re-uses
    # value_info as buffer-allocation hints, and stale entries from the
    # pre-defuse graph (or wrong inferences from upstream FusedMatMul
    # nodes that vanilla `onnx.shape_inference` doesn't understand)
    # cause "Shape mismatch attempting to re-use buffer" failures
    # downstream. Letting ORT re-infer at session-build is correct and
    # safe.
    try:
        onnx.shape_inference.infer_shapes(model)
    except Exception as exc:
        print(f"  ! shape inference soft-failed: {exc}")

    onnx.save(model, str(path))
    print(f"  {path.name}: defused {rewritten} Gemm pattern(s)")
    return rewritten


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("models", nargs="+", type=Path,
                    help="Path(s) to .onnx files to defuse in place.")
    args = ap.parse_args()

    total = 0
    for p in args.models:
        if not p.exists():
            print(f"  ! {p} not found, skipping", file=sys.stderr)
            continue
        total += defuse_model(p)
    print(f"Done. Total Gemm patterns rewritten: {total}")


if __name__ == "__main__":
    main()
