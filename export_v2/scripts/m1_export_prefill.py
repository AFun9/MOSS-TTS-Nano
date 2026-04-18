"""M1 - export PrefillWrapper to ONNX (dynamo path)."""
from __future__ import annotations
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "export_v2"))

from wrappers.prefill import PrefillWrapper, io_names, patch_attention  # noqa: E402

CKPT = ROOT / "MOSS-TTS-Nano-100M"
OUT_DIR = ROOT / "export_v2" / "_build" / "tts_fp32"
OUT_PATH = OUT_DIR / "moss_tts_prefill.onnx"


def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    print(f"[load] {CKPT}")
    lm = AutoModelForCausalLM.from_pretrained(
        str(CKPT), trust_remote_code=True, dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    patch_attention(lm)
    cfg = lm.config
    num_layers = int(cfg.gpt2_config.n_layer)
    n_vq = int(cfg.n_vq)
    pad = int(cfg.audio_pad_token_id)

    wrapper = PrefillWrapper(lm).eval()

    B, T = 1, 8
    input_ids = torch.zeros(B, T, n_vq + 1, dtype=torch.int32)
    input_ids[..., 0] = torch.arange(10, 10 + T, dtype=torch.int32)
    input_ids[..., 1:] = pad
    attention_mask = torch.ones(B, T, dtype=torch.int32)

    print(f"[fwd] sanity-checking PrefillWrapper(input_ids={tuple(input_ids.shape)})")
    with torch.no_grad():
        outs = wrapper(input_ids, attention_mask)
    print(f"[fwd] hidden={tuple(outs[0].shape)}  pkv0_k={tuple(outs[1].shape)}")
    assert tuple(outs[0].shape) == (B, T, cfg.gpt2_config.hidden_size)
    assert tuple(outs[1].shape) == (B, T, cfg.gpt2_config.n_head, cfg.gpt2_config.hidden_size // cfg.gpt2_config.n_head)
    assert len(outs) == 1 + 2 * num_layers

    in_names, out_names, _ = io_names(num_layers)
    print(f"[export] {OUT_PATH}")

    from torch.export import Dim
    batch = Dim("batch", min=1, max=8)
    prefill_seq = Dim("prefill_seq", min=1, max=2048)
    dynamic_shapes = {
        "input_ids": {0: batch, 1: prefill_seq},
        "attention_mask": {0: batch, 1: prefill_seq},
    }
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        str(OUT_PATH),
        input_names=in_names,
        output_names=out_names,
        dynamic_shapes=dynamic_shapes,
        opset_version=17,
        dynamo=True,
    )
    print(f"[export] done. size={OUT_PATH.stat().st_size/1024:.1f} KB")

    import onnx
    m = onnx.load(str(OUT_PATH), load_external_data=False)
    print(f"[onnx] producer={m.producer_name} {m.producer_version}  opset={[(o.domain or 'ai.onnx', o.version) for o in m.opset_import]}")
    print(f"[onnx] inputs:   {[(i.name, [d.dim_value or d.dim_param or '?' for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]}")
    print(f"[onnx] outputs:  {[o.name for o in m.graph.output]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
