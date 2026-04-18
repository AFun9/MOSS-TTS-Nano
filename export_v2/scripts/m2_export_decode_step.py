"""M2 - export DecodeStepWrapper to ONNX (dynamo path)."""
from __future__ import annotations
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "export_v2"))

from wrappers.decode_step import DecodeStepWrapper, io_names  # noqa: E402
from wrappers.prefill import patch_attention                  # noqa: E402

CKPT = ROOT / "MOSS-TTS-Nano-100M"
OUT_DIR = ROOT / "export_v2" / "_build" / "tts_fp32"
OUT_PATH = OUT_DIR / "moss_tts_decode_step.onnx"


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
    n_heads = int(cfg.gpt2_config.n_head)
    head_dim = int(cfg.gpt2_config.hidden_size // n_heads)
    n_vq = int(cfg.n_vq)
    pad = int(cfg.audio_pad_token_id)

    wrapper = DecodeStepWrapper(lm).eval()

    B, step_seq, past_seq = 1, 1, 8
    input_ids = torch.zeros(B, step_seq, n_vq + 1, dtype=torch.int32)
    input_ids[..., 0] = torch.tensor([200], dtype=torch.int32)
    input_ids[..., 1:] = pad
    past_valid_lengths = torch.tensor([past_seq], dtype=torch.int32)
    past_kv = []
    torch.manual_seed(0)
    for _ in range(num_layers):
        past_kv.append(torch.randn(B, past_seq, n_heads, head_dim))
        past_kv.append(torch.randn(B, past_seq, n_heads, head_dim))

    print(f"[fwd] sanity-checking DecodeStepWrapper(step_seq={step_seq}, past_seq={past_seq})")
    with torch.no_grad():
        outs = wrapper(input_ids, past_valid_lengths, *past_kv)
    print(f"[fwd] hidden={tuple(outs[0].shape)}  pkv0_k={tuple(outs[1].shape)}")
    assert tuple(outs[0].shape) == (B, step_seq, cfg.gpt2_config.hidden_size)
    assert tuple(outs[1].shape) == (B, past_seq + step_seq, n_heads, head_dim)
    assert len(outs) == 1 + 2 * num_layers

    in_names, out_names, _ = io_names(num_layers)

    print(f"[export] {OUT_PATH}")
    from torch.export import Dim
    batch = Dim("batch", min=1, max=8)
    step = Dim("step_seq", min=1, max=512)
    past = Dim("past_seq", min=2, max=4096)
    # `*past_kv` is one vararg argument, so its dynamic spec is a tuple of
    # length 2L, one entry per actual tensor.
    kv_specs = tuple({0: batch, 1: past} for _ in range(2 * num_layers))
    dynamic_shapes = (
        {0: batch, 1: step},   # input_ids
        {0: batch},            # past_valid_lengths
        kv_specs,              # *past_kv  (vararg, must be tuple)
    )

    torch.onnx.export(
        wrapper,
        (input_ids, past_valid_lengths, *past_kv),
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
    print(f"[onnx] inputs:   {[(i.name, [d.dim_value or d.dim_param or '?' for d in i.type.tensor_type.shape.dim]) for i in m.graph.input][:4]} ...")
    print(f"[onnx] outputs:  {[o.name for o in m.graph.output][:4]} ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
