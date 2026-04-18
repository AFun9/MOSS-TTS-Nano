"""M3.2 - export LocalCachedStepWrapper to ONNX (dynamo path)."""
from __future__ import annotations
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "export_v2"))

from wrappers.local_cached_step import LocalCachedStepWrapper, io_names  # noqa: E402
from wrappers.prefill import patch_attention                              # noqa: E402

CKPT = ROOT / "MOSS-TTS-Nano-100M"
OUT_PATH = ROOT / "export_v2" / "_build" / "tts_fp32" / "moss_tts_local_cached_step.onnx"


def main():
    OUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    print(f"[load] {CKPT}")
    lm = AutoModelForCausalLM.from_pretrained(
        str(CKPT), trust_remote_code=True, dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    patch_attention(lm, also_local=True)

    cfg = lm.config
    n_local = int(cfg.local_transformer_layers)
    n_heads = int(cfg.gpt2_config.n_head)
    head_dim = int(cfg.gpt2_config.hidden_size // n_heads)

    wrapper = LocalCachedStepWrapper(lm).eval()

    B, past_seq = 1, 2
    global_hidden = torch.randn(B, 768)
    text_token_id = torch.tensor([10], dtype=torch.int32)
    audio_token_id = torch.tensor([5], dtype=torch.int32)
    channel_index = torch.tensor([2], dtype=torch.int32)
    step_type = torch.tensor([2], dtype=torch.int32)
    past_valid_lengths = torch.tensor([past_seq], dtype=torch.int32)
    past_kv = []
    torch.manual_seed(0)
    for _ in range(n_local):
        past_kv.append(torch.randn(B, past_seq, n_heads, head_dim))
        past_kv.append(torch.randn(B, past_seq, n_heads, head_dim))

    print("[fwd] sanity-checking ...")
    with torch.no_grad():
        outs = wrapper(global_hidden, text_token_id, audio_token_id,
                       channel_index, step_type, past_valid_lengths, *past_kv)
    print(f"[fwd] text_logits={tuple(outs[0].shape)}  audio_logits={tuple(outs[1].shape)}")
    assert tuple(outs[1].shape) == (B, 16, 1024)
    assert len(outs) == 2 + 2 * n_local

    in_names, out_names = io_names(n_local)
    print(f"[export] {OUT_PATH}")

    from torch.export import Dim
    batch = Dim("batch", min=1, max=8)
    past = Dim("local_past_seq", min=0, max=128)
    kv_specs = tuple({0: batch, 1: past} for _ in range(2 * n_local))
    dynamic_shapes = (
        {0: batch},  # global_hidden
        {0: batch}, {0: batch}, {0: batch}, {0: batch}, {0: batch},
        kv_specs,
    )
    torch.onnx.export(
        wrapper,
        (global_hidden, text_token_id, audio_token_id,
         channel_index, step_type, past_valid_lengths, *past_kv),
        str(OUT_PATH),
        input_names=in_names,
        output_names=out_names,
        dynamic_shapes=dynamic_shapes,
        opset_version=17,
        dynamo=True,
    )
    print(f"[export] done. size={OUT_PATH.stat().st_size/1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
