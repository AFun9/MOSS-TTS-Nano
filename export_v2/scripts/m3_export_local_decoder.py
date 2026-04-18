"""M3.1 - export LocalDecoderWrapper to ONNX (dynamo path)."""
from __future__ import annotations
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "export_v2"))

from wrappers.local_decoder import LocalDecoderWrapper, io_names  # noqa: E402
from wrappers.prefill import patch_attention                       # noqa: E402

CKPT = ROOT / "MOSS-TTS-Nano-100M"
OUT_DIR = ROOT / "export_v2" / "_build" / "tts_fp32"
OUT_PATH = OUT_DIR / "moss_tts_local_decoder.onnx"


def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    print(f"[load] {CKPT}")
    lm = AutoModelForCausalLM.from_pretrained(
        str(CKPT), trust_remote_code=True, dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    patch_attention(lm, also_local=True)

    cfg = lm.config
    n_vq = int(cfg.n_vq)
    H = int(cfg.gpt2_config.hidden_size)

    wrapper = LocalDecoderWrapper(lm).eval()

    B = 1
    global_hidden = torch.randn(B, H)
    text_token_id = torch.tensor([10], dtype=torch.int32)
    audio_prefix_token_ids = torch.zeros(B, n_vq - 1, dtype=torch.int32)

    print(f"[fwd] sanity-checking LocalDecoderWrapper")
    with torch.no_grad():
        tl, al = wrapper(global_hidden, text_token_id, audio_prefix_token_ids)
    print(f"[fwd] text_logits={tuple(tl.shape)}  audio_logits={tuple(al.shape)}")
    assert tl.shape == (B, wrapper.vocab_size)
    assert al.shape == (B, n_vq, 1024)

    in_names, out_names = io_names()

    print(f"[export] {OUT_PATH}")
    from torch.export import Dim
    batch = Dim("batch", min=1, max=8)
    dynamic_shapes = (
        {0: batch},   # global_hidden
        {0: batch},   # text_token_id
        {0: batch},   # audio_prefix_token_ids
    )
    torch.onnx.export(
        wrapper,
        (global_hidden, text_token_id, audio_prefix_token_ids),
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
