#!/usr/bin/env bash
# Re-create hardlinks android/models/ <- export_v2/release/ for the v1.0.0 bundle.
# Run after `make_release.py` regenerates export_v2/release/ (which breaks inodes).
#
# Layout produced (flat, ~164.6 MB total, zero extra disk via hardlinks):
#
#   android/models/
#     moss_tts_prefill.onnx
#     moss_tts_decode_step.onnx
#     moss_tts_local_decoder.onnx
#     moss_tts_local_cached_step.onnx
#     moss_tts_shared.data                 (~143 MB, INT8 weights for 4 TTS graphs)
#     moss_audio_tokenizer_encode.onnx
#     moss_audio_tokenizer_decode_full.onnx
#     moss_audio_tokenizer_decode_step.onnx
#     moss_audio_tokenizer_shared.data     (~22 MB, INT8 weights for 3 codec graphs)
#     manifest.json
#     tokenizer.model
#
# All `.onnx` and `.data` files are gitignored (see android/models/.gitignore).

set -euo pipefail
cd "$(dirname "$0")/.."

SRC=export_v2/release
DST=android/models

mkdir -p "$DST"

FILES=(
  # TTS graphs (graph defs, weights live in moss_tts_shared.data)
  moss_tts_prefill.onnx
  moss_tts_decode_step.onnx
  moss_tts_local_decoder.onnx
  moss_tts_local_cached_step.onnx
  moss_tts_shared.data

  # Codec graphs (graph defs, weights live in moss_audio_tokenizer_shared.data)
  moss_audio_tokenizer_encode.onnx
  moss_audio_tokenizer_decode_full.onnx
  moss_audio_tokenizer_decode_step.onnx
  moss_audio_tokenizer_shared.data

  # Metadata
  manifest.json
  tokenizer.model
)

for f in "${FILES[@]}"; do
  if [[ ! -f "$SRC/$f" ]]; then
    echo "ERROR: missing $SRC/$f - run \`python export_v2/scripts/make_release.py\` first" >&2
    exit 1
  fi
  ln -f "$SRC/$f" "$DST/$f"
done

echo "Hardlinked ${#FILES[@]} files: $SRC/ -> $DST/"
ls -lh "$DST" | tail -n +2
