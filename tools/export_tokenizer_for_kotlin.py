#!/usr/bin/env python3
"""Export `tokenizer.model` into a Kotlin-friendly JSON + a golden fixture.

Outputs:
  - `<android>/app/src/main/assets/tokenizer_kotlin.json`
        Vocab table + meta needed by the Kotlin BPE re-implementation.
        Bundled into the APK (~700 KiB) so it always matches the wire
        format the app expects, independent of the downloaded ONNX bundle.
  - `<android>/app/src/test/resources/tokenizer_golden.json`
        Multi-language fixtures used by the JVM unit test
        `TokenizerGoldenTest` to byte-match the Python encoder.

Run after a fresh ONNX export:

    python tools/export_tokenizer_for_kotlin.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "onnx_export" / "tokenizer.model"
OUT_RUNTIME = REPO / "android" / "app" / "src" / "main" / "assets" / "tokenizer_kotlin.json"
OUT_GOLDEN = REPO / "android" / "app" / "src" / "test" / "resources" / "tokenizer_golden.json"

PIECE_TYPE_NAMES = {1: "NORMAL", 2: "UNKNOWN", 3: "CONTROL", 4: "USER_DEFINED", 5: "UNUSED", 6: "BYTE"}


def export_runtime(sp: spm.SentencePieceProcessor, model: pb.ModelProto) -> dict:
    """The shape consumed by Kotlin `Tokenizer.load(json)`."""
    pieces = []
    byte_start = -1
    for i, p in enumerate(model.pieces):
        # type: 1 NORMAL · 2 UNK · 3 CONTROL · 4 USER_DEFINED · 5 UNUSED · 6 BYTE
        pieces.append({"piece": p.piece, "score": float(p.score), "type": int(p.type)})
        if p.type == 6 and byte_start == -1:
            byte_start = i

    if byte_start == -1:
        raise SystemExit("no BYTE pieces found; byte_fallback expected for this model")
    # Sanity: the 256 byte tokens must be contiguous and named <0xXX>
    for off in range(256):
        idx = byte_start + off
        expected = f"<0x{off:02X}>"
        if model.pieces[idx].piece != expected:
            raise SystemExit(f"byte_fallback table broken at id {idx}: got {model.pieces[idx].piece!r}")

    norm = model.normalizer_spec
    return {
        "vocab_size": sp.GetPieceSize(),
        "unk_id": sp.unk_id(),
        "bos_id": sp.bos_id(),
        "eos_id": sp.eos_id(),
        "pad_id": sp.pad_id(),
        "byte_fallback_start": byte_start,
        "trainer": {
            "model_type": int(model.trainer_spec.model_type),
            "model_type_name": {1: "UNIGRAM", 2: "BPE", 3: "WORD", 4: "CHAR"}[model.trainer_spec.model_type],
            "byte_fallback": bool(model.trainer_spec.byte_fallback),
        },
        "normalizer": {
            "name": norm.name,
            "add_dummy_prefix": bool(norm.add_dummy_prefix),
            "remove_extra_whitespaces": bool(norm.remove_extra_whitespaces),
            "escape_whitespaces": bool(norm.escape_whitespaces),
        },
        "pieces": pieces,
    }


GOLDEN_TEXTS = [
    # — empty / whitespace —
    "", " ", "  ", "\t", "\n", "   leading spaces", "trailing spaces   ",
    # — English —
    "Hello World", "Hello, World!", "The quick brown fox jumps over the lazy dog.",
    "AbCdEfGhIjKlMnOp", "I'm OK with you're alright.",
    # — Numbers / dates / punctuation —
    "12345", "12.5", "3.14159265", "$1,234.56", "2024-04-17",
    "test@example.com", "https://github.com/AFun9/MOSS-TTS-Nano",
    # — Chinese (mixed script + punct) —
    "你好", "你好世界", "我喜欢吃苹果。", "今天天气真好,我们去公园吧!",
    "中文里夹杂 English words 和 12345 数字。",
    "今晚的月亮很美——真的非常美。",
    # — Japanese (hiragana + katakana + kanji) —
    "こんにちは", "おはようございます", "コンピューター", "東京駅",
    "今日は天気がいいですね。", "アニメとマンガが大好き！",
    # — Korean —
    "안녕하세요", "한국어로 말할 수 있어요?", "오늘 날씨 정말 좋네요.",
    # — Russian —
    "Привет", "Здравствуйте, мир!", "Сегодня хорошая погода.",
    # — Spanish / French / German (accents) —
    "¿Cómo estás?", "Mañana voy a la playa.",
    "Bonjour, comment ça va ?", "Les œuvres de Molière",
    "Schöne Grüße aus München", "Über alles in der Welt",
    # — Arabic (RTL) —
    "مرحبا بالعالم", "السلام عليكم",
    # — Thai (no spaces) —
    "สวัสดีครับ", "ผมไปโรงเรียน",
    # — Vietnamese (combining) —
    "Xin chào", "Tiếng Việt rất hay",
    # — Edge cases —
    "🙂", "Hello 🌍 World 🚀", "café", "naïve façade",
    "FULLWIDTH ＡＢＣ１２３",  # half/full-width
    "smart \u201cquotes\u201d and \u2018apos\u2019",
    "em dash — and ellipsis …",
    "soft\u00ADhyphen", "non\u00A0break\u00A0space",
    "control \x07 bell", "ZWJ\u200dZWNJ\u200c",
    # — Long Chinese paragraph —
    "在过去的一年里,人工智能技术取得了惊人的进展,从大语言模型到多模态生成,各个领域都涌现出令人瞩目的应用。",
    # — Long English paragraph —
    "Streaming inference reduces first-chunk latency by emitting audio as soon as the first window of tokens is decoded, instead of waiting for the entire sequence to complete.",
    # — Mixed script burst —
    "中英日韩 mixed: hello 世界 こんにちは 안녕하세요 12345.",
    # — Special tokens visible as text —
    "<|im_start|>user\nhi<|im_end|>",
]


def export_golden(sp: spm.SentencePieceProcessor) -> list:
    out = []
    for t in GOLDEN_TEXTS:
        ids = sp.encode(t, out_type=int)
        pieces = sp.encode(t, out_type=str)
        out.append({"text": t, "ids": ids, "pieces": pieces})
    return out


def main() -> int:
    if not SRC.is_file():
        print(f"missing {SRC}", file=sys.stderr)
        return 1
    sp = spm.SentencePieceProcessor()
    sp.Load(str(SRC))
    model = pb.ModelProto()
    model.ParseFromString(SRC.read_bytes())

    runtime = export_runtime(sp, model)
    golden = export_golden(sp)

    OUT_RUNTIME.parent.mkdir(parents=True, exist_ok=True)
    OUT_GOLDEN.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_RUNTIME, "w", encoding="utf-8") as f:
        json.dump(runtime, f, ensure_ascii=False, separators=(",", ":"))
    with open(OUT_GOLDEN, "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=2)

    rt_kb = OUT_RUNTIME.stat().st_size / 1024
    gd_kb = OUT_GOLDEN.stat().st_size / 1024
    print(f"wrote {OUT_RUNTIME.relative_to(REPO)}  ({rt_kb:.1f} KiB · {len(runtime['pieces'])} pieces)")
    print(f"wrote {OUT_GOLDEN.relative_to(REPO)}  ({gd_kb:.1f} KiB · {len(golden)} fixtures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
