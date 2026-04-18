"""dump_prompt_fixtures_v2 - byte-equal Python <-> Kotlin PromptBuilder fixtures.

Replaces the v1 fixture file (`android/app/src/test/resources/prompt_fixtures.json`)
which targeted the old custom 5-session protocol. The v1.0.0 release ships
a single `voice_clone` layout driven entirely by the manifest's
`prompt_templates`, so this dumper keeps the harness minimal:

    - Loads `export_v2/release/manifest.json` (the source of truth on-device).
    - Calls `demo_generate.build_input_ids(manifest, text_ids, codes)` to get
      the canonical [1, T, 17] int32 prompt.
    - Generates a handful of (text_ids, F) combinations using a deterministic
      formula for both inputs, so the Kotlin side can rebuild the exact same
      inputs without needing the tokenizer or any audio assets.

Each fixture is dumped as the full int64 input_ids list plus a SHA-256 of
the byte buffer for fast first-line-of-defence comparisons.

Usage:
    python tools/dump_prompt_fixtures_v2.py
        # writes android/app/src/test/resources/prompt_fixtures_v2.json
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "export_v2" / "scripts"))

# Re-uses the canonical builder; if demo_generate.py drifts from the
# on-device protocol this fixture starts failing immediately, which is
# exactly the regression alarm we want.
import demo_generate  # noqa: E402

RELEASE_DIR = ROOT / "export_v2" / "release"
OUT = ROOT / "android" / "app" / "src" / "test" / "resources" / "prompt_fixtures_v2.json"


def fake_codes(frames: int, n_vq: int, codebook_size: int, seed: int = 0) -> np.ndarray:
    """Deterministic [F, n_vq] int code grid - matches Kotlin's PromptBuilderTest.fakeCodes."""
    flat = np.empty(frames * n_vq, dtype=np.int64)
    for i in range(flat.size):
        flat[i] = (i * 31 + seed) % codebook_size
    return flat.reshape(frames, n_vq)


def fake_text_ids(start: int, length: int, stride: int = 7) -> np.ndarray:
    """Deterministic text id sequence (avoids needing the tokenizer here)."""
    return np.array([(start + i * stride) for i in range(length)], dtype=np.int32)


def fingerprint(input_ids: np.ndarray) -> dict:
    flat = input_ids.flatten().astype(np.int64)
    sha = hashlib.sha256(flat.tobytes()).hexdigest()
    return {
        "shape": list(input_ids.shape),
        "length": int(flat.size),
        "first_16": [int(x) for x in flat[:16].tolist()],
        "last_16": [int(x) for x in flat[-16:].tolist()],
        "sum": int(flat.sum()),
        "sha256": sha,
    }


def build_fixture(name: str, manifest: dict, text_ids: np.ndarray, codes: np.ndarray) -> dict:
    # demo_generate signature is (manifest, prompt_audio_codes, text_token_ids).
    arr = demo_generate.build_input_ids(manifest, codes, text_ids)  # [1, T, 17] int32
    fp = fingerprint(arr)
    return {
        "id": name,
        "input": {
            "text_token_ids": [int(x) for x in text_ids.tolist()],
            "audio_prompt": {
                "frames": int(codes.shape[0]),
                "n_vq": int(codes.shape[1]),
                "codes": [int(x) for x in codes.flatten().tolist()],
            },
        },
        "input_ids_full": [int(x) for x in arr.flatten().tolist()],
        "fingerprint": fp,
    }


def main() -> int:
    manifest_path = RELEASE_DIR / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} missing - run make_release.py first", file=sys.stderr)
        return 1
    with open(manifest_path) as f:
        manifest = json.load(f)

    cfg = manifest["tts_config"]
    n_vq = cfg["n_vq"]
    codebook = cfg["audio_codebook_sizes"][0]

    # Five fixtures span the interesting axes:
    #   - tiny / mid / large reference frames (tests audio block sizing)
    #   - tiny / mid / large text id sequences
    #   - non-trivial seeds so codes vary between fixtures
    cases = [
        ("tiny",          1,   1, 0),
        ("short",         5,   3, 7),
        ("mid",          20,  12, 13),
        ("long_text",     8,  64, 21),
        ("long_audio", 100,  16, 99),
    ]

    fixtures = []
    for name, frames, text_len, seed in cases:
        codes = fake_codes(frames, n_vq, codebook, seed=seed)
        text_ids = fake_text_ids(start=1000 + seed, length=text_len)
        fixtures.append(build_fixture(name, manifest, text_ids, codes))

    out = {
        "schema": "moss-tts-nano-prompt-fixtures-v2/1",
        "release_version": manifest.get("release_version", "unknown"),
        "n_vq": n_vq,
        "audio_codebook_size": codebook,
        "fixtures": fixtures,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(fixtures)} fixtures, {size_kb:.1f} KiB)")
    for fx in fixtures:
        fp = fx["fingerprint"]
        print(f"  {fx['id']:>10}  shape={fp['shape']}  len={fp['length']:>6}  sha={fp['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
