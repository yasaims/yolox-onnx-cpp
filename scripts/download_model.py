#!/usr/bin/env python3
"""Download a YOLOX ONNX model into models/.

Standard-library only (urllib), so it runs anywhere Python 3 does without
extra dependencies. Verifies the download against a pinned SHA256 so a
corrupted or unexpected file is never silently used, and skips re-downloading
if a valid file already exists.

Usage:
    python scripts/download_model.py            # fetches yolox_nano.onnx
    python scripts/download_model.py --model tiny
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

# Pinned to the YOLOX 0.1.1rc0 release assets. Update alongside the SHA256
# below if the upstream release is ever re-tagged.
MODELS = {
    "nano": {
        "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx",
        "sha256": "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d",
        "filename": "yolox_nano.onnx",
    },
    "tiny": {
        "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx",
        "sha256": "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
        "filename": "yolox_tiny.onnx",
    },
}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(model_key: str) -> Path:
    spec = MODELS[model_key]
    dest = MODELS_DIR / spec["filename"]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if dest.exists() and sha256sum(dest) == spec["sha256"]:
        print(f"[skip] {dest} already present and verified")
        return dest

    print(f"[download] {spec['url']} -> {dest}")
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(spec["url"], tmp_path)

    actual = sha256sum(tmp_path)
    if actual != spec["sha256"]:
        tmp_path.unlink(missing_ok=True)
        print(f"[error] SHA256 mismatch: expected {spec['sha256']}, got {actual}", file=sys.stderr)
        sys.exit(1)

    tmp_path.replace(dest)
    print(f"[ok] verified SHA256 for {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS.keys()), default="nano",
                         help="Which YOLOX variant to download (default: nano)")
    args = parser.parse_args()
    download(args.model)


if __name__ == "__main__":
    main()
