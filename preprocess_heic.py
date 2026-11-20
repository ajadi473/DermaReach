#!/usr/bin/env python3
"""
Optional helper: convert HEIC phone captures to JPEG for DermaReach training.

Many clinic images in this project were originally iPhone HEIC files. Keras
ImageDataGenerator expects JPEG/PNG, so convert first, then train.

Requires: pillow-heif (or pillow with HEIF support) and Pillow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError as exc:  # pragma: no cover - runtime dependency hint
    raise SystemExit(
        "pillow-heif is required for HEIC conversion.\n"
        "Install with: pip install pillow-heif pillow"
    ) from exc


def convert_tree(source: Path, destination: Path, keep_structure: bool = True) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    converted = 0
    for path in source.rglob("*"):
        if path.suffix.lower() not in {".heic", ".heif"}:
            continue
        relative = path.relative_to(source) if keep_structure else Path(path.name)
        out_path = destination / relative.with_suffix(".jpg")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as img:
            img.convert("RGB").save(out_path, format="JPEG", quality=95)
        converted += 1
        print(f"Converted {path} -> {out_path}")
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HEIC/HEIF images to JPEG.")
    parser.add_argument("--source", type=Path, required=True, help="Folder with HEIC files.")
    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
        help="Output folder for JPEG files.",
    )
    args = parser.parse_args()
    count = convert_tree(args.source, args.destination)
    print(f"Converted {count} image(s).")


if __name__ == "__main__":
    main()
