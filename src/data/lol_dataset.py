"""Helpers for reading paired low/normal-light images (e.g. LOL).

Takes explicit low_dir/high_dir paths rather than assuming any particular
top-level LOL folder layout — the exact internal structure of the
downloaded archive is confirmed by verify_lol.py once the data is present.
"""
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(d: Path):
    return sorted([p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS])


def load_pairs(low_dir, high_dir):
    """Pair images in low_dir and high_dir by matching filename (stem).

    Raises ValueError if a low-light image has no normal-light partner —
    fail loudly rather than silently misaligning pairs.
    """
    low_files = list_images(Path(low_dir))
    high_files = {p.stem: p for p in list_images(Path(high_dir))}
    pairs = []
    for lp in low_files:
        if lp.stem not in high_files:
            raise ValueError(f"no matching high-light image for {lp.name} in {high_dir}")
        pairs.append((lp, high_files[lp.stem]))
    return pairs
