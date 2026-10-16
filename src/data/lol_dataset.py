"""Generic paired low/normal-light image dataset.

Deliberately takes explicit low_dir/high_dir paths rather than assuming any
particular top-level LOL folder layout — the exact internal structure of the
downloaded archive is confirmed by verify_lol.py once the data is present.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(d: Path):
    return sorted([p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS])


class PairedLowLightDataset(Dataset):
    """Pairs images in low_dir and high_dir by matching filename (stem).

    Raises ValueError at construction time if the two directories don't
    contain exactly the same set of filenames — fail loudly rather than
    silently misaligning pairs.
    """

    def __init__(self, low_dir, high_dir, crop_size: int | None = None,
                 augment=None, subset_fraction: float = 1.0, seed: int = 0):
        self.low_dir = Path(low_dir)
        self.high_dir = Path(high_dir)
        low_files = list_images(self.low_dir)
        high_files = list_images(self.high_dir)

        low_stems = {p.stem for p in low_files}
        high_stems = {p.stem for p in high_files}
        if low_stems != high_stems:
            missing_in_high = low_stems - high_stems
            missing_in_low = high_stems - low_stems
            raise ValueError(
                f"low/high filename mismatch between {low_dir} and {high_dir}: "
                f"{len(missing_in_high)} in low only, {len(missing_in_low)} in high only "
                f"(examples: {list(missing_in_high)[:3]}, {list(missing_in_low)[:3]})"
            )

        stems = sorted(low_stems)
        if subset_fraction < 1.0:
            rng = np.random.default_rng(seed)
            n = max(1, int(round(len(stems) * subset_fraction)))
            stems = list(rng.choice(stems, size=n, replace=False))
            stems.sort()

        low_by_stem = {p.stem: p for p in low_files}
        high_by_stem = {p.stem: p for p in high_files}
        self.pairs = [(low_by_stem[s], high_by_stem[s]) for s in stems]

        self.crop_size = crop_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def _load(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return img

    def __getitem__(self, idx):
        low_path, high_path = self.pairs[idx]
        low = self._load(low_path)
        high = self._load(high_path)

        if self.crop_size is not None:
            h, w = low.shape[:2]
            cs = self.crop_size
            if h < cs or w < cs:
                raise ValueError(f"image {low_path} is {h}x{w}, smaller than crop_size {cs}")
            y0 = np.random.randint(0, h - cs + 1)
            x0 = np.random.randint(0, w - cs + 1)
            low = low[y0:y0 + cs, x0:x0 + cs]
            high = high[y0:y0 + cs, x0:x0 + cs]

        if self.augment is not None:
            low, high = self.augment(low, high)

        low_t = torch.from_numpy(np.ascontiguousarray(low.transpose(2, 0, 1)))
        high_t = torch.from_numpy(np.ascontiguousarray(high.transpose(2, 0, 1)))
        return low_t, high_t, low_path.name
