"""Build the Phase 5 training sets from the clean re-split.

  fogmix     every train and valid image gets one condition, drawn uniformly
             from clean / light / medium / heavy, and is degraded with
             fog_sim.py. Same images and labels as the baseline, only degraded.
  fogmix_cp  like fogmix, but before the fog, half of the train images get
             1-2 constellations copied in from other train images
             (copy-paste). Valid is the same as fogmix (no pasting).

Copy-paste for a night sky: the pasted crop's own background level (its
per-channel median) is subtracted and only the light above it is added to
the target image, so stars are added without a visible rectangle. A crop is
placed only where it does not overlap any box already in the image.

The test split is not touched: both data.yaml files point at the clean test
images, and the fogged test sets from Phase 3 are used for scoring.

Example:
  python -m src.constellation.make_train_sets --src data/raw/constellations/resplit \
      --out data/raw/constellations/phase5
"""
import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.constellation.fog_sim import SEVERITIES, apply_fog, image_rng

CONDITIONS = ["clean", *SEVERITIES]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def rng_for(name: str, tag: str, seed: int) -> np.random.Generator:
    d = hashlib.sha256(f"{seed}:{tag}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(d[:8], "little"))


def read_labels(path: Path) -> list[list[float]]:
    if not path.exists():
        return []
    return [[float(v) for v in line.split()[:5]] for line in path.read_text().splitlines() if line.strip()]


def write_labels(path: Path, rows: list[list[float]]) -> None:
    path.write_text("".join(f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n" for r in rows))


def to_px(r, w, h):
    _, cx, cy, bw, bh = r
    return int((cx - bw / 2) * w), int((cy - bh / 2) * h), int(np.ceil((cx + bw / 2) * w)), int(np.ceil((cy + bh / 2) * h))


def crop_bank(images: list[Path], labels_dir: Path, max_frac: float) -> list[tuple]:
    """(source stem, class, crop) for every box at most max_frac of the image in each direction."""
    bank = []
    for f in images:
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        for r in read_labels(labels_dir / f"{f.stem}.txt"):
            if r[3] <= max_frac and r[4] <= max_frac:
                x0, y0, x1, y1 = to_px(r, w, h)
                x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
                if x1 - x0 >= 8 and y1 - y0 >= 8:
                    bank.append((f.stem, int(r[0]), img[y0:y1, x0:x1].copy()))
    return bank


def overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def paste(img, rows, bank, rng, stem, n_max, tries=30):
    """Add 1..n_max crops from other images where they overlap no existing box."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    boxes = [to_px(r, w, h) for r in rows]
    pasted = []
    for _ in range(int(rng.integers(1, n_max + 1))):
        src_stem, cls, crop = bank[int(rng.integers(len(bank)))]
        if src_stem == stem:
            continue
        ch, cw = crop.shape[:2]
        for _ in range(tries):
            x0, y0 = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
            box = (x0, y0, x0 + cw, y0 + ch)
            if not any(overlaps(box, b) for b in boxes):
                c = crop.astype(np.float32)
                light = np.maximum(c - np.median(c.reshape(-1, 3), axis=0), 0)
                out[y0:y0 + ch, x0:x0 + cw] += light
                boxes.append(box)
                rows.append([cls, (x0 + cw / 2) / w, (y0 + ch / 2) / h, cw / w, ch / h])
                pasted.append(f"{cls}@{src_stem[:24]}")
                break
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), rows, pasted


def build(src: Path, out: Path, seed: int, cp_prob: float, cp_max: int, max_frac: float, quality: int):
    names = yaml.safe_load((src / "data.yaml").read_text())["names"]
    manifest = []
    train_imgs = sorted(p for p in (src / "train/images").iterdir() if p.suffix.lower() in IMG_EXTS)
    bank = crop_bank(train_imgs, src / "train/labels", max_frac)
    print(f"copy-paste bank: {len(bank)} crops")
    jobs = [("fogmix", "train", False), ("fogmix", "valid", False), ("fogmix_cp", "train", True)]
    for set_name, split, do_cp in jobs:
        img_out, lab_out = out / set_name / split / "images", out / set_name / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lab_out.mkdir(parents=True, exist_ok=True)
        images = sorted(p for p in (src / split / "images").iterdir() if p.suffix.lower() in IMG_EXTS)
        for i, f in enumerate(images):
            img = cv2.imread(str(f))
            rows = read_labels(src / split / "labels" / f"{f.stem}.txt")
            # The condition depends only on the image, so fogmix and fogmix_cp
            # give each train image the same fog level.
            cond = CONDITIONS[int(rng_for(f.name, "condition", seed).integers(len(CONDITIONS)))]
            pasted = []
            if do_cp:
                r = rng_for(f.name, "paste", seed)
                if r.random() < cp_prob:
                    img, rows, pasted = paste(img, rows, bank, r, f.stem, cp_max)
            if cond != "clean":
                img = apply_fog(img, SEVERITIES[cond], image_rng(f.name, cond, seed))
            cv2.imwrite(str(img_out / f"{f.stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            write_labels(lab_out / f"{f.stem}.txt", rows)
            manifest.append([set_name, split, f.name, cond, ";".join(pasted)])
            if (i + 1) % 500 == 0:
                print(f"{set_name}/{split}: {i + 1}/{len(images)}", flush=True)
    # fogmix_cp validates on the same fogged valid images as fogmix.
    shutil.copytree(out / "fogmix" / "valid", out / "fogmix_cp" / "valid", dirs_exist_ok=True)
    for set_name in ("fogmix", "fogmix_cp"):
        cfg = {"path": str((out / set_name).resolve()), "train": "train/images", "val": "valid/images",
               "test": str((src / "test/images").resolve()), "names": names, "nc": len(names)}
        (out / set_name / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "split", "image", "condition", "pasted"])
        w.writerows(manifest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("data/raw/constellations/resplit"))
    ap.add_argument("--out", type=Path, default=Path("data/raw/constellations/phase5"))
    ap.add_argument("--seed", type=int, default=1, help="differs from the test fog seed (0)")
    ap.add_argument("--cp-prob", type=float, default=0.5, help="share of train images that get pasted crops")
    ap.add_argument("--cp-max", type=int, default=2, help="most crops pasted into one image")
    ap.add_argument("--max-frac", type=float, default=0.5, help="largest crop, as a fraction of the image side")
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()
    build(args.src, args.out, args.seed, args.cp_prob, args.cp_max, args.max_frac, args.quality)
    return 0


if __name__ == "__main__":
    sys.exit(main())
