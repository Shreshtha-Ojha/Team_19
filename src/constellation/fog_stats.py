"""Measure how much each fog severity degrades the images, and draw a figure.

Star retention: point sources are found in the clean image and in the fogged
image with the same detector (white top-hat, then local maxima above
k * robust noise sigma). Reference stars in the clean image must also be at
least --clean-min-amp grey levels above the local background, so JPEG texture
is not counted as stars. A clean star counts as retained when a fogged
detection lies within --match-px pixels.

Noise in the fogged image also produces detections, some of which land near
a clean star by chance. The chance rate is measured by repeating the match
with the fogged detections shifted by a fixed offset; both numbers are
reported, and retention above chance is their difference.

Example:
  python -m src.constellation.fog_stats --clean data/raw/constellations/resplit \
      --fog data/raw/constellations/fog --out results/constellation/phase3
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_stars(gray: np.ndarray, k: float, min_amp: float) -> np.ndarray:
    """Return (N, 2) array of (x, y) point-source positions."""
    g = gray.astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, kernel)
    th = cv2.GaussianBlur(th, (0, 0), 0.8)
    mad = np.median(np.abs(th - np.median(th)))
    thresh = max(k * 1.4826 * mad, min_amp)
    peaks = (th == cv2.dilate(th, np.ones((5, 5), np.uint8))) & (th > thresh)
    ys, xs = np.nonzero(peaks)
    return np.stack([xs, ys], axis=1)


def retained(clean: np.ndarray, fog: np.ndarray, r: float) -> int:
    if len(clean) == 0 or len(fog) == 0:
        return 0
    d2 = ((clean[:, None, :] - fog[None, :, :]) ** 2).sum(-1)
    return int((d2.min(axis=1) <= r * r).sum())


def label_mask(label: Path, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), bool)
    if label.exists():
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                x0, x1 = int((cx - bw / 2) * w), int(np.ceil((cx + bw / 2) * w))
                y0, y1 = int((cy - bh / 2) * h), int(np.ceil((cy + bh / 2) * h))
                m[max(0, y0):y1, max(0, x0):x1] = True
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", type=Path, default=Path("data/raw/constellations/resplit"))
    ap.add_argument("--fog", type=Path, default=Path("data/raw/constellations/fog"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=Path("results/constellation/phase3"))
    ap.add_argument("--k", type=float, default=5.0, help="detection threshold in noise sigmas")
    ap.add_argument("--min-amp", type=float, default=4.0, help="minimum top-hat amplitude (grey levels)")
    ap.add_argument("--clean-min-amp", type=float, default=20.0,
                    help="minimum top-hat amplitude for reference stars in the clean image")
    ap.add_argument("--match-px", type=float, default=3.0)
    ap.add_argument("--chance-offset", type=int, nargs=2, default=(37, 23),
                    help="pixel shift (x y) applied to fogged detections to measure chance matches")
    ap.add_argument("--figure-images", nargs="*", default=None,
                    help="file-name prefixes to show in the figure (default: 4 spread across the split)")
    args = ap.parse_args()

    sevs = sorted((d.name for d in args.fog.iterdir() if (d / args.split / "images").is_dir()),
                  key=lambda s: ["light", "medium", "heavy"].index(s) if s in ("light", "medium", "heavy") else 9)
    clean_dir = args.clean / args.split / "images"
    names = sorted(f.stem for f in clean_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
    clean_files = {f.stem: f for f in clean_dir.iterdir() if f.suffix.lower() in IMG_EXTS}

    off = np.array(args.chance_offset)
    rows = {s: {"stars_clean": 0, "stars_kept": 0, "stars_kept_in_boxes": 0, "stars_clean_in_boxes": 0,
                "chance": 0, "chance_in_boxes": 0,
                "mean_grey": [], "bg_noise_sd": [], "psnr": []} for s in sevs}
    clean_mean = []
    for n in names:
        c = cv2.imread(str(clean_files[n]))
        cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        clean_mean.append(float(cg.mean()))
        cs = find_stars(cg, args.k, args.clean_min_amp)
        box = label_mask(args.clean / args.split / "labels" / f"{n}.txt", *cg.shape)
        in_box = box[cs[:, 1], cs[:, 0]] if len(cs) else np.zeros(0, bool)
        for s in sevs:
            f = cv2.imread(str(args.fog / s / args.split / "images" / f"{n}.jpg"))
            fg = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            fs = find_stars(fg, args.k, args.min_amp)
            r = rows[s]
            r["stars_clean"] += len(cs)
            r["stars_kept"] += retained(cs, fs, args.match_px)
            r["stars_clean_in_boxes"] += int(in_box.sum())
            r["stars_kept_in_boxes"] += retained(cs[in_box], fs, args.match_px)
            r["chance"] += retained(cs, fs + off, args.match_px)
            r["chance_in_boxes"] += retained(cs[in_box], fs + off, args.match_px)
            r["mean_grey"].append(float(fg.mean()))
            hp = fg.astype(np.float32) - cv2.medianBlur(fg, 5).astype(np.float32)
            r["bg_noise_sd"].append(float(1.4826 * np.median(np.abs(hp - np.median(hp)))))
            r["psnr"].append(float(cv2.PSNR(c, f)))

    summary = {"split": args.split, "n_images": len(names), "detector": {
               "k": args.k, "min_amp": args.min_amp, "clean_min_amp": args.clean_min_amp,
               "match_px": args.match_px, "chance_offset": list(args.chance_offset)}, "clean_mean_grey": float(np.mean(clean_mean)), "severities": {}}
    for s, r in rows.items():
        n, nb = max(1, r["stars_clean"]), max(1, r["stars_clean_in_boxes"])
        summary["severities"][s] = {
            "star_retention": r["stars_kept"] / n,
            "chance_retention": r["chance"] / n,
            "retention_above_chance": (r["stars_kept"] - r["chance"]) / n,
            "star_retention_in_boxes": r["stars_kept_in_boxes"] / nb,
            "chance_retention_in_boxes": r["chance_in_boxes"] / nb,
            "retention_above_chance_in_boxes": (r["stars_kept_in_boxes"] - r["chance_in_boxes"]) / nb,
            "stars_clean": r["stars_clean"], "stars_clean_in_boxes": r["stars_clean_in_boxes"],
            "mean_grey": float(np.mean(r["mean_grey"])),
            "bg_noise_sd": float(np.mean(r["bg_noise_sd"])),
            "psnr_db": float(np.mean(r["psnr"])),
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"fog_stats_{args.split}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Figure: rows = example images, columns = clean + severities. A brightened
    # copy (x4 gain) of each row is shown underneath, because the raw frames are
    # too dark to read on screen.
    picks = ([next(n for n in names if n.startswith(p)) for p in args.figure_images]
             if args.figure_images else [names[i] for i in np.linspace(0, len(names) - 1, 4).astype(int)])
    tile = 300
    font = cv2.FONT_HERSHEY_SIMPLEX
    rows_img = []
    for n in picks:
        imgs = [cv2.imread(str(clean_files[n]))] + \
               [cv2.imread(str(args.fog / s / args.split / "images" / f"{n}.jpg")) for s in sevs]
        for gain in (1, 4):
            cells = []
            for title, im in zip(["clean"] + sevs, imgs):
                t = cv2.resize(im, (tile, tile), interpolation=cv2.INTER_AREA)
                t = np.clip(t.astype(np.float32) * gain, 0, 255).astype(np.uint8)
                label = title + ("" if gain == 1 else "  (x4 display gain)")
                cv2.putText(t, label, (6, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cells.append(t)
            rows_img.append(np.hstack(cells))
    fig = np.vstack(rows_img)
    (args.out / "figures").mkdir(exist_ok=True)
    cv2.imwrite(str(args.out / "figures" / f"fog_examples_{args.split}.jpg"), fig, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return 0


if __name__ == "__main__":
    sys.exit(main())
