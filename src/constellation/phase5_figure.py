"""Figure for Phase 5: test images under medium fog, original vs fog-trained detector.

Columns: clean image with true labels, fogged image with the original
(Phase 2) detector's boxes, fogged image with the fog-trained (fogmix)
detector's boxes. Images are picked by a fixed rule (every --step-th test
image in name order), not by how good the result looks. Display gain as in
phase4_figure.py; the detectors see the images without it.

  python -m src.constellation.phase5_figure
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", type=Path, default=Path("data/raw/constellations/resplit/test"))
    ap.add_argument("--fog", type=Path, default=Path("data/raw/constellations/fog/medium/test/images"))
    ap.add_argument("--baseline", type=Path, default=Path("results/constellation/phase2/baseline/weights/best.pt"))
    ap.add_argument("--fogmix", type=Path, default=Path("results/constellation/phase5/fogmix/weights/best.pt"))
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--step", type=int, default=30)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--gain", type=float, default=3.0)
    ap.add_argument("--out", type=Path, default=Path("results/constellation/phase5/figures/detections_medium.jpg"))
    args = ap.parse_args()

    from ultralytics import YOLO
    models = {"original detector": YOLO(str(args.baseline)), "fog-trained detector": YOLO(str(args.fogmix))}
    names = models["original detector"].names
    tile, font = 360, cv2.FONT_HERSHEY_SIMPLEX
    images = sorted(p for p in (args.clean / "images").iterdir() if p.suffix == ".jpg")
    picked = images[::args.step][:args.rows]

    def show(img, boxes, colour, title):
        g = min(args.gain, max(1.0, 40.0 / max(img.mean(), 1e-3)))
        out = np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)
        for (x0, y0, x1, y1), label in boxes:
            cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), colour, 2)
            cv2.putText(out, label, (int(x0), max(14, int(y0) - 4)), font, 0.55, colour, 2, cv2.LINE_AA)
        t = cv2.resize(out, (tile, tile), interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 0), (tile, 26), (0, 0, 0), -1)
        cv2.putText(t, title, (6, 18), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return t

    rows = []
    for p in picked:
        clean = cv2.imread(str(p))
        h, w = clean.shape[:2]
        gt = []
        lab = args.clean / "labels" / f"{p.stem}.txt"
        for line in lab.read_text().splitlines() if lab.exists() else []:
            c, cx, cy, bw, bh = (float(v) for v in line.split()[:5])
            gt.append((((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h), names[int(c)]))
        fog = cv2.imread(str(args.fog / f"{p.stem}.jpg"))
        cells = [show(clean, gt, (80, 220, 120), f"clean, true labels ({len(gt)})")]
        for title, m in models.items():
            r = m.predict(fog, conf=args.conf, verbose=False, device="cpu")[0]
            boxes = [(b, f"{names[int(c)]} {s:.2f}") for b, c, s in
                     zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
            cells.append(show(fog, boxes, (0, 220, 255), f"fog, {title} ({len(boxes)})"))
        rows.append(np.hstack(cells))
    fig = np.vstack(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), fig, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Saved {args.out}: {[p.stem[:20] for p in picked]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
