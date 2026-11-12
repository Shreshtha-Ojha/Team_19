"""Figure for Phase 4: one test image per fog level, under each enhancement
method, with the baseline detector's boxes drawn on.

Uses the enhanced images written by enhance_eval.py. For display only, dark
frames are brightened by up to --gain (never past a mean grey of 40, so
already-bright outputs are not blown out); the detector sees the images
without that gain.

  python -m src.constellation.phase4_figure --image 2022-02-02
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="2022-02-02", help="file-name prefix of the test image")
    ap.add_argument("--methods", nargs="+", default=["none", "clahe", "pipeline", "nlm_k3"])
    ap.add_argument("--conditions", nargs="+", default=["clean", "light", "medium"])
    ap.add_argument("--work", type=Path, default=Path("data/raw/constellations/enh/test"))
    ap.add_argument("--weights", type=Path,
                    default=Path("results/constellation/phase2/baseline/weights/best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--gain", type=float, default=3.0)
    ap.add_argument("--out", type=Path,
                    default=Path("results/constellation/phase4/figures/enhancement_examples.jpg"))
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    tile, font = 320, cv2.FONT_HERSHEY_SIMPLEX
    rows = []
    for cond in args.conditions:
        cells = []
        for m in args.methods:
            img_dir = args.work / m / cond / "test" / "images"
            path = next(p for p in sorted(img_dir.iterdir()) if p.name.startswith(args.image))
            img = cv2.imread(str(path))
            res = model.predict(img, conf=args.conf, verbose=False, device="cpu")[0]
            g = min(args.gain, max(1.0, 40.0 / max(img.mean(), 1e-3)))
            shown = np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)
            for (x0, y0, x1, y1), c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(),
                                              res.boxes.conf.tolist()):
                cv2.rectangle(shown, (int(x0), int(y0)), (int(x1), int(y1)), (0, 220, 255), 2)
                cv2.putText(shown, f"{model.names[int(c)]} {s:.2f}", (int(x0), max(14, int(y0) - 4)),
                            font, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
            t = cv2.resize(shown, (tile, tile), interpolation=cv2.INTER_AREA)
            cv2.putText(t, f"{cond} / {m}  ({len(res.boxes)} boxes)", (6, 18), font, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(t)
        rows.append(np.hstack(cells))
    fig = np.vstack(rows)
    cv2.putText(fig, f"display gain up to x{args.gain:g}; boxes at conf >= {args.conf}", (6, fig.shape[0] - 8),
                font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), fig, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
