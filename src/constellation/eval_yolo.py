"""Evaluate trained weights on one split and save the metrics as JSON.

--only-regex restricts the evaluation to images whose file name matches, for
example '^\\d{4}-\\d{2}-\\d{2}-' for the daily sky screenshots only.

Example:
  python eval_yolo.py --weights best.pt --data data.yaml \
      --out results/constellation/phase2/baseline_test_all.json
"""
import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--only-regex", help="keep only images whose file name matches")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="GPU index, or cpu")
    args = ap.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO

    cfg = yaml.safe_load(args.data.read_text())
    base = Path(cfg.get("path") or args.data.parent)
    img_dir = base / cfg[args.split]
    images = sorted(p.resolve() for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if args.only_regex:
        images = [p for p in images if re.search(args.only_regex, p.name)]
    if not images:
        print(f"ERROR: no images selected from {img_dir}", file=sys.stderr)
        return 1

    names = cfg["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    instances = Counter()
    for img in images:
        label = img.parent.parent / "labels" / f"{img.stem}.txt"
        for line in label.read_text().splitlines():
            if line.strip():
                instances[names[int(line.split()[0])]] += 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "images.txt").write_text("\n".join(map(str, images)) + "\n")
        run_cfg = {**cfg, "path": str(base.resolve()), args.split: str(tmp / "images.txt")}
        (tmp / "data.yaml").write_text(yaml.safe_dump(run_cfg, sort_keys=False))
        m = YOLO(str(args.weights)).val(
            data=str(tmp / "data.yaml"), split=args.split, imgsz=args.imgsz, batch=args.batch,
            device=args.device, plots=False, verbose=False, project=str(tmp / "runs"), name="val")

    box = m.box
    per_class = {c: {"instances": instances[c], "ap50": None, "ap50_95": None} for c in names}
    for i, cls in enumerate(box.ap_class_index):
        per_class[names[int(cls)]].update(ap50=float(box.ap50[i]), ap50_95=float(box.ap[i]))

    result = {
        "weights": str(args.weights),
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "data": str(args.data), "split": args.split, "only_regex": args.only_regex,
        "n_images": len(images), "n_instances": sum(instances.values()),
        "mAP50": float(box.map50), "mAP50_95": float(box.map),
        "precision": float(box.mp), "recall": float(box.mr), "per_class": per_class,
        "imgsz": args.imgsz, "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("n_images", "n_instances", "mAP50", "mAP50_95",
                                             "precision", "recall")}, indent=2))
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
