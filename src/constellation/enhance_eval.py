"""Enhance clean and fogged images, then score the unchanged baseline on them.

For every (method, condition) pair the enhanced images are written to
<work>/<split>/<method>/<condition>/<split>/images (labels copied), and
eval_yolo.py is run on them. Existing outputs are reused, so the script can
be re-run after an interruption.

Conditions: "clean" (re-split images) and each fog severity under --fog.

Protocol: run with --split valid over all methods to choose settings, then
with --split test only for the methods being reported.

Run with the detection environment, from the repository root:
  .venv-det/bin/python -m src.constellation.enhance_eval --split valid \
      --out results/constellation/phase4/valid
"""
import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import yaml

from src.constellation.enhance import METHODS

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_KEY = {"train": "train", "valid": "val", "test": "test"}


def _enhance_one(job):
    method, src, dst = job
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    out = METHODS[method](img)
    cv2.imwrite(str(dst), out, [cv2.IMWRITE_PNG_COMPRESSION, 1])


def build_set(method: str, cond_root: Path, split: str, out_root: Path, names: list, pool) -> Path:
    img_in, lab_in = cond_root / split / "images", cond_root / split / "labels"
    img_out, lab_out = out_root / split / "images", out_root / split / "labels"
    images = sorted(f for f in img_in.iterdir() if f.suffix.lower() in IMG_EXTS)
    done = out_root / ".done"
    if not done.exists():
        img_out.mkdir(parents=True, exist_ok=True)
        lab_out.mkdir(parents=True, exist_ok=True)
        # PNG output: no second round of JPEG loss on top of the method.
        list(pool.map(_enhance_one, [(method, f, img_out / f"{f.stem}.png") for f in images],
                      chunksize=4))
        for f in images:
            lab = lab_in / f"{f.stem}.txt"
            if lab.exists():
                shutil.copyfile(lab, lab_out / lab.name)
        done.write_text(str(len(images)))
    data = {"path": str(out_root.resolve()), "names": names, "nc": len(names),
            SPLIT_KEY[split]: f"{split}/images"}
    for k in ("train", "val", "test"):
        data.setdefault(k, f"{split}/images")
    (out_root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return out_root / "data.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", type=Path, default=Path("data/raw/constellations/resplit"))
    ap.add_argument("--fog", type=Path, default=Path("data/raw/constellations/fog"))
    ap.add_argument("--conditions", nargs="+", default=["clean", "light", "medium", "heavy"])
    ap.add_argument("--split", default="valid", choices=("valid", "test"))
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--weights", type=Path,
                    default=Path("results/constellation/phase2/baseline/weights/best.pt"))
    ap.add_argument("--work", type=Path, default=Path("data/raw/constellations/enh"))
    ap.add_argument("--out", type=Path, required=True, help="folder for the per-run JSON results")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    unknown = set(args.methods) - set(METHODS)
    if unknown:
        print(f"unknown methods: {sorted(unknown)}", file=sys.stderr)
        return 1
    names = yaml.safe_load((args.clean / "data.yaml").read_text())["names"]
    eval_script = Path(__file__).with_name("eval_yolo.py")
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for method in args.methods:
            for cond in args.conditions:
                res = args.out / f"{method}__{cond}.json"
                if not res.exists():
                    cond_root = args.clean if cond == "clean" else args.fog / cond
                    data = build_set(method, cond_root, args.split,
                                     args.work / args.split / method / cond, names, pool)
                    r = subprocess.run([sys.executable, str(eval_script), "--weights", str(args.weights),
                                        "--data", str(data), "--split", SPLIT_KEY[args.split],
                                        "--device", args.device, "--out", str(res)],
                                       capture_output=True, text=True)
                    if r.returncode != 0:
                        print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
                        return 1
                d = json.loads(res.read_text())
                summary.setdefault(method, {})[cond] = {"mAP50": d["mAP50"], "mAP50_95": d["mAP50_95"]}
                print(f"{method:22s} {cond:7s} mAP50 {d['mAP50']:.3f}  mAP50-95 {d['mAP50_95']:.3f}",
                      flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
