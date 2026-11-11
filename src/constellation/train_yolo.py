"""Fine-tune a YOLO detector on the constellation dataset.

Every experiment (baseline, fog, fog + copy-paste) uses this script, so the
only thing that differs between them is --data and --name. Early stopping is
off by default so that each run gets the same number of epochs.

Example (Colab):
  python train_yolo.py --data /content/data/data.yaml --name baseline \
      --project /content/drive/MyDrive/constellation/runs
Continue an interrupted run by repeating the command with --resume.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/raw/constellations/resplit/data.yaml"))
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--project", type=Path, default=Path("runs/constellation"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=0, help="0 turns early stopping off")
    ap.add_argument("--device", default="0", help="GPU index, or cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO

    project = args.project.resolve()
    run_dir = project / args.name
    start = time.time()
    if args.resume:
        last = run_dir / "weights" / "last.pt"
        if not last.exists():
            print(f"ERROR: nothing to resume at {last}", file=sys.stderr)
            return 1
        YOLO(str(last)).train(resume=True, device=args.device, workers=args.workers)
    else:
        if run_dir.exists():
            print(f"ERROR: {run_dir} already exists. Use --resume or a new --name.", file=sys.stderr)
            return 1
        YOLO(args.model).train(
            data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, seed=args.seed, deterministic=True, patience=args.patience,
            device=args.device, workers=args.workers, project=str(project), name=args.name)

    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"ERROR: training finished but {best} is missing", file=sys.stderr)
        return 1
    # Read the settings the run really used, which on --resume are the original
    # run's rather than this command line's.
    used = yaml.safe_load((run_dir / "args.yaml").read_text())
    info = {
        "name": args.name, "model": used["model"], "data": used["data"],
        "data_yaml_sha256": sha256(Path(used["data"])), "epochs": used["epochs"],
        "imgsz": used["imgsz"], "batch": used["batch"], "seed": used["seed"],
        "patience": used["patience"],
        "resumed": args.resume, "seconds_this_session": round(time.time() - start, 1),
        "best_sha256": sha256(best), "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (run_dir / "train_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
