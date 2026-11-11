"""Package the re-split dataset and build the Colab notebook for the baseline.

The training and evaluation scripts are embedded in the notebook, so Colab
needs only the notebook and the dataset zip. The zip's checksum is embedded
too, so Colab refuses to train on a different dataset.

Run from the repository root:
  python scripts/build_colab_phase2.py
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ULTRALYTICS = "8.4.210"
EXPECTED_COUNTS = {"train": 3294, "valid": 97, "test": 92}
DAILY_REGEX = r"^\d{4}-\d{2}-\d{2}-"


def make_zip(src: Path, dst: Path) -> None:
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(src.rglob("*")):
            if p.is_file() and p.suffix != ".cache":
                z.write(p, p.relative_to(src))


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("data/raw/constellations/resplit"))
    ap.add_argument("--zip", type=Path, default=Path("data/raw/constellations/constellation_resplit.zip"))
    ap.add_argument("--out", type=Path, default=Path("colab/phase2_baseline.ipynb"))
    args = ap.parse_args()

    make_zip(args.src, args.zip)
    digest = hashlib.sha256(args.zip.read_bytes()).hexdigest()
    train_src = Path("src/constellation/train_yolo.py").read_text()
    eval_src = Path("src/constellation/eval_yolo.py").read_text()

    cells = [
        md("# Phase 2: YOLOv8n baseline\n\n"
           "1. Choose **Runtime > Change runtime type > T4 GPU**.\n"
           "2. Put `constellation_resplit.zip` in a Google Drive folder named `constellation`.\n"
           "3. Choose **Runtime > Run all** and allow Drive access.\n\n"
           "If the session disconnects, run all again: training continues from the last epoch.\n"
           "When it finishes, download `constellation/phase2_results.zip` from Drive.\n"),
        code(f"!pip -q install ultralytics=={ULTRALYTICS} pyyaml\n"
             "import torch\n"
             "assert torch.cuda.is_available(), 'No GPU: choose Runtime > Change runtime type > T4 GPU'\n"
             "print(torch.cuda.get_device_name(0))\n"),
        code("from google.colab import drive\ndrive.mount('/content/drive')\n"),
        code("import hashlib, pathlib, zipfile, yaml\n"
             "DRIVE = pathlib.Path('/content/drive/MyDrive/constellation')\n"
             "DATA = pathlib.Path('/content/data')\n"
             "ZIP = DRIVE / 'constellation_resplit.zip'\n"
             "assert ZIP.exists(), f'Upload the dataset zip to {ZIP}'\n"
             f"EXPECTED_SHA256 = '{digest}'\n"
             "actual = hashlib.sha256(ZIP.read_bytes()).hexdigest()\n"
             "assert actual == EXPECTED_SHA256, f'Wrong or damaged zip: {actual}'\n"
             "if not (DATA / 'data.yaml').exists():\n"
             "    with zipfile.ZipFile(ZIP) as z:\n"
             "        z.extractall(DATA)\n"
             "cfg = yaml.safe_load((DATA / 'data.yaml').read_text())\n"
             "cfg['path'] = str(DATA)\n"
             "(DATA / 'data.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))\n"
             f"EXPECTED_COUNTS = {EXPECTED_COUNTS!r}\n"
             "counts = {s: len(list((DATA / s / 'images').iterdir())) for s in EXPECTED_COUNTS}\n"
             "assert counts == EXPECTED_COUNTS, counts\n"
             "print('Dataset OK:', counts)\n"),
        code("%%writefile train_yolo.py\n" + train_src),
        code("%%writefile eval_yolo.py\n" + eval_src),
        code("RUNS = DRIVE / 'runs'\n"
             "RUN = RUNS / 'baseline'\n"
             "if (RUN / 'train_info.json').exists():\n"
             "    print('Training already finished, skipping.')\n"
             "else:\n"
             "    resume = '--resume' if (RUN / 'weights' / 'last.pt').exists() else ''\n"
             "    !python train_yolo.py --data {DATA}/data.yaml --name baseline --project {RUNS} "
             "--epochs 100 --imgsz 640 --batch 16 --seed 0 --device 0 --workers 2 {resume}\n"
             "assert (RUN / 'train_info.json').exists(), 'Training did not finish; read the log above.'\n"),
        code("OUT = DRIVE / 'results' / 'phase2'\n"
             "BEST = RUN / 'weights' / 'best.pt'\n"
             "!python eval_yolo.py --weights {BEST} --data {DATA}/data.yaml --split test "
             "--out {OUT}/baseline_test_all.json --device 0\n"
             f"DAILY = r'{DAILY_REGEX}'\n"
             "!python eval_yolo.py --weights {BEST} --data {DATA}/data.yaml --split test "
             "--only-regex '{DAILY}' --out {OUT}/baseline_test_daily.json --device 0\n"
             "assert (OUT / 'baseline_test_all.json').exists() and (OUT / 'baseline_test_daily.json').exists()\n"),
        code("with zipfile.ZipFile(DRIVE / 'phase2_results.zip', 'w') as z:\n"
             "    for p in sorted(OUT.glob('*.json')):\n"
             "        z.write(p, p.name)\n"
             "    for name in ('results.csv', 'args.yaml', 'train_info.json', 'weights/best.pt'):\n"
             "        z.write(RUN / name, f'baseline/{name}')\n"
             "print('Done. Download', DRIVE / 'phase2_results.zip')\n"),
    ]
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
                "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                             "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(notebook, indent=1))
    print(f"zip: {args.zip} sha256={digest}\nnotebook: {args.out}")


if __name__ == "__main__":
    main()
