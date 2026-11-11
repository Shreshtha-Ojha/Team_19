"""Build the Colab notebook for Phase 5 (retrain with fog, and fog + copy-paste).

Colab reuses constellation_resplit.zip from Phase 2 (already in Drive) and
builds the fogged / copy-paste training sets itself with the embedded
scripts, so nothing new has to be uploaded. Test images are not used in
Colab at all; the returned models are scored locally on the Phase 3 test sets.

Run from the repository root:
  python scripts/build_colab_phase5.py
"""
import argparse
import hashlib
import json
from pathlib import Path

ULTRALYTICS = "8.4.210"
EPOCHS = 50
RUNS = ["fogmix", "fogmix_cp"]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, default=Path("data/raw/constellations/constellation_resplit.zip"))
    ap.add_argument("--out", type=Path, default=Path("colab/phase5_fog_training.ipynb"))
    args = ap.parse_args()

    digest = hashlib.sha256(args.zip.read_bytes()).hexdigest()
    files = {name: Path(f"src/constellation/{name}").read_text()
             for name in ("fog_sim.py", "make_train_sets.py", "train_yolo.py")}

    cells = [
        md("# Phase 5: retrain with fog, and with fog + copy-paste\n\n"
           "1. Choose **Runtime > Change runtime type > T4 GPU**.\n"
           "2. `constellation_resplit.zip` should still be in the Drive folder `constellation` from Phase 2.\n"
           "3. Choose **Runtime > Run all** and allow Drive access.\n\n"
           f"It builds two training sets (about 15 minutes), then trains two models for {EPOCHS} epochs "
           "each (about 40 minutes each). If the session disconnects, run all again: finished runs are "
           "skipped and an interrupted one continues from its last epoch.\n"
           "When it finishes, download `constellation/phase5_results.zip` from Drive.\n"),
        code(f"!pip -q install ultralytics=={ULTRALYTICS} pyyaml\n"
             "import torch\n"
             "assert torch.cuda.is_available(), 'No GPU: choose Runtime > Change runtime type > T4 GPU'\n"
             "print(torch.cuda.get_device_name(0))\n"),
        code("from google.colab import drive\ndrive.mount('/content/drive')\n"),
        code("import hashlib, pathlib, zipfile, yaml\n"
             "DRIVE = pathlib.Path('/content/drive/MyDrive/constellation')\n"
             "CLEAN = pathlib.Path('/content/data')\n"
             "ZIP = DRIVE / 'constellation_resplit.zip'\n"
             "assert ZIP.exists(), f'Upload the dataset zip to {ZIP}'\n"
             f"EXPECTED_SHA256 = '{digest}'\n"
             "actual = hashlib.sha256(ZIP.read_bytes()).hexdigest()\n"
             "assert actual == EXPECTED_SHA256, f'Wrong or damaged zip: {actual}'\n"
             "if not (CLEAN / 'data.yaml').exists():\n"
             "    with zipfile.ZipFile(ZIP) as z:\n"
             "        z.extractall(CLEAN)\n"
             "print('Clean dataset ready')\n"),
        code("import os\n"
             "os.makedirs('/content/src/constellation', exist_ok=True)\n"
             "%cd /content\n"),
    ]
    for name, src in files.items():
        cells.append(code(f"%%writefile /content/src/constellation/{name}\n" + src))
    cells += [
        code("SETS = pathlib.Path('/content/phase5')\n"
             "if not (SETS / 'fogmix_cp' / 'data.yaml').exists():\n"
             "    !python -m src.constellation.make_train_sets --src {CLEAN} --out {SETS}\n"
             "assert (SETS / 'fogmix_cp' / 'data.yaml').exists(), 'Building the training sets failed; read the log above.'\n"),
        code("RUNS = DRIVE / 'runs'\n"
             f"for name in {RUNS!r}:\n"
             "    run = RUNS / name\n"
             "    if (run / 'train_info.json').exists():\n"
             "        print(name, 'already finished, skipping.')\n"
             "        continue\n"
             "    resume = '--resume' if (run / 'weights' / 'last.pt').exists() else ''\n"
             "    data = SETS / name / 'data.yaml'\n"
             "    !python src/constellation/train_yolo.py --data {data} --name {name} --project {RUNS} "
             f"--epochs {EPOCHS} --imgsz 640 --batch 16 --seed 0 --device 0 --workers 2 {{resume}}\n"
             "    assert (run / 'train_info.json').exists(), f'{name} did not finish; read the log above.'\n"),
        code("with zipfile.ZipFile(DRIVE / 'phase5_results.zip', 'w') as z:\n"
             "    z.write(SETS / 'manifest.csv', 'manifest.csv')\n"
             f"    for name in {RUNS!r}:\n"
             "        for f in ('results.csv', 'args.yaml', 'train_info.json', 'weights/best.pt'):\n"
             "            z.write(RUNS / name / f, f'{name}/{f}')\n"
             "print('Done. Download', DRIVE / 'phase5_results.zip')\n"),
    ]
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
                "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                             "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(notebook, indent=1))
    print(f"notebook: {args.out} (dataset sha256 {digest[:12]}…)")


if __name__ == "__main__":
    main()
