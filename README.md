# Constellation Detection in Foggy, Low-Light Sky Images

Course project: *Low-Light Image Enhancement using Spatial and Copy-Paste
Techniques*. A YOLOv8n detector finds constellations in night-sky images. We
then measure how fog and low light break it, and how much classical
enhancement and training-time augmentation (fog, copy-paste) recover.

All numbers below come from result files saved under `results/constellation/`.

## Status

| Phase | What | State |
|---|---|---|
| 1 | Dataset check and leak-free re-split | done |
| 2 | YOLOv8n baseline (Colab T4) | done |
| 3 | Fog + low-light simulator, baseline under fog | done |
| 4 | Classical enhancement before detection | in progress |
| 5 | Retrain with fog / fog + copy-paste | planned |
| 6 | Results table, figures | planned |
| 7 | Constellation stick figures | planned |

## Dataset

Roboflow Universe, `course-work-cytun/constellation-detector` v1 (Public
Domain), 16 classes. The published splits share source images, so every
phase uses the block re-split made by `src/constellation/resplit_dataset.py`
(seed 0): 3294 train, 97 valid and 92 test images.

## Results so far

Baseline YOLOv8n (100 epochs, 640 px), test split, 92 images, 283 boxes:

| Test images | mAP50 | mAP50-95 |
|---|---|---|
| Clean | 0.866 | 0.454 |
| Light fog | 0.070 | 0.032 |
| Medium fog | 0.022 | 0.006 |
| Heavy fog | 0.019 | 0.005 |

Applying one part of "light" at a time shows that sensor noise causes most
of the drop (mAP50 0.140 with noise alone, 0.789 with fog alone, 0.842 with
darkening alone). The dataset images are almost noise-free, so the baseline
never saw grain during training.

![Fog severities](results/constellation/phase3/figures/fog_examples_test.jpg)

## Fog and low-light simulator

`src/constellation/fog_sim.py` works in linear light and uses the haze model
I = J·t + A·(1 − t), with patchy optical depth, glow around bright sources,
reduced exposure, and Poisson + read noise. Each image's random seed comes
from its file name, so the output is identical every run.

```bash
python -m src.constellation.fog_sim --split test            # writes data/raw/constellations/fog/
python -m src.constellation.fog_stats --out results/constellation/phase3
```

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# detection (evaluation runs on CPU; training runs on Colab):
python -m venv .venv-det
.venv-det/bin/pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cpu
.venv-det/bin/pip install -r requirements-detection.txt
```

Evaluate a model on one test set:

```bash
.venv-det/bin/python src/constellation/eval_yolo.py \
    --weights results/constellation/phase2/baseline/weights/best.pt \
    --data data/raw/constellations/fog/light/data.yaml --split test --device cpu \
    --out results/constellation/phase3/baseline_test_light.json
```

## Layout

- `src/constellation/`: dataset checks, re-split, training, evaluation, fog simulator
- `src/classical/`, `src/pipeline/`: classical enhancement operators (CLAHE,
  Retinex, gamma, homomorphic filtering, denoising) used in Phase 4
- `colab/`: notebooks for GPU training
- `results/constellation/phaseN/`: saved outputs of each phase
