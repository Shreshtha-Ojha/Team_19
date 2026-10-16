# Low-Light Image Enhancement: Spatial Methods vs. a Small U-Net, with a Copy-Blend Ablation

College project for Image Processing & Computer Vision. Compares classical
spatial-domain low-light enhancement methods against a small supervised
U-Net, and runs an ablation on whether Copy-Blend augmentation (Shyam et al.,
arXiv:2103.05889) improves the network, including a reduced-training-data
("25%") setting where augmentation should matter most.

## Environment

- Python 3.12 (system Python is PEP 668-managed; use the venv below)
- No CUDA GPU on the development machine (verified: `torch.cuda.is_available()`
  is `False`; host GPU is an integrated Intel Iris Xe). All training/eval code
  detects the device at runtime (`torch.cuda.is_available()`) rather than
  assuming CPU or GPU, so it will use a GPU automatically if run on a machine
  that has one.
- Exact installed versions are pinned in `requirements.txt` (generated via
  `pip freeze` after installing, not hand-written).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(PyTorch was installed from the CPU-only wheel index,
`--index-url https://download.pytorch.org/whl/cpu`, since no CUDA GPU is
present on the development machine; use the standard index instead if
installing on a CUDA machine.)

## 1. Dataset — LOL (manual download required)

See `data/README.md` for the exact download links (Google Drive / Baidu Pan —
both require a manual step, verified against the authors' project page) and
where to place the extracted archive (`data/raw/LOL/`).

After placing it, discover and verify the structure:

```bash
python src/data/verify_lol.py
```

This reports the actual folder layout found (it does not assume a specific
layout) and checks that every low-light image has a matching normal-light
image by filename. **The exact `--low-dir`/`--high-dir` paths used in the
commands below depend on what this script reports** — fill them in once you
have real output.

## 2. Classical spatial-domain baselines

Implemented in `src/classical/`: `gamma.py`, `hist_eq.py` (global histogram
equalization on LAB L-channel), `clahe.py` (CLAHE on LAB L-channel),
`retinex.py` (single- and multi-scale Retinex), `denoise.py` (optional
bilateral/median denoise). These are plain functions, exercised directly by
`evaluate.py` and `make_figure.py` — there's no separate "run baselines"
script needed.

## 3. Train the U-Net (four ablation runs)

```bash
# full data, standard augmentation (flip/crop)
python src/train.py --low-dir <LOW> --high-dir <HIGH> \
    --aug standard --subset-fraction 1.0 --epochs 60 \
    --out checkpoints/unet_standard_full.pt

# full data, standard + copy-blend
python src/train.py --low-dir <LOW> --high-dir <HIGH> \
    --aug copyblend --subset-fraction 1.0 --epochs 60 \
    --out checkpoints/unet_copyblend_full.pt

# 25% of training data, standard augmentation
python src/train.py --low-dir <LOW> --high-dir <HIGH> \
    --aug standard --subset-fraction 0.25 --epochs 60 \
    --out checkpoints/unet_standard_25pct.pt

# 25% of training data, standard + copy-blend
python src/train.py --low-dir <LOW> --high-dir <HIGH> \
    --aug copyblend --subset-fraction 0.25 --epochs 60 \
    --out checkpoints/unet_copyblend_25pct.pt
```

All four runs use the same seed (`--seed 0`, the default) and the same
epoch count, differing only in `--aug` and `--subset-fraction`, as required
for a clean ablation.

**Measured CPU training speed** (benchmarked on this machine, i7-1255U,
no GPU — see conversation log, not a vendor spec): with 128x128 crops and
batch size 8, ~0.23s/training step. For the LOL training set (485 pairs if
`our485` is in fact 485 images — to be confirmed by `verify_lol.py`), that's
~14s/epoch for the full set and ~3.5s/epoch for the 25% subset, i.e. all
four 60-epoch runs combined should take well under an hour on this hardware.
Each run also writes a `<name>.log.json` next to its checkpoint with the
actual measured per-epoch time and loss — the real numbers, not an estimate.

### Copy-Blend augmentation — implementation notes

Implemented in `src/data/copy_blend.py` from the **paper's description only**
(arXiv:2103.05889), not from the authors' reference code: the repository the
paper cites, `github.com/PS06/Copy_Blend`, returned HTTP 404 when checked
(2026-10-16), and the GitHub account `PS06` has no public repository by that
name. The module docstring lists, in detail, every place the paper
underspecifies the method (the numeric blend-intensity value; what "and vice
versa" means for the training target) and the documented, reasoned choice
made for each. Read it before trusting results from this ablation as a
faithful reproduction — it is a good-faith re-implementation, not a verified
match to the original code.

## 4. Evaluate: PSNR/SSIM on the test set

```bash
python src/evaluate.py --low-dir <TEST_LOW> --high-dir <TEST_HIGH> \
    --checkpoints checkpoints/unet_standard_full.pt checkpoints/unet_copyblend_full.pt \
                  checkpoints/unet_standard_25pct.pt checkpoints/unet_copyblend_25pct.pt \
    --out results/metrics.csv
```

Computes PSNR and SSIM (via `skimage.metrics`) for every classical method and
every checkpoint passed in, over every image in the test set. Writes
per-image metrics to `results/metrics.csv` and the per-method mean to
`results/metrics_summary.csv`. These are the only files any results summary
in this README should ever be written from.

## 5. Comparison figure

```bash
python src/make_figure.py --low-dir <TEST_LOW> --high-dir <TEST_HIGH> \
    --checkpoints checkpoints/unet_standard_full.pt checkpoints/unet_copyblend_full.pt \
    --num-images 4 --out results/figures/comparison.png
```

## Results

**Pending** — no training or evaluation run has been executed yet on the
real LOL dataset (still waiting on the manual download, see Section 1). This
section will be filled in after real runs, quoting only numbers that appear
in `results/metrics.csv` / `results/metrics_summary.csv`.

## Project structure

```
lowlight-enhancement/
├── requirements.txt
├── README.md
├── data/
│   ├── raw/LOL/            # place downloaded dataset here (see data/README.md)
│   └── README.md
├── src/
│   ├── data/
│   │   ├── lol_dataset.py  # generic paired-image dataset
│   │   ├── verify_lol.py   # discovers & verifies actual folder layout
│   │   └── copy_blend.py   # paper-based re-implementation (see notes above)
│   ├── classical/          # gamma, hist_eq, clahe, retinex, denoise
│   ├── model/unet.py       # SmallUNet
│   ├── train.py
│   ├── evaluate.py
│   └── make_figure.py
├── results/
│   ├── metrics.csv
│   ├── metrics_summary.csv
│   └── figures/
└── checkpoints/
```
