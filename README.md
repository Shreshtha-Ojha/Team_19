# Low-Light Image Enhancement with Classical Image Processing

College project for Image Processing & Computer Vision. Takes a dark photo
and produces an enhanced one using only classical spatial-domain techniques —
no deep learning, no training data. Every step is a standard, explainable
operation (point transforms, histogram processing, spatial filtering, colour
space conversions, Retinex).

**Status:** single-operation baselines are implemented and evaluated. The
multi-stage enhancement pipeline and the demo webpage are in progress.

## Environment

Python 3.12. System Python is PEP 668-managed, so use a venv. Exact versions
are pinned in `requirements.txt` (generated via `pip freeze`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset — LOL

Used only for **measuring** quality (PSNR/SSIM against a reference image) and
for tuning pipeline parameters; nothing is trained. See `data/README.md` for
download links and where to put it, then verify the layout:

```bash
python src/data/verify_lol.py
```

## Single-operation baselines

Implemented in `src/classical/`:

| File | Method |
|---|---|
| `gamma.py` | Gamma correction (power-law point transform) |
| `hist_eq.py` | Global histogram equalization on the LAB L-channel |
| `clahe.py` | CLAHE on the LAB L-channel |
| `retinex.py` | Single- and multi-scale Retinex |
| `denoise.py` | Bilateral / median denoising |

### Evaluate (PSNR/SSIM on the LOL test set)

Run from `src/`:

```bash
cd src
python evaluate.py --low-dir ../data/raw/LOL/lol_dataset/eval15/low \
    --high-dir ../data/raw/LOL/lol_dataset/eval15/high --out ../results/metrics.csv
```

Writes per-image metrics to `results/metrics.csv` and per-method means to
`results/metrics_summary.csv`.

### Comparison figure

```bash
cd src
python make_figure.py --low-dir ../data/raw/LOL/lol_dataset/eval15/low \
    --high-dir ../data/raw/LOL/lol_dataset/eval15/high \
    --num-images 4 --out ../results/figures/comparison.png
```

## Results so far

Mean over the 15 LOL test images, from `results/metrics_summary.csv`:

| Method | PSNR (dB) | SSIM |
|---|---|---|
| CLAHE | 9.13 | 0.357 |
| Gamma (γ = 2.2) | 13.23 | 0.590 |
| Histogram equalization | 14.26 | 0.425 |
| Multi-scale Retinex | 15.50 | 0.591 |
| Single-scale Retinex | 15.57 | 0.597 |

No single operation is enough on its own. The next step is to combine them
into one pipeline: illumination estimation → adaptive brightening → denoising
→ local contrast → colour correction.

## Project structure

```
lowlight-enhancement/
├── requirements.txt
├── README.md
├── data/
│   ├── raw/LOL/            # place downloaded dataset here (see data/README.md)
│   └── README.md
├── src/
│   ├── classical/          # gamma, hist_eq, clahe, retinex, denoise
│   ├── data/
│   │   ├── lol_dataset.py  # pairs low/high images by filename
│   │   └── verify_lol.py   # discovers & verifies actual folder layout
│   ├── evaluate.py
│   └── make_figure.py
└── results/
    ├── metrics.csv
    ├── metrics_summary.csv
    └── figures/
```
