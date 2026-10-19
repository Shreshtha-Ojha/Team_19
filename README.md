# Low-Light Image Enhancement with Classical Image Processing

College project for Image Processing & Computer Vision. Takes a dark photo
and produces an enhanced one using only classical techniques — no deep
learning, no training. Every step is a standard, explainable operation:
Retinex theory, point transforms, spatial filtering, histogram processing
and colour-space conversions.

It comes with a **web demo**: take or upload a photo on your phone and watch
it get enhanced step by step, ending in a before/after slider.

![Comparison on the LOL test set](results/figures/comparison.png)

## The pipeline

`src/pipeline/stages.py` — each stage is a function `stage(img, info, params) -> img`.

| # | Stage | What it does | Syllabus topic |
|---|---|---|---|
| 1 | Illumination map | Brightest of B, G, R per pixel, smoothed with a guided filter (edge-preserving) | Retinex, spatial filtering |
| 2 | Adaptive brightening | Gamma on the illumination map; gamma found by bisection so mean brightness hits a target. Same gain on B, G, R so colours don't shift | Point operations |
| 3 | Denoising | Non-local means, strength from an estimated noise level (Immerkær), applied mostly where the photo was dark | Noise, spatial filtering |
| 4 | Local contrast | Gentle CLAHE on the LAB L channel | Histogram processing |
| 5 | Colour correction | Partial gray-world white balance + HSV saturation | Colour spaces |
| 6 | Sharpening *(off by default)* | Unsharp masking | Spatial filtering |

Plus **homomorphic filtering** (`src/classical/homomorphic.py`), a
frequency-domain alternative, as a comparison method.

### Parameter tuning

All parameters were chosen by a small grid search (`src/tune_pipeline.py`)
on 100 images from the LOL training set (`our485`), stage by stage in
pipeline order, maximising PSNR + 10·SSIM. The 15 test images were never
used for tuning. Full log: `results/tune.log`; chosen values:
`results/tuned_params.json`.

What tuning found:
- **Denoising is the most valuable stage after brightening** (SSIM 0.49 → 0.74 on the training images).
- **Colour correction helps** (+0.7 dB, mostly from white balance).
- **CLAHE and saturation help only at their gentlest settings**, and **every sharpening amount lowered both scores**, so sharpening is off by default. LOL's reference images are soft and muted, so these steps move the output away from the reference even when it looks punchier.

## Results

Mean over the 15 LOL test images (`eval15`), from `results/metrics_summary.csv`:

| Method | PSNR (dB) | SSIM |
|---|---|---|
| CLAHE alone | 9.13 | 0.357 |
| Gamma (γ = 2.2) | 13.23 | 0.590 |
| Histogram equalization | 14.26 | 0.425 |
| Multi-scale Retinex | 15.50 | 0.591 |
| Single-scale Retinex | 15.57 | 0.597 |
| Homomorphic filtering | 16.15 | 0.383 |
| **Pipeline, stages 1–2** (illumination + brighten) | 15.69 | 0.485 |
| **+ denoise** | 16.45 | **0.729** |
| **+ contrast** | 16.43 | 0.718 |
| **+ colour — default pipeline** | **16.47** | 0.716 |
| + sharpen (amount 0.4) | 16.23 | 0.691 |

The full pipeline has the best PSNR of all methods and a much better SSIM
than any single operation. On the test set, contrast and colour change the
scores very little (they helped slightly on the 100 training images: 16.37 →
17.19 dB), but they visibly remove colour casts — compare the "pipeline"
and "ssr"/"homomorphic" columns in the figure.

Speed: ~0.5 s for a 600×400 LOL image, ~1.7 s for a 12 MP phone photo
(downscaled to 1600 px) on a laptop CPU (i7-1255U).

## Usage

### Setup

Python 3.12. System Python is PEP 668-managed, so use a venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All commands below are run from `src/`.

### Enhance a photo

```bash
python enhance_image.py dark.jpg -o enhanced.jpg
python enhance_image.py dark.jpg -o enhanced.jpg --steps-dir steps/   # save every stage
python enhance_image.py dark.jpg -o enhanced.jpg --skip colour --sharpen 0.4
```

### Web demo

```bash
uvicorn web.server:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. To use it from a phone, connect the phone to the
same Wi-Fi and open `http://<laptop-ip>:8000` (find the IP with `ip addr` /
`ipconfig`; on WSL you may need to forward the port from Windows).

To host it publicly, the included `Dockerfile` runs as-is as a
[Hugging Face Docker Space](https://huggingface.co/docs/hub/spaces-sdks-docker) (port 7860).

### Dataset — LOL

Used only for measuring quality and tuning parameters. See `data/README.md`
for download links, then check the layout with `python data/verify_lol.py`.

### Reproduce the numbers

```bash
D=../data/raw/LOL/lol_dataset
python tune_pipeline.py --low-dir $D/our485/low --high-dir $D/our485/high   # ~20 min
python evaluate.py --low-dir $D/eval15/low --high-dir $D/eval15/high --out ../results/metrics.csv
python make_figure.py --low-dir $D/eval15/low --high-dir $D/eval15/high --out ../results/figures/comparison.png
```

## Project structure

```
lowlight-enhancement/
├── Dockerfile               # hosting the web demo
├── requirements.txt
├── data/README.md           # dataset download instructions
├── src/
│   ├── pipeline/
│   │   ├── stages.py        # the six stages + tuned parameters
│   │   └── pipeline.py      # runs stages in order, keeps intermediate images
│   ├── classical/           # single-operation baselines + homomorphic filter
│   ├── data/                # LOL pairing + layout check
│   ├── web/
│   │   ├── server.py        # FastAPI backend
│   │   └── static/          # the webpage + sample photos
│   ├── enhance_image.py     # command-line tool
│   ├── tune_pipeline.py     # parameter grid search (training images only)
│   ├── evaluate.py          # PSNR/SSIM on the test set
│   └── make_figure.py
└── results/                 # metrics, tuning log, comparison figure
```
