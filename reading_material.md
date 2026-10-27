# Reading Material: Low-Light Image Enhancement

This is the study guide for the whole repository. It explains **what** the
project does, **why** each piece exists, **how** every algorithm works (with
the maths and worked numbers), and **where** in the code each thing lives.

Read it top to bottom once. After that, use the table of contents to look
things up. Every claim here was checked against the code at commit `d8e0eea`.
The example numbers in sections 5 and 6 come from actually running the
pipeline on the three sample photos in `src/web/static/samples/`.

---

## Table of contents

1. [The project in one page](#1-the-project-in-one-page)
2. [Background you need first](#2-background-you-need-first)
3. [Repository map](#3-repository-map)
4. [Architecture: how the pieces fit together](#4-architecture-how-the-pieces-fit-together)
5. [The enhancement pipeline, stage by stage](#5-the-enhancement-pipeline-stage-by-stage)
6. [The pipeline runner and the presets](#6-the-pipeline-runner-and-the-presets)
7. [Zoom, motion deblur and text view](#7-zoom-motion-deblur-and-text-view)
8. [The single-step "classical" baselines](#8-the-single-step-classical-baselines)
9. [The web app (backend and frontend)](#9-the-web-app-backend-and-frontend)
10. [The command-line tool](#10-the-command-line-tool)
11. [Measuring quality: dataset, PSNR, SSIM, evaluation](#11-measuring-quality-dataset-psnr-ssim-evaluation)
12. [How the settings were chosen (tuning)](#12-how-the-settings-were-chosen-tuning)
13. [Reading the results](#13-reading-the-results)
14. [The deblur experiment](#14-the-deblur-experiment)
15. [Hosting and deployment](#15-hosting-and-deployment)
16. [Project history](#16-project-history)
17. [Limits, quirks and loose ends](#17-limits-quirks-and-loose-ends)
18. [How to change or extend things](#18-how-to-change-or-extend-things)
19. [Questions you will probably be asked (with answers)](#19-questions-you-will-probably-be-asked-with-answers)
20. [Glossary](#20-glossary)
21. [Suggested study plan](#21-suggested-study-plan)

---

## 1. The project in one page

**Problem.** Photos taken in poor light are dark, grainy (noisy), have weak
contrast and often a colour cast (yellow from indoor bulbs, orange from
street lamps). We want to turn such a photo into one that looks clear and
natural.

**Constraint.** The project uses **classical image processing only**. That
means no neural networks, no training and no learned weights. Every step is
a formula you could write on a whiteboard: gamma curves, filters,
histograms, colour-space conversions and Fourier transforms. (An earlier
version had a small U-Net neural network; it was removed on purpose; see
[section 16](#16-project-history).)

**What it delivers.**

1. **A six-stage enhancement pipeline** (`src/pipeline/`):
   illumination map → brighten → denoise → local contrast → colour
   correction → sharpen.
2. **A "zoom into a detail" tool** (`src/pipeline/zoom.py`) for small regions
   such as number plates. It can undo straight-line motion blur with
   **Wiener deconvolution**, enlarge the region, sharpen it, and show a
   black-and-white "text view".
3. **A web app** (`src/web/`): FastAPI backend plus a single HTML page. You
   upload a photo or take one with your phone. The page shows every stage
   one after another, gives you a before/after slider, and has the zoom
   tool.
4. **A command-line tool** (`src/enhance_image.py`).
5. **Measurement scripts.** These score the pipeline against six
   single-step classical methods on the public **LOL dataset**, using
   **PSNR** and **SSIM**. A tuning script picked the pipeline's settings
   with a grid search.
6. **Free hosting** on Hugging Face Spaces, using a `Dockerfile` and a
   deploy script.

**Headline result** (15 LOL test images, never used for tuning):

| | PSNR (dB) | SSIM |
|---|---|---|
| Best single method (by PSNR): homomorphic filtering | 16.15 | 0.383 |
| Best single method (by SSIM): single-scale Retinex | 15.57 | 0.597 |
| **Full pipeline** | **16.47** | **0.716** |

---

## 2. Background you need first

If these ideas are already familiar, skip to section 3. Everything later
builds on them.

### 2.1 How an image is stored

- A colour image is a 3-D array of shape `(height, width, 3)`.
- **OpenCV stores channels as B, G, R** (blue, green, red), not RGB. All
  code in this repo works in BGR. It only converts to RGB where another
  library expects it: matplotlib in `make_figure.py` and Pillow in
  `server.py`.
- **Two number formats are used:**
  - `uint8`: whole numbers 0–255. This is how files are loaded and saved.
  - `float32` in `[0, 1]`: used inside the pipeline so the maths does not
    overflow or round at every step.
- The converters live in `src/pipeline/stages.py`:
  ```python
  def to_float(img_u8):  return img_u8.astype(np.float32) / 255.0
  def to_u8(img_f):      return np.clip(img_f * 255.0 + 0.5, 0, 255).astype(np.uint8)
  ```
  The `+ 0.5` makes the conversion round to the nearest integer. Without
  it, `astype` would truncate (0.999 → 0), and the image would get slightly
  darker every time it went float → uint8.

### 2.2 Point operations vs. neighbourhood operations

- **Point operation:** each output pixel depends only on the same input
  pixel. Gamma correction is an example.
- **Spatial (neighbourhood) filter:** each output pixel depends on a window
  of pixels around it. Blurring, guided filtering and sharpening are
  examples.
- **Frequency-domain operation:** transform the image with the FFT
  (Fast Fourier Transform), multiply by a filter, and transform back.
  Homomorphic filtering and Wiener deconvolution work this way.

### 2.3 Gamma correction

Gamma correction applies `out = in^γ` to values in `[0, 1]`.

- If γ < 1, dark values are pushed up a lot and bright values only a
  little. The image gets **brighter**.
- If γ > 1, the image gets darker.
- If γ = 1, nothing changes.

Example with γ = 0.5: 0.04 → 0.2 (5× brighter), but 0.81 → 0.9 (only 1.1×).

### 2.4 Histograms and equalization

A **histogram** counts how many pixels have each brightness value. A dark
photo has its histogram squeezed into the low end.

**Histogram equalization** remaps brightness values so the histogram
spreads over the whole 0–255 range. The mapping is the **cumulative
distribution function (CDF)** of the histogram.

**CLAHE** (Contrast-Limited Adaptive Histogram Equalization) improves on
this in two ways:

- It equalizes **small tiles** separately, so contrast is improved locally.
- It **clips** each tile's histogram before equalizing, so noise in flat
  areas is not blown up.

### 2.5 Colour spaces used here

| Space | Channels | Why it is used here |
|---|---|---|
| **BGR** | blue, green, red | OpenCV's native format |
| **LAB** | L = lightness, A = green↔red, B = blue↔yellow | Contrast can be changed on L without touching colour (CLAHE, histogram equalization, zoom) |
| **HSV** | hue, saturation, value | Saturation can be scaled directly (colour stage) |
| **YCrCb** | Y = luma, Cr/Cb = colour differences | Brightness channel for homomorphic filtering |
| **Gray** | one channel | Noise estimation, brightness measurement, text view |

### 2.6 The Retinex idea (the core idea of the project)

Retinex theory (Edwin Land, 1970s) models an image as:

```
I(x, y) = R(x, y) × L(x, y)
```

- `R` is **reflectance**: what the objects actually look like (their colour
  and texture). It does not depend on lighting.
- `L` is **illumination**: how much light fell on each point. It usually
  changes slowly across the scene.

A dark photo has a good `R` and a tiny `L`. If we can estimate `L`, we can
**change the lighting without changing the objects**. Stages 1 and 2 of the
pipeline do exactly this.

### 2.7 Noise

Camera sensors add random noise. In a dark photo the signal is weak, so
noise is a large fraction of it. **Brightening multiplies the noise by the
same factor as the signal.** That is why every brightened photo needs
denoising.

### 2.8 Convolution and the Fourier transform

- Blurring an image is a **convolution** with a small kernel, written
  `g = f * h`.
- The **convolution theorem** says convolution in space equals
  multiplication in frequency: `G = F · H`.
- So in principle, undoing a blur is just division: `F = G / H`. In
  practice plain division explodes the noise wherever `H` is close to zero.
  **Wiener deconvolution** fixes this (see [section 7.2](#72-motion-deblur-wiener-deconvolution)).

---

## 3. Repository map

```
lowlight-enhancement/
├── README.md                    user-facing docs: run, host, results
├── reading_material.md          this file
├── requirements.txt             runtime deps (app + CLI)
├── requirements-dev.txt         + scikit-image, pandas, matplotlib for measurement scripts
├── Dockerfile                   container recipe for Hugging Face Spaces
├── .dockerignore / .gitignore
├── scripts/deploy_hf.sh         pushes the app to a Hugging Face Space
├── data/README.md               how to download the LOL dataset (optional)
├── results/
│   ├── metrics.csv              per-image PSNR/SSIM, 11 methods × 15 images
│   ├── metrics_summary.csv      mean per method
│   ├── tune.log                 full log of the settings search
│   ├── tuned_params.json        the chosen settings
│   └── figures/
│       ├── comparison.png       visual comparison grid on LOL images
│       └── deblur_demo.png      synthetic number-plate deblur figure
└── src/                         ← all commands are run from here
    ├── pipeline/
    │   ├── stages.py            the 6 stages + PipelineParams + guided filter + noise estimator
    │   ├── pipeline.py          STAGES list, presets, enhance()
    │   └── zoom.py              motion PSF, Wiener deblur, enlarge, text view, zoom_region()
    ├── classical/               single-step methods (baselines; CLAHE is also reused by stage 4)
    │   ├── gamma.py  hist_eq.py  clahe.py  retinex.py  homomorphic.py
    │   └── denoise.py           bilateral/median helpers (not used anywhere now)
    ├── data/
    │   ├── lol_dataset.py       pairs low/high images by filename
    │   └── verify_lol.py        checks the downloaded dataset layout
    ├── web/
    │   ├── server.py            FastAPI backend: /, /api/enhance, /api/zoom, /static
    │   └── static/
    │       ├── index.html       the entire frontend (HTML + CSS + JS, no framework)
    │       └── samples/1-3.jpg  sample dark photos (600×400)
    ├── enhance_image.py         CLI
    ├── evaluate.py              PSNR/SSIM of every method on a test set
    ├── make_figure.py           builds results/figures/comparison.png
    ├── tune_pipeline.py         grid search for PipelineParams
    └── deblur_demo.py           synthetic plate experiment for Wiener deblur
```

**Why you must run things from `src/`.** The code imports packages as
`from pipeline.stages import ...` and `from classical.clahe import ...`.
Those names only resolve when `src/` is the current directory, or is on
`PYTHONPATH`. The Dockerfile sets `WORKDIR /home/user/app/src` for the same
reason.

`classical/` and `data/` have no `__init__.py`. Python 3.3+ still imports
them as "namespace packages".

**Two different "data" folders:**

- `src/data/` is Python code (dataset helpers).
- `data/` at the repository root is where the downloaded dataset goes
  (`data/raw/`, ignored by git).

---

## 4. Architecture: how the pieces fit together

### 4.1 The layers

```
                ┌───────────────────────────────────────────────┐
  Users  ──────▶│  Browser: src/web/static/index.html           │
                │  (upload / camera / samples, step cards,      │
                │   before-after slider, zoom box, deblur UI)   │
                └──────────────┬────────────────────────────────┘
                               │ HTTP (multipart form / JSON)
                ┌──────────────▼────────────────────────────────┐
                │  Backend: src/web/server.py (FastAPI)         │
                │  decode + downscale, in-memory photo cache,   │
                │  presets, JSON with base64 JPEGs              │
                └──────┬───────────────────────┬────────────────┘
                       │                       │
         ┌─────────────▼──────────┐  ┌─────────▼────────────────┐
  CLI ──▶│ pipeline/pipeline.py   │  │ pipeline/zoom.py         │
         │ enhance(), PRESETS     │  │ zoom_region(), Wiener    │
         └─────────────┬──────────┘  └─────────┬────────────────┘
                       │                       │ (uses estimate_noise,
         ┌─────────────▼──────────┐            │  to_float, to_u8)
         │ pipeline/stages.py     │◀───────────┘
         │ 6 stages, params,      │
         │ guided filter, noise   │──▶ classical/clahe.py
         └────────────────────────┘

  Offline / research side (needs requirements-dev.txt + LOL dataset):
    tune_pipeline.py ──▶ results/tuned_params.json, tune.log  ──▶ (copied by hand into PipelineParams defaults)
    evaluate.py      ──▶ results/metrics.csv, metrics_summary.csv
    make_figure.py   ──▶ results/figures/comparison.png
    deblur_demo.py   ──▶ results/figures/deblur_demo.png
```

### 4.2 Design principles in the code

1. **One stage = one function with one signature.** Every stage is
   `stage(img, info, params) -> img`. Because of this, the runner,
   the tuner, the server and the CLI can all treat stages the same way.
2. **All tunable numbers live in one dataclass,** `PipelineParams`. The
   presets, the tuner and the CLI make modified copies with
   `dataclasses.replace(...)`. They never edit globals.
3. **Stages talk to each other through the `info` dict.** Stage 1 writes
   `info["illumination"]`. Stages 2 and 3 read it. Stage 2 writes
   `info["gamma"]` and stage 3 writes `info["noise_sigma"]`, so the CLI and
   the tests can report them.
4. **Every intermediate image is kept** (`keep_steps=True`). This is what
   lets the web page show the step-by-step story.
5. **No training, so no hidden state.** The only "learned" thing is the
   choice of a few numbers from a grid search. Those numbers are written
   in plain sight as dataclass defaults.

### 4.3 The data flow of one enhancement

```
uint8 BGR photo
   │ to_float
   ▼
float32 BGR in [0,1] ─▶ Stage 1 estimate_illumination ─▶ info["illumination"] = t (H×W map)
   │                         (image unchanged)
   ▼
Stage 2 brighten          img ← img × t^(γ−1)        (γ found by bisection; info["gamma"])
   ▼
Stage 3 denoise           img ← w·NLM(img) + (1−w)·img,  w = (1−t)^p   (info["noise_sigma"])
   ▼
Stage 4 local_contrast    CLAHE on LAB's L channel
   ▼
Stage 5 colour_correct    gray-world gains (partial, capped) then HSV saturation × s
   ▼
Stage 6 sharpen           img + a·(img − GaussianBlur(img))
   │ to_u8
   ▼
uint8 BGR result  (+ list of (stage name, image) for display)
```

---

## 5. The enhancement pipeline, stage by stage

File: `src/pipeline/stages.py`. Each stage below covers its purpose, the
idea behind it, the code, the maths, the parameters, and why it sits where
it does in the order.

### 5.0 `PipelineParams`: every setting in one place

```python
@dataclass
class PipelineParams:
    guide_radius: int = 30          # stage 1
    guide_eps: float = 1e-3         # stage 1
    target_brightness: float = 0.40 # stage 2
    illum_floor: float = 0.02       # stage 1/2
    denoise_strength: float = 1.5   # stage 3
    denoise_dark_power: float = 1.0 # stage 3
    clahe_clip: float = 0.75        # stage 4
    clahe_tiles: int = 8            # stage 4
    white_balance_strength: float = 0.5  # stage 5
    wb_max_gain: float | None = None     # stage 5
    saturation: float = 0.9         # stage 5
    sharpen_amount: float = 0.4     # stage 6
    sharpen_sigma: float = 1.5      # stage 6
```

These defaults are the values the tuning script picked on 100 LOL images
(`results/tuned_params.json`). Together with "sharpen off", they make up
the **`benchmark`** preset.

### 5.0.1 Why the stages are in this order

The order is a deliberate design decision, and you should be able to
defend it:

- **Brighten first.** Contrast, colour and denoising all behave badly on
  near-black pixels. Colour ratios of tiny numbers are unstable, and CLAHE
  on a dark histogram amplifies noise. So the light is fixed first.
- **Denoise before contrast and colour.** CLAHE and saturation boosts would
  also amplify the grain. Removing it first keeps it from growing.
- **Sharpen last.** Sharpening boosts high frequencies, and noise is high
  frequency. Sharpening before denoising would sharpen the noise and then
  force the denoiser to fight it.

---

### 5.1 Stage 1: illumination map (`estimate_illumination`)

**Purpose:** estimate `L`, "how much light fell here", at every pixel. This
stage **does not change the image.** It only stores a map in
`info["illumination"]`.

**Step 1: initial estimate, the brightest channel.**

```python
t0 = img.max(axis=2)        # for each pixel: max(B, G, R)
```

*Why the max?* A surface cannot reflect more light than falls on it.
Reflectance is between 0 and 1 for each channel, so every channel value is
at most the illumination. The largest of the three is therefore the
tightest lower bound on `L` that we can read straight off the pixel. A
white wall gives max ≈ L. A red apple gives R ≈ L. Using the max (instead
of, say, the grey average) avoids treating saturated colours as "dark".

This is the same initial estimate used by the well-known LIME method
(Guo et al., 2017, *Low-light Image Enhancement via Illumination Map
Estimation*). Stages 1 and 2 of this pipeline closely follow LIME's
structure: max-RGB illumination, a refinement step, then gamma on the
illumination.

**Step 2: refine with a guided filter.**

```python
t = guided_filter(t0, t0, _scaled_radius(p.guide_radius, img), p.guide_eps)
info["illumination"] = np.clip(t, p.illum_floor, 1.0)
```

*Why smooth at all?* Raw `t0` contains all the texture of the scene: text
on a book spine, wood grain. If we divided by that texture, we would
divide the texture away and the result would look flat and strange.
Illumination should change slowly.

*Why not a plain Gaussian blur?* Blur smears the map across object edges.
Near the edge of a bright lamp against a dark wall, the blurred map would
say "medium light" on both sides. The wall next to the lamp would get too
little boost, and the lamp's edge too much. The result is a visible
**halo**. The guided filter smooths **inside** regions but keeps
**edges** sharp.

**How the guided filter works** (He, Sun, Tang 2010; `guided_filter()` in
`stages.py`). In every square window of side `2r+1`, it assumes the output
is a linear function of the guide image `I`:

```
q = a · I + b        (a, b constant inside the window)
```

Solving least squares with a regularizer ε gives:

```
a = cov(I, p) / (var(I) + ε)
b = mean(p) − a · mean(I)
```

Here `p` is the image being filtered. Each pixel lies in many windows, so
the final output averages `a` and `b` over all of them:
`q = mean(a)·I + mean(b)`.

In this code the guide and the input are the **same image** (`t0, t0`),
so `cov(I,p) = var(I)` and:

```
a = var / (var + ε)
```

- **Flat window** (variance much smaller than ε): `a ≈ 0`, so `q ≈ mean`.
  The window is smoothed.
- **Window across an edge** (variance much larger than ε): `a ≈ 1`, so
  `q ≈ I`. The edge is kept.

With `ε = 1e-3`, the switch-over happens at a standard deviation of about
√0.001 ≈ 0.03, roughly 8 grey levels out of 255. Differences smaller than
that count as "texture to smooth". Larger ones count as "edge to keep".

**Speed.** Every mean is computed with `cv2.boxFilter`. A box filter costs
the same per pixel whatever the radius, because it uses running sums. So a
61×61 window is as cheap as a 3×3 one.

**`_scaled_radius`.** The radius was tuned on LOL images that are 600 px
wide:

```python
max(1, round(r * max(img.shape[:2]) / 600))
```

At radius 30, a 600 px image uses a 61×61 window. A 1600 px web upload
uses radius 80 (161×161), so the smoothing covers the same *fraction* of
the scene whatever the resolution.

**`illum_floor = 0.02`.** Pitch-black pixels would have `t ≈ 0`. Stage 2
divides by a power of `t`, so a floor stops the gain from becoming
enormous.

**What the web page shows for this stage.** `pipeline.illumination_heatmap`
colours the map with OpenCV's INFERNO colormap: dark purple means no
light, yellow means well lit. This is the only stage whose displayed image
is not the image passed to the next stage.

---

### 5.2 Stage 2: adaptive brightening (`brighten`)

**Purpose:** relight the scene. Dark regions get a large boost. Bright
regions get almost none.

**The core formula** (`_apply_gamma`):

```python
gain = t ** (gamma - 1.0)
out  = clip(img * gain, 0, 1)        # same gain on B, G and R
```

*Derivation.* By Retinex, `img ≈ R · t`. We want to keep `R` and replace
the illumination `t` by a brighter `t^γ` (with γ < 1):

```
out = R · t^γ = (img / t) · t^γ = img · t^(γ−1)
```

So gamma correction is applied to the **illumination map**, not to the
image.

**Worked numbers.** Sample photo 1 picked γ = 0.267, so the gain is
`t^(−0.733)`:

| Illumination `t` | Gain | Meaning |
|---|---|---|
| 0.02 (floor) | ≈ 17.6× | darkest corners, capped by the floor |
| 0.05 | ≈ 9.0× | deep shadow |
| 0.20 | ≈ 3.3× | dim area |
| 0.50 | ≈ 1.7× | mid-lit |
| 0.90 | ≈ 1.08× | near a lamp: almost untouched, so it doesn't blow out |

**Why this beats plain gamma on the image** (the `gamma` baseline):

1. **Hue is preserved.** Plain gamma bends each of B, G, R separately along
   a curve, which changes their ratios: colours wash out and shift. Here
   all three channels at a pixel are multiplied by **the same** number, so
   the ratios B:G:R stay the same. (Exception: a channel that would go
   above 1 gets clipped.)
2. **Local detail is preserved.** The gain comes from the *smoothed* map,
   so neighbouring pixels in a region get almost the same gain. The texture
   inside the region (the `R` part) keeps its relative contrast instead of
   being squashed by a curve.
3. **γ is chosen automatically for each photo.** See below.

**Choosing γ automatically, by bisection.** The goal is that the result's
mean grey level equals `target_brightness` (0.40 by default).

```python
small   = resize(img, 0.25×)       # speed: search on a quarter-size copy
t_small = resize(t,   same size)

def mean_after(g): return gray(_apply_gamma(small, t_small, g)).mean()

lo, hi = 0.05, 1.0
if mean_after(1.0) >= target:      # γ=1 means "unchanged": already bright enough
    gamma = 1.0
else:
    for _ in range(20):
        mid = (lo + hi) / 2
        if mean_after(mid) > target: lo = mid    # too bright → γ can go higher
        else:                        hi = mid    # too dark   → γ must go lower
    gamma = (lo + hi) / 2
info["gamma"] = gamma
```

- Bisection works because `mean_after(γ)` is **monotonic**: a smaller γ
  always means a brighter image.
- 20 halvings shrink the interval from 0.95 to about 10⁻⁶, which is far
  more precise than needed.
- A darker photo automatically gets a smaller γ, meaning a stronger boost.
  The sample photos came out at γ = 0.267, 0.313 and 0.322 (their input
  means were 0.098, 0.077 and 0.095).
- γ is never below 0.05, so the gain is at most `0.02^(−0.95)` ≈ 41×.

**Note.** The target is measured right after this stage. Later stages
(CLAHE especially) change the mean a little. On the samples, the final
outputs averaged 0.44–0.48, not exactly 0.40.

---

### 5.3 Stage 3: denoising (`denoise`, `estimate_noise`)

**Purpose:** remove the grain that brightening amplified, mostly in the
areas that were dark.

**Step 1: measure the noise in this image** (Immerkær 1996, *Fast Noise
Variance Estimation*):

```python
mask = [[ 1, -2,  1],
        [-2,  4, -2],
        [ 1, -2,  1]]
resp  = filter2D(gray, mask)[1:-1, 1:-1]          # drop the 1-px border
sigma = sqrt(pi/2) * mean(|resp|) / 6
```

- The mask is the difference of two Laplacian kernels. Smooth image
  content (flat areas, gentle gradients) largely cancels out, so what is
  left is mostly pixel-to-pixel noise.
- For Gaussian noise with standard deviation σ passed through this mask,
  the output's standard deviation is 6σ, because √(1+4+1+4+16+4+1+4+1) =
  √36 = 6. The mean absolute value of a Gaussian is σ·√(2/π). Dividing
  `mean|resp|` by 6 and multiplying by √(π/2) recovers σ.
- The estimate is made on the **already brightened** image, which is the
  noise we actually need to remove. On the samples, σ was 0.008–0.014,
  that is, 2–3.6 grey levels.

**Step 2: non-local means (NLM).**

```python
h   = max(1.0, sigma * 255 * p.denoise_strength)
den = cv2.fastNlMeansDenoisingColored(to_u8(img), None, h, h, 7, 21)
```

- *How NLM works.* For each pixel, compare the 7×7 patch around it with
  every 7×7 patch inside a 21×21 search area. Average the centre pixels of
  those patches, weighting each by how similar its patch is. Similar
  patches (the same texture somewhere nearby) count a lot. Different ones
  count very little.
- Edges and repeated textures survive, because only patches that look
  alike get averaged together. A Gaussian blur would mix everything.
- `h` is the filter strength in grey levels. A larger `h` means more
  patches are treated as "similar", giving stronger smoothing. Tying `h`
  to the measured σ makes denoising **adaptive**: a clean photo gets light
  denoising and a grainy one gets strong denoising.
  `denoise_strength` is a multiplier on top of that (benchmark 1.5,
  natural 1.0, detail 0.5).

**Step 3: blend by darkness.**

```python
w   = (1 - t) ** p.denoise_dark_power      # t = illumination map from stage 1
out = w * den + (1 - w) * img
```

Areas that were dark had their noise multiplied by a big gain, so they get
mostly the denoised version. Areas that were bright stay mostly original,
which keeps their real detail sharp. With `denoise_dark_power = 0`, `w = 1`
everywhere, giving uniform denoising. The tuner tried 0, 0.5 and 1.0. 1.0
won on every strength level (`results/tune.log`, lines 34–48).

**Why this stage matters.** It gives the single biggest improvement in the
evaluation: SSIM rises from 0.485 to 0.729. SSIM penalises noise heavily
(see [section 13](#13-reading-the-results)).

---

### 5.4 Stage 4: local contrast (`local_contrast`)

```python
return to_float(clahe_enhance(to_u8(img), p.clahe_clip, p.clahe_tiles))
```

`classical/clahe.py` converts BGR → LAB, runs OpenCV's CLAHE on **L only**,
merges the channels back, and converts LAB → BGR.

**Why it is needed.** Brightening with a smooth gain lifts the shadows
but also *compresses* the overall tonal range, so the image can look flat
and hazy. CLAHE restores contrast tile by tile.

**How CLAHE works:**

1. Split the L channel into `clahe_tiles × clahe_tiles` tiles (8×8 = 64
   tiles).
2. In each tile, compute the histogram. **Clip** every bin at a limit and
   spread the clipped excess evenly over all bins.
3. Equalize each tile using the CDF of its clipped histogram.
4. Blend between neighbouring tiles' mappings with bilinear interpolation,
   so no tile boundaries are visible.

**What the clip limit means.** OpenCV's `clipLimit` is relative to the
average bin height. The lower it is, the more of the histogram gets
flattened. A flat histogram has a straight-line CDF, and a straight-line
CDF maps every value to itself. So a **small clip limit means a mapping
close to "do nothing"**, and a large one means strong, aggressive
equalization. The benchmark value 0.75 is very gentle. The tuner tried
0.75–2.0, and the score dropped steadily as the value rose (`tune.log`,
lines 51–55), because LOL's reference photos are fairly soft. The
real-photo presets use 1.5 (natural) and 2.5 (detail).

**Why only L.** Equalizing B, G and R separately changes their ratios, so
colours shift. LAB separates lightness from colour, so contrast can be
changed without touching hue.

---

### 5.5 Stage 5: colour correction (`colour_correct`)

**Part A: gray-world white balance.**

```python
means = img.reshape(-1, 3).mean(axis=0)            # mean B, mean G, mean R
gains = means.mean() / max(means, 1e-6)            # push each mean to the common average
gains = 1 + p.white_balance_strength * (gains - 1) # partial strength
if p.wb_max_gain:
    gains = clip(gains, 1/p.wb_max_gain, p.wb_max_gain)
out = clip(img * gains, 0, 1)
```

- *Assumption:* across a whole typical scene, the colours average out to
  neutral grey. If the average is yellowish (bulb light), the light is
  yellow, and we cancel it by scaling the channels until their means are
  equal.
- *Worked example:* means B=0.30, G=0.40, R=0.50 (a warm cast). The
  average is 0.40, so the full gains are B×1.333, G×1.0, R×0.8. At strength
  0.5 they become B×1.167, G×1.0, R×0.9. With `wb_max_gain = 1.2` they stay
  inside [0.833, 1.2].
- *Why partial and capped?* The assumption fails when a scene really is
  mostly one colour: a sunset, a red wall, a forest. Full gray-world would
  turn a sunset grey. Strength 0.5 and a ×1.2 cap (in the real-photo
  presets) limit the damage. The benchmark preset uses no cap, because
  LOL's indoor scenes rarely trigger the problem.

**Part B: saturation.**

```python
hsv = cvtColor(out, BGR2HSV)       # for float input OpenCV uses H∈[0,360], S,V∈[0,1]
hsv[..., 1] = clip(hsv[..., 1] * p.saturation, 0, 1)
return cvtColor(hsv, HSV2BGR)
```

The benchmark uses 0.9, slightly *less* saturated. The tuner chose it
because LOL's reference photos are muted, so a less saturated output
matches them better. Natural uses 1.1 (a slight boost) and detail uses 1.0.

---

### 5.6 Stage 6: sharpening (`sharpen`), via unsharp masking

```python
blurred = GaussianBlur(img, sigma = p.sharpen_sigma)
return clip(img + p.sharpen_amount * (img - blurred), 0, 1)
```

- `img − blurred` is the high-frequency part of the image: edges and fine
  detail. Adding a fraction of it back makes edges crisper.
- `sharpen_sigma` sets what counts as "detail". A smaller sigma means
  finer detail. `sharpen_amount` sets how much is added back.
- The name "unsharp masking" comes from film darkrooms, where a blurred
  ("unsharp") copy was used as a mask.

**It is off in the benchmark preset.** On LOL, every amount tried lowered
both PSNR and SSIM (`tune.log`, lines 80–83). The reference photos are
soft, and sharpening adds overshoot and amplifies leftover noise, neither
of which the reference contains. Real photos *look* better with some
sharpening, so `natural` (0.3) and `detail` (0.8, sigma 1.0) turn it on.
This is a nice example of a metric and human judgement disagreeing.

---

## 6. The pipeline runner and the presets

File: `src/pipeline/pipeline.py`.

### 6.1 The stage list

```python
STAGES = [
    ("illumination", stages.estimate_illumination, "Finding where the light is"),
    ("brighten",     stages.brighten,              "Lifting the shadows"),
    ("denoise",      stages.denoise,               "Cleaning up grain"),
    ("contrast",     stages.local_contrast,        "Bringing back depth"),
    ("colour",       stages.colour_correct,        "Restoring natural colours"),
    ("sharpen",      stages.sharpen,               "Crisper details"),
]
REQUIRED        = {"illumination", "brighten"}       # can never be switched off
DEFAULT_ENABLED = all stages except "sharpen"
```

The third item in each tuple is the step title shown on the web page.

### 6.2 `enhance()`

```python
def enhance(img_u8, params=None, enabled=None, keep_steps=True) -> Result:
```

1. `params` defaults to `PipelineParams()`, the benchmark values.
2. `enabled` defaults to `DEFAULT_ENABLED`. If you pass your own set,
   `REQUIRED` is **always added** to it. You cannot switch off stages 1 and
   2, because without them there is no enhancement at all. Unknown stage
   names raise `ValueError`.
3. Converts to float, then loops over `STAGES` in order, skipping disabled
   ones: `img = fn(img, info, params)`.
4. If `keep_steps` is true, it saves `(name, uint8 image)` after each
   stage. For "illumination" it saves the heatmap instead of the
   (unchanged) image.
5. Returns `Result(final, steps, info)`. `info` contains `illumination`,
   `gamma` and (if denoise ran) `noise_sigma`.

`keep_steps=False` is used by the evaluation, tuning and figure scripts.
They only need the final image, so it saves memory and JPEG work.

### 6.3 Presets

```python
PRESETS = {
  "benchmark": (PipelineParams(), DEFAULT_ENABLED),
  "natural":   (PipelineParams(denoise_strength=1.0, clahe_clip=1.5, saturation=1.1,
                               wb_max_gain=1.2, sharpen_amount=0.3), all stages),
  "detail":    (PipelineParams(denoise_strength=0.5, clahe_clip=2.5, saturation=1.0,
                               wb_max_gain=1.2, sharpen_amount=0.8, sharpen_sigma=1.0), all stages),
}
```

Full comparison (a blank cell means "same as the default"):

| Setting | benchmark | natural | detail | Effect |
|---|---|---|---|---|
| guide_radius / guide_eps | 30 / 1e-3 | | | smoothness of the light map |
| target_brightness | 0.40 | | | how bright the result is (the web slider changes this) |
| denoise_strength | 1.5 | 1.0 | 0.5 | less denoising keeps small text from being smeared |
| denoise_dark_power | 1.0 | | | |
| clahe_clip | 0.75 | 1.5 | 2.5 | local contrast punch |
| white_balance_strength | 0.5 | | | |
| wb_max_gain | none | 1.2 | 1.2 | protects strongly coloured scenes |
| saturation | 0.9 | 1.1 | 1.0 | |
| sharpen | off | 0.3, σ 1.5 | 0.8, σ 1.0 | |

**Who uses what:**

- **benchmark:** the numbers in the README, `evaluate.py`, `make_figure.py`,
  and any call to `enhance()` without arguments. It is tuned against a
  metric.
- **natural:** the CLI default and the web default. Set by eye for real
  photos.
- **detail:** "Max detail" on the web page, and `--preset detail` in the
  CLI. Meant for text and number plates.

The web app only offers natural and detail (`WEB_PRESETS` in `server.py`).

---

## 7. Zoom, motion deblur and text view

File: `src/pipeline/zoom.py`. The web endpoint `/api/zoom` uses it, and so
does `deblur_demo.py`.

**Goal:** the user drags a box around something small, such as a number
plate or a sign, and gets the most readable version possible.

### 7.1 Where the crop comes from

The server does **not** crop the final enhanced image. It recomputes
stages 1–2 only and crops the **brightened** image:

```python
f      = stages.estimate_illumination(to_float(img), info, params)
bright = to_u8(stages.brighten(f, info, params))
out    = zoom_region(bright[y0:y1, x0:x1], ...)
```

The reason: denoising in stage 3 smooths fine detail away, and a
character a few pixels wide could be erased. The zoom tool does its own
gentler denoising later, after deblurring. The picker on the page shows the
final image, but crop coordinates match because every stage keeps the
image size the same.

### 7.2 Motion deblur: Wiener deconvolution

**Modelling the blur.** If the camera or car moved in a straight line
during the exposure, every point of light is smeared into a short line.
That line is the **point-spread function (PSF)**:

```python
def motion_psf(length, angle_deg):
    size = int(ceil(length)) | 1        # "| 1" sets the lowest bit → always odd → line centred
    psf  = zeros((size, size))
    draw an anti-aliased line of `length` px through the centre at `angle_deg` (0 = horizontal)
    return psf / psf.sum()              # sums to 1 → blur doesn't change overall brightness
```

The blurred image is `g = f * h + noise`, where `f` is the sharp image and
`h` is the PSF.

**Undoing it.** In the frequency domain, `G = F·H + N`. Naive inverse
filtering, `F̂ = G / H`, fails. A motion-blur `H` has frequencies where it
is almost zero (the blur wiped them out completely), and dividing the
noise `N` by almost-zero gives enormous values.

**The Wiener filter** is the linear filter that minimises the mean squared
error when noise is present:

```
F̂ = conj(H) / (|H|² + NSR) · G
```

- Where `|H|` is large (frequencies the blur kept), this behaves like
  `1/H`: true inversion.
- Where `|H|` is tiny (frequencies the blur destroyed), the `NSR` term
  dominates and the output goes towards 0 instead of exploding.
- **NSR** is the noise-to-signal ratio. `NSR = 0` is plain inverse
  filtering. A larger NSR is safer and less noisy but deblurs less. It is
  the main trade-off knob.

**Implementation details** (`wiener_deblur`):

1. **Reflect-pad** the image by twice the PSF size. The FFT treats the
   image as if it repeated forever like wallpaper, so the left edge
   "touches" the right edge. Without padding, deblurring would create
   ringing stripes along the borders. Padding by mirror reflection makes
   the wrap-around seam smooth. The padding is cut off at the end.
2. Put the PSF in an array the size of the padded image, then `np.roll` it
   so the PSF's **centre is at pixel (0,0)**. The FFT treats (0,0) as the
   origin. Without the roll, the output would come out shifted by half the
   kernel size.
3. Compute `H = fft2(kernel)` and the Wiener filter once. Apply it to each
   colour channel: `ifft2(fft2(channel) · wiener)`. Take the real part and
   clip to [0,1].

**On the web page.** Three sliders: smear length (2–40 px), direction
(0–180°) and strength (0–1). Strength is converted to NSR on a log scale:

```js
const nsrOf = (s) => 0.05 * Math.pow(0.002 / 0.05, s);   // s=0 → 0.05 (gentle), s=1 → 0.002 (strong)
```

A log scale is used because NSR's effect is multiplicative: going from
0.05 to 0.01 matters about as much as going from 0.01 to 0.002. In the
backend, deblurring only runs if `deblur_length >= 2`. When the checkbox is
off, the page sends length 0.

### 7.3 The rest of `zoom_region`

```python
img = to_float(crop)
if deblur_length >= 2: img = wiener_deblur(img, motion_psf(len, angle), nsr)

h   = max(3.0, estimate_noise(img) * 255 * 1.5)                     # adaptive, at least 3
img = NLM(img, h, template 5, search 15)                            # denoise at ORIGINAL size
big = enlarge(img)                                                  # Lanczos, up to 8×, target 800 px

L channel of big ← CLAHE(clip 2.0, tiles 4×4)                       # strong local contrast
sharp = big + 0.8 · (big − GaussianBlur(big, σ=2))                  # strong unsharp mask

plain = nearest-neighbour resize of the original crop to the same size
return {"plain": plain, "enhanced": sharp, "text": text_view(sharp)}
```

Why each step is in this place:

- **Denoise before enlarging.** Deblurring amplifies noise. If we enlarged
  first, each noise speck would become a blob, and the sharpening would
  then sharpen it.
- **`enlarge`** (an interpolation topic): the scale is
  `min(8, 800 / long side)`. Crops already 800 px or larger are left
  alone. **Lanczos-4** interpolation uses a windowed-sinc kernel over 8×8
  neighbours. It is sharper than bilinear or bicubic, with a little
  ringing.
- **`plain`** uses **nearest-neighbour** on purpose. It shows the honest
  raw pixels as blocks, so you can see how much information was really
  there.
- **Stronger CLAHE and sharpening** than in the main pipeline, because the
  goal here is readability, not natural looks.

### 7.4 Text view: adaptive thresholding (a segmentation topic)

```python
gray  = GaussianBlur(gray(img), σ=1.5)                   # suppress speckle first
block = max(11, (min(h, w) // 4) | 1)                    # odd neighbourhood size
bw    = adaptiveThreshold(gray, 255, GAUSSIAN_C, THRESH_BINARY, block, C=10)
```

- A **global** threshold (one cut-off for the whole region) fails when the
  lighting is uneven. With headlight glare on one side, the "white" plate
  on the dark side can be darker than the "black" letters on the bright
  side.
- **Adaptive** thresholding compares each pixel with a Gaussian-weighted
  mean of its own neighbourhood. A pixel becomes black only if it is at
  least `C = 10` grey levels darker than its surroundings. So letters come
  out black, and faint noise in flat areas stays white.
- The block size grows with the region (a quarter of the shorter side), so
  a neighbourhood is bigger than a character's stroke.

### 7.5 What zoom cannot do

No step creates information that is not in the pixels. A plate 15 px wide
gets a cleaner outline but stays unreadable. Aim for plates at least
80–100 px wide. Deblurring with the wrong length or angle makes things
*worse* (see [section 14](#14-the-deblur-experiment)).

---

## 8. The single-step "classical" baselines

Folder: `src/classical/`. These are the standard textbook methods. The
pipeline is compared against them to show that combining steps
thoughtfully beats any single step. All of them take and return uint8 BGR.

| File | Function | Settings used in evaluation | What it does |
|---|---|---|---|
| `gamma.py` | `gamma_correct(img, 2.2)` | γ = 2.2 | Lookup table `out = 255·(in/255)^(1/2.2)` applied to every channel. Note the convention: the argument is inverted, so 2.2 *brightens*. |
| `hist_eq.py` | `hist_eq` | — | Global `equalizeHist` on LAB's L channel. |
| `clahe.py` | `clahe_enhance(img, 2.0, 8)` | clip 2.0, 8×8 tiles | CLAHE on L. **Also used inside stage 4.** |
| `retinex.py` | `single_scale_retinex(img, 80)` | σ = 80 | `log(I) − log(Gauss_σ * I)` per channel, then each channel min-max stretched to 0–255. |
| `retinex.py` | `multi_scale_retinex(img, (15, 80, 250))` | 3 scales | Average of SSR at 3 scales: small σ keeps detail, large σ keeps the overall tones. Then the same stretch. |
| `homomorphic.py` | `homomorphic_filter(img, 30, 0.5, 1.5)` | D0 = 30, γL = 0.5, γH = 1.5 | Frequency domain; see below. |
| `denoise.py` | `bilateral_denoise`, `median_denoise` | **not used anywhere** | Left over from the first version. |

### 8.1 Retinex (SSR/MSR) in a bit more detail

Retinex estimates illumination as a heavy Gaussian blur of the image, and
takes a log ratio: `log R = log I − log L`. The `+1` / `log1p` avoids
`log(0)`. Because each channel is stretched to its own min/max on its own,
colour balance can shift, and the output contrast depends on extreme
pixels. That is part of why SSR/MSR score moderately here. Notice the
contrast with the pipeline: the pipeline also uses the Retinex model, but
it estimates `L` with an edge-preserving filter (no halos) and **keeps**
the reflectance, only relighting the scene, instead of throwing the
illumination away.

### 8.2 Homomorphic filtering (frequency domain)

Same model, `I = L·R`, but it separates the two in frequency space:

1. Take Y (luma) from YCrCb, in float.
2. `log1p(Y)` turns the product into a sum: `log L + log R`.
3. FFT. Illumination changes slowly, so it is low frequency. Reflectance
   and detail are high frequency.
4. Multiply by a Gaussian **high-emphasis** filter:
   `H(D) = (γH − γL)·(1 − exp(−c·D²/D0²)) + γL`. At D = 0 (lowest
   frequencies) it is γL = 0.5, which compresses the illumination. At
   high D it tends towards γH = 1.5, which boosts detail. D0 sets where the
   change-over happens.
5. Inverse FFT, then `expm1` to undo the log.
6. Stretch to [0,1] ignoring the extreme 0.5% at each end.
7. Scale B, G, R by the same ratio `Y_new / Y`, which preserves hue (the
   same trick as stage 2).

It scores well on PSNR (16.15) but poorly on SSIM (0.383). It brightens
correctly on average, but boosting all high frequencies also boosts noise
and creates texture that SSIM penalises.

---

## 9. The web app (backend and frontend)

### 9.1 Backend: `src/web/server.py` (FastAPI)

**Start:** `cd src && uvicorn web.server:app --host 0.0.0.0 --port 8000`.
`--host 0.0.0.0` makes it reachable from other devices on the network,
such as your phone.

**Routes:**

| Method & path | Handler | Purpose |
|---|---|---|
| `GET /` | `index()` | Returns `static/index.html` |
| `GET /static/...` | `StaticFiles` mount | Sample images (and anything else in `static/`) |
| `POST /api/enhance` | `api_enhance` | Enhance a new upload, or re-run a cached one with new settings |
| `POST /api/zoom` | `api_zoom` | Zoom/deblur a rectangle of a cached photo |

**Constants:**

- `MAX_SIDE = 1600`: uploads are downscaled so their long side is at most
  1600 px. A 12 MP phone photo would take several seconds; 1600 px takes
  about 1–3 s.
- `MAX_UPLOAD = 25 MB`.
- `CACHE_SIZE = 16`.
- `WEB_PRESETS = ("natural", "detail")`.

**`decode_upload(bytes)`:**

1. Pillow opens the bytes. `ImageOps.exif_transpose` applies the EXIF
   orientation tag. Phones often save portrait photos sideways with a
   "rotate me" tag, and OpenCV's decoder would ignore it.
2. `.convert("RGB")` handles PNGs with transparency, greyscale images and
   so on.
3. `thumbnail((1600,1600), LANCZOS)` shrinks the image only if it is
   bigger, keeping the aspect ratio.
4. Convert RGB → BGR for OpenCV.
5. Anything Pillow can't open gives HTTP 400, "That file doesn't look like
   an image."

**The in-memory photo cache** (`_images`, an `OrderedDict`):

- `remember(img)` stores the decoded photo under a random `uuid4` hex key.
  If there are more than 16 entries, the oldest is evicted (LRU
  behaviour).
- `recall(key)` returns it and marks it most-recently used. If it is
  missing, the server answers **HTTP 410 Gone**: "This photo has expired
  on the server. Please upload it again."
- *Why have a cache?* When you move the brightness slider, switch preset or
  zoom, the browser sends only the `id`, not the photo again. That saves
  upload time, which matters on mobile data.
- *Limits of this design:* the cache lives in one process's memory. It is
  lost on restart, and it is **not shared between worker processes**.
  Running uvicorn with several workers would cause random 410s. Hugging
  Face runs a single process, so it works there.

**`POST /api/enhance`** takes form fields: `file` *or* `id`, `preset`
(default "natural") and optional `brightness`.

1. With a `file`: read at most 25 MB + 1 byte. If it is over the limit,
   return HTTP 413. Otherwise decode it and `remember` it to get a new
   `id`. With an `id` only: `recall`. With neither: HTTP 400.
2. `preset_params`: reject unknown presets. If `brightness` was sent,
   override `target_brightness`, clamped to [0.15, 0.7].
3. Run `enhance(img, params, enabled)` and time it.
4. Respond with JSON:
   ```json
   {
     "id": "...", "width": 600, "height": 400, "seconds": 0.21,
     "brightness": 0.4,
     "input":  "data:image/jpeg;base64,...",      // quality 88
     "final":  "data:image/jpeg;base64,...",      // quality 94
     "steps": [ {"name": "illumination", "title": "Finding where the light is",
                 "explain": "A map of how much light ...", "image": "data:..."}, ... ]
   }
   ```
   The images are sent as **base64 data URLs** embedded in the JSON. That
   is about 33% bigger than raw bytes. In exchange, there are no extra
   image endpoints, no temporary files and no cleanup, and the page can
   set `img.src = data.final` directly. The **Download** button simply
   links to the same data URL.
   `EXPLAIN` holds the plain-English explanation for each stage card.

**`POST /api/zoom`** takes a JSON body validated by the Pydantic model
`ZoomRequest`:

```python
id: str; x: int; y: int; w: int (>0); h: int (>0)
preset: str = "natural"; brightness: float | None
deblur_length: 0..60; deblur_angle: -180..180; deblur_nsr: (0, 1]
```

FastAPI rejects values outside those ranges automatically with HTTP 422.
The handler:

1. Clamps the box to the image.
2. Rejects boxes smaller than 8×4 px.
3. Recomputes stages 1–2 with the same preset and brightness.
4. Crops, calls `zoom_region`, and returns three data URLs: `plain`,
   `enhanced` and `text`.

**Concurrency, in detail.** `api_enhance` is declared `async def` (it needs
`await file.read`), but the image processing inside it is ordinary
blocking code. While one photo is being enhanced, that request blocks the
server's event loop, so other requests wait. `api_zoom` is a plain `def`,
which FastAPI runs in a thread pool. For a demo with a handful of users
this is fine.

### 9.2 Frontend: `src/web/static/index.html`

One file: HTML, CSS and vanilla JavaScript. No framework and no build
step.

**Page sections** (only one of the first three is visible at a time, via
`show()`):

1. `#start`: buttons for **Take a photo** (`<input type=file
   accept="image/*" capture="environment">` opens the rear camera on
   phones) and **Upload a photo**, a drag-and-drop zone, and three sample
   images.
2. `#loading`: a spinner.
3. `#results`, which contains:
   - **Controls:** a segmented "Natural / Max detail" toggle and a
     **Brightness** slider (0.20–0.65, step 0.05).
   - **Step by step:** one card per stage (image, "Step i of n", title,
     explanation). The cards appear one by one, 900 ms apart
     (`STEP_DELAY`), scrolling along. A "Skip to result" link cancels the
     animation.
   - **Before & after:** two stacked images. The "before" image is
     clipped with CSS `clip-path: inset(0 calc(100% - var(--pos)) 0 0)`.
     An invisible full-size range input updates the `--pos` CSS variable,
     so dragging reveals more or less of the original.
   - **Download** and **Try another photo** buttons, plus size and timing
     info.
   - **Zoom into a detail:** the final image in a "picker" where you drag
     a rectangle. Results appear as three figures (Just enlarged /
     Enhanced / Text view), with the **Fix motion blur** panel under them.

**Client state:** `state = { id, preset, brightness, box, boxFrac, imgW, imgH }`.

**Main flows:**

- **New photo** (`enhance(blob)`): build a `FormData` with `file` and
  `preset`, reset brightness to null (so the server uses the preset's
  default) and clear the zoom box. POST `/api/enhance`, then
  `render(data, animate=true)`. Samples are fetched as a blob first, then
  sent the same way as an upload.
- **Setting change** (`rerun()`): sends `id`, `preset` and `brightness`.
  Each call increments `rerunSeq`, and a response is ignored unless it
  belongs to the newest call. This avoids a **race condition**: if you
  move the slider twice quickly, a slow first response must not overwrite
  the newer one. After rendering, if a zoom box exists, the zoom is
  refreshed too. The brightness slider triggers a rerun on `change` (when
  you release it), not on every `input` movement.
- **Zoom selection:** pointer events (`pointerdown`, `pointermove`,
  `pointerup` with `setPointerCapture`) work for mouse and touch alike. The
  box is drawn in screen pixels. On release it is converted to *image*
  pixels by multiplying by `imgW / displayedWidth` (and the same for
  height). It is also saved as fractions (`boxFrac`), so it can be redrawn
  correctly after a window resize.
- **Deblur sliders:** each `input` event updates the labels and schedules a
  zoom request after **150 ms of no movement** (a *debounce*), so dragging
  doesn't flood the server. `zoomSeq` drops stale responses, like
  `rerunSeq` does.
- **Errors:** `post()` turns network failures into "Couldn't reach the
  server", and shows the server's `detail` message when it is a string.

**Accessibility touches:** `aria-pressed` on the toggle buttons,
`role="alert"` on errors, `aria-live` on the loading message,
`prefers-reduced-motion` support, and a mobile layout below 680 px.

### 9.3 One request, end to end

1. The browser loads `/` and gets `index.html`.
2. You tap sample 1. JavaScript fetches `/static/samples/1.jpg` as a blob
   and POSTs it to `/api/enhance` with `preset=natural`.
3. The server decodes it with Pillow (EXIF fix, ≤1600 px), stores it as
   `id=abc…` and runs the six natural-preset stages (≈0.2–0.4 s for
   600×400).
4. The server encodes the input, six step images and the final image as
   JPEG data URLs and returns them as JSON.
5. The page builds seven cards ("Your photo" plus six steps), animates
   them in, and fills in the slider, Download link and zoom picker.
6. You drag the brightness slider to 0.55 and release. The page POSTs
   `id=abc…, preset=natural, brightness=0.55`. The server recalls the photo
   from the cache, re-runs the pipeline and responds. The page re-renders
   without animation.
7. You drag a box over some text. The page POSTs JSON to `/api/zoom`. The
   server recomputes stages 1–2, crops, runs `zoom_region` and returns
   three images.
8. You tick "Fix motion blur" and move the sliders. Debounced `/api/zoom`
   calls update the images live.

---

## 10. The command-line tool

File: `src/enhance_image.py`. Run it from `src/`.

```bash
python enhance_image.py dark.jpg -o enhanced.jpg                     # natural preset
python enhance_image.py dark.jpg -o out.jpg --preset detail
python enhance_image.py dark.jpg -o out.jpg --steps-dir steps/       # 0_input.jpg, 1_illumination.jpg, ...
python enhance_image.py dark.jpg -o out.jpg --skip colour contrast   # switch optional stages off
python enhance_image.py dark.jpg -o out.jpg --sharpen 0.5            # force sharpening at 0.5
python enhance_image.py dark.jpg -o out.jpg --max-size 0             # keep full resolution
```

How it works:

1. `load_image` reads with OpenCV. If the long side is over `--max-size`
   (default 2000), it downscales with `INTER_AREA`, the best method for
   shrinking.
2. Take `(params, enabled)` from the preset and remove the `--skip` stages.
   `--skip` only accepts non-required stages; argparse enforces this with
   `choices`.
3. `--sharpen X` adds "sharpen" to the enabled set and replaces
   `sharpen_amount`.
4. `enhance(...)`, with `keep_steps` only if `--steps-dir` was given.
   Print the size, the time and the chosen γ.
5. Write the output, and optionally every step image.

Phone photos come out the right way up. `cv2.imread` applies the EXIF
orientation tag by default, just as the web app does through Pillow.

---

## 11. Measuring quality: dataset, PSNR, SSIM, evaluation

### 11.1 The LOL dataset

- "LOw-Light" dataset (Wei, Wang, Yang, Liu, BMVC 2018). It has **pairs of
  the same scene**: one shot in low light, one in normal light. Images are
  600×400.
- `our485/` has 485 pairs. The tuner used 100 of them, randomly picked
  with seed 0.
- `eval15/` has 15 pairs, used **only** for the reported numbers.
- It is not included in the repo (`data/raw/` is git-ignored). Download
  instructions are in `data/README.md`. It is only needed to reproduce the
  numbers, not to run the app.
- `src/data/verify_lol.py` walks `data/raw/LOL/` and counts images per
  folder. For sibling folders named like low/high or low/normal, it checks
  that the filenames match one-to-one and prints a few image shapes.
- `src/data/lol_dataset.py`: `load_pairs(low_dir, high_dir)` pairs files by
  filename stem. It raises an error if a low image has no partner
  (*"fail loudly rather than silently misaligning pairs"*).

**Why keep tuning and test data separate?** If the settings were chosen by
looking at the same 15 images that are then reported, the score would be
optimistic. The settings would be fitted to those particular photos. This
is the classical-processing version of the train/test split.

### 11.2 PSNR: Peak Signal-to-Noise Ratio

```
MSE  = mean over all pixels and channels of (output − reference)²
PSNR = 10 · log10(255² / MSE)     [dB]
```

- It measures pixel-by-pixel fidelity. Higher is better, and identical
  images give infinity.
- To get a feel for the numbers, convert back to an RMS error
  `= 255 / 10^(PSNR/20)`:
  - 16.47 dB (pipeline): about **38** grey levels average error.
  - 9.13 dB (CLAHE alone): about **89** grey levels.
- Weakness: PSNR ignores *structure*. A uniform brightness offset and
  random noise of the same energy get the same score, even though they
  look completely different. PSNR is very sensitive to overall brightness,
  which matters a lot in low-light work.

### 11.3 SSIM: Structural Similarity

SSIM compares local windows of the two images on three things: luminance
(mean), contrast (standard deviation) and structure (correlation):

```
SSIM(x, y) = [(2μxμy + C1)(2σxy + C2)] / [(μx² + μy² + C1)(σx² + σy² + C2)]
```

It is averaged over all windows, and here over the colour channels too
(`channel_axis=2`). It ranges up to 1, which means identical. It matches
human perception better than PSNR and **punishes noise and lost
structure** much more heavily. That is why the denoise stage moves it so
much.

Both metrics come from `skimage.metrics`, computed on uint8 images with
`data_range=255`.

### 11.4 `evaluate.py`

From `src/`:

```bash
D=../data/raw/LOL/lol_dataset
python evaluate.py --low-dir $D/eval15/low --high-dir $D/eval15/high --out ../results/metrics.csv
```

- Methods: the 6 classical ones, plus a **cumulative ablation** of the
  pipeline:
  - `pipeline_2_brighten`: stages 1–2
  - `pipeline_3_denoise`: + denoise
  - `pipeline_4_contrast`: + contrast
  - `pipeline_5_colour`: + colour, which is the full default (benchmark)
    pipeline
  - `pipeline_6_sharpen_0.4`: everything plus sharpening at 0.4, to show
    its effect
- An **ablation study** adds (or removes) one component at a time to
  measure what each contributes.
- For every (method, image) pair it computes PSNR and SSIM against the
  reference. If shapes differ it raises an error rather than silently
  resizing.
- Writes `metrics.csv` (11 methods × 15 images = 165 rows) and
  `metrics_summary.csv` (the mean per method).
- The pipeline rows use `PipelineParams()` defaults, i.e. the benchmark
  settings.

### 11.5 `make_figure.py`

This builds `results/figures/comparison.png` (the image at the top of the
README):

- It picks `--num-images` (default 4) random test images, using seed 0 for
  reproducibility.
- Each row shows: low input | gamma | hist_eq | clahe | ssr | msr |
  homomorphic | pipeline (ours) | ground truth.
- "pipeline (ours)" is `enhance(img)` with defaults, which is the
  **benchmark** preset.

---

## 12. How the settings were chosen (tuning)

File: `src/tune_pipeline.py`. Output: `results/tuned_params.json` and the
console log saved as `results/tune.log`.

**Method: a greedy, stage-by-stage grid search.**

```python
SEARCH = [
  ([estimate_illumination, brighten], {"guide_radius": [5,15,30], "guide_eps": [1e-3,1e-2],
                                       "target_brightness": [0.35,0.4,0.45,0.5,0.55]}),   # 30 combos
  ([denoise],        {"denoise_strength": [1,1.5,2,2.5,3], "denoise_dark_power": [0,0.5,1]}),  # 15
  ([local_contrast], {"clahe_clip": [0.75,1,1.25,1.5,2]}),                                 # 5
  ([colour_correct], {"white_balance_strength": [0,.25,.5,.75,1], "saturation": [.9,1,1.1,1.2]}),  # 20
  ([sharpen],        {"sharpen_amount": [0,0.2,0.4,0.6]}),                                 # 4
]
```

1. Load 100 random pairs from `our485`.
2. For each group, in pipeline order: try **every combination** in that
   group's grid (`itertools.product`), running only that group's stages on
   top of the outputs saved from the previous group. Keep the combination
   with the best score. Its outputs (and `info` dicts) become the input for
   the next group.
3. **Score = mean PSNR + 10 × mean SSIM.** The factor 10 puts them on
   similar scales, because a good change moves PSNR by about 1 dB and SSIM
   by about 0.1. Without it, PSNR would dominate.

**Why greedy rather than a full grid?** Searching everything at once would
mean 30 × 15 × 5 × 20 × 4 = **180,000** combinations, each run on 100
images. The greedy approach tries 30 + 15 + 5 + 20 + 4 = **74** candidate
settings. Each stage's work is done once per candidate, and earlier stages
are not recomputed. The trade-off is that it can miss interactions between
stages. For example, a stronger denoise might pair best with stronger
CLAHE, and greedy search wouldn't find that. It took about 20 minutes.

**Why stages 1 and 2 are tuned together:** stage 1 changes nothing in the
image by itself, so it can't be scored alone. Its effect only shows
through brightening.

**What it chose** (from `tune.log`):

| Group | Chosen | Score on search set (PSNR / SSIM) |
|---|---|---|
| illumination + brighten | radius 30, eps 1e-3, target 0.40 | 15.72 / 0.488 |
| denoise | strength 1.5, dark power 1.0 | 16.37 / 0.737 |
| contrast | clip 0.75 (the smallest tried) | 16.51 / 0.738 |
| colour | WB 0.5, saturation 0.9 | 17.19 / 0.751 |
| sharpen | 0.0 (off) | 17.19 / 0.751 |

Interesting patterns in the log:

- Larger guide radius was always better (5 < 15 < 30), and eps 1e-3 beat
  1e-2. LOL rewards a smooth, edge-aware light map.
- Brightness targets above 0.40 lowered the score: 0.45 already raised
  PSNR slightly but cost more SSIM. The LOL references aren't very bright.
- CLAHE clip and sharpening both scored best at the *lowest* value
  tried. The best value may lie at or beyond the edge of the grid, so it
  is worth saying the grid was not extended.

The chosen numbers were then written by hand into `PipelineParams`
defaults. Sharpening 0.0 corresponds to "sharpen stage disabled" in
`DEFAULT_ENABLED`, while `sharpen_amount` keeps 0.4 for when it is turned
on.

**Search score vs. test score:** 17.19 / 0.751 on the tuning images
against 16.47 / 0.716 on the 15 untouched test images. A small drop on
unseen data is normal and expected; it is exactly why the test set is
kept separate.

---

## 13. Reading the results

`results/metrics_summary.csv` (means over the 15 eval images):

| Method | PSNR | SSIM | Comment |
|---|---|---|---|
| CLAHE alone | 9.13 | 0.357 | Only redistributes contrast locally. It doesn't lift the overall level enough, so the output stays far too dark compared with the reference. |
| Gamma 2.2 alone | 13.23 | 0.590 | Global brightening, but fixed. Not strong enough for very dark images and too strong for brighter ones. Colours wash out. |
| Histogram equalization | 14.26 | 0.425 | Brightens strongly but amplifies noise and makes a harsh tonal mapping. |
| MSR | 15.50 | 0.591 | Good relighting. The min-max stretch shifts tones and colours. |
| SSR | 15.57 | 0.597 | Similar. Best single method on SSIM. |
| Homomorphic | 16.15 | 0.383 | Good average brightness (high PSNR), but boosts high frequencies including noise (low SSIM). |
| Pipeline 1–2 (brighten) | 15.69 | 0.485 | Correct relighting, but the amplified noise is still there, so SSIM is lower than gamma's. |
| **+ denoise** | 16.45 | **0.729** | **The biggest jump.** Removing amplified noise is what SSIM rewards. |
| + contrast | 16.43 | 0.718 | Essentially neutral on LOL (it was tuned to the gentlest setting). Helps real photos look less flat. |
| **+ colour (= benchmark)** | **16.47** | 0.716 | Best PSNR overall. |
| + sharpen 0.4 | 16.23 | 0.691 | Worse on both, because the references are soft. |

**The story to tell:**

1. Relighting with an edge-aware illumination map (stages 1–2) already
   matches the best single methods on PSNR.
2. Brightening always amplifies noise. The **adaptive, darkness-weighted
   denoising** is the single most valuable addition.
3. The full pipeline beats every single method on PSNR, and is far ahead
   on SSIM (0.716 vs 0.597 for the best single method).
4. Metrics are not everything. Contrast and sharpening look better to
   people on real photos but score neutral or worse on LOL. That is why
   there are separate presets: one tuned for the metric and two set by eye.

**Speed** (README, Intel i7-1255U): about 0.3–0.5 s for 600×400 and about
1.7 s for a 12 MP photo. Locally, the three samples took about 0.2 s each.
NLM denoising is the most expensive stage.

---

## 14. The deblur experiment

File: `src/deblur_demo.py`. Output: `results/figures/deblur_demo.png`.

**Why it is synthetic.** To *measure* deblurring you need to know the
sharp truth. Real CCTV plates are too small and their true text is
unknown. So the script:

1. Draws a plate (240×60, text "MH12 AB 3456", with a 30 px grey border):
   the **ground truth**.
2. Blurs it with a known PSF (15 px at 10°) using `cv2.filter2D`. The line
   PSF is symmetric about its centre, so correlation and convolution give
   the same result here.
3. Adds Gaussian noise with σ = 0.01 (fixed seed 0, so it is
   reproducible).
4. Tries Wiener with NSR ∈ {0.001, 0.005, 0.01, 0.03} and keeps the best
   PSNR. This uses the ground truth to choose, an "oracle" choice that is
   fine for a demo but not something possible on real photos. In the app,
   the user tunes it by eye.
5. Deliberately deblurs with an angle that is **50° wrong**.
6. Runs the full `zoom_region` on the plate area, and stacks five labelled
   rows into the figure: original, blurred, wrong direction, deblurred +
   zoom, text view.

**Results** (README):

| | PSNR (dB) |
|---|---|
| Blurred | 16.39 |
| Wiener, correct PSF | **21.28** (+4.9 dB) |
| Wiener, direction 50° wrong | 8.59 (much worse than doing nothing) |

**Lesson:** deconvolution is powerful when the blur model is right, and
harmful when it is wrong. That is why the UI says "Wrong settings make it
worse."

---

## 15. Hosting and deployment

### 15.1 Dependencies

`requirements.txt` (app and CLI), all with exact versions pinned so the
setup is reproducible:

| Package | Why |
|---|---|
| numpy | arrays and FFT |
| opencv-python-headless | image processing. The *headless* build has no GUI libraries, which is smaller and works in servers and containers |
| pillow | robust decoding of uploads plus the EXIF rotation fix |
| fastapi | web framework |
| uvicorn | ASGI server that runs FastAPI |
| python-multipart | required by FastAPI for `Form(...)` and `UploadFile` |

`requirements-dev.txt` adds `scikit-image` (PSNR/SSIM), `pandas` (CSV
summaries) and `matplotlib` (the comparison figure). Only the measurement
scripts need these. `deblur_demo.py` also imports scikit-image.

### 15.2 Dockerfile

```dockerfile
FROM python:3.12-slim
RUN useradd -m -u 1000 user          # HF Spaces run containers as uid 1000
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt   # copied before src → layer cache
COPY --chown=user src ./src
WORKDIR /home/user/app/src           # imports need src/ as cwd
EXPOSE 7860                          # HF Spaces' expected port
CMD ["uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "7860"]
```

Copying `requirements.txt` and installing before copying `src/` means
code changes don't trigger a reinstall of every package (Docker layer
caching). `.dockerignore` leaves out `.venv`, `.git`, `data`, `results`
and `__pycache__` so the build context stays small.

### 15.3 `scripts/deploy_hf.sh`

A Hugging Face Space is a git repository. HF builds the Dockerfile in it
and runs the container. The script:

1. Takes `<user>/<space>`, finds the repo root and the current commit
   hash, and creates a temp directory (deleted on exit by `trap`).
2. Shallow-clones the Space repo. `SPACE_URL` can override the URL for
   testing.
3. Deletes everything in the clone except `.git` and `.gitattributes`,
   then copies in `src/`, `Dockerfile` and `requirements.txt`, and strips
   `__pycache__`.
4. Writes a `README.md` with the **YAML front matter HF requires**:
   `sdk: docker`, `app_port: 7860`, title, emoji and colours. The body
   records which GitHub commit was deployed.
5. If nothing changed, it says so and exits. Otherwise it commits and
   pushes; git prompts for the HF username and a write-access token.

`set -euo pipefail` makes the script stop at the first error, treat unset
variables as errors, and fail when any part of a pipe fails.

Free Spaces sleep after about 2 days without visitors, and the first
visit after that takes about a minute to wake. Open it a few minutes
before a demo.

---

## 16. Project history

From `git log`, oldest first:

1. `03d10a0`: **First version.** Classical baselines compared against a
   small **U-Net** (a convolutional neural network) trained on LOL, plus
   an ablation of a "Copy-Blend" data-augmentation trick.
2. `a26360d`: **Pivot to classical-only.** The U-Net, training script and
   Copy-Blend were removed.
3. `9687658`: Added the six-stage pipeline, tuning, CLI, evaluation
   ablation and the step-by-step web demo.
4. `3d3528c`: Added presets, the region zoom, Wiener motion deblur and the
   text view.
5. `e5dce3d`: README rewritten around running the app; removed leftover
   "training" wording.
6. `d8e0eea`: Free hosting on Hugging Face Spaces (Dockerfile, deploy
   script, split requirements).

**Why the pivot matters for presentations.** The project's selling point
is that *every step is explainable* with course material. Nothing is a
black box, and you can justify every number, either by the maths or by the
tuning log.

---

## 17. Limits, quirks and loose ends

These are worth knowing, both for honest discussion and so they don't
surprise you.

**Inherent limits:**

- Brightening can only reveal what the sensor recorded. In very dark
  areas the signal is mostly noise, so pitch-black photos come out grainy
  even after denoising.
- Gray-world white balance can be wrong for scenes dominated by one
  colour. The partial strength and cap reduce this but don't remove it.
- Zoom cannot invent characters. Deblur only handles **straight-line,
  uniform** motion with a **user-guessed** PSF. It does not handle camera
  shake (curved paths), out-of-focus blur, or blur that differs across the
  image.
- PSNR/SSIM against LOL reflects LOL's particular "look": soft and muted.
  That is why the benchmark preset looks flat on real photos.

**Code-level quirks (true at commit `d8e0eea`):**

- `src/classical/denoise.py` (bilateral and median) is not imported
  anywhere. It is a leftover.
- `enhance()` with no arguments uses **benchmark** settings, but the CLI
  and web default is **natural**. `make_figure.py` therefore shows
  benchmark output.
- In the CLI, `if args.sharpen:` treats `--sharpen 0` as "not given", so
  it cannot be used to force sharpening to zero. Use the benchmark preset
  or `--skip sharpen` instead.
- The `target_brightness` is hit exactly only right after stage 2. The
  final output ends up somewhat brighter (0.44–0.48 on the samples).
- The web cache is per-process and in memory (see 9.1). The enhance
  endpoint blocks the event loop while it computes.
- `/api/zoom` recomputes stages 1–2 on every call, including every
  debounced slider move. The cost is small (no denoising), but nothing is
  cached.
- `verify_lol.py`'s fallback message ("tell me the actual paths…") is
  worded as if talking to an assistant. It is harmless.
- The docstring examples in `evaluate.py` and `make_figure.py` show paths
  like `data/raw/LOL/eval15/...`. The README's commands, with
  `lol_dataset/` in the path and run from `src/`, are the correct ones.
- The tuning grid hit its lower edge for `clahe_clip` and
  `sharpen_amount`, so the true optimum may lie outside the values tried.

---

## 18. How to change or extend things

**Change how the web app looks or behaves by default:** edit `PRESETS` in
`src/pipeline/pipeline.py`. For example, raise natural's `clahe_clip` for
more punch.

**Add a new pipeline stage**, for example "gamma tone curve":

1. In `stages.py`, write `def my_stage(img, info, p): ... return img`,
   working in float [0,1] BGR.
2. Add any settings as fields of `PipelineParams`, with defaults.
3. Insert `("mystage", stages.my_stage, "Caption for web")` in the right
   place in `STAGES`.
4. **Add an entry to `EXPLAIN` in `src/web/server.py`**. Otherwise
   `/api/enhance` crashes with a `KeyError` when it builds the step cards.
5. Decide whether it belongs in `DEFAULT_ENABLED` and in each preset's
   enabled set.
6. Optionally add a group to `SEARCH` in `tune_pipeline.py` and re-run
   `evaluate.py`. `PIPELINE_METHODS` builds the cumulative ablation rows
   from `STAGE_NAMES` automatically.

**Add a new baseline method:** write it in `src/classical/` (uint8 BGR in,
uint8 BGR out) and register it in `CLASSICAL_METHODS` in both
`evaluate.py` and `make_figure.py`.

**Re-tune:** install `requirements-dev.txt`, download LOL, run
`tune_pipeline.py`, then copy the chosen values from
`results/tuned_params.json` into the `PipelineParams` defaults.
**Re-run `evaluate.py` after that.** The tuner does not update the
defaults on its own.

**Run things quickly in Python** (from `src/`):

```python
import cv2
from pipeline.pipeline import enhance, PRESETS
img = cv2.imread("web/static/samples/1.jpg")
params, enabled = PRESETS["natural"]
r = enhance(img, params, enabled)
print(r.info["gamma"], r.info["noise_sigma"])
for name, step in r.steps: cv2.imwrite(f"/tmp/{name}.jpg", step)
cv2.imwrite("/tmp/final.jpg", r.final)
```

---

## 19. Questions you will probably be asked (with answers)

**Q: Why not just use a neural network? They score higher on LOL.**
A: The aim was a method where every step is explainable and needs no
training data. It runs on a CPU in a fraction of a second, has no weights
to ship, and you can see exactly what each stage did, which the web page
shows. Learned methods can score higher on benchmarks, but they are black
boxes and can hallucinate detail. That last point matters for something
like a number plate.

**Q: What is the illumination map and why take the max of R, G, B?**
A: It estimates how much light fell on each point (the Retinex `L`). A
surface can't reflect more than the light falling on it, so the brightest
channel is the tightest per-pixel lower bound on the light. Using the max
stops strongly coloured objects from being mistaken for dark ones.

**Q: Why a guided filter instead of a Gaussian blur on that map?**
A: A Gaussian blur smears light across object edges, so brightening
would create halos. The guided filter smooths inside regions but keeps
edges, because its local linear model passes high-variance (edge) windows
through almost unchanged.

**Q: How is the brightening different from ordinary gamma correction?**
A: Gamma is applied to the smooth illumination map, not to each colour
channel. Every channel at a pixel gets the same gain, so hue is
preserved. Neighbouring pixels get similar gains, so texture contrast is
preserved. And γ is found automatically (by bisection) so that every photo
reaches the same target brightness.

**Q: Why bisection? Is it guaranteed to work?**
A: Mean brightness decreases steadily as γ increases (for γ in [0.05, 1]),
so the function is monotonic and bisection converges. 20 iterations give
about 10⁻⁶ precision. It runs on a quarter-size copy for speed.

**Q: How does the denoiser know how strong to be?**
A: It estimates the noise standard deviation with Immerkær's method. A
3×3 difference-of-Laplacians mask cancels image structure and leaves
noise, and a closed-form scale factor turns the mean absolute response
into σ. The NLM strength `h` is proportional to that σ, and the result is
blended in more where the original was darker.

**Q: Why non-local means?**
A: It averages pixels whose *surrounding patches* look alike, not just
nearby pixels. Edges and repeated textures survive far better than with
Gaussian, median or bilateral filtering.

**Q: Why CLAHE only on the L channel?**
A: Equalizing R, G and B separately changes their ratios, which shifts
colours. LAB separates lightness from colour.

**Q: What does the clip limit do?**
A: It caps each tile's histogram bins before equalizing, which limits how
much contrast (and noise) can be amplified. Lower means gentler.

**Q: Gray-world white balance fails on sunsets. How do you handle that?**
A: It is applied at half strength, and the real-photo presets cap each
channel's gain at ×1.2.

**Q: Why is sharpening off in the benchmark?**
A: On LOL, every amount tried lowered both PSNR and SSIM, because the
reference photos are soft. Real photos look better with it, so the natural
and detail presets use it. It is a deliberate metric-versus-perception
decision.

**Q: Explain the Wiener filter.**
A: Blur is multiplication by `H` in frequency space. Dividing by `H` undoes
it but blows up noise where `H ≈ 0`. Wiener uses `conj(H)/(|H|² + NSR)`:
it acts like inversion where the signal survived and shrinks towards zero
where it didn't. NSR trades sharpness against noise.

**Q: Why pad before the FFT?**
A: The FFT assumes the image repeats periodically, so its edges wrap around
and touch. Reflect-padding stops ringing at those edges.

**Q: Why is the zoom crop taken from the brightened image rather than the
final one?**
A: The pipeline's denoising can smooth away characters a few pixels wide.
The zoom path deblurs first (if asked), then applies its own denoising.

**Q: How do you know the tuning didn't overfit?**
A: Tuning used 100 images from `our485`. The reported numbers come from the
separate 15-image `eval15` set, which was never used for any choice. The
score dropped a little on the test set (17.19 → 16.47 dB), which is the
expected small generalisation gap.

**Q: Why greedy tuning?**
A: A full grid would be 180,000 combinations; greedy needs 74. The
trade-off is that it can miss interactions between stages.

**Q: What are PSNR and SSIM, and why report both?**
A: PSNR measures per-pixel error (log scale, in dB). SSIM compares local
brightness, contrast and structure, which is closer to human perception.
They can disagree: homomorphic filtering has a high PSNR but low SSIM
because of amplified noise.

**Q: What's the biggest contributor to the result?**
A: Denoising: SSIM goes from 0.485 to 0.729.

**Q: What can't it do?**
A: Recover detail that was never recorded. Read a tiny plate. Fix
non-linear or out-of-focus blur. Handle scenes where gray-world is badly
wrong.

**Q: How does the web app avoid uploading the photo again on every slider
change?**
A: The server keeps the last 16 decoded photos in memory under random ids.
The page sends only the id plus new settings. Sequence counters drop stale
responses, and the deblur sliders are debounced by 150 ms.

---

## 20. Glossary

| Term | Meaning |
|---|---|
| **Ablation** | Adding or removing one component at a time to measure what it contributes |
| **Adaptive thresholding** | Thresholding each pixel against its own neighbourhood's mean instead of one global value |
| **BGR** | OpenCV's channel order: blue, green, red |
| **Bisection** | Root-finding by repeatedly halving an interval known to contain the answer |
| **Box filter** | Average over a square window. Constant cost per pixel whatever the size |
| **CDF** | Cumulative distribution function. The mapping used by histogram equalization |
| **CLAHE** | Contrast-Limited Adaptive Histogram Equalization |
| **Data URL** | A file embedded in a string: `data:image/jpeg;base64,...` |
| **Debounce** | Waiting until input stops for a short time before acting |
| **Deconvolution** | Undoing a convolution (blur) |
| **EXIF orientation** | Metadata tag telling viewers how to rotate a photo |
| **FFT** | Fast Fourier Transform. Converts an image to its frequency representation |
| **Gamma correction** | Power-law mapping `out = in^γ` |
| **Gray-world** | White balance that assumes the scene averages to grey |
| **Guided filter** | Edge-preserving smoothing using a local linear model of a guide image |
| **Homomorphic filtering** | Log → FFT → high-emphasis filter → inverse FFT → exp |
| **Illumination / reflectance** | Light falling on a scene vs. the surfaces' own appearance (Retinex) |
| **Lanczos interpolation** | Windowed-sinc resampling. A sharp, high-quality way to enlarge |
| **LAB / HSV / YCrCb** | Colour spaces separating lightness, or saturation, from colour |
| **LOL dataset** | 500 low/normal-light photo pairs (485 + 15) |
| **NLM (non-local means)** | Denoising by averaging pixels whose surrounding patches look alike |
| **NSR** | Noise-to-signal ratio, the Wiener filter's regularization term |
| **PSF** | Point-spread function: what a single point of light looks like after blurring |
| **PSNR** | Peak signal-to-noise ratio, pixel-error metric in dB |
| **Retinex** | Model `I = R × L`, and the family of methods built on it |
| **SSIM** | Structural similarity index, a perceptual metric up to 1 |
| **Unsharp masking** | Sharpening by adding back `image − blurred image` |
| **Wiener filter** | Minimum-mean-squared-error linear deconvolution filter |

---

## 21. Suggested study plan

1. **Use the app first.** Run `cd src && uvicorn web.server:app --port
   8000`, try the three samples in both presets, move the brightness
   slider, and try zooming. Seeing every stage makes the code much easier
   to read.
2. **Read `src/pipeline/stages.py` top to bottom** together with
   [section 5](#5-the-enhancement-pipeline-stage-by-stage). It is only 211
   lines and is the heart of the project.
3. **Read `src/pipeline/pipeline.py`** (78 lines) with
   [section 6](#6-the-pipeline-runner-and-the-presets).
4. **Experiment.** Run the CLI with `--steps-dir steps/` and open the step
   images. Try `--skip denoise` and look at the grain. Try `--preset
   benchmark` vs `natural`.
5. **Read `src/pipeline/zoom.py`** with [section 7](#7-zoom-motion-deblur-and-text-view),
   then look at `results/figures/deblur_demo.png`.
6. **Read `src/web/server.py`** and skim the `<script>` part of
   `index.html`, with [section 9](#9-the-web-app-backend-and-frontend).
7. **Read `results/tune.log`** next to [section 12](#12-how-the-settings-were-chosen-tuning).
   Find the line where each chosen value won.
8. **Look at `results/metrics_summary.csv` and `comparison.png`** with
   [section 13](#13-reading-the-results), and practise telling "the story".
9. **Answer the questions in [section 19](#19-questions-you-will-probably-be-asked-with-answers)
   out loud** without looking. Anything you stumble on, go back to its
   section.

Small exercises to check your understanding:

- Work out the brightening gain for `t = 0.1` when γ = 0.3. *(Answer:
  0.1^(−0.7) ≈ 5.0×.)*
- Why would `denoise_dark_power = 0` hurt bright regions? *(It denoises
  them fully even though their noise wasn't amplified, which smooths
  away real detail.)*
- What happens to `wiener_deblur` if NSR = 0 and the image is noisy?
  *(Plain inverse filtering: noise explodes at frequencies where H ≈ 0.)*
- If you ran uvicorn with `--workers 4`, what would users see, and why?
  *(Random "photo expired" 410 errors, because each worker has its own
  in-memory cache.)*
