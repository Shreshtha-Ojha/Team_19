"""Classical enhancement methods applied before the detector (Phase 4).

Every method maps a uint8 BGR image to a uint8 BGR image, so any of them can
be put in front of the unchanged baseline detector.

Two groups:
  generic  - course operators used as they are: gamma, global histogram
             equalisation, CLAHE, multi-scale Retinex, and the six-stage
             low-light pipeline in src/pipeline (Retinex illumination map +
             adaptive gamma + NLM + CLAHE + colour). These brighten for a
             human viewer.
  sky      - methods aimed at undoing the fog + low-light model instead:
             non-local means denoising with strength set from the image's own
             estimated noise, removal of the fog background (airlight and
             glow), and a contrast gain for the attenuated stars.

The baseline was trained on dark, noise-free sky images, so the sky methods
try to bring a degraded image back towards that look, not to brighten it.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from classical.clahe import clahe_enhance  # noqa: E402
from classical.gamma import gamma_correct  # noqa: E402
from classical.hist_eq import hist_eq  # noqa: E402
from classical.retinex import multi_scale_retinex  # noqa: E402
from pipeline.pipeline import enhance as pipeline_enhance  # noqa: E402
from pipeline.stages import estimate_noise  # noqa: E402


def noise_sigma(img: np.ndarray) -> float:
    """Estimated noise std in grey levels (Immerkaer 1996)."""
    return estimate_noise(img.astype(np.float32) / 255.0) * 255.0


def nlm(img: np.ndarray, k: float) -> np.ndarray:
    """Non-local means with filter strength h = k * estimated noise sigma.

    On a clean image sigma is tiny, so h stays at its floor of 1 grey level
    and the image is almost unchanged.
    """
    h = max(1.0, k * noise_sigma(img))
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)


def remove_background(img: np.ndarray, ksize: int, target: float) -> np.ndarray:
    """Subtract the part of the smooth sky background that lies above `target`.

    The background is a per-channel median over ksize x ksize pixels: stars
    are a few pixels wide, so the median ignores them and keeps the airlight
    and glow. Only the excess over `target` (the dark-sky level of clean
    images) is removed, so dark regions and the Moon's disc are kept.
    """
    bg = cv2.medianBlur(img, ksize).astype(np.float32)
    excess = np.maximum(bg - target, 0.0)
    return np.clip(img.astype(np.float32) - excess, 0, 255)


def sky_restore(img: np.ndarray, k: float = 2.0, bg_ksize: int = 31, gain: float = 1.5,
                target: float = 7.0) -> np.ndarray:
    """Denoise -> remove fog background -> stretch star contrast about the sky level."""
    den = nlm(img, k) if k > 0 else img
    flat = remove_background(den, bg_ksize, target) if bg_ksize > 0 else den.astype(np.float32)
    out = target + gain * (flat - target)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _pipeline(img: np.ndarray) -> np.ndarray:
    return pipeline_enhance(img, keep_steps=False).final


METHODS = {
    "none": lambda im: im,
    # generic
    "gamma": lambda im: gamma_correct(im, 2.2),
    "hist_eq": hist_eq,
    "clahe": lambda im: clahe_enhance(im, 2.0, 8),
    "msr": multi_scale_retinex,
    "pipeline": _pipeline,
    # denoising only
    "median3": lambda im: cv2.medianBlur(im, 3),
    "nlm_k1": lambda im: nlm(im, 1.0),
    "nlm_k2": lambda im: nlm(im, 2.0),
    "nlm_k3": lambda im: nlm(im, 3.0),
    # k4-k6 were added after k3 came out best on valid with k3 at the edge of the grid.
    "nlm_k4": lambda im: nlm(im, 4.0),
    "nlm_k5": lambda im: nlm(im, 5.0),
    "nlm_k6": lambda im: nlm(im, 6.0),
    # background removal only (ablation of sky_restore)
    "bgsub_31": lambda im: sky_restore(im, k=0, bg_ksize=31, gain=1.0),
}
# sky_restore grid; the setting is chosen on the validation split.
for _k in (2, 3):
    for _b in (31, 61):
        for _g in (1.0, 2.0):
            METHODS[f"sky_k{_k}_b{_b}_g{_g:g}"] = (
                lambda im, k=_k, b=_b, g=_g: sky_restore(im, k=k, bg_ksize=b, gain=g))
