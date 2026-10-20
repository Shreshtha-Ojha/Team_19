"""The six stages of the low-light enhancement pipeline.

Every stage has the same shape:

    stage(img, info, params) -> img

  img    : float32 BGR image, values in [0, 1]
  info   : dict shared between stages (e.g. stage 1 stores the illumination
           map in info["illumination"] for stages 2 and 3 to use)
  params : PipelineParams with every tunable number in one place

Order: illumination -> brighten -> denoise -> contrast -> colour -> sharpen.
Brightening comes first because the other steps behave badly on near-black
pixels; denoising comes before contrast/colour so they don't amplify noise;
sharpening is last so it doesn't sharpen noise.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from classical.clahe import clahe_enhance


@dataclass
class PipelineParams:
    # Defaults are the values chosen by tune_pipeline.py on 100 LOL training
    # images (results/tuned_params.json); the test set was never used.
    # Stage 1: illumination map
    guide_radius: int = 30          # guided-filter window radius (pixels, at 600px width)
    guide_eps: float = 1e-3         # guided-filter smoothness; larger = smoother map
    # Stage 2: adaptive brightening
    target_brightness: float = 0.40  # mean grey level the output should reach
    illum_floor: float = 0.02       # avoids dividing by ~0 in pitch-black pixels
    # Stage 3: denoising
    denoise_strength: float = 1.5   # multiplies the automatically estimated noise level
    denoise_dark_power: float = 1.0  # how strongly denoising favours dark regions
    # Stage 4: local contrast
    clahe_clip: float = 0.75
    clahe_tiles: int = 8
    # Stage 5: colour
    white_balance_strength: float = 0.5  # 0 = off, 1 = full gray-world
    wb_max_gain: float | None = None     # cap per-channel gain (protects sunsets etc.); None = no cap
    saturation: float = 0.9
    # Stage 6: sharpening
    sharpen_amount: float = 0.4   # stage is off by default, see pipeline.DEFAULT_ENABLED
    sharpen_sigma: float = 1.5


def to_float(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.astype(np.float32) / 255.0


def to_u8(img_f: np.ndarray) -> np.ndarray:
    return np.clip(img_f * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _scaled_radius(r: int, img: np.ndarray) -> int:
    """Scale a radius tuned for ~600px-wide LOL images to this image's size."""
    return max(1, round(r * max(img.shape[:2]) / 600))


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Edge-preserving smoothing (He, Sun, Tang 2010), single-channel guide.

    Fits a local linear model src ≈ a*guide + b in every (2r+1)^2 window, so
    the output follows edges in `guide` while being smooth inside regions.
    """
    k = (2 * r + 1, 2 * r + 1)
    mean_I = cv2.boxFilter(guide, -1, k)
    mean_p = cv2.boxFilter(src, -1, k)
    corr_Ip = cv2.boxFilter(guide * src, -1, k)
    corr_II = cv2.boxFilter(guide * guide, -1, k)
    var_I = corr_II - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return cv2.boxFilter(a, -1, k) * guide + cv2.boxFilter(b, -1, k)


# --- Stage 1 -----------------------------------------------------------------

def estimate_illumination(img, info, p: PipelineParams):
    """Retinex view: image = reflectance x illumination.

    Initial estimate: the brightest of B, G, R at each pixel (a surface can't
    reflect more light than falls on it). Refined with a guided filter so the
    map is smooth inside objects but keeps their boundaries — otherwise
    brightening would create halos around edges.
    """
    t0 = img.max(axis=2)
    t = guided_filter(t0, t0, _scaled_radius(p.guide_radius, img), p.guide_eps)
    info["illumination"] = np.clip(t, p.illum_floor, 1.0)
    return img  # this stage only measures; the image is unchanged


# --- Stage 2 -----------------------------------------------------------------

def _apply_gamma(img, t, gamma):
    # Replace illumination t by t**gamma: out = img / t * t**gamma.
    # Same gain on B, G, R, so hue is preserved.
    gain = np.power(t, gamma - 1.0)
    return np.clip(img * gain[..., None], 0.0, 1.0)


def brighten(img, info, p: PipelineParams):
    """Gamma correction on the illumination map, not on the image itself.

    Dark regions (small t) get a large gain, already-bright regions (t near 1)
    almost none, so lamps and windows don't blow out. Gamma is chosen
    automatically by bisection so the result's mean brightness hits
    `target_brightness`: a darker photo gets a stronger boost.
    """
    t = info["illumination"]
    small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    t_small = cv2.resize(t, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)

    def mean_after(g):
        return cv2.cvtColor(_apply_gamma(small, t_small, g), cv2.COLOR_BGR2GRAY).mean()

    lo, hi = 0.05, 1.0  # gamma=1 leaves the image unchanged
    if mean_after(hi) >= p.target_brightness:
        gamma = hi  # already bright enough
    else:
        for _ in range(20):
            mid = (lo + hi) / 2
            if mean_after(mid) > p.target_brightness:
                lo = mid
            else:
                hi = mid
        gamma = (lo + hi) / 2
    info["gamma"] = gamma
    return _apply_gamma(img, t, gamma)


# --- Stage 3 -----------------------------------------------------------------

def estimate_noise(img) -> float:
    """Noise standard deviation (Immerkaer 1996, "Fast Noise Variance Estimation").

    The 3x3 mask below is the difference of two Laplacians, so it cancels
    smooth image structure and mostly responds to pixel-level noise.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    resp = cv2.filter2D(gray, -1, mask)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(resp).mean() / 6.0)


def denoise(img, info, p: PipelineParams):
    """Non-local means, weighted towards regions that were dark.

    Brightening multiplies sensor noise by the same gain as the signal, so
    the shadows get grainy while bright regions stay clean. NLM averages
    similar-looking patches from across the image. Its strength comes from
    the noise level estimated on this image, and the result is blended in
    per pixel: fully where the original was dark, little where it was bright.
    """
    sigma = estimate_noise(img)
    info["noise_sigma"] = sigma
    h = max(1.0, sigma * 255.0 * p.denoise_strength)
    den = cv2.fastNlMeansDenoisingColored(to_u8(img), None, h, h, 7, 21)
    w = np.power(1.0 - info["illumination"], p.denoise_dark_power)[..., None]
    return w * to_float(den) + (1.0 - w) * img


# --- Stage 4 -----------------------------------------------------------------

def local_contrast(img, info, p: PipelineParams):
    """Gentle CLAHE on the lightness (L) channel of LAB only.

    Restores the depth that brightening flattens out. Working on L alone
    leaves the colour channels untouched, so no colour shifts.
    """
    return to_float(clahe_enhance(to_u8(img), p.clahe_clip, p.clahe_tiles))


# --- Stage 5 -----------------------------------------------------------------

def colour_correct(img, info, p: PipelineParams):
    """Gray-world white balance, then a mild saturation boost.

    Gray-world assumes the scene averages to neutral grey, so each channel is
    scaled to make the channel means equal — this removes yellow/blue casts
    from indoor or street lighting. Applied at partial strength because the
    assumption fails for scenes dominated by one colour (a sunset would turn
    grey), so the gain can also be capped with wb_max_gain. Saturation is then
    scaled in HSV (tuning on LOL picked 0.9, i.e. slightly muted: the LOL
    reference images are less saturated than the brightened output).
    """
    means = img.reshape(-1, 3).mean(axis=0)
    gains = means.mean() / np.maximum(means, 1e-6)
    gains = 1.0 + p.white_balance_strength * (gains - 1.0)
    if p.wb_max_gain:
        gains = np.clip(gains, 1.0 / p.wb_max_gain, p.wb_max_gain)
    out = np.clip(img * gains, 0.0, 1.0)

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * p.saturation, 0.0, 1.0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# --- Stage 6 -----------------------------------------------------------------

def sharpen(img, info, p: PipelineParams):
    """Unsharp masking: out = img + amount * (img - blurred).

    (img - blurred) is the high-frequency detail; adding some back recovers
    crispness lost to denoising.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), p.sharpen_sigma)
    return np.clip(img + p.sharpen_amount * (img - blurred), 0.0, 1.0)
