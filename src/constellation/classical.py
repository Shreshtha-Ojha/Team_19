"""Classical spatial-domain operators compared in Phase 4.

Every public function maps a uint8 BGR image to a uint8 BGR image:

    gamma_correct        power-law brightening
    hist_eq              global histogram equalisation on LAB L
    clahe_enhance        CLAHE on LAB L
    multi_scale_retinex  MSR (Jobson, Rahman, Woodell 1997)
    brighten_pipeline    a five-stage low-light brightening pipeline:
                         Retinex illumination map -> adaptive gamma ->
                         dark-weighted NLM -> CLAHE -> grey-world colour

estimate_noise gives the noise level used to set the NLM strength.
"""
import cv2
import numpy as np


def _check_u8(img: np.ndarray) -> None:
    if img.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {img.dtype}")


def to_float(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.astype(np.float32) / 255.0


def to_u8(img_f: np.ndarray) -> np.ndarray:
    return np.clip(img_f * 255.0 + 0.5, 0, 255).astype(np.uint8)


# --- Single operators ----------------------------------------------------------

def gamma_correct(img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """out = (in / 255) ** (1 / gamma) * 255, so gamma > 1 brightens."""
    _check_u8(img)
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
    return table[img]


def _on_lightness(img: np.ndarray, fn) -> np.ndarray:
    """Apply fn to the L channel of LAB only, so hue and chroma are unchanged."""
    _check_u8(img)
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge([fn(l), a, b]), cv2.COLOR_LAB2BGR)


def hist_eq(img: np.ndarray) -> np.ndarray:
    return _on_lightness(img, cv2.equalizeHist)


def clahe_enhance(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return _on_lightness(img, clahe.apply)


def multi_scale_retinex(img: np.ndarray, sigmas=(15.0, 80.0, 250.0)) -> np.ndarray:
    """R = log(I) - log(I * G_sigma), averaged over sigmas, then stretched per channel."""
    _check_u8(img)
    img_f = img.astype(np.float32) + 1.0
    acc = np.zeros_like(img_f)
    for sigma in sigmas:
        acc += np.log1p(img_f) - np.log1p(cv2.GaussianBlur(img_f, (0, 0), sigma))
    acc /= len(sigmas)
    out = np.zeros_like(acc)
    for c in range(acc.shape[2]):
        ch = acc[:, :, c]
        out[:, :, c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6) * 255.0
    return out.astype(np.uint8)


def estimate_noise(img: np.ndarray) -> float:
    """Noise std of a float [0, 1] BGR image (Immerkaer 1996).

    The 3x3 mask is a difference of two Laplacians: it cancels smooth image
    structure and mostly responds to pixel-level noise.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    resp = cv2.filter2D(gray, -1, mask)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(resp).mean() / 6.0)


# --- Brightening pipeline -------------------------------------------------------
# Settings were tuned earlier for ordinary low-light photos and are used
# unchanged here; Phase 4 tests whether such brightening helps the detector.
GUIDE_RADIUS = 30        # guided-filter radius at 600 px width
GUIDE_EPS = 1e-3
TARGET_BRIGHTNESS = 0.40  # mean grey level the brightened image should reach
ILLUM_FLOOR = 0.02
DENOISE_STRENGTH = 1.5    # multiplies the estimated noise level
CLAHE_CLIP = 0.75
WB_STRENGTH = 0.5         # 0 = off, 1 = full grey-world
SATURATION = 0.9


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Edge-preserving smoothing (He, Sun, Tang 2010) with a single-channel guide."""
    k = (2 * r + 1, 2 * r + 1)
    mean_i = cv2.boxFilter(guide, -1, k)
    mean_p = cv2.boxFilter(src, -1, k)
    var_i = cv2.boxFilter(guide * guide, -1, k) - mean_i * mean_i
    cov_ip = cv2.boxFilter(guide * src, -1, k) - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return cv2.boxFilter(a, -1, k) * guide + cv2.boxFilter(b, -1, k)


def _apply_gamma(img, t, gamma):
    # Replace illumination t by t**gamma; the same gain on B, G, R keeps hue.
    return np.clip(img * np.power(t, gamma - 1.0)[..., None], 0.0, 1.0)


def brighten_pipeline(img_u8: np.ndarray) -> np.ndarray:
    img = to_float(img_u8)

    # 1. Illumination map: max of B, G, R, smoothed by a guided filter.
    t0 = img.max(axis=2)
    r = max(1, round(GUIDE_RADIUS * max(img.shape[:2]) / 600))
    t = np.clip(guided_filter(t0, t0, r, GUIDE_EPS), ILLUM_FLOOR, 1.0)

    # 2. Gamma on the illumination map, gamma found by bisection so the mean
    #    brightness (measured at quarter size) reaches TARGET_BRIGHTNESS.
    small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    t_small = cv2.resize(t, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)

    def mean_after(g):
        return cv2.cvtColor(_apply_gamma(small, t_small, g), cv2.COLOR_BGR2GRAY).mean()

    lo, hi = 0.05, 1.0
    if mean_after(hi) >= TARGET_BRIGHTNESS:
        gamma = hi
    else:
        for _ in range(20):
            mid = (lo + hi) / 2
            if mean_after(mid) > TARGET_BRIGHTNESS:
                lo = mid
            else:
                hi = mid
        gamma = (lo + hi) / 2
    img = _apply_gamma(img, t, gamma)

    # 3. NLM denoising, blended in most strongly where the input was dark.
    h = max(1.0, estimate_noise(img) * 255.0 * DENOISE_STRENGTH)
    den = cv2.fastNlMeansDenoisingColored(to_u8(img), None, h, h, 7, 21)
    w = (1.0 - t)[..., None]
    img = w * to_float(den) + (1.0 - w) * img

    # 4. Gentle CLAHE on lightness.
    img = to_float(clahe_enhance(to_u8(img), CLAHE_CLIP, 8))

    # 5. Partial grey-world white balance, then a slight saturation change.
    means = img.reshape(-1, 3).mean(axis=0)
    gains = 1.0 + WB_STRENGTH * (means.mean() / np.maximum(means, 1e-6) - 1.0)
    img = np.clip(img * gains, 0.0, 1.0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * SATURATION, 0.0, 1.0)
    return to_u8(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
