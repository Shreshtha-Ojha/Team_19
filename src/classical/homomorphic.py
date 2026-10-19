"""Homomorphic filtering (frequency domain).

Image model: I = L * R (illumination x reflectance). Taking the log turns
the product into a sum, log I = log L + log R, so the two can be separated
with a linear filter: illumination varies slowly (low frequencies) and
reflectance/detail varies quickly (high frequencies). A Gaussian
high-emphasis filter in the Fourier domain scales low frequencies by
gamma_low < 1 and high frequencies by gamma_high > 1:

    H(u, v) = (gamma_high - gamma_low) * (1 - exp(-c * D^2 / D0^2)) + gamma_low

Applied to the luminance (Y of YCrCb) only; B, G, R are then scaled by the
same ratio Y_new / Y so hue is preserved.
"""
import cv2
import numpy as np


def _high_emphasis(shape, d0: float, gamma_low: float, gamma_high: float, c: float = 1.0):
    rows, cols = shape
    u = np.fft.fftfreq(rows)[:, None] * rows
    v = np.fft.fftfreq(cols)[None, :] * cols
    d2 = u ** 2 + v ** 2
    return (gamma_high - gamma_low) * (1.0 - np.exp(-c * d2 / (d0 ** 2))) + gamma_low


def homomorphic_filter(img: np.ndarray, d0: float = 30.0, gamma_low: float = 0.5,
                       gamma_high: float = 1.5) -> np.ndarray:
    """img: uint8 BGR image."""
    if img.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {img.dtype}")
    img_f = img.astype(np.float32) / 255.0
    y = cv2.cvtColor(img_f, cv2.COLOR_BGR2YCrCb)[..., 0]

    log_y = np.log1p(y)
    spec = np.fft.fft2(log_y)
    filtered = np.real(np.fft.ifft2(spec * _high_emphasis(y.shape, d0, gamma_low, gamma_high)))
    y_new = np.expm1(filtered)

    # Stretch to [0, 1], ignoring the extreme 0.5% at each end.
    lo, hi = np.percentile(y_new, [0.5, 99.5])
    y_new = np.clip((y_new - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    ratio = y_new / np.maximum(y, 1e-3)
    out = np.clip(img_f * ratio[..., None], 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)
