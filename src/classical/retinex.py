"""Single-scale and multi-scale Retinex (SSR / MSR), spatial domain.

Standard formulation (Jobson, Rahman, Woodell 1997):
    R(x,y) = log(I(x,y)) - log(I(x,y) * G_sigma(x,y))
where G_sigma is a Gaussian kernel and * denotes convolution.
MSR averages SSR output over several sigmas, then applies simple color
restoration by rescaling to [0, 255] per channel.
"""
import cv2
import numpy as np


def _single_scale_retinex(img_f: np.ndarray, sigma: float) -> np.ndarray:
    """img_f: float32 image, any single channel or multi-channel, values > 0."""
    blurred = cv2.GaussianBlur(img_f, (0, 0), sigma)
    retinex = np.log1p(img_f) - np.log1p(blurred)
    return retinex


def _normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    for c in range(x.shape[2]):
        ch = x[:, :, c]
        mn, mx = ch.min(), ch.max()
        out[:, :, c] = (ch - mn) / (mx - mn + 1e-6) * 255.0
    return out.astype(np.uint8)


def single_scale_retinex(img: np.ndarray, sigma: float = 80.0) -> np.ndarray:
    """img: uint8 BGR image."""
    if img.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {img.dtype}")
    img_f = img.astype(np.float32) + 1.0
    retinex = _single_scale_retinex(img_f, sigma)
    return _normalize_to_uint8(retinex)


def multi_scale_retinex(img: np.ndarray, sigmas=(15.0, 80.0, 250.0)) -> np.ndarray:
    """img: uint8 BGR image. Averages SSR across the given sigma scales."""
    if img.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {img.dtype}")
    img_f = img.astype(np.float32) + 1.0
    acc = np.zeros_like(img_f)
    for sigma in sigmas:
        acc += _single_scale_retinex(img_f, sigma)
    acc /= len(sigmas)
    return _normalize_to_uint8(acc)
