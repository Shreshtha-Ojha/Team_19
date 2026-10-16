"""Gamma correction enhancement (spatial domain, per-pixel power-law)."""
import numpy as np


def gamma_correct(img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Apply gamma correction to brighten a low-light image.

    img: uint8 BGR or grayscale image, any range [0, 255].
    gamma: >1 brightens (since output = (input/255)^(1/gamma) * 255 convention
           used here so a single gamma>1 argument means "brighten").
    """
    if img.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {img.dtype}")
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return table[img]
