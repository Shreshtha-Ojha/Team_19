"""Global histogram equalization (spatial domain)."""
import cv2
import numpy as np


def hist_eq(img: np.ndarray) -> np.ndarray:
    """Global histogram equalization, applied to the L channel in LAB space
    to equalize luminance without distorting color (hue/chroma untouched).

    img: uint8 BGR image.
    """
    if img.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {img.dtype}")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
