"""CLAHE (Contrast Limited Adaptive Histogram Equalization) on the LAB L channel."""
import cv2
import numpy as np


def clahe_enhance(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """img: uint8 BGR image."""
    if img.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {img.dtype}")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
