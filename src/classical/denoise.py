"""Optional denoise step (bilateral or median), applied after an enhancement step."""
import cv2
import numpy as np


def bilateral_denoise(img: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    if img.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {img.dtype}")
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def median_denoise(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    if img.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {img.dtype}")
    return cv2.medianBlur(img, ksize)
