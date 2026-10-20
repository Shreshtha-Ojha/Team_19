"""Zoom into a small region (e.g. a number plate) and make it as readable as possible.

Steps for the selected region:
  1. (optional) motion deblur — Wiener deconvolution with a straight-line
     blur kernel whose length and angle the user sets
  2. denoise (non-local means), then enlarge with Lanczos interpolation
  3. strong local contrast (CLAHE) and sharpening (unsharp mask)
  4. "text view": adaptive thresholding to black and white, which makes
     characters stand out from uneven lighting

No step adds information that isn't in the pixels: if a plate is only a
few pixels tall, its characters cannot be recovered by any of this.
"""
import cv2
import numpy as np

from pipeline.stages import estimate_noise, to_float, to_u8


def motion_psf(length: float, angle_deg: float) -> np.ndarray:
    """Point-spread function of straight-line motion blur: a normalised line
    `length` pixels long at `angle_deg` (0 = horizontal)."""
    length = max(1.0, float(length))
    size = int(np.ceil(length)) | 1  # odd, so the line is centred
    psf = np.zeros((size, size), np.float32)
    c = size // 2
    dx = np.cos(np.deg2rad(angle_deg)) * (length - 1) / 2
    dy = -np.sin(np.deg2rad(angle_deg)) * (length - 1) / 2
    cv2.line(psf, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)), 1.0, 1, cv2.LINE_AA)
    return psf / psf.sum()


def wiener_deblur(img: np.ndarray, psf: np.ndarray, nsr: float) -> np.ndarray:
    """Wiener deconvolution, per channel, in the frequency domain:

        F_hat = conj(H) / (|H|^2 + NSR) * G

    G: blurred image spectrum, H: blur kernel spectrum, NSR: noise-to-signal
    ratio. NSR = 0 would be plain inverse filtering, which explodes noise
    wherever H is near zero; a larger NSR is safer but deblurs less.
    The image is padded by reflection first so the FFT's wrap-around
    doesn't create ringing at the borders.
    """
    pad = psf.shape[0] * 2
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    h, w = padded.shape[:2]

    kernel = np.zeros((h, w), np.float32)
    kh, kw = psf.shape
    kernel[:kh, :kw] = psf
    kernel = np.roll(kernel, (-(kh // 2), -(kw // 2)), axis=(0, 1))  # centre at (0,0)
    H = np.fft.fft2(kernel)
    wiener = np.conj(H) / (np.abs(H) ** 2 + nsr)

    out = np.empty_like(padded)
    for c in range(padded.shape[2]):
        out[..., c] = np.real(np.fft.ifft2(np.fft.fft2(padded[..., c]) * wiener))
    return np.clip(out[pad:-pad, pad:-pad], 0.0, 1.0)


def enlarge(img: np.ndarray, target_long_side: int = 800, max_scale: float = 8.0) -> np.ndarray:
    scale = min(max_scale, target_long_side / max(img.shape[:2]))
    if scale <= 1.0:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)


def text_view(img_u8: np.ndarray) -> np.ndarray:
    """Black-and-white view via adaptive (local mean) thresholding.

    A single global threshold fails when a plate is lit unevenly (headlight
    glare on one side); comparing each pixel with its own neighbourhood
    handles that.
    """
    gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.5)
    block = max(11, (min(gray.shape) // 4) | 1)
    # C=10: a pixel must be clearly darker than its surroundings to count as ink,
    # so faint noise in flat areas stays white.
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 10)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


def zoom_region(crop_u8: np.ndarray, deblur_length: float = 0.0, deblur_angle: float = 0.0,
                deblur_nsr: float = 0.01) -> dict:
    """crop_u8: uint8 BGR region, ideally taken from the brightened (not
    denoised) image so fine detail hasn't been smoothed away yet."""
    img = to_float(crop_u8)
    if deblur_length >= 2:
        img = wiener_deblur(img, motion_psf(deblur_length, deblur_angle), deblur_nsr)

    # Denoise at the original size (deblurring amplifies noise), so the
    # sharpening below doesn't sharpen grain.
    h = max(3.0, estimate_noise(img) * 255.0 * 1.5)
    img = to_float(cv2.fastNlMeansDenoisingColored(to_u8(img), None, h, h, 5, 15))
    big = enlarge(img)

    lab = cv2.cvtColor(to_u8(big), cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab[..., 0] = clahe.apply(lab[..., 0])
    big = to_float(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))

    blurred = cv2.GaussianBlur(big, (0, 0), 2.0)
    sharp = to_u8(np.clip(big + 0.8 * (big - blurred), 0.0, 1.0))
    plain = cv2.resize(crop_u8, (sharp.shape[1], sharp.shape[0]), interpolation=cv2.INTER_NEAREST)
    return {"plain": plain, "enhanced": sharp, "text": text_view(sharp)}
