"""Controlled test of the motion-deblur + zoom tools on a synthetic number plate.

Real CCTV plates are usually too small to test on, and their true text is
unknown, so this draws a plate with known text, blurs it with a known motion,
adds noise, and measures how much Wiener deconvolution recovers — including
what happens when the blur direction is guessed wrong.

Usage (from src/):
    python deblur_demo.py --out ../results/figures/deblur_demo.png
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

from pipeline.stages import to_float, to_u8
from pipeline.zoom import motion_psf, wiener_deblur, zoom_region


def make_plate(text="MH12 AB 3456"):
    plate = np.full((60, 240, 3), 235, np.uint8)
    cv2.rectangle(plate, (2, 2), (237, 57), (0, 0, 0), 2)
    cv2.putText(plate, text, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (20, 20, 20), 2, cv2.LINE_AA)
    return cv2.copyMakeBorder(plate, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(60, 60, 60))


def label(img, text):
    out = cv2.copyMakeBorder(img, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(out, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=float, default=15)
    ap.add_argument("--angle", type=float, default=10)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--out", default="../results/figures/deblur_demo.png")
    args = ap.parse_args()

    clean = to_float(make_plate())
    psf = motion_psf(args.length, args.angle)
    blurred = cv2.filter2D(clean, -1, psf, borderType=cv2.BORDER_REFLECT)
    rng = np.random.default_rng(0)
    blurred = np.clip(blurred + rng.normal(0, args.noise, blurred.shape), 0, 1).astype(np.float32)

    print(f"blur: {args.length}px at {args.angle} deg, noise sigma {args.noise}")
    print(f"  blurred                     PSNR {psnr(clean, blurred, data_range=1):.2f} dB")
    best = None
    for nsr in [0.001, 0.005, 0.01, 0.03]:
        rec = wiener_deblur(blurred, psf, nsr)
        p = psnr(clean, rec, data_range=1)
        print(f"  Wiener, NSR={nsr:<6}         PSNR {p:.2f} dB")
        if best is None or p > best[0]:
            best = (p, nsr, rec)
    wrong = wiener_deblur(blurred, motion_psf(args.length, args.angle + 50), best[1])
    print(f"  Wiener, angle off by 50 deg PSNR {psnr(clean, wrong, data_range=1):.2f} dB")

    crop = (slice(30, 90), slice(30, 270))
    zoomed = zoom_region(to_u8(blurred)[crop], args.length, args.angle, best[1])
    w = zoomed["enhanced"].shape[1]
    fit = lambda im: cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1])), interpolation=cv2.INTER_NEAREST)
    fig = np.vstack([
        label(fit(to_u8(clean)[crop]), "original"),
        label(fit(to_u8(blurred)[crop]), "motion-blurred + noise"),
        label(fit(to_u8(wrong)[crop]), "deblurred with the WRONG direction"),
        label(zoomed["enhanced"], f"deblurred (NSR {best[1]}) + zoom tool"),
        label(zoomed["text"], "text view"),
    ])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
