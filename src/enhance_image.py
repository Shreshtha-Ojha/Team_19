"""Enhance a single dark photo.

Usage (from src/):
    python enhance_image.py dark.jpg -o enhanced.jpg                      # "natural" preset
    python enhance_image.py dark.jpg -o enhanced.jpg --preset detail      # text, plates
    python enhance_image.py dark.jpg -o enhanced.jpg --steps-dir steps/   # also save every stage
    python enhance_image.py dark.jpg -o enhanced.jpg --skip colour
    python enhance_image.py dark.jpg -o enhanced.jpg --sharpen 0.4         # override sharpening amount

Large photos are downscaled to --max-size on the long side first (default
2000px): denoising cost grows with pixel count, and phone photos are often
12MP+. Pass --max-size 0 to keep full resolution.
"""
import argparse
import time
from dataclasses import replace
from pathlib import Path

import cv2

from pipeline.pipeline import PRESETS, REQUIRED, STAGE_NAMES, enhance


def load_image(path, max_size: int):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read image: {path}")
    long_side = max(img.shape[:2])
    if max_size and long_side > max_size:
        scale = max_size / long_side
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--steps-dir", help="also save each stage's image here")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=[s for s in STAGE_NAMES if s not in REQUIRED], help="stages to switch off")
    ap.add_argument("--preset", choices=list(PRESETS), default="natural",
                    help="natural: real photos; detail: text and plates; benchmark: LOL-tuned settings")
    ap.add_argument("--sharpen", type=float, metavar="AMOUNT", help="override the sharpening amount")
    ap.add_argument("--max-size", type=int, default=2000)
    args = ap.parse_args()

    img = load_image(args.input, args.max_size)
    t0 = time.time()
    params, enabled = PRESETS[args.preset]
    enabled = set(enabled) - set(args.skip)
    if args.sharpen:
        enabled.add("sharpen")
        params = replace(params, sharpen_amount=args.sharpen)
    result = enhance(img, params, enabled, keep_steps=bool(args.steps_dir))
    print(f"enhanced {img.shape[1]}x{img.shape[0]} in {time.time() - t0:.2f}s "
          f"(gamma={result.info['gamma']:.2f})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, result.final)
    print(f"saved {args.output}")

    if args.steps_dir:
        d = Path(args.steps_dir)
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "0_input.jpg"), img)
        for i, (name, step) in enumerate(result.steps, start=1):
            cv2.imwrite(str(d / f"{i}_{name}.jpg"), step)
        print(f"saved {len(result.steps) + 1} step images to {d}/")


if __name__ == "__main__":
    main()
