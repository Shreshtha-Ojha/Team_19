"""Pick the pipeline's settings by trying values on LOL reference photos (our485).

This is a plain grid search: a handful of hand-chosen values per setting are
tried, and the best-scoring ones are kept. Stages are tuned
greedily in pipeline order — each stage's parameters are chosen to give the
best score on the output so far, with the earlier stages already fixed — so
every stage's outputs are computed once per candidate value, not once per
combination.

Score = PSNR + 10 * SSIM (both averaged over the images). The factor 10
puts the two on a similar scale: on LOL, a good change moves PSNR by ~1 dB
and SSIM by ~0.1. The test set (eval15) is never used here.

Usage (from src/):
    python tune_pipeline.py --low-dir ../data/raw/LOL/lol_dataset/our485/low \
        --high-dir ../data/raw/LOL/lol_dataset/our485/high --out ../results/tuned_params.json
"""
import argparse
import json
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from data.lol_dataset import load_pairs
from pipeline import stages
from pipeline.stages import PipelineParams, to_float, to_u8

# Stages grouped with the parameters tuned at that point and their candidates.
# Illumination and brightening are tuned together: brightening depends on the
# illumination map, and stage 1 alone changes nothing in the image.
SEARCH = [
    ([stages.estimate_illumination, stages.brighten], {
        "guide_radius": [5, 15, 30],
        "guide_eps": [1e-3, 1e-2],
        "target_brightness": [0.35, 0.4, 0.45, 0.5, 0.55],
    }),
    ([stages.denoise], {
        "denoise_strength": [1.0, 1.5, 2.0, 2.5, 3.0],
        "denoise_dark_power": [0.0, 0.5, 1.0],
    }),
    ([stages.local_contrast], {
        "clahe_clip": [0.75, 1.0, 1.25, 1.5, 2.0],
    }),
    ([stages.colour_correct], {
        "white_balance_strength": [0.0, 0.25, 0.5, 0.75, 1.0],
        "saturation": [0.9, 1.0, 1.1, 1.2],
    }),
    ([stages.sharpen], {
        "sharpen_amount": [0.0, 0.2, 0.4, 0.6],
    }),
]


def score(preds, gts):
    p = np.mean([psnr(g, to_u8(x), data_range=255) for x, g in zip(preds, gts)])
    s = np.mean([ssim(g, to_u8(x), data_range=255, channel_axis=2) for x, g in zip(preds, gts)])
    return p + 10 * s, p, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low-dir", required=True)
    ap.add_argument("--high-dir", required=True)
    ap.add_argument("--num-images", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="../results/tuned_params.json")
    args = ap.parse_args()

    pairs = load_pairs(args.low_dir, args.high_dir)
    rng = np.random.default_rng(args.seed)
    idx = sorted(rng.choice(len(pairs), size=min(args.num_images, len(pairs)), replace=False))
    pairs = [pairs[i] for i in idx]
    print(f"tuning on {len(pairs)} images from {args.low_dir}")

    gts = [cv2.imread(str(h)) for _, h in pairs]
    # Each entry: (current image, shared info dict) for one photo.
    state = [(to_float(cv2.imread(str(l))), {}) for l, _ in pairs]
    params = PipelineParams()
    log = []

    for fns, grid in SEARCH:
        names = list(grid)
        best = None
        for values in product(*grid.values()):
            cand = replace(params, **dict(zip(names, values)))
            outs, infos = [], []
            for img, info in state:
                info = dict(info)
                for fn in fns:
                    img = fn(img, info, cand)
                outs.append(img)
                infos.append(info)
            sc, p, s = score(outs, gts)
            print(f"  {dict(zip(names, values))}  psnr={p:.2f} ssim={s:.3f} score={sc:.2f}")
            if best is None or sc > best[0]:
                best = (sc, p, s, cand, list(zip(outs, infos)))
        sc, p, s, params, state = best
        chosen = {n: getattr(params, n) for n in names}
        print(f"stage {[f.__name__ for f in fns]} -> {chosen}  psnr={p:.2f} ssim={s:.3f}\n")
        log.append({"stages": [f.__name__ for f in fns], "chosen": chosen,
                    "search_psnr": p, "search_ssim": s})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"params": asdict(params), "per_stage": log,
                   "num_images": len(pairs), "seed": args.seed}, f, indent=2)
    print(f"tuned params -> {out}")


if __name__ == "__main__":
    main()
