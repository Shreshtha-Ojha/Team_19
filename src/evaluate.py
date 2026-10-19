"""Evaluate classical enhancement methods and the full pipeline on a paired test set.

Besides the single-operation baselines, the pipeline is scored cumulatively:
stages 1-2 only, then + denoise, + contrast, + colour (the default
pipeline), and finally + sharpen, so the summary shows what each stage adds.

PSNR/SSIM computed with scikit-image (skimage.metrics), on uint8 [0,255]
RGB images. Writes one row per method with per-image and averaged metrics
to results/metrics.csv — every number here comes from an actual run over
the images in the given test directories, nothing is hand-entered.

Usage example:
    python src/evaluate.py \
        --low-dir data/raw/LOL/eval15/low --high-dir data/raw/LOL/eval15/high \
        --out results/metrics.csv
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from classical.clahe import clahe_enhance
from classical.gamma import gamma_correct
from classical.hist_eq import hist_eq
from classical.homomorphic import homomorphic_filter
from classical.retinex import multi_scale_retinex, single_scale_retinex
from data.lol_dataset import load_pairs
from pipeline.pipeline import STAGE_NAMES, enhance
from pipeline.stages import PipelineParams

CLASSICAL_METHODS = {
    "gamma": lambda img: gamma_correct(img, 2.2),
    "hist_eq": hist_eq,
    "clahe": clahe_enhance,
    "ssr": single_scale_retinex,
    "msr": multi_scale_retinex,
    "homomorphic": homomorphic_filter,
}


def pipeline_up_to(k: int):
    enabled = set(STAGE_NAMES[:k])
    return lambda img: enhance(img, enabled=enabled, keep_steps=False).final


# Cumulative ablation: "pipeline_2_brighten" = stages 1-2, ..., "pipeline_5_colour" =
# the default pipeline. Sharpening is off by default, so its row uses an explicit
# amount (0.4) to show its effect.
PIPELINE_METHODS = {
    f"pipeline_{k}_{STAGE_NAMES[k - 1]}": pipeline_up_to(k) for k in range(2, len(STAGE_NAMES))
}
PIPELINE_METHODS["pipeline_6_sharpen_0.4"] = lambda img: enhance(
    img, PipelineParams(sharpen_amount=0.4), set(STAGE_NAMES), keep_steps=False).final


def eval_classical(pairs, method_name, fn):
    rows = []
    for low_path, high_path in pairs:
        low_bgr = cv2.imread(str(low_path), cv2.IMREAD_COLOR)
        high_bgr = cv2.imread(str(high_path), cv2.IMREAD_COLOR)
        pred_bgr = fn(low_bgr)
        if pred_bgr.shape != high_bgr.shape:
            raise ValueError(f"shape mismatch for {low_path}: pred {pred_bgr.shape} vs gt {high_bgr.shape}")
        p = psnr(high_bgr, pred_bgr, data_range=255)
        s = ssim(high_bgr, pred_bgr, data_range=255, channel_axis=2)
        rows.append({"method": method_name, "image": low_path.name, "psnr": p, "ssim": s})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low-dir", required=True)
    ap.add_argument("--high-dir", required=True)
    ap.add_argument("--out", default="results/metrics.csv")
    args = ap.parse_args()

    pairs = load_pairs(args.low_dir, args.high_dir)
    print(f"test pairs: {len(pairs)}")

    all_rows = []
    for name, fn in {**CLASSICAL_METHODS, **PIPELINE_METHODS}.items():
        print(f"evaluating method: {name}")
        all_rows.extend(eval_classical(pairs, name, fn))

    df = pd.DataFrame(all_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    summary = df.groupby("method")[["psnr", "ssim"]].mean().reset_index()
    print("\n=== Summary (mean over test set) ===")
    print(summary.to_string(index=False))

    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nPer-image metrics -> {out_path}")
    print(f"Summary metrics    -> {summary_path}")


if __name__ == "__main__":
    main()
