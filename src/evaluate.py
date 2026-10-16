"""Evaluate classical baselines and trained checkpoints on a paired test set.

PSNR/SSIM computed with scikit-image (skimage.metrics), on uint8 [0,255]
RGB images. Writes one row per method with per-image and averaged metrics
to results/metrics.csv — every number here comes from an actual run over
the images in the given test directories, nothing is hand-entered.

Usage example:
    python src/evaluate.py \
        --low-dir data/raw/LOL/eval15/low --high-dir data/raw/LOL/eval15/high \
        --checkpoints checkpoints/unet_standard_full.pt checkpoints/unet_copyblend_full.pt \
        --out results/metrics.csv
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from classical.clahe import clahe_enhance
from classical.gamma import gamma_correct
from classical.hist_eq import hist_eq
from classical.retinex import multi_scale_retinex, single_scale_retinex
from data.lol_dataset import list_images
from model.unet import SmallUNet

CLASSICAL_METHODS = {
    "gamma": lambda img: gamma_correct(img, 2.2),
    "hist_eq": hist_eq,
    "clahe": clahe_enhance,
    "ssr": single_scale_retinex,
    "msr": multi_scale_retinex,
}


def load_pairs(low_dir: Path, high_dir: Path):
    low_files = list_images(Path(low_dir))
    high_files = {p.stem: p for p in list_images(Path(high_dir))}
    pairs = []
    for lp in low_files:
        if lp.stem not in high_files:
            raise ValueError(f"no matching high-light image for {lp.name} in {high_dir}")
        pairs.append((lp, high_files[lp.stem]))
    return pairs


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


def load_checkpoint_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    base_ch = ckpt["args"].get("base_ch", 16)
    model = SmallUNet(base_ch=base_ch)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["args"]


def eval_unet(pairs, method_name, model, device):
    rows = []
    with torch.no_grad():
        for low_path, high_path in pairs:
            low_bgr = cv2.imread(str(low_path), cv2.IMREAD_COLOR)
            high_bgr = cv2.imread(str(high_path), cv2.IMREAD_COLOR)
            low_rgb = cv2.cvtColor(low_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t = torch.from_numpy(low_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
            pred = model(t).squeeze(0).cpu().numpy().transpose(1, 2, 0)
            pred_rgb = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
            high_rgb = cv2.cvtColor(high_bgr, cv2.COLOR_BGR2RGB)
            if pred_rgb.shape != high_rgb.shape:
                raise ValueError(f"shape mismatch for {low_path}: pred {pred_rgb.shape} vs gt {high_rgb.shape}")
            p = psnr(high_rgb, pred_rgb, data_range=255)
            s = ssim(high_rgb, pred_rgb, data_range=255, channel_axis=2)
            rows.append({"method": method_name, "image": low_path.name, "psnr": p, "ssim": s})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low-dir", required=True)
    ap.add_argument("--high-dir", required=True)
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--skip-classical", action="store_true")
    ap.add_argument("--out", default="results/metrics.csv")
    args = ap.parse_args()

    pairs = load_pairs(args.low_dir, args.high_dir)
    print(f"test pairs: {len(pairs)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_rows = []
    if not args.skip_classical:
        for name, fn in CLASSICAL_METHODS.items():
            print(f"evaluating classical method: {name}")
            all_rows.extend(eval_classical(pairs, name, fn))

    for ckpt_path in args.checkpoints:
        ckpt_path = Path(ckpt_path)
        method_name = ckpt_path.stem
        print(f"evaluating checkpoint: {method_name}")
        model, _ = load_checkpoint_model(ckpt_path)
        model.to(device)
        all_rows.extend(eval_unet(pairs, method_name, model, device))

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
