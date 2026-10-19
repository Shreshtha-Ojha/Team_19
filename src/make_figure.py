"""Generate a side-by-side comparison figure for a few test images:
low-light input | classical methods | full pipeline | ground truth.

Usage example:
    python src/make_figure.py \
        --low-dir data/raw/LOL/eval15/low --high-dir data/raw/LOL/eval15/high \
        --num-images 4 --out results/figures/comparison.png
"""
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from classical.clahe import clahe_enhance
from classical.gamma import gamma_correct
from classical.hist_eq import hist_eq
from classical.homomorphic import homomorphic_filter
from classical.retinex import multi_scale_retinex, single_scale_retinex
from data.lol_dataset import list_images
from pipeline.pipeline import enhance

CLASSICAL_METHODS = {
    "gamma": lambda img: gamma_correct(img, 2.2),
    "hist_eq": hist_eq,
    "clahe": clahe_enhance,
    "ssr": single_scale_retinex,
    "msr": multi_scale_retinex,
    "homomorphic": homomorphic_filter,
    "pipeline (ours)": lambda img: enhance(img, keep_steps=False).final,
}



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low-dir", required=True)
    ap.add_argument("--high-dir", required=True)
    ap.add_argument("--num-images", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/figures/comparison.png")
    args = ap.parse_args()

    low_files = list_images(Path(args.low_dir))
    high_by_stem = {p.stem: p for p in list_images(Path(args.high_dir))}

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(low_files, size=min(args.num_images, len(low_files)), replace=False)

    col_names = ["low (input)"] + list(CLASSICAL_METHODS.keys()) + ["high (ground truth)"]
    n_rows = len(chosen)
    n_cols = len(col_names)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, low_path in enumerate(chosen):
        high_path = high_by_stem[low_path.stem]
        low_bgr = cv2.imread(str(low_path), cv2.IMREAD_COLOR)
        high_bgr = cv2.imread(str(high_path), cv2.IMREAD_COLOR)

        images = [cv2.cvtColor(low_bgr, cv2.COLOR_BGR2RGB)]
        for fn in CLASSICAL_METHODS.values():
            images.append(cv2.cvtColor(fn(low_bgr), cv2.COLOR_BGR2RGB))
        images.append(cv2.cvtColor(high_bgr, cv2.COLOR_BGR2RGB))

        for col, (img, name) in enumerate(zip(images, col_names)):
            ax = axes[row][col]
            ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(name, fontsize=10)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"saved figure to {out_path}")


if __name__ == "__main__":
    main()
