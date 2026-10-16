"""Train the SmallUNet enhancer.

Device is detected at runtime (torch.cuda.is_available()), never assumed.
On this project's development machine no CUDA GPU was found, so expect CPU.

Usage example:
    python src/train.py \
        --low-dir data/raw/LOL/our485/low --high-dir data/raw/LOL/our485/high \
        --aug copyblend --subset-fraction 0.25 --epochs 60 \
        --out checkpoints/unet_copyblend_25pct.pt
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.copy_blend import CopyBlendConfig, copy_blend
from data.lol_dataset import PairedLowLightDataset
from model.unet import SmallUNet


def standard_flip_aug(low: np.ndarray, high: np.ndarray, rng: np.random.Generator):
    if rng.random() < 0.5:
        low, high = low[:, ::-1].copy(), high[:, ::-1].copy()
    if rng.random() < 0.5:
        low, high = low[::-1, :].copy(), high[::-1, :].copy()
    return low, high


def make_augment_fn(mode: str, seed: int):
    rng = np.random.default_rng(seed)
    cb_cfg = CopyBlendConfig()

    if mode == "none":
        return None
    if mode == "standard":
        return lambda low, high: standard_flip_aug(low, high, rng)
    if mode == "copyblend":
        def fn(low, high):
            low, high = standard_flip_aug(low, high, rng)
            return copy_blend(low, high, cb_cfg, rng)
        return fn
    raise ValueError(f"unknown aug mode: {mode}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--low-dir", required=True)
    p.add_argument("--high-dir", required=True)
    p.add_argument("--aug", choices=["none", "standard", "copyblend"], default="standard")
    p.add_argument("--subset-fraction", type=float, default=1.0)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-ch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    augment = make_augment_fn(args.aug, args.seed)
    dataset = PairedLowLightDataset(
        args.low_dir, args.high_dir,
        crop_size=args.crop_size, augment=augment,
        subset_fraction=args.subset_fraction, seed=args.seed,
    )
    print(f"train pairs: {len(dataset)} (subset_fraction={args.subset_fraction})")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = SmallUNet(base_ch=args.base_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.L1Loss()

    history = []
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0
        for low, high, _ in loader:
            low, high = low.to(device), high.to(device)
            opt.zero_grad()
            pred = model(low)
            loss = loss_fn(pred, high)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        epoch_time = time.time() - t0
        avg_loss = epoch_loss / max(1, n_batches)
        history.append({"epoch": epoch, "loss": avg_loss, "epoch_time_s": epoch_time})
        print(f"epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}  time={epoch_time:.1f}s")

    total_time = time.time() - t_start
    print(f"total training time: {total_time:.1f}s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "args": vars(args),
    }, out_path)

    log_path = out_path.with_suffix(".log.json")
    with open(log_path, "w") as f:
        json.dump({"args": vars(args), "history": history, "total_time_s": total_time}, f, indent=2)

    print(f"saved checkpoint to {out_path}")
    print(f"saved training log to {log_path}")


if __name__ == "__main__":
    main()
