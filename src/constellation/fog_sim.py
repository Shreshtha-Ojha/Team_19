"""Simulate fog and low light on night-sky images.

The model works in linear light (sRGB decoded) and follows the atmospheric
scattering model used for haze, I = J*t + A*(1 - t), with three additions
that matter for a night sky:

  1. Patchy fog: the optical depth varies smoothly across the frame, so the
     transmission t(x) = exp(-beta * (1 + patch * F(x))) is not constant.
  2. Scattered starlight: light removed from the direct beam reappears as a
     wide glow around bright sources (Moon, bright stars), and the direct
     beam itself is slightly blurred.
  3. Low light: the scene is scaled by an exposure factor, then sensor noise
     is added (Poisson shot noise for a given full-well count, plus Gaussian
     read noise), and the result is quantised to 8 bits.

Each image gets its own random generator, seeded from the file name, the
severity and --seed, so every run produces identical images.

Labels are copied unchanged: fog does not move the constellations.

Example:
  python -m src.constellation.fog_sim --src data/raw/constellations/resplit \
      --split test --out data/raw/constellations/fog
"""
import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class FogParams:
    beta: float        # mean optical depth; t = exp(-beta) on average
    patch: float       # relative spread of optical depth across the frame (0 = uniform)
    airlight: float    # fog brightness in linear light (0..1)
    blur: float        # Gaussian sigma (px) of the direct beam
    glow: float        # fraction of scattered light that reappears as a halo
    glow_sigma: float  # Gaussian sigma (px) of the halo
    exposure: float    # scene scale for low light (1 = unchanged)
    full_well: float   # photons at linear value 1 after exposure; lower = more shot noise
    read_noise: float  # Gaussian read noise, std in linear light


SEVERITIES = {
    "light":  FogParams(beta=0.4, patch=0.3, airlight=0.010, blur=0.6, glow=0.15,
                        glow_sigma=12, exposure=0.6, full_well=4000, read_noise=0.0010),
    "medium": FogParams(beta=0.9, patch=0.4, airlight=0.018, blur=1.0, glow=0.20,
                        glow_sigma=16, exposure=0.4, full_well=2000, read_noise=0.0015),
    "heavy":  FogParams(beta=1.6, patch=0.5, airlight=0.028, blur=1.5, glow=0.25,
                        glow_sigma=20, exposure=0.25, full_well=1000, read_noise=0.0020),
}

# One component of "light" at a time, to see which part hurts detection.
# "identity" passes images through the same decode/encode path with no
# degradation, as a control for the pipeline itself.
_OFF = dict(beta=0.0, patch=0.0, airlight=0.0, blur=0.0, glow=0.0, glow_sigma=1.0,
            exposure=1.0, full_well=1e9, read_noise=0.0)
_L = SEVERITIES["light"]
ABLATIONS = {
    "identity": FogParams(**_OFF),
    "light_fog_only": FogParams(**{**_OFF, "beta": _L.beta, "patch": _L.patch, "airlight": _L.airlight,
                                   "blur": _L.blur, "glow": _L.glow, "glow_sigma": _L.glow_sigma}),
    "light_dark_only": FogParams(**{**_OFF, "exposure": _L.exposure}),
    "light_noise_only": FogParams(**{**_OFF, "full_well": _L.full_well, "read_noise": _L.read_noise}),
}
ALL_PARAMS = {**SEVERITIES, **ABLATIONS}

# Light-pollution tint of the airlight, BGR: slightly warm grey.
AIRLIGHT_TINT_BGR = np.array([0.85, 0.95, 1.0], dtype=np.float32)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def image_rng(name: str, severity: str, seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{severity}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def smooth_field(h: int, w: int, rng: np.random.Generator, cell: int = 80) -> np.ndarray:
    """Smooth random field in [-1, 1] with features about `cell` pixels wide."""
    gh, gw = max(2, h // cell + 2), max(2, w // cell + 2)
    coarse = rng.standard_normal((gh, gw)).astype(np.float32)
    field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), cell / 3)
    return field / (np.abs(field).max() + 1e-8)


def blur(x: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(x, (0, 0), sigma) if sigma > 0 else x


def apply_fog(img_bgr: np.ndarray, p: FogParams, rng: np.random.Generator) -> np.ndarray:
    """uint8 BGR in, uint8 BGR out."""
    if img_bgr.dtype != np.uint8 or img_bgr.ndim != 3:
        raise ValueError("expected a uint8 BGR image")
    h, w = img_bgr.shape[:2]
    lin = srgb_to_linear(img_bgr.astype(np.float32) / 255.0).astype(np.float32)

    depth = p.beta * (1.0 + p.patch * smooth_field(h, w, rng))
    t = np.exp(-np.maximum(depth, 0.0))[..., None]

    direct = t * blur(lin, p.blur)
    halo = p.glow * (1.0 - t) * blur(lin, p.glow_sigma)
    air = p.airlight * AIRLIGHT_TINT_BGR * (1.0 - t)
    scene = p.exposure * (direct + halo + air)

    photons = rng.poisson(np.clip(scene, 0, None) * p.full_well).astype(np.float32)
    noisy = photons / p.full_well + rng.normal(0.0, p.read_noise, scene.shape).astype(np.float32)
    return np.round(linear_to_srgb(noisy) * 255.0).astype(np.uint8)


def fog_split(src: Path, split: str, out: Path, severity: str, seed: int, quality: int) -> dict:
    p = ALL_PARAMS[severity]
    img_in, lab_in = src / split / "images", src / split / "labels"
    img_out, lab_out = out / severity / split / "images", out / severity / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)
    images = sorted(f for f in img_in.iterdir() if f.suffix.lower() in IMG_EXTS)
    for f in images:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cannot read {f}")
        fogged = apply_fog(img, p, image_rng(f.name, severity, seed))
        cv2.imwrite(str(img_out / (f.stem + ".jpg")), fogged, [cv2.IMWRITE_JPEG_QUALITY, quality])
        lab = lab_in / (f.stem + ".txt")
        if lab.exists():
            shutil.copyfile(lab, lab_out / lab.name)
    return {"images": len(images)}


def write_data_yaml(src: Path, out_dir: Path, splits: list[str]) -> None:
    cfg = yaml.safe_load((src / "data.yaml").read_text())
    new = {"path": str(out_dir.resolve()), "names": cfg["names"], "nc": cfg.get("nc", len(cfg["names"]))}
    for key, split in (("train", "train"), ("val", "valid"), ("test", "test")):
        # Splits that were not fogged point at the clean images, so the yaml always loads.
        new[key] = (f"{split}/images" if split in splits
                    else str((src / split / "images").resolve()))
    (out_dir / "data.yaml").write_text(yaml.safe_dump(new, sort_keys=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("data/raw/constellations/resplit"))
    ap.add_argument("--out", type=Path, default=Path("data/raw/constellations/fog"))
    ap.add_argument("--split", nargs="+", default=["test"], choices=("train", "valid", "test"))
    ap.add_argument("--severity", nargs="+", default=list(SEVERITIES), choices=list(ALL_PARAMS),
                    help="fog levels, or ablations that apply one part of 'light' at a time")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality of the output")
    args = ap.parse_args()

    report = {"seed": args.seed, "jpeg_quality": args.quality, "src": str(args.src), "severities": {}}
    for sev in args.severity:
        counts = {s: fog_split(args.src, s, args.out, sev, args.seed, args.quality) for s in args.split}
        write_data_yaml(args.src, args.out / sev, args.split)
        report["severities"][sev] = {"params": asdict(ALL_PARAMS[sev]), "splits": counts}
        print(sev, counts)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"fog_report_{'_'.join(args.severity)}.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
