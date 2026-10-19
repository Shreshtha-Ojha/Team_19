"""Runs the enhancement stages in order and keeps every intermediate image.

    from pipeline.pipeline import enhance
    result = enhance(img_bgr_uint8)
    result.final            # enhanced uint8 BGR image
    result.steps            # [(stage name, uint8 BGR image), ...]
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from pipeline import stages
from pipeline.stages import PipelineParams, to_float, to_u8

# (name, function, caption shown on the webpage)
STAGES = [
    ("illumination", stages.estimate_illumination, "Finding where the light is"),
    ("brighten", stages.brighten, "Lifting the shadows"),
    ("denoise", stages.denoise, "Cleaning up grain"),
    ("contrast", stages.local_contrast, "Bringing back depth"),
    ("colour", stages.colour_correct, "Restoring natural colours"),
    ("sharpen", stages.sharpen, "Crisper details"),
]
STAGE_NAMES = [name for name, _, _ in STAGES]
# Stages 1 and 2 are the core of the method; the rest can be switched off.
REQUIRED = {"illumination", "brighten"}
# Sharpening is off by default: on LOL every sharpening amount tried lowered
# both PSNR and SSIM (results/tune.log). It stays available as an option.
DEFAULT_ENABLED = set(STAGE_NAMES) - {"sharpen"}


@dataclass
class Result:
    final: np.ndarray
    steps: list = field(default_factory=list)
    info: dict = field(default_factory=dict)


def illumination_heatmap(t: np.ndarray) -> np.ndarray:
    """Colour-mapped view of the illumination map (dark blue = dark, yellow = bright)."""
    return cv2.applyColorMap(to_u8(t), cv2.COLORMAP_INFERNO)


def enhance(img_u8: np.ndarray, params: PipelineParams | None = None,
            enabled: set[str] | None = None, keep_steps: bool = True) -> Result:
    """img_u8: uint8 BGR image. `enabled`: stage names to run (default: DEFAULT_ENABLED)."""
    params = params or PipelineParams()
    enabled = DEFAULT_ENABLED if enabled is None else set(enabled) | REQUIRED
    unknown = enabled - set(STAGE_NAMES)
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}")

    img = to_float(img_u8)
    info: dict = {}
    result = Result(final=img_u8, info=info)
    for name, fn, _ in STAGES:
        if name not in enabled:
            continue
        img = fn(img, info, params)
        if keep_steps:
            shown = illumination_heatmap(info["illumination"]) if name == "illumination" else to_u8(img)
            result.steps.append((name, shown))
    result.final = to_u8(img)
    return result
