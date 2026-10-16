"""Copy-Blend augmentation for paired low-light enhancement data.

Re-implemented from the method description in:
    Shyam, Sengar, Yoon, Kim, "Evaluating COPY-BLEND Augmentation for
    Low Level Vision Tasks", arXiv:2103.05889 (2021).

IMPORTANT — this is a paper-based re-implementation, NOT a port of the
authors' reference code. The code repository cited in the paper
(github.com/PS06/Copy_Blend) returned HTTP 404 when checked on 2026-10-16,
and the GitHub account PS06 has no such public repository, so the
reference implementation could not be read or matched against.

Known gaps in the paper, and the choices made here to resolve them:

1. Blend intensity. The paper's formula is
     I'_in = (1-a)*I_in + a*I_out
   where a in [0,1] is a per-pixel mask built from two controls: maximum
   patch size (MP_max) and "blend intensity" (beta in [0,1]). No numeric
   value for beta is given for the low-light experiment. `blend_intensity`
   below is left as an explicit, documented hyperparameter (default 1.0 =
   hard copy-paste, no partial blending) rather than a guessed reproduction
   of their number.

2. Patch geometry. For the low-light task the paper states: 1 square
   patch, maximum patch size 0.2 (relative to training image width/height),
   randomly positioned. Matched here as the defaults.

3. What "vice versa" means for the training target — UNRESOLVED IN THE
   PAPER. The paper says the method "copies patches from noisy images and
   blends onto a clean image and vice versa" but never states whether the
   noisy-onto-clean direction corrupts the ground-truth target used for the
   loss, or only changes which image serves as the augmented *input* while
   the ground truth stays the true clean image. I checked this directly
   (fetched the arXiv HTML and asked for the relevant passage) and no
   algorithm box or training-loss description resolves it.

   Resolution used here: the ground-truth target is ALWAYS the true,
   unmodified normal-light image. Only the input is constructed two ways:
     - mode "clean_onto_noisy" (the direction the paper's formula literally
       describes): paste a patch of the clean image into the noisy input.
     - mode "noisy_onto_clean" (the "vice versa" direction): start from the
       clean image and paste a noisy patch into it, producing a mostly-clean
       input with one locally degraded patch.
   In both modes the target is `high` untouched. This mirrors CutBlur
   (Yoo et al. 2020), which the paper itself cites approvingly and which
   keeps ground truth unmodified — corrupting the target would give the
   network an incorrect supervision signal, which is very unlikely to be
   the authors' intent even though the text does not say so explicitly.
   This is a documented design decision, not a verified reproduction.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class CopyBlendConfig:
    num_patches: int = 1
    max_patch_size: float = 0.2   # fraction of H/W, per paper's low-light setting
    blend_intensity: float = 1.0  # beta in [0,1]; NOT specified numerically in the paper
    vice_versa_prob: float = 0.5  # probability of using the "noisy onto clean" direction
    prob: float = 0.5             # probability of applying the augmentation at all


def _random_patch_bbox(h: int, w: int, max_patch_size: float, rng: np.random.Generator):
    patch_h = max(1, int(round(rng.uniform(0.05, max_patch_size) * h)))
    patch_w = max(1, int(round(rng.uniform(0.05, max_patch_size) * w)))
    y0 = int(rng.integers(0, max(1, h - patch_h + 1)))
    x0 = int(rng.integers(0, max(1, w - patch_w + 1)))
    return y0, x0, patch_h, patch_w


def copy_blend(low: np.ndarray, high: np.ndarray, cfg: CopyBlendConfig, rng: np.random.Generator):
    """Apply Copy-Blend to one paired (low-light, normal-light) image pair.

    low, high: float32 arrays, same H x W x C, same value range (e.g. [0,1]).
    Returns (input_aug, target), where target is always the unmodified
    normal-light image `high` (see module docstring, point 3).
    """
    if low.shape != high.shape:
        raise ValueError(f"low/high shape mismatch: {low.shape} vs {high.shape}")
    if rng.random() >= cfg.prob:
        return low.copy(), high.copy()

    h, w = low.shape[:2]
    vice_versa = rng.random() < cfg.vice_versa_prob
    base = high.copy() if vice_versa else low.copy()
    donor = low if vice_versa else high
    beta = cfg.blend_intensity

    for _ in range(cfg.num_patches):
        y0, x0, ph, pw = _random_patch_bbox(h, w, cfg.max_patch_size, rng)
        sl = (slice(y0, y0 + ph), slice(x0, x0 + pw))
        base[sl] = (1 - beta) * base[sl] + beta * donor[sl]

    return base, high.copy()
