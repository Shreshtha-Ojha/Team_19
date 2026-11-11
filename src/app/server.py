"""Local demo: pick a test image, add fog, enhance it, and see what the detector finds.

Run from the repository root with the detection environment:
  .venv-det/bin/uvicorn src.app.server:app --port 8000
then open http://localhost:8000
"""
import base64
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from src.constellation.enhance import METHODS
from src.constellation.fog_sim import SEVERITIES, apply_fog, image_rng

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "data/raw/constellations/resplit/test"
FOG = ROOT / "data/raw/constellations/fog"
WEIGHTS = ROOT / "results/constellation/phase2/baseline/weights/best.pt"
RESULTS = ROOT / "results/constellation"
NAMES = yaml.safe_load((ROOT / "data/raw/constellations/resplit/data.yaml").read_text())["names"]

FOG_LEVELS = ["clean", *SEVERITIES]
# The few methods worth showing; keys are entries of enhance.METHODS.
SHOWN_METHODS = {
    "none": "No enhancement",
    "nlm_k3": "Denoise (non-local means)",
    "clahe": "CLAHE",
    "pipeline": "Brighten (classical pipeline)",
}

app = FastAPI()


@lru_cache(maxsize=1)
def model():
    from ultralytics import YOLO
    return YOLO(str(WEIGHTS))


def test_images() -> dict[str, Path]:
    return {p.stem: p for p in sorted((CLEAN / "images").iterdir()) if p.suffix == ".jpg"}


def ground_truth(stem: str, w: int, h: int) -> list[dict]:
    out = []
    lab = CLEAN / "labels" / f"{stem}.txt"
    for line in lab.read_text().splitlines() if lab.exists() else []:
        c, cx, cy, bw, bh = line.split()[:5]
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        out.append({"cls": NAMES[int(c)], "box": [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]})
    return out


def load_fogged(path: Path, level: str) -> np.ndarray:
    if level == "clean":
        return cv2.imread(str(path))
    saved = FOG / level / "test" / "images" / f"{path.stem}.jpg"
    if saved.exists():  # the exact images used for the reported numbers
        return cv2.imread(str(saved))
    fogged = apply_fog(cv2.imread(str(path)), SEVERITIES[level], image_rng(path.name, level, 0))
    ok, buf = cv2.imencode(".jpg", fogged, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def draw(img: np.ndarray, boxes: list[dict], colour, gain: float) -> str:
    # Screen-only boost for dark frames; capped so bright outputs are not blown out.
    g = min(gain, max(1.0, 40.0 / max(float(img.mean()), 1e-3)))
    shown = np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)
    for b in boxes:
        x0, y0, x1, y1 = (int(v) for v in b["box"])
        cv2.rectangle(shown, (x0, y0), (x1, y1), colour, 2)
        label = b["cls"] + (f" {b['conf']:.2f}" if "conf" in b else "")
        cv2.putText(shown, label, (x0 + 2, max(14, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    colour, 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


@app.get("/")
def index():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.get("/api/options")
def options():
    return {"images": list(test_images()), "fog": FOG_LEVELS, "methods": SHOWN_METHODS}


@app.get("/api/run")
def run(image: str, fog: str = "light", method: str = "nlm_k3", gain: float = 3.0, conf: float = 0.25):
    imgs = test_images()
    if image not in imgs or fog not in FOG_LEVELS or method not in SHOWN_METHODS:
        raise HTTPException(400, "unknown image, fog level or method")
    gain = float(np.clip(gain, 1, 10))
    clean = cv2.imread(str(imgs[image]))
    h, w = clean.shape[:2]
    gt = ground_truth(image, w, h)
    degraded = load_fogged(imgs[image], fog)
    enhanced = METHODS[method](degraded)
    r = model().predict(enhanced, conf=conf, verbose=False, device="cpu")[0]
    dets = sorted(({"cls": NAMES[int(c)], "conf": float(s), "box": b, "correct": False}
                   for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())),
                  key=lambda d: -d["conf"])
    # Same rule as mAP50: a detection is correct if it has the right class and
    # overlaps a not-yet-matched true box with IoU >= 0.5 (highest confidence first).
    matched = set()
    for d in dets:
        cands = [(iou(d["box"], g["box"]), i) for i, g in enumerate(gt)
                 if g["cls"] == d["cls"] and i not in matched]
        best = max(cands, default=(0.0, None))
        if best[0] >= 0.5:
            d["correct"] = True
            matched.add(best[1])
    return {
        "clean": draw(clean, gt, (80, 220, 120), gain),
        "degraded": draw(degraded, [], (0, 0, 0), gain),
        "result": draw(enhanced, dets, (0, 200, 255), gain),
        "detections": dets,
        "n_true": len(gt),
        "n_found": len(matched),
        "missed": sorted(g["cls"] for i, g in enumerate(gt) if i not in matched),
    }


def iou(a: list, b: list) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@app.get("/api/results")
def results():
    """Saved test-split numbers (mAP50), so the page shows the measured results."""
    p4 = RESULTS / "phase4/test/summary.json"
    if p4.exists():
        table = json.loads(p4.read_text())
    else:  # Phase 4 not run yet: fall back to the Phase 3 baseline numbers
        table = {"none": {lvl: json.loads((RESULTS / f"phase3/baseline_test_{lvl}.json").read_text())
                          for lvl in FOG_LEVELS}}
    rows = [{"method": SHOWN_METHODS[m], **{lvl: table[m][lvl]["mAP50"] for lvl in FOG_LEVELS}}
            for m in SHOWN_METHODS if m in table]
    return {"levels": FOG_LEVELS, "rows": rows}
