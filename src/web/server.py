"""Web demo: upload or take a dark photo, see each enhancement stage, then
zoom into a detail (e.g. a number plate) with optional motion deblurring.

Run (from src/):
    uvicorn web.server:app --host 0.0.0.0 --port 8000

then open http://localhost:8000, or http://<laptop-ip>:8000 from a phone on
the same Wi-Fi.
"""
import base64
import io
import time
import uuid
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from pipeline import stages
from pipeline.pipeline import PRESETS, STAGES, enhance
from pipeline.stages import to_float, to_u8
from pipeline.zoom import zoom_region

STATIC = Path(__file__).parent / "static"
MAX_SIDE = 1600          # downscale big phone photos so a request takes ~1-3s
MAX_UPLOAD = 25 * 1024 * 1024
WEB_PRESETS = ("natural", "detail")
CACHE_SIZE = 16

# Plain-English explanation shown under each stage on the page.
EXPLAIN = {
    "illumination": "A map of how much light reached each part of the scene. "
                    "Dark purple means almost no light, yellow means well lit.",
    "brighten": "Dark areas get a big boost while already-bright areas barely change, "
                "so lamps and windows don't blow out.",
    "denoise": "Brightening also brightens the camera's grain. It's smoothed away, "
               "mostly in the areas that were darkest.",
    "contrast": "Contrast is restored region by region so the picture doesn't look flat.",
    "colour": "Colour casts from indoor or street lighting are removed and faded colours restored.",
    "sharpen": "Edges are made crisper so fine detail and text stand out.",
}

app = FastAPI(title="Low-light enhancement demo")

# Recently uploaded photos (after downscaling), so changing a setting or
# zooming doesn't need the photo uploaded again. Oldest entries are dropped.
_images: "OrderedDict[str, np.ndarray]" = OrderedDict()


def remember(img: np.ndarray) -> str:
    key = uuid.uuid4().hex
    _images[key] = img
    while len(_images) > CACHE_SIZE:
        _images.popitem(last=False)
    return key


def recall(key: str) -> np.ndarray:
    if key not in _images:
        raise HTTPException(410, "This photo has expired on the server. Please upload it again.")
    _images.move_to_end(key)
    return _images[key]


def decode_upload(data: bytes) -> np.ndarray:
    """Bytes -> uint8 BGR, honouring the phone's EXIF rotation."""
    try:
        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        raise HTTPException(400, "That file doesn't look like an image.")
    pil.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def to_data_url(img: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(500, "could not encode image")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def preset_params(preset: str, brightness: float | None):
    if preset not in WEB_PRESETS:
        raise HTTPException(400, f"unknown preset: {preset}")
    params, enabled = PRESETS[preset]
    if brightness is not None:
        params = replace(params, target_brightness=float(np.clip(brightness, 0.15, 0.7)))
    return params, enabled


@app.post("/api/enhance")
async def api_enhance(file: UploadFile | None = None, id: str | None = Form(None),
                      preset: str = Form("natural"), brightness: float | None = Form(None)):
    """Either upload `file`, or pass the `id` of an earlier upload to re-run with new settings."""
    if file is not None:
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "Image is too large (max 25 MB).")
        img = decode_upload(data)
        id = remember(img)
    elif id:
        img = recall(id)
    else:
        raise HTTPException(400, "Send a photo.")

    params, enabled = preset_params(preset, brightness)
    t0 = time.time()
    result = enhance(img, params, enabled)
    elapsed = time.time() - t0

    captions = {name: caption for name, _, caption in STAGES}
    return {
        "id": id,
        "width": img.shape[1],
        "height": img.shape[0],
        "seconds": round(elapsed, 2),
        "brightness": params.target_brightness,
        "input": to_data_url(img),
        "final": to_data_url(result.final, quality=94),
        "steps": [
            {"name": name, "title": captions[name], "explain": EXPLAIN[name], "image": to_data_url(step)}
            for name, step in result.steps
        ],
    }


class ZoomRequest(BaseModel):
    id: str
    x: int
    y: int
    w: int = Field(gt=0)
    h: int = Field(gt=0)
    preset: str = "natural"
    brightness: float | None = None
    deblur_length: float = Field(0.0, ge=0, le=60)
    deblur_angle: float = Field(0.0, ge=-180, le=180)
    deblur_nsr: float = Field(0.01, gt=0, le=1)


@app.post("/api/zoom")
def api_zoom(req: ZoomRequest):
    img = recall(req.id)
    H, W = img.shape[:2]
    x0, y0 = max(0, req.x), max(0, req.y)
    x1, y1 = min(W, req.x + req.w), min(H, req.y + req.h)
    if x1 - x0 < 8 or y1 - y0 < 4:
        raise HTTPException(400, "Select a slightly bigger area.")

    # Crop from the brightened image (stages 1-2), before denoising has had a
    # chance to smooth away fine detail such as characters.
    params, _ = preset_params(req.preset, req.brightness)
    info: dict = {}
    f = stages.estimate_illumination(to_float(img), info, params)
    bright = to_u8(stages.brighten(f, info, params))
    out = zoom_region(bright[y0:y1, x0:x1], req.deblur_length, req.deblur_angle, req.deblur_nsr)
    return {k: to_data_url(v, quality=92) for k, v in out.items()}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
