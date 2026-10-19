"""Web demo: upload or take a dark photo, see each enhancement stage.

Run (from src/):
    uvicorn web.server:app --host 0.0.0.0 --port 8000

then open http://localhost:8000, or http://<laptop-ip>:8000 from a phone on
the same Wi-Fi.
"""
import base64
import io
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from pipeline.pipeline import STAGES, enhance

STATIC = Path(__file__).parent / "static"
MAX_SIDE = 1600          # downscale big phone photos so a request takes ~1-3s
MAX_UPLOAD = 25 * 1024 * 1024

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
    "sharpen": "Fine detail softened by noise removal is brought back.",
}

app = FastAPI(title="Low-light enhancement demo")


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


@app.post("/api/enhance")
async def api_enhance(file: UploadFile):
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image is too large (max 25 MB).")
    img = decode_upload(data)

    t0 = time.time()
    result = enhance(img)
    elapsed = time.time() - t0

    captions = {name: caption for name, _, caption in STAGES}
    return {
        "width": img.shape[1],
        "height": img.shape[0],
        "seconds": round(elapsed, 2),
        "input": to_data_url(img),
        "final": to_data_url(result.final, quality=94),
        "steps": [
            {"name": name, "title": captions[name], "explain": EXPLAIN[name], "image": to_data_url(step)}
            for name, step in result.steps
        ],
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
