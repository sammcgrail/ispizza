"""Pizza detector — FastAPI service wrapping RF-DETR (COCO-pretrained, 'pizza' class).
Serves the web UI at / and a detection endpoint at /detect. CPU inference on ARM64."""
import io
import os
import time

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from rfdetr import RFDETRBase

try:
    from rfdetr.util.coco_classes import COCO_CLASSES
except Exception:  # pragma: no cover
    COCO_CLASSES = None

APP_DIR = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="Pizza Detector")

print("[pizza] loading RF-DETR base (COCO)...", flush=True)
_t0 = time.time()
model = RFDETRBase()
print(f"[pizza] model ready in {time.time() - _t0:.1f}s", flush=True)


def class_name(cid):
    cid = int(cid)
    if COCO_CLASSES is None:
        return str(cid)
    if isinstance(COCO_CLASSES, dict):
        return COCO_CLASSES.get(cid, str(cid))
    try:
        return COCO_CLASSES[cid]
    except Exception:
        return str(cid)


@app.get("/")
def index():
    return FileResponse(os.path.join(APP_DIR, "index.html"))


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/detect")
async def detect(file: UploadFile = File(...), threshold: float = Query(0.35, ge=0.05, le=0.95)):
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        return JSONResponse({"error": f"could not read image: {e}"}, status_code=400)

    t = time.time()
    try:
        det = model.predict(img, threshold=threshold)
    except Exception as e:
        return JSONResponse({"error": f"inference failed: {e}"}, status_code=500)
    elapsed = int((time.time() - t) * 1000)

    detections = []
    for cid, conf, box in zip(det.class_id, det.confidence, det.xyxy):
        if "pizza" in class_name(cid).lower():
            x1, y1, x2, y2 = (float(v) for v in box)
            detections.append({"label": "pizza", "confidence": round(float(conf), 4), "box": [x1, y1, x2, y2]})
    detections.sort(key=lambda d: -d["confidence"])

    return {
        "detections": detections,
        "image_width": img.width,
        "image_height": img.height,
        "elapsed_ms": elapsed,
    }
