"""Pizza detector — FastAPI service wrapping RF-DETR (COCO-pretrained, 'pizza' class).

Serves the web UI at /, the benchmark report at /bench, and detection at /detect.
CPU inference on ARM64 costs ~3.5s an image, so the serving path is built around that
cost: inference runs in a threadpool (never on the event loop), one at a time, behind
a bounded queue, a size-capped upload and a per-IP rate limit.
"""
import asyncio
import io
import os
import time
from collections import deque

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from rfdetr import RFDETRBase
from starlette.concurrency import run_in_threadpool

try:
    from rfdetr.util.coco_classes import COCO_CLASSES
except Exception:  # pragma: no cover
    COCO_CLASSES = None

APP_DIR = os.path.dirname(os.path.abspath(__file__))
# bench/ is mounted read-only from the repo, so re-running bench.py refreshes the
# public report without rebuilding the image.
BENCH_DIR = os.path.join(APP_DIR, "bench")

# --- serving limits -------------------------------------------------------
MAX_UPLOAD_BYTES = 25 * 1024 * 1024   # stop reading a body past this
MAX_PIXELS = 80_000_000               # decompression-bomb guard (PIL raises above this)
MAX_INFLIGHT = 4                      # detect requests in flight before we shed load
RATE_LIMIT = 20                       # requests per IP...
RATE_WINDOW = 60                      # ...per this many seconds
RATE_MAX_TRACKED = 4096               # cap the limiter's own memory

Image.MAX_IMAGE_PIXELS = MAX_PIXELS

app = FastAPI(title="Pizza Detector")

print("[pizza] loading RF-DETR base (COCO)...", flush=True)
_t0 = time.time()
model = RFDETRBase()
print(f"[pizza] model ready in {time.time() - _t0:.1f}s", flush=True)

_sem = asyncio.Semaphore(1)   # the model is CPU-bound on 2 cores: one at a time
_hits: dict[str, deque] = {}  # ip -> recent request timestamps
_inflight = 0                 # event loop is single-threaded, so a plain int is safe
_busy = False


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


def client_ip(request: Request):
    """The IP Caddy saw, or None for a direct hit on the port.

    Caddy APPENDS the real peer to any inbound X-Forwarded-For, so the LAST entry is
    the one it observed and a client-supplied header cannot forge it. No header at all
    means the request never went through Caddy — the high app ports are closed at the
    Hetzner cloud firewall, so that traffic is internal (bench.py, health checks) and
    is deliberately not rate limited.
    """
    xff = request.headers.get("x-forwarded-for")
    if not xff:
        return None
    return xff.split(",")[-1].strip() or None


def rate_limited(ip: str) -> bool:
    now = time.monotonic()
    if len(_hits) > RATE_MAX_TRACKED:
        for stale in [k for k, v in _hits.items() if not v or now - v[-1] > RATE_WINDOW]:
            _hits.pop(stale, None)
    q = _hits.setdefault(ip, deque())
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return True
    q.append(now)
    return False


class BadImage(Exception):
    """Undecodable upload, or one too big to be worth decoding."""

    def __init__(self, msg, status=400):
        super().__init__(msg)
        self.status = status


def _decode_and_predict(raw: bytes, threshold: float):
    """CPU-bound; ALWAYS called via run_in_threadpool so the event loop stays free.

    Decoding is in here too — a 25MB JPEG takes real CPU to turn into pixels, and
    doing that on the loop would stall every other connection just as surely as
    inference would.
    """
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Image.DecompressionBombError:
        raise BadImage("image has too many pixels", 413)
    except Exception as e:
        raise BadImage(f"could not read image: {e}", 400)

    det = model.predict(img, threshold=threshold)
    out = []
    for cid, conf, box in zip(det.class_id, det.confidence, det.xyxy):
        if "pizza" in class_name(cid).lower():
            x1, y1, x2, y2 = (float(v) for v in box)
            out.append({"label": "pizza", "confidence": round(float(conf), 4), "box": [x1, y1, x2, y2]})
    out.sort(key=lambda d: -d["confidence"])
    return out, img.width, img.height


@app.middleware("http")
async def cap_body(request: Request, call_next):
    """Reject an oversized body from its Content-Length, before Starlette parses it.

    Without this the multipart parser would happily spool the whole upload to disk
    first and only then hand us something to refuse. Chunked uploads carry no
    Content-Length and fall through to the streaming cap in /detect.
    """
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_UPLOAD_BYTES + (1 << 20):
        return JSONResponse(
            {"error": f"image too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)}MB)"},
            status_code=413)
    return await call_next(request)


@app.get("/")
def index():
    return FileResponse(os.path.join(APP_DIR, "index.html"))


@app.get("/bench")
def bench_page():
    path = os.path.join(APP_DIR, "bench.html")
    if not os.path.exists(path):
        return JSONResponse({"error": "benchmark report not built"}, status_code=404)
    return FileResponse(path)


@app.get("/bench_data.json")
def bench_data():
    path = os.path.join(BENCH_DIR, "data.json")
    if not os.path.exists(path):
        return JSONResponse({"error": "benchmark has not been run"}, status_code=404)
    return FileResponse(path, media_type="application/json",
                        headers={"Cache-Control": "public, max-age=300"})


@app.get("/health")
def health():
    return {"ok": True, "busy": _busy, "inflight": _inflight}


@app.post("/detect")
async def detect(request: Request, file: UploadFile = File(...),
                 threshold: float = Query(0.35, ge=0.05, le=0.95)):
    global _inflight, _busy

    ip = client_ip(request)
    if ip and rate_limited(ip):
        return JSONResponse({"error": "rate limit — 20 images a minute, give it a sec"},
                            status_code=429, headers={"Retry-After": str(RATE_WINDOW)})

    # Shed load before doing any work: at ~3.5s an image a queue is a liability.
    if _inflight >= MAX_INFLIGHT:
        return JSONResponse({"error": "detector is busy, try again in a moment"},
                            status_code=503, headers={"Retry-After": "5"})

    raw = bytearray()
    while True:
        chunk = await file.read(1 << 20)
        if not chunk:
            break
        raw.extend(chunk)
        if len(raw) > MAX_UPLOAD_BYTES:
            return JSONResponse(
                {"error": f"image too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)}MB)"},
                status_code=413)

    _inflight += 1
    try:
        async with _sem:
            _busy = True
            t = time.time()
            try:
                detections, width, height = await run_in_threadpool(
                    _decode_and_predict, bytes(raw), threshold)
            except BadImage as e:
                return JSONResponse({"error": str(e)}, status_code=e.status)
            except Exception as e:
                return JSONResponse({"error": f"inference failed: {e}"}, status_code=500)
            finally:
                elapsed = int((time.time() - t) * 1000)
                _busy = False
    finally:
        _inflight -= 1

    return {
        "detections": detections,
        "image_width": width,
        "image_height": height,
        "elapsed_ms": elapsed,
    }


_assets = os.path.join(BENCH_DIR, "assets")
if os.path.isdir(_assets):
    app.mount("/bench_assets", StaticFiles(directory=_assets), name="bench_assets")
else:  # keeps the route honest if the bench has never been run
    print(f"[pizza] no {_assets} — /bench_assets is unmounted", flush=True)
