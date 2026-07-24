# Pizza Detector (ispizza)

Detects pizza in an image (paste / drag / upload) and draws bounding boxes on any
pizza — works on single slices and full pies. Live: https://ispizza.sebland.com

- **Model:** RF-DETR base (Roboflow, Apache-2.0), COCO-pretrained — built-in `pizza`
  class, no training.
- **Backend:** FastAPI `server.py`, CPU inference ~3-4s/image. `POST /detect`
  (multipart `file`) -> `{detections:[{label,confidence,box:[x1,y1,x2,y2]}],...}`.
  Inference runs in a threadpool, one at a time, behind a 25MB upload cap, a
  decompression-bomb guard, a 4-deep queue and a 20/min per-IP rate limit.
- **Frontend:** self-contained `index.html` (paste/drag/upload, canvas boxes) and
  `bench.html`, the public benchmark report at `/bench`.
- **Deploy:** `docker compose up -d --build` (port 20070), Caddy reverse-proxies it.
  `bench/` is bind-mounted read-only, so re-running the benchmark refreshes `/bench`
  without a rebuild.

## Benchmark

`bench.py` is the frozen regression suite — it scores every image in `dataset/` through
the live endpoint at a low scan threshold (0.05, so a 0.000 is genuine blindness rather
than a cut-off artefact) and reduces the run to one number.

```
python3 bench.py            # run + diff against bench/baseline.json (exit 1 on drift)
python3 bench.py --freeze   # accept current behaviour as the new baseline
python3 bench.py --thumbs   # also regenerate bench/assets/*.webp for the report page
```

Ground truth is "should the detector fire on this image?": `pizza` yes; `nearpizza`
(look-alikes), `depiction` (package art, closed boxes — pizza-shaped pixels, no pizza)
and `poisoned` (filename lies) no. `synthetic` (AI-generated) is excluded from every
score and reported separately. Per-image drift over ±0.02, or any change to the summary
metrics, fails the run.

Findings live in the report at `/bench`. The short version: it is a *topped baked dough*
detector, not a pizza detector. No threshold separates the classes (the best fake
outscores every real pizza), it fires 0.922 on a freezer aisle of printed package art,
and it is totally blind to a tray of pizza bagels on a busy table — including at 17
progressively tighter crops, so tiled inference cannot rescue it. Both cheap fixes are
dead by experiment; the remaining path is training data.

Test images are Creative Commons from Wikimedia Commons (sources in
`dataset/manifest.json`, image bytes gitignored — `bench.py` re-fetches any that are
missing). `dataset.py` is the collector; `eval.py` -> `test/report.html` is the older
small-set harness.
