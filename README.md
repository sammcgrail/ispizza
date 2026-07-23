# Pizza Detector (ispizza)

Detects pizza in an image (paste / drag / upload) and draws bounding boxes on any
pizza — works on single slices and full pies. Live: https://ispizza.sebland.com

- **Model:** RF-DETR base (Roboflow, Apache-2.0), COCO-pretrained — built-in `pizza`
  class, no training.
- **Backend:** FastAPI `server.py`, CPU inference ~3-4s/image. `POST /detect`
  (multipart `file`) -> `{detections:[{label,confidence,box:[x1,y1,x2,y2]}],...}`.
- **Frontend:** self-contained `index.html` (paste/drag/upload, canvas boxes).
- **Deploy:** `docker compose up -d --build` (port 20070), Caddy reverse-proxies it.
- **Tests:** `eval.py` -> `test/report.html` (pies/slices/non-pizza). Test images are
  Creative Commons from Wikimedia Commons; sources in `test/manifest.json`.
