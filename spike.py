"""Feasibility spike: does RF-DETR run on this box and flag pizza on slice + pie?"""
import sys, time
from PIL import Image
from rfdetr import RFDETRBase

try:
    from rfdetr.util.coco_classes import COCO_CLASSES
except Exception:
    COCO_CLASSES = None


def cname(cid):
    cid = int(cid)
    if COCO_CLASSES is None:
        return str(cid)
    if isinstance(COCO_CLASSES, dict):
        return COCO_CLASSES.get(cid, str(cid))
    try:
        return COCO_CLASSES[cid]
    except Exception:
        return str(cid)


t0 = time.time()
model = RFDETRBase()
print(f"model loaded in {time.time()-t0:.1f}s")

for path in sys.argv[1:]:
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"{path}: cannot open ({e})")
        continue
    t = time.time()
    det = model.predict(img, threshold=0.3)
    dt = (time.time() - t) * 1000
    items = [(cname(c), float(cf), [round(float(x)) for x in b])
             for c, cf, b in zip(det.class_id, det.confidence, det.xyxy)]
    pizza = [i for i in items if "pizza" in i[0].lower()]
    print(f"\n{path}  ({dt:.0f}ms, {img.width}x{img.height})")
    print(f"  all: {[(n, round(cf,2)) for n,cf,_ in items]}")
    print(f"  PIZZA: {'YES ' + str([(round(cf,2), box) for _,cf,box in pizza]) if pizza else 'no'}")
