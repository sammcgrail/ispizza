#!/usr/bin/env python3
"""Is the cluttered-scene miss a KNOWLEDGE gap or a RESOLUTION gap?

The party-table image scores 0.000 full-frame. If the model simply can't
resolve the pizza after its input downscale, then cropping/tiling should
find it with NO retraining. If tiles also score 0.000, it genuinely doesn't
know the case and only training will fix it.

That distinction decides fine-tune vs augment.
"""
import io, sys
from pathlib import Path
import httpx
from PIL import Image

API = "http://localhost:20070/detect"
SRC = Path("/root/pizza/dataset/pizza__bmo__busy-party-table.jpg")


def score(im, tag):
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=95)
    buf.seek(0)
    try:
        d = httpx.post(API, files={"file": ("t.jpg", buf, "image/jpeg")}, timeout=180).json()
    except Exception as e:
        print(f"  {tag:34s} ERROR {e}")
        return 0.0
    dets = d.get("detections", [])
    top = dets[0]["confidence"] if dets else 0.0
    print(f"  {tag:34s} {im.width:5d}x{im.height:<5d} n={len(dets):<3d} top={top:.3f}")
    return top


def main():
    im = Image.open(SRC)
    W, H = im.size
    print(f"source {W}x{H}\n")

    print("baseline:")
    score(im, "full frame")

    # the tray of mini pizzas sits bottom-right
    print("\nmanual crops (bottom-right quadrant, tightening):")
    crops = {
        "bottom-right half": (W // 2, H // 2, W, H),
        "bottom-right quarter": (int(W * 0.55), int(H * 0.6), W, H),
        "tray tight": (int(W * 0.60), int(H * 0.68), int(W * 0.98), int(H * 0.99)),
    }
    best_crop = 0.0
    for tag, box in crops.items():
        best_crop = max(best_crop, score(im.crop(box), tag))

    # systematic tiling: what a real tiled-inference pass would do
    print("\nsystematic tiling (overlapping):")
    best_tile, best_tag = 0.0, ""
    for n in (2, 3):
        tw, th = W // n, H // n
        ov = 0.25
        for r in range(n):
            for c in range(n):
                x0 = max(0, int(c * tw - tw * ov)); y0 = max(0, int(r * th - th * ov))
                x1 = min(W, int((c + 1) * tw + tw * ov)); y1 = min(H, int((r + 1) * th + th * ov))
                t = im.crop((x0, y0, x1, y1))
                s = score(t, f"{n}x{n} tile r{r}c{c}")
                if s > best_tile:
                    best_tile, best_tag = s, f"{n}x{n} r{r}c{c}"

    print("\n" + "=" * 60)
    print(f"full frame          : 0.000 (known miss)")
    print(f"best manual crop    : {best_crop:.3f}")
    print(f"best tile           : {best_tile:.3f}  ({best_tag})")
    if max(best_crop, best_tile) >= 0.35:
        print("\n=> RESOLUTION GAP. The model already knows this pizza; it just")
        print("   can't see it after downscale. Tiled inference fixes it with")
        print("   ZERO retraining. Augment, don't fine-tune.")
    else:
        print("\n=> KNOWLEDGE GAP. Even at full resolution it doesn't fire.")
        print("   Only training data fixes this one.")


if __name__ == "__main__":
    main()
