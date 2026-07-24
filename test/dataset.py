#!/usr/bin/env python3
"""Collect every claws-submitted image into a labelled fine-tune dataset.

Unlike claws_bench.py (which reuses imgNN names per run and clobbers), this
writes STABLE provenance-tagged filenames:  <label>__<submitter>__<slug>.jpg
and emits manifest.json with url, submitter, label, and detector score.

  labels:  pizza      = real photograph of real pizza  (positive)
           nearpizza  = real photograph of a look-alike (hard negative)
           synthetic  = AI-generated, scored separately, NEVER a training positive

Usage: python3 dataset.py [--scan-only]
"""
import io
import json
import re
import sys
import time
from pathlib import Path

import httpx
from PIL import Image

API = "http://localhost:20070/detect"
UA = {"User-Agent": "ispizza-benchmark/1.0 (https://ispizza.sebland.com; sammcgrail@gmail.com) python-httpx"}
OUT = Path("/root/pizza/dataset")
W = "https://upload.wikimedia.org/wikipedia/commons/"

# (label, submitter, slug, url_or_localpath)
MANIFEST = [
    # ---------- bmo, round 1 ----------
    ("pizza", "bmo", "margherita-plate",      W + "d/de/Margherita_pizza_on_plate.jpg"),
    ("pizza", "bmo", "margherita-sf",         W + "7/7b/Pizza_Margherita_-_San_Francisco%2C_CA.jpg"),
    ("pizza", "bmo", "slices-two",            W + "b/bc/Two_pizza_slices.jpg"),
    ("pizza", "bmo", "slices-white-supreme",  W + "8/87/White_slice_and_supreme_slice.jpg"),
    ("pizza", "bmo", "deepdish",              W + "8/85/Deep-Dish_Pizza.jpg"),
    ("pizza", "bmo", "deepdish-giordanos",    W + "4/4b/Giordano%27s_Deep_Dish_Pizza.jpg"),
    ("pizza", "bmo", "white-prosciutto",      W + "b/b2/Pizza_prosciutto_bianca.jpg"),
    ("pizza", "bmo", "white-cheese",          W + "e/eb/White_cheese_pizza.jpg"),
    ("pizza", "bmo", "busy-party-table",      W + "d/d6/Graduation_party_table_with_steel_tub_full_of_drinks%2C_decorative_personalized_frame_snacks_burger_sliders_pizza_bagels_glass_mason_jar_%2816892699357%29.jpg"),
    ("nearpizza", "bmo", "quiche-lorraine-04", W + "5/5e/Quiche_lorraine_04.jpg"),
    ("nearpizza", "bmo", "quiche-plumart",     W + "9/99/Quiche_Lorraine_-_Julien_Plumart_2025-05-12.jpg"),
    # ---------- gclaw, round 2 ----------
    ("pizza", "gclaw", "detroit-rectangular", W + "d/db/Outsiders_Pizza_Company%2C_Detroit_Style_Pizza%2C_King_Soopers%2C_Governors_Ranch.jpg"),
    ("pizza", "gclaw", "slices-2",            W + "1/1b/2_Pizza_Slices.jpg"),
    ("pizza", "gclaw", "chicago-deepdish",    W + "f/fa/Chicago_Deep_Dish_%283850674567%29.jpg"),
    ("nearpizza", "gclaw", "quiche-lorraine", W + "7/7e/Quiche_Lorraine.jpg"),
    # ---------- gclaw, round 3 (near-pizza) ----------
    ("nearpizza", "gclaw", "flammkuchen",     W + "1/14/20161004_Flammkuchen_003_%2829519056094%29.jpg"),
    ("nearpizza", "gclaw", "focaccia-tomato", W + "0/08/Tomato_and_olive_focaccia.jpg"),
    ("nearpizza", "gclaw", "galette-apple",   W + "7/70/Apple_galette_%283926990412%29.jpg"),
    ("nearpizza", "gclaw", "manakish-zaatar", W + "d/d6/Manakish_%40_Marina_Caf%C3%A9_%40_Evening_%40_Abu_Dhabi_%2816042146541%29.jpg"),
    # ---------- brendbot: AI-GENERATED, quarantined ----------
    ("synthetic", "brendbot", "ai-margherita", "/root/seb/discord-attachments/1530299636986675342.jpg"),
    # ---------- app's own bundled sample ----------
    ("synthetic", "app", "bundled-sample", "/root/pizza/pizza.png"),
]


def fetch(src: str) -> bytes:
    if src.startswith("/"):
        return Path(src).read_bytes()
    for attempt in range(4):
        r = httpx.get(src, headers=UA, timeout=45, follow_redirects=True)
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.content
    raise RuntimeError("rate-limited (NOT a detector miss)")


def main():
    scan_only = "--scan-only" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, who, slug, src in MANIFEST:
        name = f"{label}__{who}__{slug}.jpg"
        p = OUT / name
        row = {"label": label, "submitter": who, "slug": slug, "source": src, "file": str(p)}
        try:
            if not p.exists():
                raw = fetch(src)
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                im.thumbnail((1600, 1600))
                im.save(p, "JPEG", quality=92)
            im = Image.open(p)
            row["dims"] = f"{im.width}x{im.height}"
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"  FAIL {name}: {row['error']}", flush=True)
            rows.append(row)
            continue
        if not scan_only:
            try:
                with open(p, "rb") as fh:
                    d = httpx.post(API, files={"file": (name, fh, "image/jpeg")}, timeout=180).json()
                dets = d.get("detections", [])
                row["score"] = dets[0]["confidence"] if dets else 0.0
                row["n_det"] = len(dets)
            except Exception as e:
                row["error"] = f"detect: {type(e).__name__}: {e}"
        rows.append(row)
        print(f"  {label:9s} {row.get('score', float('nan')):.3f}  {slug}", flush=True)

    (OUT / "manifest.json").write_text(json.dumps(rows, indent=1))

    scored = [r for r in rows if "score" in r]
    pizza = [r for r in scored if r["label"] == "pizza"]
    near = [r for r in scored if r["label"] == "nearpizza"]
    syn = [r for r in scored if r["label"] == "synthetic"]
    print("\n" + "=" * 58)
    print(f"dataset: {len(rows)} images -> {OUT}")
    print(f"  pizza (positives)      {len(pizza)}")
    print(f"  nearpizza (hard negs)  {len(near)}")
    print(f"  synthetic (quarantine) {len(syn)}")
    if pizza and near:
        miss = [r for r in pizza if r["score"] == 0]
        fp = [r for r in near if r["score"] > 0.35]
        print(f"\nat shipped threshold 0.35:")
        print(f"  real pizza MISSED:      {len(miss)}/{len(pizza)}  {[r['slug'] for r in miss]}")
        print(f"  near-pizza FALSE POS:   {len(fp)}/{len(near)}")
        overlap = [r for r in pizza if 0 < r["score"] <= max((x["score"] for x in near), default=0)]
        print(f"  real pizza scoring at/below the best fake: {len(overlap)}/{len(pizza)}")
        print("\n  -> hard negatives to fine-tune ON (fakes it scores highest):")
        for r in sorted(near, key=lambda x: -x["score"])[:5]:
            print(f"       {r['score']:.3f}  {r['slug']}")


if __name__ == "__main__":
    main()
