#!/usr/bin/env python3
"""Frozen benchmark for ispizza — one number, run on demand.

Scores every image in the claws-sourced dataset through the LIVE /detect endpoint
at a low scan threshold (so a "miss" is a real 0.000, not a thresholding artefact),
computes the headline metrics, writes bench_data.json for the public /bench page,
and diffs against a frozen baseline so any change to the model or the serving path
shows up as a number instead of prose.

  python3 bench.py            # run + compare to bench_baseline.json (exit 1 on drift)
  python3 bench.py --freeze   # run + write bench_baseline.json (accept current behaviour)
  python3 bench.py --thumbs   # also regenerate bench_assets/*.webp

Ground truth: `expected` = "should the detector fire on this image?"
  pizza      -> True    real photograph of real pizza
  nearpizza  -> False   real photograph of a look-alike (hard negative)
  depiction  -> False   pizza-shaped pixels, no pizza present (package art, closed boxes)
  poisoned   -> False   filename claims pizza, image is not
  synthetic  -> EXCLUDED from every score; AI-generated, reported separately
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "dataset" / "manifest.json"
BENCH = ROOT / "bench"          # mounted read-only into the container at /app/bench
OUT = BENCH / "data.json"       # served at /bench_data.json
BASELINE = BENCH / "baseline.json"
THUMBS = BENCH / "assets"       # served at /bench_assets/
API = "http://localhost:20070/detect"
UA = {"User-Agent": "ispizza-benchmark/1.0 (https://ispizza.sebland.com; sammcgrail@gmail.com) python-httpx"}

SCAN_THRESHOLD = 0.05   # low, so a 0.000 is a genuine blindness and not a cut-off
SHIP_THRESHOLD = 0.35   # what the app ships with
TOLERANCE = 0.02        # per-image score drift that counts as a regression

EXPECTED = {"pizza": True, "nearpizza": False, "depiction": False, "poisoned": False}

CAPTIONS = {
    "margherita-plate": "Margherita on a plate",
    "margherita-sf": "Margherita, San Francisco",
    "margherita-wiki": "Margherita, Naples",
    "slices-two": "Two slices",
    "slices-2": "Two slices, boxed",
    "slices-white-supreme": "White slice + supreme slice",
    "deepdish": "Deep dish",
    "deepdish-giordanos": "Giordano's deep dish",
    "chicago-deepdish": "Chicago deep dish",
    "detroit-rectangular": "Detroit style (rectangular)",
    "white-prosciutto": "Prosciutto bianca (no red sauce)",
    "white-cheese": "White cheese pizza (no red sauce)",
    "busy-party-table": "Pizza bagels on a busy party table",
    "beer-pizza-bg": "Pizza behind a beer glass, out of focus",
    "hp-crusts-only": "Pizza crusts, toppings eaten off",
    "hp-dark-blurred": "Dark, motion-blurred, half out of frame",
    "rev-in-delivery-box": "Real pizza in a delivery box",
    "rev-held-mid-bite": "Held mid-bite, occluded by a hand",
    "rev-broccoli-no-red": "Broccoli pizza (no red sauce)",
    "quiche-lorraine": "Quiche Lorraine",
    "quiche-lorraine-04": "Quiche Lorraine",
    "quiche-plumart": "Quiche Lorraine, patisserie",
    "quiche-2009": "Quiche Lorraine, sliced",
    "flammkuchen": "Flammkuchen",
    "t2-flammkuchen-els": "Alsatian flammkuchen",
    "t2-tarte-flambee": "Tarte flambée, Strasbourg",
    "tarte-bavaria": "Tarte flambée, Bavaria",
    "focaccia-tomato": "Tomato and olive focaccia",
    "farinata-focaccia": "Farinata and focaccia",
    "galette-apple": "Apple galette (sweet, no cheese)",
    "manakish-zaatar": "Manakish za'atar",
    "t1-lahmacun": "Lahmacun",
    "t1-lahmacun-acili": "Acılı lahmacun",
    "mini-lahmacun-pide": "Mini lahmacun and pide",
    "t3-khachapuri": "Khachapuri, Mingrelian (round)",
    "khachapuri-adjaruli-BOAT": "Khachapuri, Adjaruli (boat-shaped)",
    "t4-plain-flatbread": "Belokranjska pogača (cheese-dusted)",
    "BARE-flatbread": "Bare herbed flatbread, no toppings",
    "BARE-pita": "Bare pita, no toppings",
    "garlic-naan": "Garlic naan",
    "pissaladiere": "Pissaladière",
    "tomato-tart": "Tomato tart",
    "bruschetta": "Bruschetta",
    "fruit-tart": "Fruit tart",
    "tarte-tatin": "Tarte tatin",
    "closed-box": "Closed pizza boxes, no pizza visible",
    "cart-of-boxes": "Delivery cart of closed boxes",
    "freezer-aisle-packageart": "Freezer aisle — every pizza is printed package art",
    "toastie-mislabelled": "Filed as “woman eating a slice of pizza”. It is a folded toastie.",
    "ai-margherita": "AI-generated margherita",
    "bundled-sample": "The app's own bundled sample image",
}


def commons_page(src: str) -> str:
    """upload.wikimedia.org/.../Foo.jpg -> the Commons File: page (for attribution)."""
    if "upload.wikimedia.org" not in src:
        return ""
    return "https://commons.wikimedia.org/wiki/File:" + src.rsplit("/", 1)[-1]


def ensure_local(row: dict) -> Path:
    """Images are gitignored (bulk CC media); re-fetch from source when missing."""
    p = Path(row["file"])
    if p.exists():
        return p
    src = row["source"]
    if src.startswith("/"):
        raise FileNotFoundError(f"{p} missing and its source {src} is a local path")
    p.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        r = httpx.get(src, headers=UA, timeout=60, follow_redirects=True)
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        im.thumbnail((1600, 1600))
        im.save(p, "JPEG", quality=92)
        return p
    raise RuntimeError(f"rate-limited fetching {src} (NOT a detector miss)")


def score_one(path: Path) -> dict:
    with open(path, "rb") as fh:
        r = httpx.post(API, files={"file": (path.name, fh, "image/jpeg")},
                       params={"threshold": SCAN_THRESHOLD}, timeout=300)
    r.raise_for_status()
    d = r.json()
    dets = d.get("detections", [])
    return {
        "score": round(float(dets[0]["confidence"]), 4) if dets else 0.0,
        "n_det": len(dets),
        "elapsed_ms": int(d.get("elapsed_ms", 0)),
        "dims": f"{d.get('image_width')}x{d.get('image_height')}",
    }


def auroc(pos: list, neg: list) -> float:
    """Mann-Whitney U with tie correction. Threshold-free separability: 0.5 = coin flip."""
    if not pos or not neg:
        return float("nan")
    pairs = 0.0
    for a in pos:
        for b in neg:
            pairs += 1.0 if a > b else (0.5 if a == b else 0.0)
    return pairs / (len(pos) * len(neg))


def accuracy_at(rows: list, t: float) -> tuple:
    tp = sum(1 for r in rows if r["expected"] and r["score"] >= t)
    tn = sum(1 for r in rows if not r["expected"] and r["score"] < t)
    fp = sum(1 for r in rows if not r["expected"] and r["score"] >= t)
    fn = sum(1 for r in rows if r["expected"] and r["score"] < t)
    return (tp + tn) / len(rows), tp, tn, fp, fn


def build(thumbs: bool) -> dict:
    manifest = json.loads(MANIFEST.read_text())
    images = []
    for row in manifest:
        label = row["label"]
        try:
            path = ensure_local(row)
            res = score_one(path)
        except Exception as e:
            print(f"  FAIL {row['slug']}: {type(e).__name__}: {e}", flush=True)
            continue
        rec = {
            "slug": row["slug"],
            "label": label,
            "submitter": row["submitter"],
            "expected": EXPECTED.get(label),
            "caption": CAPTIONS.get(row["slug"], row["slug"].replace("-", " ")),
            "source": commons_page(row["source"]),
            "thumb": f"bench_assets/{label}__{row['submitter']}__{row['slug']}.webp",
            **res,
        }
        if rec["expected"] is not None:
            rec["detected"] = rec["score"] >= SHIP_THRESHOLD
            rec["correct"] = rec["detected"] == rec["expected"]
        images.append(rec)
        if thumbs:
            THUMBS.mkdir(parents=True, exist_ok=True)
            im = Image.open(path).convert("RGB")
            im.thumbnail((360, 360))
            im.save(THUMBS / Path(rec["thumb"]).name, "WEBP", quality=80, method=5)
        flag = "" if rec.get("correct", True) else "  <-- WRONG"
        print(f"  {label:9s} {rec['score']:.4f}  {rec['slug']}{flag}", flush=True)

    scored = [r for r in images if r["expected"] is not None]
    pos = [r["score"] for r in scored if r["expected"]]
    neg = [r["score"] for r in scored if not r["expected"]]
    acc, tp, tn, fp, fn = accuracy_at(scored, SHIP_THRESHOLD)

    best_t, best_acc = SHIP_THRESHOLD, acc
    for cand in sorted({round(r["score"] + 0.0001, 4) for r in scored} | {SHIP_THRESHOLD}):
        if not 0.05 <= cand <= 0.95:
            continue
        a = accuracy_at(scored, cand)[0]
        if a > best_acc:
            best_t, best_acc = cand, a

    # The bar any detector has to clear: answering "not pizza" to everything.
    majority = sum(1 for r in scored if not r["expected"]) / len(scored)
    best_fake = max((r for r in scored if not r["expected"]), key=lambda r: r["score"])
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": "RF-DETR base (COCO-pretrained), CPU",
        "scan_threshold": SCAN_THRESHOLD,
        "ship_threshold": SHIP_THRESHOLD,
        "summary": {
            "n_images": len(images),
            "n_scored": len(scored),
            "n_positive": len(pos),
            "n_negative": len(neg),
            "n_correct": tp + tn,
            "accuracy": round(acc, 4),
            "false_positives": fp,
            "false_negatives": fn,
            "auroc": round(auroc(pos, neg), 4),
            "best_threshold": round(best_t, 4),
            "best_accuracy": round(best_acc, 4),
            "majority_baseline": round(majority, 4),
            "beats_majority": bool(best_acc > majority),
            "best_fake_slug": best_fake["slug"],
            "best_fake_score": best_fake["score"],
            "reals_at_or_below_best_fake": sum(1 for s in pos if s <= best_fake["score"]),
            "median_ms": sorted(r["elapsed_ms"] for r in images)[len(images) // 2] if images else 0,
        },
        "images": images,
    }


def compare(new: dict, old: dict) -> int:
    """Return count of regressions. Per-image drift beyond TOLERANCE or a summary change."""
    o = {r["slug"]: r for r in old["images"]}
    drift, gone = [], []
    for r in new["images"]:
        prev = o.pop(r["slug"], None)
        if prev is None:
            drift.append(f"  NEW    {r['slug']:32s}            -> {r['score']:.4f}")
            continue
        if abs(r["score"] - prev["score"]) > TOLERANCE:
            drift.append(f"  DRIFT  {r['slug']:32s} {prev['score']:.4f} -> {r['score']:.4f}")
    gone = [f"  MISSING {s}" for s in o]

    changed = [f"  {k}: {v} -> {new['summary'][k]}"
               for k, v in old["summary"].items()
               if k in new["summary"] and new["summary"][k] != v and k != "median_ms"]

    print("\n" + "=" * 62)
    if not drift and not gone and not changed:
        print("BASELINE MATCH — no drift beyond +/-%.2f, summary identical." % TOLERANCE)
        return 0
    print("REGRESSION vs bench_baseline.json")
    for line in drift + gone:
        print(line)
    if changed:
        print("  summary changes:")
        for line in changed:
            print("  " + line)
    return len(drift) + len(gone) + len(changed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true", help="write bench_baseline.json from this run")
    ap.add_argument("--thumbs", action="store_true", help="regenerate bench_assets/*.webp")
    args = ap.parse_args()

    data = build(thumbs=args.thumbs)
    BENCH.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1))
    s = data["summary"]
    print("\n" + "=" * 62)
    print(f"SCORE  {s['n_correct']}/{s['n_scored']} = {s['accuracy'] * 100:.1f}% at the shipped {SHIP_THRESHOLD} threshold")
    print(f"       {s['false_positives']} false positives, {s['false_negatives']} misses")
    print(f"AUROC  {s['auroc']:.3f}   (0.5 = coin flip; threshold-free separability)")
    print(f"BEST   {s['best_accuracy'] * 100:.1f}% at threshold {s['best_threshold']:.2f} — the ceiling for any usable threshold")
    print(f"BAR    {s['majority_baseline'] * 100:.1f}% = answering 'not pizza' to every image; "
          f"the detector {'beats' if s['beats_majority'] else 'DOES NOT BEAT'} it")
    print(f"FAKE   best fake '{s['best_fake_slug']}' {s['best_fake_score']:.4f}; "
          f"{s['reals_at_or_below_best_fake']}/{s['n_positive']} real pizzas score at or below it")
    print(f"-> {OUT}")

    if args.freeze:
        BASELINE.write_text(json.dumps(data, indent=1))
        print(f"-> froze baseline {BASELINE}")
        return 0
    if not BASELINE.exists():
        print("\nno baseline yet — run with --freeze to accept this as the reference")
        return 0
    return 1 if compare(data, json.loads(BASELINE.read_text())) else 0


if __name__ == "__main__":
    sys.exit(main())
