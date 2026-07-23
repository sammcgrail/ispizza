#!/usr/bin/env python
"""Evaluation harness for the pizza detector.

Mirrors the production logic in server.py exactly:
  - RFDETRBase() (COCO-pretrained), loaded ONCE
  - model.predict(pil_rgb_image, threshold=0.35)
  - a detection counts as pizza iff the COCO class name contains "pizza"

Runs every image listed in test/manifest.json, prints a per-image table and
group tallies, saves box-annotated copies to test/annotated/, and writes a
dark-theme visual report to test/report.html.

Exit codes: 0 = all pies+slices detected (negatives soft-reported unless
egregious); 1 = a pie/slice was missed, or >=3 negatives false-fired.
"""
import argparse
import datetime
import html
import json
import os
import sys
import time

import numpy as np
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase

try:
    from rfdetr.util.coco_classes import COCO_CLASSES
except Exception:  # pragma: no cover
    COCO_CLASSES = None

APP_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(APP_DIR, "test")
ANNOT_DIR = os.path.join(TEST_DIR, "annotated")
REPORT = os.path.join(TEST_DIR, "report.html")
GROUP_ORDER = {"pie": 0, "slice": 1, "neg": 2}


def class_name(cid):
    """Same dict-or-list handling as server.py."""
    cid = int(cid)
    if COCO_CLASSES is None:
        return str(cid)
    if isinstance(COCO_CLASSES, dict):
        return COCO_CLASSES.get(cid, str(cid))
    try:
        return COCO_CLASSES[cid]
    except Exception:
        return str(cid)


def annotate(img, pizza, others):
    """Draw thick green boxes for pizza dets, thin gray for everything else."""
    scene = np.array(img)
    base = max(img.size) / 1000.0
    thick = max(2, round(3 * base))
    tscale = min(1.6, max(0.5, 0.55 * base))
    green, gray = sv.Color.from_hex("#22c55e"), sv.Color.from_hex("#8a93a3")
    for dets, labels, color, th in (
        (others, [f"{class_name(c)} {cf:.2f}" for c, cf in zip(others.class_id, others.confidence)], gray, max(1, thick // 2)),
        (pizza, [f"PIZZA {cf:.2f}" for cf in pizza.confidence], green, thick),
    ):
        if len(dets) == 0:
            continue
        scene = sv.BoxAnnotator(color=color, thickness=th).annotate(scene, dets)
        scene = sv.LabelAnnotator(color=color, text_color=sv.Color.BLACK,
                                  text_scale=tscale, text_thickness=max(1, th // 2)
                                  ).annotate(scene, dets, labels=labels)
    out = Image.fromarray(scene)
    if max(out.size) > 1600:  # keep the report light; inference already ran on the original
        out.thumbnail((1600, 1600))
    return out


def subset(det, idx):
    """Fresh sv.Detections from row indices (robust across supervision versions)."""
    if len(idx) == 0:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.asarray(det.xyxy, dtype=np.float32).reshape(-1, 4)[idx],
        class_id=np.asarray(det.class_id, dtype=int)[idx],
        confidence=np.asarray(det.confidence, dtype=np.float32)[idx],
    )


def main():
    ap = argparse.ArgumentParser(description="Pizza detector eval")
    ap.add_argument("--threshold", type=float, default=0.35, help="app default 0.35")
    args = ap.parse_args()

    with open(os.path.join(TEST_DIR, "manifest.json")) as f:
        manifest = json.load(f)["images"]
    manifest.sort(key=lambda m: (GROUP_ORDER.get(m["group"], 9), m["file"]))
    missing = [m["file"] for m in manifest if not os.path.exists(os.path.join(TEST_DIR, m["file"]))]
    if missing:
        sys.exit(f"manifest lists missing files: {missing}")
    os.makedirs(ANNOT_DIR, exist_ok=True)

    print(f"[eval] loading RF-DETR base (COCO)... threshold={args.threshold}", flush=True)
    t0 = time.time()
    model = RFDETRBase()
    load_s = time.time() - t0
    print(f"[eval] model ready in {load_s:.1f}s", flush=True)

    rows = []
    for m in manifest:
        img = Image.open(os.path.join(TEST_DIR, m["file"])).convert("RGB")
        t = time.time()
        det = model.predict(img, threshold=args.threshold)  # production call
        elapsed_ms = int((time.time() - t) * 1000)

        n = len(det.class_id) if det.class_id is not None else 0
        pizza_idx = [i for i in range(n) if "pizza" in class_name(det.class_id[i]).lower()]
        other_idx = [i for i in range(n) if i not in pizza_idx]
        confs = sorted((float(det.confidence[i]) for i in pizza_idx), reverse=True)
        detected = bool(pizza_idx)
        expected = m["expected"] == "pizza"
        seen = sorted(((class_name(det.class_id[i]), float(det.confidence[i])) for i in range(n)),
                      key=lambda x: -x[1])

        annotate(img, subset(det, pizza_idx), subset(det, other_idx)).save(
            os.path.join(ANNOT_DIR, m["file"]), quality=88)

        rows.append({
            "file": m["file"], "group": m["group"], "desc": m.get("desc", ""),
            "expected": m["expected"], "detected": detected,
            "top_confidence": round(confs[0], 4) if confs else None,
            "n_boxes": len(pizza_idx), "elapsed_ms": elapsed_ms,
            "pass": detected == expected, "seen": seen[:4],
        })
        mark = "PASS" if rows[-1]["pass"] else "FAIL"
        top = f"top={confs[0]:.3f}" if confs else "top=  -  "
        print(f"  {mark}  {m['file']:34s} {m['expected']:9s} -> "
              f"{'pizza' if detected else 'no-pizza':8s} {top} "
              f"boxes={len(pizza_idx)} {elapsed_ms}ms", flush=True)

    # ---- table + tallies -------------------------------------------------
    W = "{:<34} {:<6} {:<9} {:<9} {:>8} {:>7} {:>8}  {}"
    print("\n" + W.format("file", "group", "expected", "detected", "top_conf", "boxes", "ms", "model_saw(top)"))
    print("-" * 118)
    for r in rows:
        print(W.format(r["file"], r["group"], r["expected"],
                       "pizza" if r["detected"] else "no-pizza",
                       f"{r['top_confidence']:.3f}" if r["top_confidence"] else "-",
                       r["n_boxes"], r["elapsed_ms"],
                       ", ".join(f"{n} {c:.2f}" for n, c in r["seen"][:3]) or "-"))

    def tally(g):
        sub = [r for r in rows if r["group"] == g]
        return sum(r["pass"] for r in sub), len(sub)

    pies, slices, negs = tally("pie"), tally("slice"), tally("neg")
    correct = sum(r["pass"] for r in rows)
    acc = 100.0 * correct / len(rows)
    core_missed = [r["file"] for r in rows if r["group"] in ("pie", "slice") and not r["pass"]]
    fps = [r for r in rows if r["group"] == "neg" and not r["pass"]]

    print(f"\nPIES detected:            {pies[0]}/{pies[1]}")
    print(f"SLICES detected:          {slices[0]}/{slices[1]}")
    print(f"NEGATIVES rejected:       {negs[0]}/{negs[1]}")
    print(f"OVERALL accuracy:         {correct}/{len(rows)} = {acc:.1f}%")
    if core_missed:
        print(f"CORE MISSES (pie/slice): {core_missed}")
    if fps:
        print("FALSE POSITIVES: " + ", ".join(f"{r['file']} (conf {r['top_confidence']:.3f})" for r in fps))

    write_report(rows, args.threshold, load_s, (pies, slices, negs, correct, len(rows), acc))
    print(f"\n[eval] report: {REPORT}")

    if core_missed:
        print("[eval] EXIT 1 — core capability miss (pie/slice not detected)")
        sys.exit(1)
    if len(fps) >= 3:
        print("[eval] EXIT 1 — egregious false-positive rate on negatives")
        sys.exit(1)
    sys.exit(0)


def write_report(rows, threshold, load_s, tallies):
    pies, slices, negs, correct, total, acc = tallies
    chip = {"pie": "#f59e0b", "slice": "#fb923c", "neg": "#60a5fa"}
    when = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    cards = []
    for r in rows:
        ok = r["pass"]
        seen = ", ".join(f"{html.escape(n)} {c:.2f}" for n, c in r["seen"]) or "nothing above threshold"
        conf = f"{r['top_confidence']:.3f}" if r["top_confidence"] is not None else "&mdash;"
        cards.append(f"""
    <div class="card">
      <div class="imgwrap"><img loading="lazy" src="annotated/{html.escape(r['file'])}" alt="{html.escape(r['file'])}"></div>
      <div class="body">
        <div class="row1">
          <span class="chip" style="background:{chip[r['group']]}">{r['group'].upper()}</span>
          <span class="fname">{html.escape(r['file'])}</span>
          <span class="badge {'pass' if ok else 'fail'}">{'PASS' if ok else 'FAIL'}</span>
        </div>
        <div class="desc">{html.escape(r['desc'])}</div>
        <div class="kv">expected <b>{r['expected']}</b> &rarr; detected <b>{'pizza' if r['detected'] else 'no pizza'}</b>
            &nbsp;&middot;&nbsp; top conf <b>{conf}</b> &nbsp;&middot;&nbsp; {r['n_boxes']} box(es) &nbsp;&middot;&nbsp; {r['elapsed_ms']} ms</div>
        <div class="seen">model saw: {seen}</div>
      </div>
    </div>""")
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pizza Detector — Eval Report</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; min-width: 0; }}
  body {{ margin: 0; background: #0b0f14; color: #e5e9f0; font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; -webkit-text-size-adjust: 100%; }}
  header {{ padding: 22px 20px 14px; border-bottom: 1px solid #1d2530; }}
  h1 {{ margin: 0 0 4px; font-size: 21px; }} h1 span {{ color: #22c55e; }}
  .meta {{ color: #8a93a3; font-size: 13px; }}
  .tallies {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }}
  .t {{ background: #131a23; border: 1px solid #1f2937; border-radius: 10px; padding: 10px 16px; }}
  .t .n {{ font-size: 22px; font-weight: 700; }} .t .l {{ font-size: 12px; color: #8a93a3; }}
  .t.ok .n {{ color: #22c55e; }} .t.bad .n {{ color: #ef4444; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px; padding: 16px 20px 40px; }}
  .card {{ background: #131a23; border: 1px solid #1f2937; border-radius: 12px; overflow: hidden; }}
  .imgwrap {{ background: #0d1117; }} .imgwrap img {{ display: block; width: 100%; height: 260px; object-fit: contain; }}
  .body {{ padding: 10px 12px 12px; }}
  .row1 {{ display: flex; align-items: center; gap: 8px; }}
  .chip {{ color: #0b0f14; font-size: 10.5px; font-weight: 800; padding: 2px 7px; border-radius: 99px; letter-spacing: .4px; }}
  .fname {{ font-size: 12.5px; color: #c7cdd8; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }}
  .badge {{ font-size: 11.5px; font-weight: 800; padding: 3px 9px; border-radius: 6px; letter-spacing: .5px; }}
  .badge.pass {{ background: #123723; color: #22c55e; border: 1px solid #22c55e; }}
  .badge.fail {{ background: #3a1214; color: #ef4444; border: 1px solid #ef4444; }}
  .desc {{ color: #aeb6c2; font-size: 12.5px; margin-top: 6px; }}
  .kv {{ font-size: 12.5px; color: #c7cdd8; margin-top: 6px; }} .kv b {{ color: #e5e9f0; }}
  .seen {{ font-size: 11.5px; color: #7d8797; margin-top: 5px; font-family: ui-monospace, monospace; }}
</style></head><body>
<header>
  <h1><span>Pizza Detector</span> — Evaluation Report</h1>
  <div class="meta">RF-DETR base (COCO) &middot; production predicate: class name contains "pizza" &middot; threshold {threshold} &middot; model load {load_s:.1f}s &middot; {when}</div>
  <div class="tallies">
    <div class="t {'ok' if pies[0] == pies[1] else 'bad'}"><div class="n">{pies[0]}/{pies[1]}</div><div class="l">PIES detected</div></div>
    <div class="t {'ok' if slices[0] == slices[1] else 'bad'}"><div class="n">{slices[0]}/{slices[1]}</div><div class="l">SLICES detected</div></div>
    <div class="t {'ok' if negs[0] == negs[1] else 'bad'}"><div class="n">{negs[0]}/{negs[1]}</div><div class="l">NEGATIVES rejected</div></div>
    <div class="t {'ok' if correct == total else 'bad'}"><div class="n">{acc:.1f}%</div><div class="l">OVERALL ({correct}/{total})</div></div>
  </div>
</header>
<div class="grid">{''.join(cards)}
</div>
</body></html>"""
    with open(REPORT, "w") as f:
        f.write(doc)


if __name__ == "__main__":
    main()
