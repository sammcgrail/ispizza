#!/usr/bin/env python3
"""Benchmark ispizza against a list of image URLs (sourced from the claws).

Usage:  python3 claws_bench.py urls.txt
        urls.txt lines: <url>  [| expected:yes|no]  [| note]
Outputs a markdown-ish report to stdout + writes results.json
"""
import io
import json
import sys
import time
from pathlib import Path

import httpx
from PIL import Image

API = "http://localhost:20070/detect"
# Wikimedia UA policy: they want a DESCRIPTIVE ua + contact. A generic browser
# UA from a datacenter IP gets 429'd (bmo hit the same wall curl-verifying).
UA = {"User-Agent": "ispizza-benchmark/1.0 (https://ispizza.sebland.com; sammcgrail@gmail.com) python-httpx"}
OUT = Path("/root/pizza/test/claws")
OUT.mkdir(parents=True, exist_ok=True)


def parse(line):
    parts = [p.strip() for p in line.split("|")]
    url = parts[0]
    expected, note = None, ""
    for p in parts[1:]:
        if p.lower().startswith("expected:"):
            expected = p.split(":", 1)[1].strip().lower().startswith("y")
        else:
            note = p
    return url, expected, note


def run(urls):
    results = []
    for i, (url, expected, note) in enumerate(urls, 1):
        row = {"n": i, "url": url, "expected": expected, "note": note}
        try:
            # bmo flagged wikimedia 429s — retry w/ backoff so a rate limit
            # never gets misreported as a detector miss.
            raw = None
            for attempt in range(4):
                r = httpx.get(url, headers=UA, timeout=30, follow_redirects=True)
                if r.status_code == 429:
                    wait = 5 * (attempt + 1)
                    print(f"    429 on [{i}], backing off {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                raw = r.content
                break
            if raw is None:
                raise RuntimeError("rate-limited after 4 attempts (NOT a detector miss)")
            im = Image.open(io.BytesIO(raw))
            im.verify()
            row["bytes"] = len(raw)
            p = OUT / f"img{i:02d}.{(im.format or 'jpg').lower()}"
            p.write_bytes(raw)
            row["file"] = str(p)
            row["dims"] = f"{im.width}x{im.height}"
        except Exception as e:
            row["error"] = f"fetch/decode failed: {type(e).__name__}: {e}"
            results.append(row)
            print(f"[{i}] FETCH FAIL {url} -> {row['error']}", flush=True)
            continue
        try:
            t = time.time()
            with open(p, "rb") as fh:
                resp = httpx.post(API, files={"file": (p.name, fh, "image/jpeg")}, timeout=180)
            resp.raise_for_status()
            d = resp.json()
            row["elapsed_ms"] = d.get("elapsed_ms", int((time.time() - t) * 1000))
            dets = d.get("detections", [])
            row["n_det"] = len(dets)
            row["top"] = dets[0]["confidence"] if dets else 0.0
            row["all"] = [x["confidence"] for x in dets]
            row["boxes"] = [x["box"] for x in dets]
            verdict = "PIZZA" if dets else "none"
            print(f"[{i}] {verdict:6s} top={row['top']:.3f} n={row['n_det']} {row['elapsed_ms']}ms  {note or url[:60]}", flush=True)
        except Exception as e:
            row["error"] = f"detect failed: {type(e).__name__}: {e}"
            print(f"[{i}] DETECT FAIL -> {row['error']}", flush=True)
        results.append(row)
    return results


def main():
    src = Path(sys.argv[1])
    urls = [parse(l) for l in src.read_text().splitlines() if l.strip() and not l.strip().startswith("#")]
    print(f"testing {len(urls)} images against {API}\n")
    res = run(urls)
    Path("/root/pizza/test/results.json").write_text(json.dumps(res, indent=1))

    ok = [r for r in res if "error" not in r]
    hits = [r for r in ok if r.get("n_det", 0) > 0]
    misses = [r for r in ok if r.get("n_det", 0) == 0]
    print("\n" + "=" * 60)
    print(f"tested {len(ok)} / fetched-ok, {len(res)-len(ok)} failed to fetch")
    print(f"detected pizza in {len(hits)}, missed {len(misses)}")
    if hits:
        confs = sorted(r["top"] for r in hits)
        print(f"confidence: min {confs[0]:.3f} / median {confs[len(confs)//2]:.3f} / max {confs[-1]:.3f}")
        print(f"mean latency: {sum(r['elapsed_ms'] for r in hits)//len(hits)}ms")
    # accuracy vs expectation when provided
    lab = [r for r in ok if r.get("expected") is not None]
    if lab:
        correct = sum(1 for r in lab if (r["n_det"] > 0) == r["expected"])
        print(f"labelled accuracy: {correct}/{len(lab)}")
        for r in lab:
            if (r["n_det"] > 0) != r["expected"]:
                kind = "FALSE POSITIVE" if r["n_det"] else "FALSE NEGATIVE"
                print(f"  {kind}: [{r['n']}] {r.get('note') or r['url'][:70]} (top={r.get('top',0):.3f})")


if __name__ == "__main__":
    main()
