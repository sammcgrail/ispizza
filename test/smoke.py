#!/usr/bin/env python3
"""Smoke test for the serving path — the things the benchmark can't see.

Proves the four hardening claims against a running container:
  1. /health stays instant while an inference is in flight (event loop is free)
  2. oversized uploads are refused from Content-Length, before the body is parsed
  3. undecodable bodies fail fast with 400
  4. the per-IP rate limit fires for proxied traffic, and never for internal traffic

Usage: python3 test/smoke.py [base_url]   (default http://localhost:20070)
"""
import concurrent.futures as cf
import sys
import time
from pathlib import Path

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:20070"
SAMPLE = next(Path("/root/pizza/dataset").glob("pizza__*.jpg"), None)
FAKE_CLIENT = {"X-Forwarded-For": "203.0.113.7"}   # what Caddy would send
fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}", flush=True)
    if not ok:
        fails.append(name)


def health_stays_live():
    """The whole point of moving inference into a threadpool."""
    if not SAMPLE:
        return check("health during inference", False, "no sample image on disk")
    with cf.ThreadPoolExecutor(2) as pool:
        with open(SAMPLE, "rb") as fh:
            job = pool.submit(httpx.post, f"{BASE}/detect",
                              files={"file": (SAMPLE.name, fh.read(), "image/jpeg")}, timeout=300)
            time.sleep(1.0)  # let inference get going
            t = time.perf_counter()
            h = httpx.get(f"{BASE}/health", timeout=30)
            dt = time.perf_counter() - t
            busy = h.json().get("busy")
            job.result()
    check("health responds <1s during inference", dt < 1.0, f"{dt * 1000:.0f}ms")
    check("health reports busy while inferring", busy is True, f"busy={busy}")


def oversize_refused():
    blob = b"\xff\xd8\xff\xe0" + b"\x00" * (26 * 1024 * 1024)
    r = httpx.post(f"{BASE}/detect", files={"file": ("big.jpg", blob, "image/jpeg")}, timeout=120)
    check("26MB upload -> 413", r.status_code == 413, f"got {r.status_code}")


def garbage_rejected():
    r = httpx.post(f"{BASE}/detect", files={"file": ("x.jpg", b"not an image", "image/jpeg")}, timeout=60)
    check("undecodable body -> 400", r.status_code == 400, f"got {r.status_code}")


def rate_limit():
    """Cheap: rejected-before-decode bodies cost no inference."""
    codes = []
    for _ in range(22):
        r = httpx.post(f"{BASE}/detect", files={"file": ("x.jpg", b"nope", "image/jpeg")},
                       headers=FAKE_CLIENT, timeout=60)
        codes.append(r.status_code)
    check("21st proxied request -> 429", 429 in codes, f"{codes.count(429)} of 22 limited")
    r = httpx.post(f"{BASE}/detect", files={"file": ("x.jpg", b"nope", "image/jpeg")}, timeout=60)
    check("internal traffic is never rate limited", r.status_code == 400, f"got {r.status_code}")


def bench_routes():
    for path, kind in (("/bench", "text/html"), ("/bench_data.json", "application/json")):
        r = httpx.get(f"{BASE}{path}", timeout=30)
        check(f"{path} serves {kind}", r.status_code == 200 and kind in r.headers.get("content-type", ""),
              f"{r.status_code} {r.headers.get('content-type')}")
    data = httpx.get(f"{BASE}/bench_data.json", timeout=30).json()
    thumb = data["images"][0]["thumb"]
    r = httpx.get(f"{BASE}/{thumb}", timeout=30)
    check("bench thumbnails serve", r.status_code == 200 and len(r.content) > 500,
          f"{r.status_code} {len(r.content)}B")


if __name__ == "__main__":
    print(f"smoke test -> {BASE}")
    for fn in (bench_routes, garbage_rejected, oversize_refused, health_stays_live, rate_limit):
        try:
            fn()
        except Exception as e:
            check(fn.__name__, False, f"{type(e).__name__}: {e}")
    print(("FAILED: " + ", ".join(fails)) if fails else "all smoke checks passed")
    sys.exit(1 if fails else 0)
