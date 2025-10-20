#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Latency benchmark for the CLIP+FAISS API.
- Supports concurrent requests with asyncio + aiohttp
- Measures QPS, avg, p50, p95 latency
- Two request modes:
   (1) multipart/form-data with an image file field 'image'
   (2) JSON body with {'image_path': '...'} if your API supports it
Usage:
  python benchmarks/latency_bench.py --endpoint http://localhost:8000/search \
      --inputs ./benchmarks/samples --concurrency 8 --warmup 10 --iterations 100
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
import numpy as np
import aiohttp

def list_image_files(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = []
    if root.is_dir():
        for p in root.rglob("*"):
            if p.suffix.lower() in exts:
                files.append(p)
    elif root.is_file():
        files = [root]
    return files

async def post_image(session, endpoint, img_path, top_k=10, mode="multipart"):
    start = time.perf_counter()
    try:
        if mode == "multipart":
            data = aiohttp.FormData()
            data.add_field("image", open(img_path, "rb"), filename=img_path.name, content_type="application/octet-stream")
            data.add_field("top_k", str(top_k))
            async with session.post(endpoint, data=data, timeout=120) as resp:
                _ = await resp.read()
                status = resp.status
        else:
            payload = {"image_path": str(img_path), "top_k": top_k}
            async with session.post(endpoint, json=payload, timeout=120) as resp:
                _ = await resp.read()
                status = resp.status
        ok = (200 <= status < 300)
    except Exception:
        ok = False
    end = time.perf_counter()
    return ok, (end - start) * 1000.0  # ms

async def worker(name, queue, session, endpoint, top_k, mode, results):
    while True:
        img_path = await queue.get()
        if img_path is None:
            queue.task_done()
            break
        ok, latency_ms = await post_image(session, endpoint, img_path, top_k, mode)
        results.append(latency_ms if ok else None)
        queue.task_done()

async def run_bench(endpoint, inputs, concurrency, warmup, iterations, top_k, mode):
    imgs = list_image_files(Path(inputs))
    if not imgs:
        raise RuntimeError(f"No images found under {inputs}")

    # Warmup
    async with aiohttp.ClientSession() as session:
        for i in range(min(warmup, len(imgs))):
            await post_image(session, endpoint, imgs[i % len(imgs)], top_k, mode)

    # Benchmark
    q = asyncio.Queue()
    for i in range(iterations):
        q.put_nowait(imgs[i % len(imgs)])
    for _ in range(concurrency):
        q.put_nowait(None)

    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(worker(f"w{k}", q, session, endpoint, top_k, mode, results))
            for k in range(concurrency)
        ]
        t0 = time.perf_counter()
        await q.join()
        t1 = time.perf_counter()
        for t in tasks:
            await t
    # Compute stats
    lat_ms = [x for x in results if x is not None]
    errors = results.count(None)
    duration = t1 - t0
    qps = iterations / duration if duration > 0 else 0.0

    def pct(arr, p):
        return float(np.percentile(arr, p)) if arr else float("nan")

    stats = {
        "iterations": iterations,
        "concurrency": concurrency,
        "duration_sec": round(duration, 3),
        "qps": round(qps, 2),
        "avg_ms": round(float(np.mean(lat_ms)) if lat_ms else float("nan"), 2),
        "p50_ms": round(pct(lat_ms, 50), 2),
        "p95_ms": round(pct(lat_ms, 95), 2),
        "errors": errors,
        "success": len(lat_ms),
    }
    return stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", type=str, default="http://localhost:8000/search")
    ap.add_argument("--inputs", type=str, default="./benchmarks/samples")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--mode", choices=["multipart", "json"], default="multipart",
                    help="Request body mode for your API")
    ap.add_argument("--save_json", type=str, default="")
    args = ap.parse_args()

    stats = asyncio.run(run_bench(
        endpoint=args.endpoint,
        inputs=args.inputs,
        concurrency=args.concurrency,
        warmup=args.warmup,
        iterations=args.iterations,
        top_k=args.top_k,
        mode=args.mode,
    ))
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
