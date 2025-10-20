#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Retrieval evaluation for CLIP+FAISS.
Assumptions:
- Ground truth CSV with columns: image_path, labels (semicolon-separated)
- API /search returns JSON with a top_k list, either:
    {"results": [{"label": "blazer", "score": 0.87}, ...]}
  or:
    {"results": [{"labels": ["blazer","outerwear"], "score": 0.87}, ...]}
Adjust parse_pred_labels() to your actual response schema.

Usage:
  python benchmarks/retrieval_eval.py --endpoint http://localhost:8000/search \
      --csv ./benchmarks/ground_truth.csv --top_k 10 --mode multipart
"""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import requests

def read_gt(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            img = Path(row["image_path"]).as_posix()
            labels = [t.strip() for t in row["labels"].split(";") if t.strip()]
            rows.append((img, labels))
    return rows

def post_image(endpoint, img_path, top_k=10, mode="multipart"):
    if mode == "multipart":
        with open(img_path, "rb") as f:
            files = {"image": (Path(img_path).name, f, "application/octet-stream")}
            data = {"top_k": str(top_k)}
            resp = requests.post(endpoint, files=files, data=data, timeout=120)
    else:
        payload = {"image_path": img_path, "top_k": top_k}
        resp = requests.post(endpoint, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()

def parse_pred_labels(resp_json, top_k=10):
    """
    Modify this parser to match your API schema.
    Expect a list of predicted labels (strings), length <= top_k.
    """
    items = resp_json.get("results", [])
    labels = []
    for it in items:
        if "label" in it:
            labels.append(it["label"])
        elif "labels" in it and isinstance(it["labels"], list) and it["labels"]:
            labels.append(it["labels"][0])
        if len(labels) >= top_k:
            break
    return labels

def precision_at_k(gt_set, pred_list, k):
    preds = set(pred_list[:k])
    if not preds:
        return 0.0
    return len(gt_set & preds) / min(k, len(preds))

def recall_at_k(gt_set, pred_list, k):
    if not gt_set:
        return 1.0
    preds = set(pred_list[:k])
    return len(gt_set & preds) / len(gt_set)

def micro_f1(gt_sets, pred_lists, k):
    tp = fp = fn = 0
    for gt, preds in zip(gt_sets, pred_lists):
        P = set(preds[:k])
        tp += len(P & gt)
        fp += len(P - gt)
        fn += len(gt - P)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", type=str, default="http://localhost:8000/search")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--mode", choices=["multipart", "json"], default="multipart")
    args = ap.parse_args()

    rows = read_gt(Path(args.csv))
    gt_sets = []
    pred_lists = []

    for img_path, labels in rows:
        r = post_image(args.endpoint, img_path, args.top_k, args.mode)
        preds = parse_pred_labels(r, args.top_k)
        gt_sets.append(set(labels))
        pred_lists.append(preds)

    
    ks = [1, 3, 5, args.top_k]
    metrics = {"precision@k": {}, "recall@k": {}}
    for k in ks:
        p = np.mean([precision_at_k(g, p, k) for g, p in zip(gt_sets, pred_lists)])
        r = np.mean([recall_at_k(g, p, k) for g, p in zip(gt_sets, pred_lists)])
        metrics["precision@k"][k] = round(float(p), 4)
        metrics["recall@k"][k] = round(float(r), 4)

    f1 = micro_f1(gt_sets, pred_lists, args.top_k)
    out = {
        "count": len(rows),
        "top_k": args.top_k,
        "precision@k": metrics["precision@k"],
        "recall@k": metrics["recall@k"],
        "micro_f1@k": round(float(f1), 4),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
