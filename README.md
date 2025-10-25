# Clothing Multi-Label Classification & Semantic Retrieval

**TL;DR**  
Multi-label tagging with CLIP/ViT encoders and FAISS-based semantic retrieval.  
A dataset of ~**7,500 images** across dozens of categories with **balanced sampling and data validation**.  
Retrieval achieves **~89% accuracy@10**, **~0.3 s median latency**, and **~40% latency reduction** through batching and caching.  
Deployed on **AWS ECS** with a **Flask API**, including health checks, basic metrics, and reproducible configuration scripts.

---

## Overview
This project provides both **multi-label classification** (tagging multiple attributes per image) and **semantic retrieval** (finding visually or conceptually similar items).  
It uses CLIP or ViT encoders for embeddings and FAISS for vector search, targeting maintainability and transparent evaluation.

---

## Features
- **Tagging & Retrieval:** Multi-label predictions with ViT or CLIP encoders, FAISS index for semantic search.  
- **Transparent Evaluation:** Accuracy@k, mAP, and latency benchmarks with balanced sampling.  
- **Operational Reliability:** `.env` configuration, health endpoints, Docker + ECS deployment.  
- **Performance Optimizations:** Request batching, LRU caching, persisted FAISS index, warm startup.

---

## Repository Structure
├─ Dockerfile
├─ docker-compose.yml
├─ server.py                               # Flask API (/predict, /search, /healthz, /metrics)
├─ predict_vit_multilabel.py               # Single/batch inference
├─ train_vit_multilabel_pro.py             # Training script (baseline)
├─ train_vit_multilabel_pro_mixupfix.py    # Training with mixup / improved regularization
├─ train_vit_multilabel_splitroots.py      # Split-root training for structured datasets
├─ optimize_thresholds.py                  # Per-class threshold optimization (F1/mAP)
├─ eval_vit_multilabel.py                  # Evaluation: accuracy@k, mAP, latency
├─ apply_per_class_thresholds.py           # Apply optimized thresholds to predictions
├─ benchmarks/                             # Latency / throughput experiments
├─ frontend/                               # (Optional) React + TypeScript UI
└─ multi_label_dataset/                    # Dataset folders and splits (ignored by Git)