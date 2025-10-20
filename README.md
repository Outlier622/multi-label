# Clothing Multi-Label Classification & Semantic Retrieval

## Overview
CLIP + FAISS multi-label tagging and image retrieval system.
7,500-image dataset across dozens of categories.

## Features
- 89% retrieval accuracy (Top-10)
- Typical latency ~0.3 s (single instance)
- 40% median latency reduction via batching & caching

## Quick Start
```bash
git clone https://github.com/Outlier622/multi-label
docker compose up --build
