# Retrieval Accuracy Report

## 1. Overview
This report summarizes the evaluation results of the **CLIP + FAISS multi-label image retrieval system**.

- **Model:** openai/clip-vit-base-patch32 (or your variant)
- **Dataset:** 7,500 images across multiple clothing categories
- **Goal:** measure retrieval precision, recall, and micro-F1 at several Top-K levels

---

## 2. Experimental Setup
| Component | Details |
|------------|----------|
| Hardware | CPU / GPU model, memory size |
| Index type | FAISS IVF / HNSW / Flat (specify parameters) |
| Metric | cosine similarity |
| Embedding dimension | e.g., 512 |
| Evaluation script | [`benchmarks/retrieval_eval.py`](retrieval_eval.py) |
| Ground-truth file | [`benchmarks/ground_truth.csv`](ground_truth.csv) |

Run command:
```bash
docker compose up -d      
python benchmarks/retrieval_eval.py \
  --endpoint http://localhost:8000/search \
  --csv benchmarks/ground_truth.csv \
  --top_k 10 \
  --mode multipart
