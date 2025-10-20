
```markdown
# Latency Benchmark Results

## 1. Overview
This report presents latency measurements of the CLIP + FAISS API under different concurrency and optimization settings.

---

## 2. Test Environment
| Parameter | Value |
|------------|--------|
| Hardware | e.g., Intel i7-12700K @ 3.6 GHz, 32 GB RAM |
| Model | CLIP ViT-B/32 (CPU inference) |
| Endpoint | `/search` |
| Benchmark script | [`benchmarks/latency_bench.py`](latency_bench.py) |
| Image set | 50 sample images in `benchmarks/samples/` |
| Iterations | 200 |
| Concurrency | 1 / 4 / 8 threads |
| Request mode | multipart (file upload) |
| Metrics | QPS, average latency, P50, P95 |

---

## 3. Command Example
```bash
python benchmarks/latency_bench.py \
  --endpoint http://localhost:8000/search \
  --inputs ./benchmarks/samples \
  --concurrency 8 \
  --warmup 20 \
  --iterations 200 \
  --mode multipart \
  --save_json ./benchmarks/latency_c8.json
