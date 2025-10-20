FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WORKDIR=/workspace

WORKDIR ${WORKDIR}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    flask gunicorn fastapi uvicorn[standard] \
    numpy pandas pillow tqdm pydantic \
    requests aiohttp \
    scikit-learn \
    faiss-cpu \
    open-clip-torch \
    transformers \
    "uvicorn[standard]"

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio

COPY . ${WORKDIR}

ENV PORT=8000
EXPOSE 8000

CMD exec gunicorn --workers=2 --threads=4 --timeout=120 --bind 0.0.0.0:${PORT} server:app
