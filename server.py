# server.py
import os, io, json
from typing import List, Optional

import numpy as np
import torch
import timm
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_tfms(img_size: int):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

class InferenceEngine:
    def __init__(self, ckpt_path: str, classes_json: str, model_name: Optional[str] = None, img_size: Optional[int] = None,
                 thresholds_csv: Optional[str] = None, global_threshold: float = 0.40):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load classes
        with open(classes_json, 'r', encoding='utf-8') as f:
            self.classes = json.load(f)
        self.num_classes = len(self.classes)
        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        ckpt_args = ckpt.get('args', {}) if isinstance(ckpt, dict) else {}
        self.model_name = model_name or ckpt_args.get('model', 'vit_base_patch16_224')
        self.img_size = img_size or ckpt_args.get('img_size', 224)
        # Build model
        self.model = timm.create_model(self.model_name, pretrained=False, num_classes=self.num_classes)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        # Transforms
        self.tfms = build_tfms(self.img_size)
        # Thresholds
        self.global_thr = float(global_threshold)
        self.per_class_thr = None
        if thresholds_csv and os.path.exists(thresholds_csv):
            import pandas as pd
            df_thr = pd.read_csv(thresholds_csv)
            self.per_class_thr = {r['Class']: float(r['Best_Threshold']) for _, r in df_thr.iterrows()}

    @torch.no_grad()
    def infer(self, pil_imgs: List[Image.Image], topk: int = 5):
        batch = torch.stack([self.tfms(im.convert('RGB')) for im in pil_imgs], dim=0).to(self.device)
        logits = self.model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()  # [B, C]
        # Apply thresholds
        if self.per_class_thr:
            thr_vec = np.array([self.per_class_thr.get(c, self.global_thr) for c in self.classes], dtype=np.float32)
            preds = (probs >= thr_vec[None, :]).astype(int)
        else:
            preds = (probs >= self.global_thr).astype(int)
        # Top-K strings
        topk_list = []
        for row in probs:
            idx = np.argsort(-row)[:topk]
            topk_list.append([{"label": self.classes[i], "prob": float(row[i])} for i in idx])
        return probs, preds, topk_list

# ---- FastAPI app ----
app = FastAPI(title="ViT Multi-Label Inference")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configure these paths for your deployment ---
CKPT_PATH = os.environ.get("CKPT_PATH", "./out_vit_multilabel_pro/best.pt")
CLASSES_JSON = os.environ.get("CLASSES_JSON", "./out_vit_multilabel_pro/classes.json")
THRESHOLDS_CSV = os.environ.get("THRESHOLDS_CSV", "./out_vit_multilabel_pro/best_thresholds.csv")  # 可选
GLOBAL_THR = float(os.environ.get("GLOBAL_THR", "0.40"))

engine = InferenceEngine(
    ckpt_path=CKPT_PATH,
    classes_json=CLASSES_JSON,
    thresholds_csv=THRESHOLDS_CSV if os.path.exists(THRESHOLDS_CSV) else None,
    global_threshold=GLOBAL_THR,
)

class PredictResponse(BaseModel):
    filename: str
    topk: list
    probs: dict
    preds: dict

@app.get("/health")
def health():
    return {"status": "ok", "device": str(engine.device), "model": engine.model_name, "img_size": engine.img_size}

@app.post("/predict", response_model=List[PredictResponse])
async def predict(files: List[UploadFile] = File(...), topk: int = Form(5)):
    pil_imgs, names = [], []
    for f in files:
        img_bytes = await f.read()
        pil_imgs.append(Image.open(io.BytesIO(img_bytes)))
        names.append(f.filename)
    probs, preds, topk_list = engine.infer(pil_imgs, topk=topk)
    out = []
    for i, name in enumerate(names):
        out.append(PredictResponse(
            filename=name,
            topk=topk_list[i],
            probs={c: float(probs[i, j]) for j, c in enumerate(engine.classes)},
            preds={c: int(preds[i, j]) for j, c in enumerate(engine.classes)},
        ))
    return out

# start： uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1