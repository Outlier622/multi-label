# optimize_thresholds.py
import argparse, os, json, math
import numpy as np
import pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import precision_recall_curve
import timm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class ValidSet(Dataset):
    def __init__(self, csv_path, image_root, class_names, image_col="filename", img_size=224):
        self.df = pd.read_csv(csv_path)
        self.root = image_root
        self.image_col = image_col
        self.class_names = class_names
        self.tfms = transforms.Compose([
            transforms.Resize(int(img_size*1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = os.path.join(self.root, row[self.image_col])
        img = Image.open(path).convert("RGB")
        x = self.tfms(img)
        y = row[self.class_names].to_numpy(dtype=np.float32)
        return x, torch.from_numpy(y)

def load_model_from_ckpt(ckpt_path, num_classes, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")  # weights from trusted source
    model_name = ckpt.get("args", {}).get("model", "vit_base_patch16_224")
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model

@torch.no_grad()
def predict_probs(model, loader, device):
    probs_all, ys_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        probs_all.append(probs)
        ys_all.append(y.numpy())
    return np.concatenate(probs_all, 0), np.concatenate(ys_all, 0)

def per_class_best_thresholds(y_true, y_prob, class_names):
    thrs, best_f1 = {}, {}
    for i, c in enumerate(class_names):
        p, r, t = precision_recall_curve(y_true[:, i], y_prob[:, i])
        f1 = (2*p*r) / (p+r+1e-8)
        j = int(np.nanargmax(f1))
        thrs[c] = float(t[j]) if j < len(t) else 0.5
        best_f1[c] = float(f1[j])
    return thrs, best_f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--class-cols", required=True, help="Comma-separated class names (order matters)")
    ap.add_argument("--image-col", default="filename")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", default="best_thresholds.csv")
    args = ap.parse_args()

    classes = [c.strip() for c in args.class_cols.split(",") if c.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = ValidSet(args.csv, args.image_root, classes, image_col=args.image_col, img_size=args.img_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    model = load_model_from_ckpt(args.ckpt, num_classes=len(classes), device=device)
    y_prob, y_true = predict_probs(model, dl, device)  # note: (probs, labels)

    thr, f1 = per_class_best_thresholds(y_true, y_prob, classes)
    out_df = pd.DataFrame({"Class": classes,
                           "Best_Threshold": [round(thr[c], 3) for c in classes],
                           "Best_F1": [round(f1[c], 3) for c in classes]})
    out_df.to_csv(args.out, index=False)
    print(f"Saved per-class thresholds to {args.out}")
    print(out_df.sort_values('Best_F1', ascending=False).head(10))

if __name__ == "__main__":
    main()
