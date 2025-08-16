import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import json
from tqdm import tqdm

def load_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "model" in checkpoint:
        model = checkpoint["model"]
    else:
        # 从 args 中恢复模型结构
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        weights = None
        num_classes = checkpoint.get("num_classes", len(checkpoint.get("class_names", [])))
        model = vit_b_16(weights=weights)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
        model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, transform, device):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to best.pt")
    parser.add_argument("--input-dir", help="Folder of images for inference")
    parser.add_argument("--input-csv", help="CSV file with image filenames")
    parser.add_argument("--image-root", help="Root folder for images if using CSV")
    parser.add_argument("--image-col", default="filename", help="CSV column for image paths")
    parser.add_argument("--classes-json", required=True, help="Path to classes.json")
    parser.add_argument("--threshold", type=float, default=None, help="Global threshold")
    parser.add_argument("--per-class-thresholds", help="CSV file with per-class thresholds")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 读取类别
    with open(args.classes_json, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    # 读取 per-class 阈值
    per_class_thr = None
    if args.per_class_thresholds:
        df_thr = pd.read_csv(args.per_class_thresholds)
        per_class_thr = {row["Class"]: row["Best_Threshold"] for _, row in df_thr.iterrows()}

    # 加载模型
    model = load_model(args.ckpt, device)

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # 读取图片路径
    if args.input_dir:
        image_paths = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        df = pd.DataFrame({args.image_col: image_paths})
    elif args.input_csv:
        df = pd.read_csv(args.input_csv)
        if args.image_root:
            df[args.image_col] = df[args.image_col].apply(lambda x: os.path.join(args.image_root, x))
    else:
        raise ValueError("Must provide either --input-dir or --input-csv")

    preds = []
    all_probs = []

    for img_path in tqdm(df[args.image_col], desc="Predicting"):
        probs = predict_image(model, img_path, transform, device)
        all_probs.append(probs)
        if per_class_thr:
            pred = (probs >= [per_class_thr[c] for c in class_names]).astype(int)
        elif args.threshold is not None:
            pred = (probs >= args.threshold).astype(int)
        else:
            pred = (probs >= 0.5).astype(int)
        preds.append(pred)

    all_probs = pd.DataFrame(all_probs, columns=[f"prob_{c}" for c in class_names])
    preds = pd.DataFrame(preds, columns=[f"pred_{c}" for c in class_names])
    out_df = pd.concat([df.reset_index(drop=True), all_probs, preds], axis=1)
    out_df.to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out}")

if __name__ == "__main__":
    main()
