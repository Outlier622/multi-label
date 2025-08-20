import argparse, pandas as pd, numpy as np
from sklearn.metrics import f1_score, average_precision_score

def load_thresholds(path):
    df = pd.read_csv(path)
    return {row["Class"]: float(row["Best_Threshold"]) for _, row in df.iterrows()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="CSV with prob_* columns (e.g., test_preds_thr0p40.csv)")
    ap.add_argument("--thr", required=True, help="best_thresholds.csv")
    ap.add_argument("--gt", required=True, help="test/_classes.csv (ground truth with 0/1 columns)")
    ap.add_argument("--image-col", default="filename", help="name of image column in GT csv")
    ap.add_argument("--out", default="test_preds_perclass.csv")
    args = ap.parse_args()

    thr = load_thresholds(args.thr)
    preds_df = pd.read_csv(args.preds)
    gt_df = pd.read_csv(args.gt)

    
    left_key = "image_path" if "image_path" in preds_df.columns else args.image_col
    preds_df[left_key] = preds_df[left_key].astype(str)
    gt_df[args.image_col] = gt_df[args.image_col].astype(str)
    df = preds_df.merge(gt_df, left_on=left_key, right_on=args.image_col, how="inner", suffixes=("", "_gt"))

    
    classes = [c.replace("prob_", "") for c in df.columns if c.startswith("prob_")]

    
    for cls in classes:
        pcol = f"prob_{cls}"
        outcol = f"pred_{cls}"
        t = thr.get(cls, 0.5)
        df[outcol] = (df[pcol].values >= t).astype(int)

    
    y_true = df[classes].values.astype(int)
    prob_cols = [f"prob_{c}" for c in classes]
    y_prob = df[prob_cols].values

    pred_cols = [f"pred_{c}" for c in classes]
    y_pred = df[pred_cols].values.astype(int)

    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

    
    ap_per_class = []
    for i in range(len(classes)):
        if y_true[:, i].sum() == 0:
            continue
        ap = average_precision_score(y_true[:, i], y_prob[:, i])
        ap_per_class.append(ap)
    mAP_macro = float(np.mean(ap_per_class)) if ap_per_class else 0.0

    df.to_csv(args.out, index=False)
    print(f"Saved per-class preds to {args.out}")
    print(f"mAP_macro={mAP_macro:.4f}  F1_micro={f1_micro:.4f}")

if __name__ == "__main__":
    main()
