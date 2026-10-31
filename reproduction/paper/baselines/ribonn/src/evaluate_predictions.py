
#!/usr/bin/env python3
import os, argparse, json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _strip_ver(tx):
    s = str(tx)
    if "." in s:
        base, maybe = s.rsplit(".", 1)
        return base if maybe.isdigit() else s
    return s

def _mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2)) if y_true.size else float("nan")

def _pearson_safe(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2: return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm * xm).sum()) * np.sqrt((ym * ym).sum())
    return float((xm * ym).sum() / denom) if denom != 0 else float("nan")

def _spearman_safe(x, y):
    xr = pd.Series(np.asarray(x, dtype=np.float64)).rank(method="average").to_numpy()
    yr = pd.Series(np.asarray(y, dtype=np.float64)).rank(method="average").to_numpy()
    return _pearson_safe(xr, yr)

def scatter_with_fit(y_true, y_pred, title, xlabel, ylabel, out_path, note="", dpi=150):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]; yp = y_pred[mask]
    if yt.size < 2: return
    
    a, b = np.polyfit(yt, yp, 1)
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    span = hi - lo
    lo -= 0.05 * span; hi += 0.05 * span
    
    plt.figure()
    plt.scatter(yt, yp, s=25, alpha=0.6)
    xfit = np.array([lo, hi])
    yfit = a * xfit + b
    plt.plot(xfit, yfit, color='red', label=f"y={a:.2f}x+{b:.2f}")
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title + f"\n{note}")
    plt.legend()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_txt", required=True)
    ap.add_argument("--gt_tsv", required=True)
    ap.add_argument("--gt_col", required=True, help="Column name in GT file containing target values")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load Predictions
    pred = pd.read_csv(args.pred_txt, sep="\t", dtype=str)
    pred["key"] = pred["tx_id"].map(_strip_ver)
    
    # Identify Prediction Column
    # It might be 'mean_predicted_TE', 'predicted_mean_te', 'predicted_TE_LCL', etc.
    # We look for a column that starts with 'predicted_' or is 'mean_predicted_TE'
    pred_col = "mean_predicted_TE"
    if pred_col not in pred.columns:
        # Try to find one
        candidates = [c for c in pred.columns if "predicted" in c]
        if len(candidates) == 1:
            pred_col = candidates[0]
        else:
            # If multiple, prefer one that matches GT
            # e.g. gt_col=TE_LCL, look for predicted_TE_LCL
            match = f"predicted_{args.gt_col.replace('^', '').replace('$', '')}"
            if match in pred.columns:
                pred_col = match
            else:
                # Fallback to mean_predicted_TE if made by my script
                if "mean_predicted_TE" in pred.columns:
                     pred_col = "mean_predicted_TE"
                else:
                    raise ValueError(f"Could not identify unique prediction column in {pred.columns}")

    pred[pred_col] = pd.to_numeric(pred[pred_col], errors='coerce')
    
    # Load GT
    gt = pd.read_csv(args.gt_tsv, sep="\t", dtype=str)
    gt["key"] = gt["tx_id"].map(_strip_ver)
    gt[args.gt_col] = pd.to_numeric(gt[args.gt_col], errors='coerce')
    
    # Merge
    merged = pred.merge(gt, on="key", suffixes=("_pred", "_gt"))
    merged = merged.dropna(subset=[pred_col, args.gt_col])
    
    y_true = merged[args.gt_col].values.astype(float)
    y_pred = merged[pred_col].values.astype(float)
    
    metrics = {
        "n": int(len(y_true)),
        "mse": _mse(y_true, y_pred),
        "pearson_r": _pearson_safe(y_true, y_pred),
        "spearman_rho": _spearman_safe(y_true, y_pred)
    }
    
    print(f"Metrics: {metrics}")
    
    # Save
    with open(f"{args.out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    note = f"R={metrics['pearson_r']:.3f}, Rho={metrics['spearman_rho']:.3f}, MSE={metrics['mse']:.3f}"
    scatter_with_fit(y_true, y_pred, f"Prediction vs GT ({args.gt_col})", "True TE", "Predicted TE", 
                     f"{args.out_dir}/scatter.pdf", note=note)

if __name__ == "__main__":
    main()
