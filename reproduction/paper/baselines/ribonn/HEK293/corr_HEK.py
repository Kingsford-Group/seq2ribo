#!/usr/bin/env python3
import os, argparse, json
import numpy as np
import pandas as pd
from pathlib import Path

# Headless-safe plotting
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
    if x.size < 2:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm * xm).sum()) * np.sqrt((ym * ym).sum())
    if denom == 0:
        return float("nan")
    return float((xm * ym).sum() / denom)

def _spearman_safe(x, y):
    xr = pd.Series(np.asarray(x, dtype=np.float64)).rank(method="average").to_numpy()
    yr = pd.Series(np.asarray(y, dtype=np.float64)).rank(method="average").to_numpy()
    return _pearson_safe(xr, yr)

def _coerce_numeric_cols(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def _axis_limits_equal(x, y, pad_ratio=0.05):
    xy = np.concatenate([np.asarray(x, float), np.asarray(y, float)])
    xy = xy[np.isfinite(xy)]
    if xy.size == 0:
        return (0.0, 1.0)
    lo, hi = float(np.min(xy)), float(np.max(xy))
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    span = hi - lo
    pad = pad_ratio * span
    lo -= pad; hi += pad
    return (lo, hi)

def scatter_with_fit(y_true, y_pred, title, xlabel, ylabel, out_path, note="", dpi=150):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]; yp = y_pred[mask]

    # Fit y = a*x + b
    a, b = np.polyfit(yt, yp, 1) if yt.size >= 2 else (np.nan, np.nan)
    lo, hi = _axis_limits_equal(yt, yp)

    plt.figure()
    plt.scatter(yt, yp, s=25, alpha=1)
    # y = x reference
    #plt.plot([lo, hi], [lo, hi], linestyle="--")
    # Least-squares fit
    if np.isfinite(a) and np.isfinite(b):
        xfit = np.array([lo, hi], dtype=np.float64)
        yfit = a * xfit + b
        plt.plot(xfit, yfit)
        plt.plot(xfit, yfit)

    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.gca().set_aspect("equal", adjustable="box")
    #plt.title(title)
    plt.xlabel("True TE", fontsize=14); plt.ylabel("Predicted TE", fontsize= 14)


    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Compare RiboNN TE predictions vs ground truth + scatter plots")
    # ap.add_argument("--pred_txt", default="HEK293/prediction_output_native.txt")
    ap.add_argument("--pred_txt", default="results/HEK293/prediction_output.txt")

    ap.add_argument("--gt_tsv", default="HEK293/test_mean_te.tsv")
    ap.add_argument("--out_dir", default=".", help="Output directory for metrics + plots")
    ap.add_argument("--plots_dir", default=None, help="If set, write plots here (defaults to out_dir)")
    ap.add_argument("--dpi", type=int, default=300, help="Plot DPI (default: 180)")
    ap.add_argument("--per_column", action="store_true",
                    help="Also compute metrics/plots for each predicted_TE_* column")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(args.plots_dir) if args.plots_dir else out_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Load predictions ---
    pred = pd.read_csv(args.pred_txt, sep="\t", dtype=str)
    if "tx_id" not in pred.columns:
        raise SystemExit("Predictions file must contain a 'tx_id' column.")
    pred["key"] = pred["tx_id"].map(_strip_ver)

    # Identify prediction columns
    pred_cols = [c for c in pred.columns if c.startswith("predicted_TE_")]
    has_mean = "mean_predicted_TE" in pred.columns
    if has_mean and "mean_predicted_TE" not in pred_cols:
        pred_cols = ["mean_predicted_TE"] + pred_cols
    _coerce_numeric_cols(pred, pred_cols + (["mean_predicted_TE"] if has_mean else []))

    # If mean not present, create from available predicted_TE_* columns
    if not has_mean and pred_cols:
        pred["mean_predicted_TE"] = pred[pred_cols].astype(float).mean(axis=1)
        has_mean = True

    # --- Load ground truth ---
    gt = pd.read_csv(args.gt_tsv, sep="\t", dtype=str)
    if "tx_id" not in gt.columns or "TE_HEK293" not in gt.columns:
        raise SystemExit("Ground truth file must have 'tx_id' and 'TE_HEK293' columns.")
    gt["key"] = gt["tx_id"].map(_strip_ver)
    _coerce_numeric_cols(gt, ["TE_HEK293"])
    gt = gt.dropna(subset=["TE_HEK293"]).copy()

    # --- Merge ---
    merged = (pred.drop_duplicates(subset=["key"])
                   .merge(gt[["key","TE_HEK293"]], on="key", how="inner"))
    merged = merged.dropna(subset=["TE_HEK293"]).copy()
    _coerce_numeric_cols(merged, ["TE_HEK293"])
    if merged.empty:
        raise SystemExit("No overlap between predictions and ground truth after matching.")

    # Save matched pairs for inspection
    merged_out = out_dir / "ribonn_matched_pairs.csv"
    merged.to_csv(merged_out, index=False)
    print(f"Matched {len(merged)} transcripts. Saved pairs to {merged_out}")

    # --- Metrics helpers ---
    results = []
    def _add_metrics(label, y_true, y_pred):
        y_true = pd.to_numeric(y_true, errors="coerce")
        y_pred = pd.to_numeric(y_pred, errors="coerce")
        mask = ~(y_true.isna() | y_pred.isna() | np.isinf(y_true) | np.isinf(y_pred))
        yt = y_true[mask].to_numpy(); yp = y_pred[mask].to_numpy()
        res = {
            "model": label,
            "n": int(yt.size),
            "mse": _mse(yt, yp),
            "pearson_r": _pearson_safe(yt, yp),
            "spearman_rho": _spearman_safe(yt, yp),
        }
        results.append(res)
        note = f"n={res['n']}  r={res['pearson_r']:.3f}  ρ={res['spearman_rho']:.3f}  MSE={res['mse']:.3f}"
        return yt, yp, note

    # --- Plot 1: "replicates" = stack all predicted_TE_* columns ---
    pred_rep_cols = [c for c in pred_cols if c != "mean_predicted_TE"]
    if pred_rep_cols:
        long = merged.melt(id_vars=["key","TE_HEK293"], value_vars=pred_rep_cols,
                           var_name="pred_col", value_name="y_pred")
        yt_rep, yp_rep, note_rep = _add_metrics("stacked_predicted_TE_*", long["TE_HEK293"], long["y_pred"])
        scatter_with_fit(
            yt_rep, yp_rep,
            "Test TE: per-cell-type predictions",
            "TE (true)", "TE (pred)",
            plots_dir / "scatter_test_replicas.png",
            note=note_rep, dpi=args.dpi
        )

    # --- Plot 2: per-transcript mean (this mirrors your second snippet) ---
    if has_mean:
        yt_tx, yp_tx, note_tx = _add_metrics("mean_predicted_TE", merged["TE_HEK293"], merged["mean_predicted_TE"])
        scatter_with_fit(
            yt_tx, yp_tx,
            "Test TE: per-transcript mean predictions",
            "TE (true)", "TE (pred)",
            plots_dir / "scatter_test_per_tx.png",
            note=note_tx, dpi=args.dpi
        )

    # --- Optional: per-column plots ---
    if args.per_column:
        for c in pred_rep_cols:
            if c in merged.columns:
                yt_c, yp_c, note_c = _add_metrics(c, merged["TE_HEK293"], merged[c])
                safe_name = c.replace("predicted_TE_", "scatter_") + ".png"
                scatter_with_fit(
                    yt_c, yp_c,
                    f"Test TE: {c}",
                    "TE (true)", "TE (pred)",
                    plots_dir / safe_name,
                    note=note_c, dpi=args.dpi
                )

    # Save metrics
    res_df = pd.DataFrame(results).sort_values("model").reset_index(drop=True)
    res_df.to_csv(out_dir / "ribonn_metrics_summary.csv", index=False)
    (out_dir / "ribonn_metrics_summary.json").write_text(json.dumps(results, indent=2))

    print("\n=== Metrics (y_true = TE_HEK293) ===")
    for r in results:
        print(f"{r['model']:>28s} | n={r['n']:4d} | MSE={r['mse']:.6f} | r={r['pearson_r']:.4f} | ρ={r['spearman_rho']:.4f}")
    print(f"\nPlots saved to: {plots_dir}")

if __name__ == "__main__":
    main()
