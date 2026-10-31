#!/usr/bin/env python3
"""
Evaluate multitask RiboNN predictions against ground truth for all cell lines.

Outputs per-cell-line JSON files in the same format as
HEK293/ribonn_metrics_summary.json, plus scatter plots.

Usage:
  python -m src.eval_multitask
"""

import os, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent

CELL_LINES = {
    "HEK293": {"gt_col": "TE_HEK293"},
    "LCL":    {"gt_col": "TE_LCL"},
    "RPE":    {"gt_col": "TE_RPE-1"},
    "IPSC":   {"gt_col": "mean_te"},
}

EXPERIMENTS = {
    "multitask_CV": {
        "gt_suffix": "test_mean_te.tsv",
        "pred_file": "prediction_output_multitask.txt",
    },
    "multitask_CDS_CV": {
        "gt_suffix": "test_mean_te_cds_filtered.tsv",
        "pred_file": "prediction_output_multitask.txt",
    },
}

PRED_COL_MAP = {
    "HEK293": "predicted_TE_HEK293",
    "LCL":    "predicted_TE_LCL",
    "RPE":    "predicted_TE_RPE-1",
    "IPSC":   "predicted_TE_IPSC",
}


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
    return (lo - pad, hi + pad)


def scatter_with_fit(y_true, y_pred, xlabel, ylabel, out_path, dpi=300):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]; yp = y_pred[mask]

    a, b = np.polyfit(yt, yp, 1) if yt.size >= 2 else (np.nan, np.nan)
    lo, hi = _axis_limits_equal(yt, yp)

    plt.figure()
    plt.scatter(yt, yp, s=25, alpha=1)
    if np.isfinite(a) and np.isfinite(b):
        xfit = np.array([lo, hi], dtype=np.float64)
        yfit = a * xfit + b
        plt.plot(xfit, yfit)

    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def evaluate_one(gt_path, pred_path, gt_col, pred_col, plot_dir, label, dpi=300):
    gt = pd.read_csv(gt_path, sep="\t", dtype=str)
    pred = pd.read_csv(pred_path, sep="\t", dtype=str)

    gt["key"] = gt["tx_id"].map(_strip_ver)
    pred["key"] = pred["tx_id"].map(_strip_ver)

    _coerce_numeric_cols(gt, [gt_col])
    _coerce_numeric_cols(pred, [pred_col])

    gt = gt.dropna(subset=[gt_col]).copy()

    merged = (pred.drop_duplicates(subset=["key"])
                  .merge(gt[["key", gt_col]], on="key", how="inner"))
    merged = merged.dropna(subset=[gt_col, pred_col]).copy()

    if merged.empty:
        return None

    yt = merged[gt_col].to_numpy(dtype=np.float64)
    yp = merged[pred_col].to_numpy(dtype=np.float64)

    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]; yp = yp[mask]

    res = {
        "model": label,
        "n": int(yt.size),
        "mse": _mse(yt, yp),
        "pearson_r": _pearson_safe(yt, yp),
        "spearman_rho": _spearman_safe(yt, yp),
    }

    scatter_with_fit(
        yt, yp,
        "True TE", "Predicted TE",
        plot_dir / f"scatter_{label}.pdf",
        dpi=dpi,
    )

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    for exp_name, exp_info in EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"  Experiment: {exp_name}")
        print(f"{'='*60}")

        for cl, cl_info in CELL_LINES.items():
            gt_path = BASE_DIR / cl / exp_info["gt_suffix"]
            pred_path = BASE_DIR / "results" / exp_name / cl / exp_info["pred_file"]

            if not pred_path.exists():
                print(f"  {cl}: predictions not found at {pred_path}, skipping.")
                continue
            if not gt_path.exists():
                print(f"  {cl}: ground truth not found at {gt_path}, skipping.")
                continue

            out_dir = BASE_DIR / "results" / exp_name / cl
            plot_dir = out_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)

            gt_col = cl_info["gt_col"]
            pred_col = PRED_COL_MAP[cl]

            results = []

            res = evaluate_one(
                gt_path, pred_path, gt_col, pred_col,
                plot_dir, label=pred_col, dpi=args.dpi,
            )
            if res:
                results.append(res)

            json_path = out_dir / "ribonn_metrics_summary.json"
            csv_path = out_dir / "ribonn_metrics_summary.csv"
            json_path.write_text(json.dumps(results, indent=2))
            pd.DataFrame(results).to_csv(csv_path, index=False)

            if res:
                print(
                    f"  {cl:<8s} | n={res['n']:4d} | "
                    f"r={res['pearson_r']:.4f} | "
                    f"ρ={res['spearman_rho']:.4f} | "
                    f"MSE={res['mse']:.4f}"
                )
            else:
                print(f"  {cl:<8s} | no overlapping transcripts")

    print("\nDone.")


if __name__ == "__main__":
    main()
