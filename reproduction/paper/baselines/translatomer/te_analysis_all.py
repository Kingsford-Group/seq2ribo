#!/usr/bin/env python3
"""
te_analysis_all.py

Calculates Pearson and Spearman correlations between Translatomer predicted ribosome sums (Unscaled and Scaled)
and Translation Efficiency (TE) for HEK293, hTERT_RPE1, LCL, and iPSC (STEM).

Unscaled: Raw sums from Translatomer predictions.
Scaled: Sums matched to Ground Truth Observed sums (per transcript).

Outputs plots and a summary JSON to ./data/
"""

import os, csv, glob, pickle, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from pathlib import Path

# Try to import util for constants, fallback if not found
try:
    from util import CODONS, CODON2IDX, STOP_IDX
    K_ANGLE_BINS = 4 # Default from util usually, or we can hardcode
except ImportError:
    # Minimal mock if util is missing (assumes we just need to load pickles)
    print("Warning: util.py not found, some features might be missing.")
    K_ANGLE_BINS = 4

PAPER = Path(__file__).resolve().parents[2]  # reproduction/paper


# ---------------- CONFIG ----------------
BASE_DIR = Path(".")  # Translatomer work directory
DATA_OUT_DIR = BASE_DIR / "data" / "te_plots"
DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Map human-readable cell line to params
CELL_LINES = {
    "HEK293": {
        "pred_pkl_counts": str(BASE_DIR / "data" / "hek293" / "eval_ready_pred_counts.pkl"),
        "te_csv": str(PAPER / "results" / "TE" / "hek293" / "run_unscaled" / "matched_mean_te_with_completed_total_train_val_test.csv"),
        "te_col": "TE_HEK293",
        "test_data_pkl": str(PAPER / "data" / "hek293" / "test_data.pkl")
    },
    "hTERT_RPE1": {
        "pred_pkl_counts": str(BASE_DIR / "data" / "rpe" / "eval_ready_pred_counts.pkl"),
        "te_csv": str(PAPER / "results" / "TE" / "rpe" / "run_unscaled" / "matched_mean_te_with_completed_total_train_val_test.csv"),
        "te_col": "TE_RPE-1",
        "test_data_pkl": str(PAPER / "data" / "rpe" / "test_data.pkl")
    },
    "LCL": {
        "pred_pkl_counts": str(BASE_DIR / "data" / "lcl" / "eval_ready_pred_counts.pkl"),
        "te_csv": str(PAPER / "results" / "TE" / "lcl" / "run_unscaled" / "matched_mean_te_with_completed_total_train_val_test.csv"),
        "te_col": "TE_LCL",
        "test_data_pkl": str(PAPER / "data" / "lcl" / "test_data.pkl")
    },
    "iPSC": {
        "pred_pkl_counts": str(BASE_DIR / "data" / "ipsc" / "eval_ready_pred_counts.pkl"),
        "te_csv": str(PAPER / "results" / "TE" / "ipsc" / "run_unscaled" / "matched_mean_te_with_completed_total_train_val_test.csv"),
        "te_col": "mean_te",
        "test_data_pkl": str(PAPER / "data" / "ipsc" / "test_data.pkl")
    }
}

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def read_te_csv(csv_path, te_col_name):
    """Map (split='test', match_tx) -> te_col value"""
    te_map = {}
    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            split_val = row.get("split", "").strip()
            if split_val == "test":
                tx = row.get("match_tx", "").strip()
                try:
                    te = float(row.get(te_col_name))
                    te_map[tx] = te
                except (ValueError, TypeError):
                    continue
    return te_map

# ---------- transforms ----------
def _minmax_fit(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0: return 0.0, 1.0
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo: hi = lo + 1.0
    return lo, hi

def _minmax_apply(a, lo, hi):
    a = np.asarray(a, dtype=np.float64)
    denom = (hi - lo) if (hi > lo) else 1.0
    return (a - lo) / denom

def _zscore_fit(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0: return 0.0, 1.0
    mu, sd = float(a.mean()), float(a.std(ddof=0))
    if sd <= 0: sd = 1.0
    return mu, sd

def _zscore_apply(a, mu, sd):
    a = np.asarray(a, dtype=np.float64)
    return (a - mu) / sd

# ---------- axes helpers ----------
def _set_equal_limits_from_transformed(ax, x_list, y_list, pad_frac=0.05):
    xs = np.concatenate([x for x in x_list if x is not None]) if x_list else np.array([])
    ys = np.concatenate([y for y in y_list if y is not None]) if y_list else np.array([])
    xs = xs[np.isfinite(xs)]; ys = ys[np.isfinite(ys)]
    if xs.size == 0 and ys.size == 0:
        return
    lo = np.nanmin([xs.min() if xs.size else np.nan,
                    ys.min() if ys.size else np.nan])
    hi = np.nanmax([xs.max() if xs.size else np.nan,
                    ys.max() if ys.size else np.nan])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return
    if lo == hi:
        delta = 1.0 if lo == 0 else abs(lo) * 0.05
        lo -= delta; hi += delta
    span = hi - lo
    lo -= pad_frac * span; hi += pad_frac * span
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')

def _fmt_tick(v):
    av = abs(float(v))
    if av >= 100:  return f"{v:.0f}"
    if av >= 10:   return f"{v:.1f}"
    if av >= 1:    return f"{v:.2f}"
    if av >= 1e-2: return f"{v:.3f}"
    return f"{v:.3g}"

def _log_ticks(y_pos_min, y_pos_max):
    """Nice raw log10 ticks between min>0 and max; decades, falling back to half-decades."""
    if not np.isfinite(y_pos_min) or not np.isfinite(y_pos_max):
        return np.array([])
    y_pos_min = max(y_pos_min, np.finfo(float).tiny)
    lo_e = int(np.floor(np.log10(y_pos_min)))
    hi_e = int(np.ceil(np.log10(y_pos_max)))
    ticks = [10.0 ** k for k in range(lo_e, hi_e + 1)]
    if len(ticks) < 3:
        halves = [10.0 ** (k + 0.5) for k in range(lo_e, hi_e)]
        ticks = sorted(set(ticks + halves))
    return np.array(ticks, dtype=np.float64)

# ---------- plotting with 1:1 equalization ----------
def _scatter_with_fit_equalized(x_raw, y_raw, xlabel, ylabel, outpath, equalize="minmax",
                                use_log1p=False):
    """
    Plot in transformed space (x equalized; y: log1p or log10 then equalized), raw tick labels,
    equal x/y limits, no identity line, no title.
    use_log1p=True: use log1p(y) so zeros are visible (good for unscaled predictions).
    use_log1p=False: use log10(y) after dropping y<=0 (good for obs counts which are always >0).
    """
    if use_log1p:
        # Keep all finite non-negative values; zeros become log1p(0)=0
        mask = np.isfinite(x_raw) & np.isfinite(y_raw) & (y_raw >= 0)
        x = x_raw[mask]; y = y_raw[mask]
        if x.size == 0:
            print("[WARN] Nothing to plot after filtering.")
            return
        y_log = np.log1p(y)
        # tick positions: raw y values (not log-space labels)
        def _y_ticks_log1p(ymin, ymax):
            # pick a few representative raw values
            candidates = [0, 1, 10, 100, 1000, 10000]
            return np.array([v for v in candidates if ymin <= v <= ymax], dtype=np.float64)
    else:
        # Only positive y for log10 behavior
        mask = y_raw > 0
        if not np.all(mask):
            print(f"[WARN] Dropping {(~mask).sum()} nonpositive y values for log-scale plotting")
        x = x_raw[mask]; y = y_raw[mask]
        if x.size == 0:
            print("[WARN] Nothing to plot after filtering.")
            return
        y_log = np.log10(y)

    if equalize == "minmax":
        x_lo, x_hi = _minmax_fit(x)
        ylo_log, yhi_log = _minmax_fit(y_log)
        x_plot = _minmax_apply(x, x_lo, x_hi)
        y_plot = _minmax_apply(y_log, ylo_log, yhi_log)
        xt_raw = np.linspace(float(np.min(x)), float(np.max(x)), num=5)
        if use_log1p:
            yt_raw = _y_ticks_log1p(float(np.min(y)), float(np.max(y)))
            yt_pos = _minmax_apply(np.log1p(yt_raw), ylo_log, yhi_log)
        else:
            yt_raw = _log_ticks(float(np.min(y)), float(np.max(y)))
            yt_pos = _minmax_apply(np.log10(yt_raw), ylo_log, yhi_log)
        xt_pos = _minmax_apply(xt_raw, x_lo, x_hi)
    elif equalize == "zscore":
        x_mu, x_sd = _zscore_fit(x)
        y_mu, y_sd = _zscore_fit(y_log)
        x_plot = _zscore_apply(x, x_mu, x_sd)
        y_plot = _zscore_apply(y_log, y_mu, y_sd)
        xt_raw = np.linspace(float(np.min(x)), float(np.max(x)), num=5)
        if use_log1p:
            yt_raw = _y_ticks_log1p(float(np.min(y)), float(np.max(y)))
            yt_pos = _zscore_apply(np.log1p(yt_raw), y_mu, y_sd)
        else:
            yt_raw = _log_ticks(float(np.min(y)), float(np.max(y)))
            yt_pos = _zscore_apply(np.log10(yt_raw), y_mu, y_sd)
        xt_pos = _zscore_apply(xt_raw, x_mu, x_sd)
    else:
        # fall back to raw plotting if needed
        x_plot = x; y_plot = y
        xt_raw = np.linspace(float(np.min(x)), float(np.max(x)), num=5)
        yt_raw = _log_ticks(float(np.min(y)), float(np.max(y)))
        xt_pos = xt_raw; yt_pos = yt_raw

    # Fit in log-space: log10(y) = m*x + b
    xs_line_plot = yhat_plot = None
    if len(x) >= 2:
        try:
            m, b = np.polyfit(x, y_log, 1)
            xs_line = np.linspace(float(np.min(x)) - 5, float(np.max(x)) + 5, 200)
            yhat_log = m * xs_line + b
            if equalize == "minmax":
                xs_line_plot = _minmax_apply(xs_line, x_lo, x_hi)
                yhat_plot    = _minmax_apply(yhat_log, ylo_log, yhi_log)
            elif equalize == "zscore":
                xs_line_plot = _zscore_apply(xs_line, x_mu, x_sd)
                yhat_plot    = _zscore_apply(yhat_log, y_mu, y_sd)
            else:
                xs_line_plot = xs_line
                yhat_plot    = 10 ** yhat_log
        except Exception:
            pass

    fig, ax = plt.subplots(dpi=300)
    ax.scatter(x_plot, y_plot, alpha=1, s=25)
    if xs_line_plot is not None and yhat_plot is not None:
        ax.plot(xs_line_plot, yhat_plot, linewidth=2)
        ax.plot(xs_line_plot, yhat_plot, linewidth=2)

    _set_equal_limits_from_transformed(ax, [x_plot, xt_pos], [y_plot, yt_pos], pad_frac=0.05)
    # (identity line intentionally omitted to match the other script)

    # Raw tick labels
    ax.set_xticks(xt_pos); ax.set_xticklabels([_fmt_tick(v) for v in xt_raw])
    ax.set_yticks(yt_pos); ax.set_yticklabels([_fmt_tick(v) for v in yt_raw])

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def get_correlations(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]
    if len(xm) < 2:
        return 0.0, 0.0, 0
    p_r, _ = pearsonr(xm, ym)
    s_r, _ = spearmanr(xm, ym)
    return p_r, s_r, len(xm)

def process_cell_line(name, cfg):
    print(f"--- Processing {name} ---")
    
    # 1. Load Translatomer Predictions (Unscaled)
    if not os.path.exists(cfg["pred_pkl_counts"]):
        print(f"Skipping {name}: Prediction PKL not found")
        return None
    pred_counts = load_pickle(cfg["pred_pkl_counts"]) # tx -> array
    
    # 2. Load TE Map
    if not os.path.exists(cfg["te_csv"]):
        print(f"Skipping {name}: TE CSV not found")
        return None
    te_map = read_te_csv(cfg["te_csv"], cfg["te_col"])
    
    # 3. Load Test Data PKL
    if not os.path.exists(cfg["test_data_pkl"]):
        print(f"Skipping {name}: Test Data PKL not found")
        return None
    test_samples = load_pickle(cfg["test_data_pkl"])
    
    data_unscaled = [] # (te, pred_sum)
    data_scaled = []   # (te, obs_sum) -> Scaled pred essentially matches Obs sum
    
    # Iterate list of samples
    skipped_not_in_pred = 0
    skipped_not_in_te = 0
    skipped_len_mismatch = 0
    matched_count = 0
    
    for i, sample in enumerate(test_samples):
        # Sample structure: (tx, cods, obs_counts, ...)
        if len(sample) < 3: continue
        tx = sample[0]
        obs_counts = sample[2]
            
        if tx not in pred_counts:
            skipped_not_in_pred += 1
            continue
        if tx not in te_map:
            skipped_not_in_te += 1
            if i < 5: print(f"DEBUG: {tx} not in TE map")
            continue
            
        # Translatomer Prediction
        pred_arr = pred_counts[tx]
        
        # Check length: Obs is NT, Pred is Codon
        # Allow exact match OR 3x match
        L_pred = len(pred_arr)
        L_obs = len(obs_counts)
        
        # If lengths match exactly, that's fine (both codon or both NT)
        # If L_obs ~ 3 * L_pred, that's fine (Obs=NT, Pred=Codon)
        if L_pred != L_obs and L_obs != 3 * L_pred and abs(L_obs - 3*L_pred) > 2:
             # Tolerance for potential partial codons or artifacts
             skipped_len_mismatch += 1
             if i < 5: print(f"DEBUG: Len mismatch {tx}: P={L_pred}, O={L_obs}")
             continue
        
        matched_count += 1
        
        pred_sum = float(np.sum(pred_arr))
        obs_sum = float(np.sum(obs_counts))
        te = te_map[tx]
        
        # Unscaled Pair
        data_unscaled.append((te, pred_sum))
        
        # Scaled Pair (Scaled to match Obs Sum)
        # If we scale Pred to match Obs sum, then Pred_Sum_New = Obs_Sum.
        data_scaled.append((te, obs_sum))

    print(f"Skipped: Pred={skipped_not_in_pred}, TE={skipped_not_in_te}, Len={skipped_len_mismatch}. Matched={matched_count}")
    
    if not data_unscaled:
        print(f"No valid matched data for {name}")
        return None

    # Convert to arrays
    data_unscaled = np.array(data_unscaled)
    data_scaled = np.array(data_scaled)
    
    # 4. Plots and Correlations
    
    # Unscaled
    # Calculate corr for JSON
    p_u, s_u, n_u = get_correlations(data_unscaled[:, 0], data_unscaled[:, 1])
    
    # Plot using reference style
    _scatter_with_fit_equalized(
        data_unscaled[:, 0], data_unscaled[:, 1],
        "True TE",
        "Total ribosome count",
        str(DATA_OUT_DIR / f"{name}_unscaled_vs_te.pdf"),
        equalize="minmax",
        use_log1p=True
    )
    
    # Scaled (Observed)
    p_s, s_s, n_s = get_correlations(data_scaled[:, 0], data_scaled[:, 1])
    
    _scatter_with_fit_equalized(
        data_scaled[:, 0], data_scaled[:, 1],
        "True TE",
        "Total ribosome count",
        str(DATA_OUT_DIR / f"{name}_scaled_vs_te.pdf"),
        equalize="minmax"
    )
    
    return {
        "unscaled": {"pearson": p_u, "spearman": s_u, "n": n_u},
        "scaled": {"pearson": p_s, "spearman": s_s, "n": n_s}
    }

def main():
    results = {}
    for name, cfg in CELL_LINES.items():
        res = process_cell_line(name, cfg)
        if res:
            results[name] = res
            
    # Save Summary
    with open(DATA_OUT_DIR / "te_correlation_summary.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nSummary saved to", DATA_OUT_DIR / "te_correlation_summary.json")

if __name__ == "__main__":
    main()
