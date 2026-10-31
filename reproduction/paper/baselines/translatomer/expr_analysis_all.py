#!/usr/bin/env python3
"""
expr_analysis_all.py

Calculates Pearson and Spearman correlations between Translatomer seq-only
predicted ribosome load (sum of per-bin predictions) and protein expression
for HEK293, iPSC, LCL, and hTERT_RPE1.

Runs each Translatomer seq-only checkpoint on the mRFP test set, sums
predicted ribo counts per transcript, and correlates with expression labels.

Outputs scatter plots and a summary JSON to ./data/expression/
"""

import os, sys, csv, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from pathlib import Path

sys.path.append('Translatomer')
from train_seqonly import PatchedTrainModuleSeqOnly, onehot_dna_fast, SEQLEN, NBINS, BIN

PAPER = Path(__file__).resolve().parents[2]  # reproduction/paper


# ---------------- CONFIG ----------------
BASE_DIR = Path(".")  # Translatomer work directory
DATA_OUT_DIR = BASE_DIR / "data" / "expression"
DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPRESSION_CSV = str(PAPER / "expression" / "mRFP_Expression.csv")

def best_checkpoint(cell):
    """Checkpoint with the lowest validation loss in checkpoints_seqonly/<cell>/ (epoch=E-val_loss=V.ckpt)."""
    ckpts = sorted((BASE_DIR / "checkpoints_seqonly" / cell).glob("*.ckpt"))
    if not ckpts:
        return ""
    return str(min(ckpts, key=lambda c: float(c.stem.split("val_loss=")[1])))


CELL_LINES = {
    "HEK293": {"checkpoint": best_checkpoint("hek293")},
    "iPSC": {"checkpoint": best_checkpoint("ipsc")},
    "LCL": {"checkpoint": best_checkpoint("lcl")},
    "hTERT_RPE1": {"checkpoint": best_checkpoint("rpe")},
}

# ---------------- DATA ----------------
class MockArgs:
    def __init__(self):
        self.model_type = 'TransModel'
        self.hparams = {}
        self.learning_rate = 1e-5


def load_expression_data(csv_path, split="test"):
    sequences, labels = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Split"].strip() == split:
                sequences.append(row["Sequence"].strip())
                labels.append(float(row["Value"]))
    return sequences, np.array(labels, dtype=np.float64)


def rna_to_dna(rna_seq: str) -> str:
    return rna_seq.replace("U", "T").replace("u", "t")


def load_model(ckpt_path, device):
    try:
        model = PatchedTrainModuleSeqOnly.load_from_checkpoint(
            ckpt_path, args=MockArgs(), map_location=device
        )
    except TypeError:
        model = PatchedTrainModuleSeqOnly.load_from_checkpoint(
            ckpt_path, map_location=device
        )
    model.to(device).eval()
    return model


@torch.inference_mode()
def predict_sequence(model, dna_seq: str, device):
    seq_len = len(dna_seq)
    pad_left = (SEQLEN - seq_len) // 2
    padded = "N" * pad_left + dna_seq + "N" * (SEQLEN - seq_len - pad_left)

    x = onehot_dna_fast(padded).unsqueeze(0).float().to(device)
    pred = model(x).squeeze(0).cpu().numpy()

    bin_start = max(0, pad_left // BIN)
    bin_end = min(NBINS, (pad_left + seq_len + BIN - 1) // BIN)

    return float(pred[bin_start:bin_end].sum())


# ---------- transforms (from te_analysis_all.py) ----------
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


# ---------- plotting ----------
def _scatter_with_fit_equalized(x_raw, y_raw, xlabel, ylabel, outpath, equalize="minmax"):
    mask = np.isfinite(x_raw) & np.isfinite(y_raw)
    x = x_raw[mask]; y = y_raw[mask]
    if x.size == 0:
        print("[WARN] Nothing to plot after filtering.")
        return

    if equalize == "minmax":
        x_lo, x_hi = _minmax_fit(x)
        y_lo, y_hi = _minmax_fit(y)
        x_plot = _minmax_apply(x, x_lo, x_hi)
        y_plot = _minmax_apply(y, y_lo, y_hi)
        xt_raw = np.linspace(float(np.min(x)), float(np.max(x)), num=5)
        yt_raw = np.linspace(float(np.min(y)), float(np.max(y)), num=5)
        xt_pos = _minmax_apply(xt_raw, x_lo, x_hi)
        yt_pos = _minmax_apply(yt_raw, y_lo, y_hi)
    else:
        x_plot = x; y_plot = y
        xt_raw = np.linspace(float(np.min(x)), float(np.max(x)), num=5)
        yt_raw = np.linspace(float(np.min(y)), float(np.max(y)), num=5)
        xt_pos = xt_raw; yt_pos = yt_raw

    xs_line_plot = yhat_plot = None
    if len(x) >= 2 and np.std(x) > 1e-12:
        try:
            m, b = np.polyfit(x, y, 1)
            xs_line = np.linspace(float(np.min(x)) - 5, float(np.max(x)) + 5, 200)
            yhat = m * xs_line + b
            if equalize == "minmax":
                xs_line_plot = _minmax_apply(xs_line, x_lo, x_hi)
                yhat_plot    = _minmax_apply(yhat, y_lo, y_hi)
            else:
                xs_line_plot = xs_line
                yhat_plot    = yhat
        except Exception:
            pass

    fig, ax = plt.subplots(dpi=300)
    ax.scatter(x_plot, y_plot, alpha=1, s=25)
    if xs_line_plot is not None and yhat_plot is not None:
        ax.plot(xs_line_plot, yhat_plot, linewidth=2)
        ax.plot(xs_line_plot, yhat_plot, linewidth=2)

    _set_equal_limits_from_transformed(ax, [x_plot, xt_pos], [y_plot, yt_pos], pad_frac=0.05)

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
        return float('nan'), float('nan'), 0
    if np.std(xm) < 1e-12 or np.std(ym) < 1e-12:
        return float('nan'), float('nan'), len(xm)
    p_r, _ = pearsonr(xm, ym)
    s_r, _ = spearmanr(xm, ym)
    return p_r, s_r, len(xm)


# ---------- per-cell-line processing ----------
def process_cell_line(name, cfg, dna_sequences, labels, device):
    print(f"\n--- Processing {name} ---")

    ckpt = cfg["checkpoint"]
    if not os.path.exists(ckpt):
        print(f"Skipping {name}: checkpoint not found at {ckpt}")
        return None

    model = load_model(ckpt, device)

    preds = []
    for i, dna_seq in enumerate(dna_sequences):
        ribo_sum = predict_sequence(model, dna_seq, device)
        preds.append(ribo_sum)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(dna_sequences)} done")

    preds = np.array(preds, dtype=np.float64)
    print(f"  Predictions: min={preds.min():.4f}, max={preds.max():.4f}, "
          f"mean={preds.mean():.4f}, std={preds.std():.4f}")

    p_r, s_r, n = get_correlations(preds, labels)
    print(f"  n={n}  Pearson r={p_r:.4f}  Spearman r={s_r:.4f}")

    _scatter_with_fit_equalized(
        preds, labels,
        "Predicted ribosome load",
        "Protein expression",
        str(DATA_OUT_DIR / f"{name}_riboload_vs_expression.pdf"),
        equalize="minmax"
    )

    del model
    torch.cuda.empty_cache()

    return {"pearson": round(float(p_r), 4),
            "spearman": round(float(s_r), 4),
            "n": n}


# ---------- main ----------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sequences, labels = load_expression_data(EXPRESSION_CSV, split="test")
    dna_sequences = [rna_to_dna(s) for s in sequences]
    print(f"Loaded {len(sequences)} test samples from {EXPRESSION_CSV}")

    results = {}
    for name, cfg in CELL_LINES.items():
        res = process_cell_line(name, cfg, dna_sequences, labels, device)
        if res:
            results[name] = res

    out_json = DATA_OUT_DIR / "expression_correlation_summary.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\n========== SUMMARY ==========")
    for name, res in results.items():
        print(f"{name:>12s}: Pearson r = {res['pearson']:.4f}  |  "
              f"Spearman r = {res['spearman']:.4f}  |  n = {res['n']}")
    print(f"\nPlots and JSON saved to {DATA_OUT_DIR}")


if __name__ == "__main__":
    main()
