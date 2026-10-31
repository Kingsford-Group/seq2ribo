"""Ribosome-profile evaluation metrics (Tx-level r, Shape r, Elemwise r and six MAEs).

Each transcript can carry several predicted profiles (one per sTASEP run, polished
independently). Predictions are averaged over runs per transcript before computing the
correlation, elementwise and group metrics; pertranscriptMAE is computed per run and then
averaged over runs.

  pertranscriptMAE  mean over transcripts of sum_j |y - y_hat| / sum_j y (0 if sum_j y = 0)
  elemwiseMAE       mean |y - y_hat| over all codon positions
  codonMAE          mean over the 64 codon slots of |Y_c - Y_hat_c| (stop codons contribute 0)
  pairMAE / angleMAE / bucketMAE
                    mean over categories of |Y_g - Y_hat_g|, with Y_g the summed counts of group g

Scaled ("sc_") metrics rescale every predicted run so that its total equals the observed total
of the transcript before any averaging.
"""

from typing import Dict, Iterable, List, Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr

N_CODONS = 64
STOP_IDX = (48, 50, 56)  # UAA, UAG, UGA in the 64-codon alphabetical order


def bucket_index(L: int) -> np.ndarray:
    b = np.zeros(L, dtype=np.int64)
    b[L // 3:(2 * L) // 3] = 1
    b[(2 * L) // 3:] = 2
    return b


def angle_bin_index(angle_dev_sum: np.ndarray, angle_bins: np.ndarray) -> np.ndarray:
    k = len(angle_bins) - 1
    return np.clip(np.digitize(np.asarray(angle_dev_sum, dtype=np.float64), angle_bins) - 1, 0, k - 1).astype(np.int64)


def _corr(a: np.ndarray, b: np.ndarray):
    if len(a) < 2 or np.std(a) <= 1e-9 or np.std(b) <= 1e-9:
        return 0.0, 0.0
    pr, _ = pearsonr(a, b)
    sr, _ = spearmanr(a, b)
    return (float(pr) if np.isfinite(pr) else 0.0), (float(sr) if np.isfinite(sr) else 0.0)


def _group_sums(values: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(idx, weights=values.astype(np.float64), minlength=n)


def make_entry(true: np.ndarray, preds: Iterable[np.ndarray], cod_ids: np.ndarray,
               pair_count: Optional[np.ndarray] = None, angle_bin: Optional[np.ndarray] = None) -> Dict:
    """Bundle one transcript: observed counts, list of predicted runs and per-codon features."""
    true = np.asarray(true, dtype=np.float64)
    preds = [np.asarray(p, dtype=np.float64) for p in preds]
    L = len(true)
    if any(len(p) != L for p in preds):
        raise ValueError("prediction length differs from observed length")
    obs_s = float(true.sum())
    sc_preds, ptx, sc_ptx = [], [], []
    for p in preds:
        ps = float(p.sum())
        p_sc = p * (obs_s / ps) if ps > 0 else np.zeros_like(p)
        sc_preds.append(p_sc)
        ptx.append(float(np.abs(p - true).sum()) / obs_s if obs_s > 0 else 0.0)
        sc_ptx.append(float(np.abs(p_sc - true).sum()) / obs_s if obs_s > 0 else 0.0)
    return {
        "true": true,
        "pred": np.mean(preds, axis=0),
        "sc_pred": np.mean(sc_preds, axis=0),
        "ptx": float(np.mean(ptx)),
        "sc_ptx": float(np.mean(sc_ptx)),
        "cod_ids": np.asarray(cod_ids, dtype=np.int64),
        "pair": None if pair_count is None else np.clip(np.asarray(pair_count, dtype=np.int64)[:L], 0, 3),
        "angle": None if angle_bin is None else np.asarray(angle_bin, dtype=np.int64)[:L],
    }


def compute_metrics(entries: List[Dict], k_angle_bins: int = 4) -> Dict[str, float]:
    """Metrics over a list of entries built with make_entry (repeats allowed, e.g. bootstrap)."""
    out = {}
    for prefix, key, ptx_key in (("", "pred", "ptx"), ("sc_", "sc_pred", "sc_ptx")):
        cod_o = np.zeros(N_CODONS); cod_p = np.zeros(N_CODONS)
        bkt_o = np.zeros(3); bkt_p = np.zeros(3)
        pair_o = np.zeros(4); pair_p = np.zeros(4)
        ang_o = np.zeros(k_angle_bins); ang_p = np.zeros(k_angle_bins)
        shape_p, shape_s, tx_o, tx_p, cat_t, cat_p, ptx = [], [], [], [], [], [], []
        for e in entries:
            t, p, c = e["true"], e[key], e["cod_ids"]
            ptx.append(e[ptx_key])
            cat_t.append(t); cat_p.append(p)
            pr, sr = _corr(t, p)
            shape_p.append(pr); shape_s.append(sr)
            tx_o.append(t.sum()); tx_p.append(p.sum())
            ns = ~np.isin(c, STOP_IDX)
            cod_o += _group_sums(t[ns], c[ns], N_CODONS); cod_p += _group_sums(p[ns], c[ns], N_CODONS)
            b = bucket_index(len(t))
            bkt_o += _group_sums(t, b, 3); bkt_p += _group_sums(p, b, 3)
            if e["pair"] is not None:
                pair_o += _group_sums(t, e["pair"], 4); pair_p += _group_sums(p, e["pair"], 4)
            if e["angle"] is not None:
                ang_o += _group_sums(t, e["angle"], k_angle_bins); ang_p += _group_sums(p, e["angle"], k_angle_bins)
        cat_t = np.concatenate(cat_t); cat_p = np.concatenate(cat_p)
        el_p, el_s = _corr(cat_t, cat_p)
        tx_pr, tx_sr = _corr(np.array(tx_o), np.array(tx_p))
        out.update({
            f"{prefix}txlvl_pearson": tx_pr,
            f"{prefix}txlvl_spearman": tx_sr,
            f"{prefix}shape_pearson": float(np.nanmean(shape_p)),
            f"{prefix}shape_spearman": float(np.nanmean(shape_s)),
            f"{prefix}elemwise_pearson": el_p,
            f"{prefix}elemwise_spearman": el_s,
            f"{prefix}pertranscriptMAE": float(np.mean(ptx)),
            f"{prefix}elemwiseMAE": float(np.mean(np.abs(cat_p - cat_t))),
            f"{prefix}codonMAE": float(np.mean(np.abs(cod_o - cod_p))),
            f"{prefix}bucketMAE": float(np.mean(np.abs(bkt_o - bkt_p))),
            f"{prefix}pairMAE": float(np.mean(np.abs(pair_o - pair_p))),
            f"{prefix}angleMAE": float(np.mean(np.abs(ang_o - ang_p))),
        })
    out["n_transcripts"] = len(entries)
    return out


def bootstrap_ci(entries: List[Dict], n_bootstrap: int = 1000, seed: int = 42, k_angle_bins: int = 4) -> Dict[str, Dict[str, float]]:
    """Percentile 95% CIs from resampling transcripts with replacement."""
    rng = np.random.RandomState(seed)
    point = compute_metrics(entries, k_angle_bins)
    names = [k for k in point if k != "n_transcripts"]
    boot = {k: [] for k in names}
    n = len(entries)
    for _ in range(n_bootstrap):
        m = compute_metrics([entries[i] for i in rng.randint(0, n, size=n)], k_angle_bins)
        for k in names:
            boot[k].append(m[k])
    return {k: {"point": point[k], "ci_lo": float(np.nanpercentile(boot[k], 2.5)),
                "ci_hi": float(np.nanpercentile(boot[k], 97.5))} for k in names}
