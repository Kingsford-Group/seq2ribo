"""Evaluation metrics for codon-level A-site count profiles.

For transcripts i with observed counts y_i and predicted counts yhat_i (length N_i):
  Tx-level r        Pearson r between per-transcript totals sum_j y_ij and sum_j yhat_ij
  Shape r           mean over transcripts of the within-transcript Pearson r(y_i, yhat_i)
                    (transcripts with a constant observed or predicted profile are skipped)
  Elemwise r        Pearson r over all codon positions of all transcripts
  pertranscriptMAE  mean over transcripts of (1/N_i) sum_j |y_ij - yhat_ij|
  elemwiseMAE       sum_ij |y_ij - yhat_ij| / sum_i N_i
  codonMAE          mean over the 61 sense codons c of |Y_c - Yhat_c|, Y_c = summed counts at codon c
  pairMAE / angleMAE / bucketMAE
                    mean over categories g of |Y_g - Yhat_g| (4 pair counts / 4 angle bins / 3 thirds)
Scaled metrics ("sc_" prefix) rescale yhat_i to the observed total of transcript i before the MAEs.
"""
from typing import Dict, List, Sequence

import numpy as np

from .simulation import bucket_index
from .utils import NONSTOP_IDX

MAE_NAMES = ("pertranscriptMAE", "elemwiseMAE", "codonMAE", "pairMAE", "angleMAE", "bucketMAE")


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def scale_to_observed(obs, pred):
    s = pred.sum()
    return pred * (obs.sum() / s) if s > 0 else np.zeros_like(pred)


def _maes(obs: List[np.ndarray], pred: List[np.ndarray], cod: List[np.ndarray], pair: List[np.ndarray],
          angle: List[np.ndarray], k_angle: int) -> Dict[str, float]:
    cod_o = np.zeros(64); cod_p = np.zeros(64)
    pair_o = np.zeros(4); pair_p = np.zeros(4)
    ang_o = np.zeros(k_angle); ang_p = np.zeros(k_angle)
    bk_o = np.zeros(3); bk_p = np.zeros(3)
    ptx, abs_sum, n_pos = [], 0.0, 0
    for y, p, c, pr, an in zip(obs, pred, cod, pair, angle):
        d = np.abs(y - p)
        ptx.append(d.mean()); abs_sum += d.sum(); n_pos += len(y)
        cod_o += np.bincount(c, weights=y, minlength=64); cod_p += np.bincount(c, weights=p, minlength=64)
        pair_o += np.bincount(pr, weights=y, minlength=4); pair_p += np.bincount(pr, weights=p, minlength=4)
        ang_o += np.bincount(an, weights=y, minlength=k_angle); ang_p += np.bincount(an, weights=p, minlength=k_angle)
        b = bucket_index(len(y))
        bk_o += np.bincount(b, weights=y, minlength=3); bk_p += np.bincount(b, weights=p, minlength=3)
    return {
        "pertranscriptMAE": float(np.mean(ptx)),
        "elemwiseMAE": abs_sum / max(n_pos, 1),
        "codonMAE": float(np.mean(np.abs(cod_o - cod_p)[NONSTOP_IDX])),
        "pairMAE": float(np.mean(np.abs(pair_o - pair_p))),
        "angleMAE": float(np.mean(np.abs(ang_o - ang_p))),
        "bucketMAE": float(np.mean(np.abs(bk_o - bk_p))),
    }


def profile_metrics(obs: Sequence[np.ndarray], pred: Sequence[np.ndarray], codon_ids: Sequence[np.ndarray],
                    pair_count: Sequence[np.ndarray], angle_bin: Sequence[np.ndarray], k_angle: int = 4,
                    scaled_only: bool = False) -> Dict[str, float]:
    """All metrics over a set of transcripts (inputs are per-transcript codon-level arrays)."""
    obs = [np.asarray(y, np.float64) for y in obs]
    pred = [np.asarray(p, np.float64) for p in pred]
    cod = [np.asarray(c, np.int64) for c in codon_ids]
    pair = [np.clip(np.asarray(p, np.int64), 0, 3) for p in pair_count]
    angle = [np.asarray(a, np.int64) for a in angle_bin]
    out = {"n_transcripts": len(obs)}
    sc = [scale_to_observed(y, p) for y, p in zip(obs, pred)]
    out.update({f"sc_{k}": v for k, v in _maes(obs, sc, cod, pair, angle, k_angle).items()})
    if scaled_only:
        return out
    out.update(_maes(obs, pred, cod, pair, angle, k_angle))
    shape = [pearson(y, p) for y, p in zip(obs, pred)]
    out["shape_r"] = float(np.nanmean(shape)) if np.isfinite(shape).any() else float("nan")
    out["shape_r_n"] = int(np.isfinite(shape).sum())
    out["txlevel_r"] = pearson([y.sum() for y in obs], [p.sum() for p in pred])
    out["elemwise_r"] = pearson(np.concatenate(obs), np.concatenate(pred))
    return out
