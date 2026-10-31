#!/usr/bin/env python3
# End-to-end fine-tune: Mamba polisher (unfrozen by default) + FULL-counts -> TE head
# Uses GEOMAPS (angle_dev_sum, pair_count) as inputs (no zeroing).
#
# Trains in a positive space by min-max scaling TE (range in CELL_CONFIG) -> [0,1],
# evaluates & saves metrics/CSV back in the original TE space.
#
# Saves:
#   - train_log.json  (per-epoch train/val MSE_raw, MSE_scaled, Pearson r, Spearman rho)
#   - te_transform.json (transform + inverse parameters)
#   - mamba_te_full_test_predictions.csv  (tx, te_true, te_pred)  # real TE space
#   - test_metrics.json  (n, MSE_raw, MSE_scaled, Pearson r, Spearman rho)
#   - mamba_te_full_best.pt / mamba_te_full_final.pt
#
# Expects PKLs from run_unscaled.py in: <pkl_root>/{train,val,test}/*_sample_*.pkl
# Expects TE labels in: --te_csv (columns: split, match_tx, mean_te)
# Expects geomaps (per your convention): --geomap_train, --geomap_val, --geomap_test
#
# No pandas, no matplotlib.

import os, csv, glob, math, json, random, argparse, pickle, hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from util import CODON2IDX  # must define the 64 codons + padding idx 64 in your util
from mamba_ssm import Mamba
K_ANGLE_BINS = 4

CELL_CONFIG = {
    "ipsc": {"te_col": "mean_te", "batch_size": 64, "epochs": 10, "te_min": -2.639472099, "te_max": 3.489203289, "ff_use_log1p": True},
    "hek293": {"te_col": "TE_HEK293", "batch_size": 64, "epochs": 10, "te_min": -2.666693891, "te_max": 3.401144287, "ff_use_log1p": False},
    "rpe": {"te_col": "TE_RPE-1", "batch_size": 64, "epochs": 10, "te_min": -3.625209297, "te_max": 4.258473735, "ff_use_log1p": True},
    "lcl": {"te_col": "TE_LCL", "batch_size": 64, "epochs": 10, "te_min": -3.145896817, "te_max": 7.362059739, "ff_use_log1p": True},
}

def derive_angle_bins_from_training(geo_map_train, k_bins, method="linear", low_q=0.0, high_q=0.995):
    if not geo_map_train:
        return np.linspace(0.0, 1.0, k_bins + 1)

    all_angles = []
    for tx, (angle_dev_sum, _pair_count) in geo_map_train.items():
        if angle_dev_sum is not None:
            all_angles.append(np.asarray(angle_dev_sum, dtype=np.float64))

    if not all_angles:
        return np.linspace(0.0, 1.0, k_bins + 1)

    vals = np.concatenate(all_angles)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.linspace(0.0, 1.0, k_bins + 1)

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))

    if method == "quantile":
        qs = np.linspace(low_q, high_q, k_bins + 1)
        edges = np.quantile(vals, qs)
        edges = np.maximum.accumulate(edges)   # non-decreasing
        edges[-1] = max(edges[-1], vmax)
    else:
        lo = float(np.quantile(vals, low_q))
        hi = float(np.quantile(vals, high_q))
        if hi <= lo:
            hi = lo + 1e-6
        edges = np.linspace(lo, hi, k_bins + 1)
        edges[-1] = max(edges[-1], vmax)

    # (optional) enforce *strictly* increasing for bucketize stability
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-6

    return edges
# ----- Default angle-bin edges (same as your prior code) -----
# DEFAULT_ANGLE_BINS = np.array(
#     [0.0, 1.0471990, 2.09439815, 3.14159722, 6.51421177],
#     dtype=np.float64
# )

# ---------------------------
# Model: Mamba polisher + TE
# ---------------------------
import torch.nn.functional as F



class RiboPolisherMamba(nn.Module):
    def __init__(
        self,
        d_model=192,
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        k_angle_bins=4
    ):
        super().__init__()
        # token & sim
        self.cod_emb  = nn.Embedding(65, d_model)  # 64 codons + padding id 64
        self.sim_proj = nn.Linear(1, d_model)

        # geo feature embeddings
        self.angle_emb  = nn.Embedding(k_angle_bins, d_model)
        self.pair_emb   = nn.Embedding(4, d_model)
        self.bucket_emb = nn.Embedding(3, d_model)

        # backbone
        self.encoder = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

        # inits
        nn.init.xavier_uniform_(self.sim_proj.weight)
        nn.init.zeros_(self.sim_proj.bias)
        nn.init.xavier_uniform_(self.angle_emb.weight)
        nn.init.xavier_uniform_(self.pair_emb.weight)
        nn.init.xavier_uniform_(self.bucket_emb.weight)

        # OPTIONAL but recommended: bias the last layer slightly positive to avoid tiny outputs at init
        last = self.head[-1]
        if isinstance(last, nn.Linear) and last.bias is not None:
            nn.init.constant_(last.bias, 1.0)

    def forward(self, cod_ids, sim_feat, pad_mask, angle_bin=None, pair_bin=None, bucket_idx=None):
        # base tokenization
        tok = self.cod_emb(cod_ids)                       # (B, L, d)
        sim = self.sim_proj(sim_feat.unsqueeze(-1))       # (B, L, d)
        x = tok + sim

        # fuse geo features by addition (no gating)
        if (angle_bin is not None) or (pair_bin is not None) or (bucket_idx is not None):
            add = 0
            if angle_bin is not None:
                add = add + self.angle_emb(angle_bin.clamp(0, self.angle_emb.num_embeddings - 1))
            if pair_bin is not None:
                add = add + self.pair_emb(pair_bin.clamp(0, self.pair_emb.num_embeddings - 1))
            if bucket_idx is not None:
                add = add + self.bucket_emb(bucket_idx.clamp(0, self.bucket_emb.num_embeddings - 1))
            x = x + add

        # mamba stack
        for i, mamba_block in enumerate(self.encoder):
            residual = x
            x = self.norms[i](x)
            x = mamba_block(x)
            x = x + residual

        raw = self.head(x).squeeze(-1)         # (B, L)
        logits = F.softplus(raw)               # >= 0, with gradients everywhere (no dead ReLU)
        logits = logits * pad_mask.float()     # keep masked positions zero
        return logits


class TEHeadFullCounts(nn.Module):
    """
    TE head that reads FULL per-position counts:
      - element-wise small MLP over counts (log1p optional)
      - masked mean pooling -> scalar
      - final linear + sigmoid to produce a value in [0,1] (scaled TE)
    """
    def __init__(self, hidden=64, use_log1p=True, positive_output=True):
        super().__init__()
        self.use_log1p = use_log1p
        self.positive_output = positive_output
        self.ff = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1)
        )
        self.post = nn.Linear(1, 1)
        self.final_act = nn.Sigmoid() if positive_output else nn.Identity()

    def forward(self, counts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.log1p(counts) if self.use_log1p else counts
        x = self.ff(x.unsqueeze(-1)).squeeze(-1)   # (B,L)
        m = mask.float()
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (x * m).sum(dim=1, keepdim=True) / denom
        out = self.post(pooled).squeeze(1)         # (B,)
        out = self.final_act(out)
        return out                                  # scaled TE in [0,1]


class MambaTEFull(nn.Module):
    def __init__(self, base: RiboPolisherMamba, hidden=64, use_log1p=True):
        super().__init__()
        self.base = base
        self.te_head = TEHeadFullCounts(hidden=hidden, use_log1p=use_log1p, positive_output=True)

    def forward(self, cod_ids, sim_feat, mask, angle_bin=None, pair_bin=None, bucket_idx=None):
        logits = self.base(cod_ids, sim_feat, mask,
                           angle_bin=angle_bin, pair_bin=pair_bin, bucket_idx=bucket_idx)
        counts = torch.expm1(torch.clamp(logits, max=20.0)) * mask.float()
        te_pred_scaled = self.te_head(counts, mask)   # in [0,1]
        return counts, te_pred_scaled

# ---------------------------
# Metrics (Pearson & Spearman)
# ---------------------------
def _pearson_safe(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if x.size < 2: return float("nan")
    xm = x - x.mean(); ym = y - y.mean()
    denom = np.sqrt((xm*xm).sum()) * np.sqrt((ym*ym).sum())
    return float((xm*ym).sum() / denom) if denom != 0 else float("nan")

def _rankdata_avg_ties(a: np.ndarray) -> np.ndarray:
    """Average-rank tie handling ('average' method), 1-based ranks."""
    n = a.size
    if n == 0: return np.array([], dtype=np.float64)
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and a[order[j]] == a[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks

def _spearman_safe(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if x.size < 2: return float("nan")
    rx = _rankdata_avg_ties(x)
    ry = _rankdata_avg_ties(y)
    return _pearson_safe(rx, ry)

# ---------------------------
# TE CSV
# ---------------------------
def read_te_csv(te_csv: Path, te_col: str = "mean_te") -> Dict[Tuple[str, str], float]:
    """
    Reads a CSV with at least columns: split, match_tx, <te_col>
    Returns: {(split, match_tx) -> <te_col>}
    """
    mp = {}
    with open(te_csv, "r", newline="") as fh:
        rd = csv.DictReader(fh)
        need = {"split", "match_tx", te_col}
        if not rd.fieldnames or not set(k.strip() for k in rd.fieldnames) >= need:
            raise RuntimeError(f"TE CSV must contain columns: split, match_tx, {te_col}")
        for r in rd:
            split = str(r["split"]).strip().lower()
            tx    = str(r["match_tx"]).strip()
            v     = str(r[te_col]).strip()
            if not split or not tx or v == "": continue
            try:
                mp[(split, tx)] = float(v)
            except Exception:
                pass
    return mp

def list_pkls_for_split(pkl_root: Path, split: str) -> List[str]:
    return sorted(glob.glob(str(pkl_root / split / "*_sample_*.pkl")))

# ---------------------------
# CDS-hash split filtering
# ---------------------------
def cds_hash_from_pkl(pkl_path: str) -> str:
    try:
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)
            cods = data[1]
        cds = "".join(cods).upper().replace("U", "T")
        return hashlib.md5(cds.encode("utf-8")).hexdigest()
    except Exception:
        return None

def compute_hashes_parallel(file_list, n_proc=16):
    with Pool(n_proc) as pool:
        hashes = list(tqdm(pool.imap(cds_hash_from_pkl, file_list),
                           total=len(file_list), desc="Hashing CDS"))
    return {fp: h for fp, h in zip(file_list, hashes) if h is not None}

def get_hashes_cached(file_list, cache_path):
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f: cache = pickle.load(f)
        except Exception: cache = {}
    missing = [f for f in file_list if f not in cache]
    if missing:
        print(f"Computing hashes for {len(missing)} missing files ...")
        new_hashes = compute_hashes_parallel(missing, n_proc=min(32, cpu_count()))
        cache.update(new_hashes)
        with open(cache_path, "wb") as f: pickle.dump(cache, f)
    return cache

def filter_target_files_excluding_ref_cds(ref_files, target_files, cache_dir,
                                          ref_name="train", target_name="test"):
    os.makedirs(cache_dir, exist_ok=True)
    ref_cache   = os.path.join(cache_dir, f"{ref_name}_cds_hashes.pkl")
    tgt_cache   = os.path.join(cache_dir, f"{target_name}_cds_hashes.pkl")
    ref_for_hash = [fp for fp in ref_files if fp.endswith("_sample_0.pkl")] or list(ref_files)
    ref_hash_map = get_hashes_cached(ref_for_hash, ref_cache)
    ref_hashes   = set(ref_hash_map.values())
    tgt_hash_map = get_hashes_cached(target_files, tgt_cache)
    kept, removed = [], []
    for fp in target_files:
        h = tgt_hash_map.get(fp)
        if h and h in ref_hashes:
            removed.append(fp)
        else:
            kept.append(fp)
    return kept, removed

# ---------------------------
# Geomap loading & helpers
# ---------------------------
def load_geomap(path: Optional[Path]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load a pickle mapping: tx -> (angle_dev_sum, pair_count)
    If path is None or missing, returns {}.
    """
    if path is None: return {}
    p = Path(path)
    if not p.exists(): return {}
    with open(p, "rb") as f:
        gm = pickle.load(f)
    return gm if isinstance(gm, dict) else {}

def compute_geo_bins_for_tx(tx: str,
                            geomap: Dict[str, Tuple[np.ndarray, np.ndarray]],
                            angle_bins: np.ndarray,
                            L: int,
                            k_angle_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute angle_bin (0..k-1) and pair_bin (0..3) for first L positions of tx.
    Falls back to zeros if tx not in geomap.
    """
    if (tx in geomap) and geomap[tx] is not None:
        angle_dev_sum, pair_count = geomap[tx]
        a = np.asarray(angle_dev_sum, dtype=np.float64)[:L]
        p = np.asarray(pair_count,    dtype=np.int64)[:L]
        kk = np.digitize(a, angle_bins) - 1
        kk = np.clip(kk, 0, k_angle_bins - 1).astype(np.int64)
        pb = np.clip(p,  0, 3).astype(np.int64)
    else:
        kk = np.zeros(L, dtype=np.int64)
        pb = np.zeros(L, dtype=np.int64)
    return kk, pb

# ---------------------------
# Dataset (uses real GEOMAPS)
# ---------------------------
class PKLTEFineTune(Dataset):
    def __init__(self,
                 pkl_files: List[str],
                 split_label: str,
                 te_map: Dict[Tuple[str,str], float],
                 geomap: Dict[str, Tuple[np.ndarray, np.ndarray]],
                 angle_bins: np.ndarray,
                 log_input: bool = True):
        self.files = []
        self.labels = []
        self.split = split_label
        self.log_input = log_input
        self.geomap = geomap
        self.angle_bins = np.asarray(angle_bins, dtype=np.float64)
        self.k_angle_bins = max(1, len(self.angle_bins) - 1)

        for p in pkl_files:
            base = os.path.basename(p)
            tx = base[: base.rfind("_sample_")]  # recover transcript id
            key = (split_label, tx)
            if key in te_map:
                self.files.append(p); self.labels.append(te_map[key])

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        with open(path, "rb") as fh:
            tx, cods, obs_counts, sim_vec_raw, sim_vec_scaled, scaling_factor, completed_total = pickle.load(fh)

        L = len(cods)
        cod_ids = np.array([CODON2IDX[c] for c in cods], dtype=np.int64)

        # use sim_vec_raw (unscaled)
        sim = np.asarray(sim_vec_raw, dtype=np.float32) if sim_vec_raw is not None else np.zeros(L, dtype=np.float32)
        sim_feat = np.log1p(sim) if self.log_input else sim

        # --- GEO features from provided geomap (no zeroing unless missing) ---
        angle_bin, pair_bin = compute_geo_bins_for_tx(
            tx, self.geomap, self.angle_bins, L, self.k_angle_bins
        )

        # bucket thirds along CDS
        bb = np.zeros(L, dtype=np.int64)
        l1 = L // 3
        l2 = (2 * L) // 3
        bb[l1:l2] = 1
        bb[l2:] = 2

        y = float(self.labels[i])

        return {
            "tx": tx,
            "length": L,
            "cod_ids": torch.tensor(cod_ids, dtype=torch.long),
            "sim_feat": torch.tensor(sim_feat, dtype=torch.float32),
            "mask": torch.ones(L, dtype=torch.bool),
            "angle_bin": torch.tensor(angle_bin, dtype=torch.long),
            "pair_bin":  torch.tensor(pair_bin,  dtype=torch.long),
            "bucket_idx":torch.tensor(bb,       dtype=torch.long),
            "te": torch.tensor(y, dtype=torch.float32),
        }

def pad_collate(batch):
    batch = sorted(batch, key=lambda x: x["length"], reverse=True)
    B, Lmax = len(batch), batch[0]["length"]
    cod = torch.full((B, Lmax), 64, dtype=torch.long)
    sim = torch.zeros((B, Lmax), dtype=torch.float32)
    msk = torch.zeros((B, Lmax), dtype=torch.bool)
    ang = torch.zeros((B, Lmax), dtype=torch.long)
    pai = torch.zeros((B, Lmax), dtype=torch.long)
    buc = torch.zeros((B, Lmax), dtype=torch.long)
    te  = torch.zeros(B, dtype=torch.float32)
    txs = []
    for i, ex in enumerate(batch):
        L = ex["length"]; txs.append(ex["tx"])
        cod[i,:L] = ex["cod_ids"]
        sim[i,:L] = ex["sim_feat"]
        msk[i,:L] = True
        ang[i,:L] = ex["angle_bin"]
        pai[i,:L] = ex["pair_bin"]
        buc[i,:L] = ex["bucket_idx"]
        te[i]     = ex["te"]
    return {"tx": txs, "cod_ids": cod, "sim_feat": sim, "mask": msk,
            "angle_bin": ang, "pair_bin": pai, "bucket_idx": buc, "te": te}

# ---------------------------
# TE transform helpers
# ---------------------------
def make_te_transform(kind: str, lo: float, hi: float, eps: float) -> Dict[str, float]:
    if kind != "minmax":
        raise ValueError("Only 'minmax' transform is implemented.")
    if not (hi > lo):
        raise ValueError("te_max must be greater than te_min.")
    return {"name": "minmax", "lo": float(lo), "hi": float(hi), "eps": float(eps)}

def te_forward_torch(y: torch.Tensor, trans: Dict[str, float]) -> torch.Tensor:
    """Map real TE -> [0,1] for training."""
    lo, hi = trans["lo"], trans["hi"]
    z = (y - lo) / (hi - lo)
    return z.clamp(0.0, 1.0)

def te_inverse_numpy(z: np.ndarray, trans: Dict[str, float]) -> np.ndarray:
    """Map [0,1] -> real TE, clipping to a safe interior with eps."""
    eps = float(trans.get("eps", 1e-6))
    zz = np.clip(np.asarray(z, dtype=np.float64), eps, 1.0 - eps)
    return zz * (trans["hi"] - trans["lo"]) + trans["lo"]

# ---------------------------
# Train / Eval
# ---------------------------
def run_epoch(model, dl, device, trans: Dict[str, float], optim=None):
    train = optim is not None
    model.train() if train else model.eval()
    total_loss_scaled = 0.0; nobs = 0
    preds_scaled, trues_scaled = [], []   # for sanity/debug
    preds_raw, trues_raw, txs = [], [], []
    mse = nn.MSELoss(reduction="sum")

    with torch.set_grad_enabled(train):
        for bo in dl:
            cod = bo["cod_ids"].to(device)
            sim = bo["sim_feat"].to(device)
            msk = bo["mask"].to(device)
            ang = bo["angle_bin"].to(device)
            pai = bo["pair_bin"].to(device)
            buc = bo["bucket_idx"].to(device)
            y_real = bo["te"].to(device)   # real TE in [-2,2] (by default)

            # forward
            _, yhat_scaled = model(cod, sim, msk, angle_bin=ang, pair_bin=pai, bucket_idx=buc)  # [0,1]
            y_scaled = te_forward_torch(y_real, trans)  # [0,1]

            loss = mse(yhat_scaled, y_scaled)

            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

            bs = y_real.size(0)
            total_loss_scaled += float(loss.item()); nobs += bs

            # detach to CPU
            yhat_s = yhat_scaled.detach().cpu().numpy()
            y_s    = y_scaled.detach().cpu().numpy()
            y_r    = y_real.detach().cpu().numpy()
            # inverse to real space for metrics
            yhat_r = te_inverse_numpy(yhat_s, trans)

            preds_scaled.append(yhat_s)
            trues_scaled.append(y_s)
            preds_raw.append(yhat_r)
            trues_raw.append(y_r)
            txs.extend(bo["tx"])

    if nobs == 0:
        return {
            "mse": float("nan"),
            "mse_scaled": float("nan"),
            "pearson_r": float("nan"),
            "spearman_r": float("nan"),
            "preds": [], "trues": [], "tx": []
        }

    yhat_scaled_all = np.concatenate(preds_scaled) if preds_scaled else np.array([])
    y_scaled_all    = np.concatenate(trues_scaled) if trues_scaled else np.array([])
    yhat_raw_all    = np.concatenate(preds_raw)    if preds_raw    else np.array([])
    y_raw_all       = np.concatenate(trues_raw)    if trues_raw    else np.array([])

    # metrics in real TE space
    r  = _pearson_safe(y_raw_all, yhat_raw_all)
    ro = _spearman_safe(y_raw_all, yhat_raw_all)

    # raw-space MSE for logging, scaled-space MSE == optimization objective / sample
    mse_raw = float(np.mean((yhat_raw_all - y_raw_all) ** 2)) if y_raw_all.size else float("nan")
    mse_scaled_mean = float(np.mean((yhat_scaled_all - y_scaled_all) ** 2)) if y_scaled_all.size else float("nan")

    return {
        "mse": mse_raw,  # keep 'mse' as raw-space MSE for compatibility
        "mse_scaled": mse_scaled_mean,
        "pearson_r": r,
        "spearman_r": ro,
        "preds": yhat_raw_all.tolist(),   # real TE space
        "trues": y_raw_all.tolist(),      # real TE space
        "tx": txs
    }

# ---------------------------
# Main
# ---------------------------

def per_transcript_mean(txs, trues, preds):
    """Average the per-simulation predictions of each transcript, then correlate with the labels."""
    acc = {}
    for tx, yt, yp in zip(txs, trues, preds):
        a = acc.setdefault(tx, [float(yt), []])
        a[1].append(float(yp))
    y = np.array([v[0] for v in acc.values()])
    p = np.array([np.mean(v[1]) for v in acc.values()])
    return {"n_tx": int(len(y)), "pearson_r": _pearson_safe(p, y), "spearman_r": _spearman_safe(p, y),
            "mse": float(np.mean((p - y) ** 2)) if len(y) else float("nan")}


def main():
    base_dir = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-line", choices=["ipsc", "hek293", "rpe", "lcl"], required=True)
    ap.add_argument("--pkl_root", default=None)
    ap.add_argument("--te_csv", default=None)
    ap.add_argument("--te_col", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--geomap_train", type=str, default=None, help="Pickle: tx -> (angle_dev_sum, pair_count) for TRAIN")
    ap.add_argument("--geomap_val", type=str, default=None, help="Pickle: tx -> (angle_dev_sum, pair_count) for VAL")
    ap.add_argument("--geomap_test", type=str, default=None, help="Pickle: tx -> (angle_dev_sum, pair_count) for TEST")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--freeze_base", action="store_true")
    ap.add_argument("--ff_hidden", type=int, default=256)
    ap.add_argument("--ff_use_log1p", action=argparse.BooleanOptionalAction, default=None,
                    help="log1p of polished counts before the per-position head (default: CELL_CONFIG)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--te_transform", choices=["minmax"], default="minmax", help="Positive-space transform for TE")
    ap.add_argument("--te_min", type=float, default=None)
    ap.add_argument("--te_max", type=float, default=None)
    ap.add_argument("--te_eps", type=float, default=1e-6)
    args = ap.parse_args()
    cfg = CELL_CONFIG[args.cell_line]
    if args.ff_use_log1p is None:
        args.ff_use_log1p = cfg["ff_use_log1p"]

    if args.pkl_root is None:
        args.pkl_root = str(base_dir / "results" / "TE" / args.cell_line / "run_unscaled")
    if args.te_csv is None:
        args.te_csv = str(Path(args.pkl_root) / "matched_mean_te_with_completed_total_train_val_test.csv")
    if args.te_col is None:
        args.te_col = cfg["te_col"]
    if args.weights is None:
        args.weights = str(base_dir / "results" / "polisher" / args.cell_line / "mamba_final_unscaled.pt")
        if not Path(args.weights).exists():
            args.weights = str(base_dir.parents[1] / "weights" / "paper" / f"{args.cell_line}_mamba_final_unscaled.pt")
    print("Base polisher weights:", args.weights)
    if args.out_dir is None:
        args.out_dir = str(base_dir / "results" / "TE" / args.cell_line / "finetune_cds")
    if args.geomap_train is None:
        args.geomap_train = str(base_dir / "data" / args.cell_line / "polisher" / "geo_map_train.pkl")
    if args.geomap_val is None:
        args.geomap_val = str(base_dir / "data" / args.cell_line / "polisher" / "geo_map_val.pkl")
    if args.geomap_test is None:
        args.geomap_test = str(base_dir / "data" / args.cell_line / "polisher" / "geo_map_test.pkl")
    if args.batch_size is None:
        args.batch_size = int(cfg["batch_size"])
    if args.epochs is None:
        args.epochs = int(cfg["epochs"])
    if args.device is None:
        args.device = "cuda"
    if args.te_min is None:
        args.te_min = float(cfg["te_min"])
    if args.te_max is None:
        args.te_max = float(cfg["te_max"])

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    PKL_ROOT = Path(args.pkl_root)
    OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)

    # Load TE labels
    te_map = read_te_csv(Path(args.te_csv), te_col=args.te_col)
    # Geomaps (per your previous usage)
    geo_train = load_geomap(Path(args.geomap_train) if args.geomap_train else None)
    geo_val   = load_geomap(Path(args.geomap_val)   if args.geomap_val   else None)
    geo_test  = load_geomap(Path(args.geomap_test)  if args.geomap_test  else None)
    # Angle bins
    ANGLE_BINS = derive_angle_bins_from_training(
    geo_train, K_ANGLE_BINS, method="linear", low_q=0.0, high_q=0.995)
    print("Derived ANGLE_BINS:", ANGLE_BINS, "len=", len(ANGLE_BINS))  
    angle_bins = ANGLE_BINS
    k_angle_bins = K_ANGLE_BINS
    # TE transform (saved for plotting and downstream use)
    trans = make_te_transform(args.te_transform, args.te_min, args.te_max, args.te_eps)
    with open(OUT / "te_transform.json", "w") as fh:
        json.dump(trans, fh, indent=2)

    # PKL lists
    train_pkls = list_pkls_for_split(PKL_ROOT, "train")
    val_pkls   = list_pkls_for_split(PKL_ROOT, "val")
    test_pkls  = list_pkls_for_split(PKL_ROOT, "test")
    print(f"PKLs -> train={len(train_pkls)}  val={len(val_pkls)}  test={len(test_pkls)}")

    # CDS-hash split filtering
    cache_dir = str(PKL_ROOT / "cds_hash_cache")
    val_pkls, val_removed = filter_target_files_excluding_ref_cds(
        train_pkls, val_pkls, cache_dir, ref_name="train", target_name="val")
    print(f"[filter] VAL vs TRAIN: removed={len(val_removed)} kept={len(val_pkls)}")

    test_no_train, test_removed_train = filter_target_files_excluding_ref_cds(
        train_pkls, test_pkls, cache_dir, ref_name="train", target_name="test")
    print(f"[filter] TEST vs TRAIN: removed={len(test_removed_train)} kept={len(test_no_train)}")

    test_pkls, test_removed_val = filter_target_files_excluding_ref_cds(
        val_pkls, test_no_train, cache_dir, ref_name="val", target_name="test")
    print(f"[filter] TEST vs VAL: removed={len(test_removed_val)} kept={len(test_pkls)}")



    # Datasets (filter to those with TE labels) and pass REAL geomaps
    train_ds = PKLTEFineTune(train_pkls, "train", te_map, geo_train, angle_bins, log_input=True)
    val_ds   = PKLTEFineTune(val_pkls,   "val",   te_map, geo_val,   angle_bins, log_input=True) if val_pkls else None
    test_ds  = PKLTEFineTune(test_pkls,  "test",  te_map, geo_test,  angle_bins, log_input=True)

    # Dataloaders
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=pad_collate, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          collate_fn=pad_collate, pin_memory=True) if val_ds else None
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                          collate_fn=pad_collate, pin_memory=True)

    # Model (unfrozen by default)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base = RiboPolisherMamba(d_model=192, n_layers=4, d_state=16, d_conv=4,
                             dropout=0.1, k_angle_bins=k_angle_bins).to(device)

    state = torch.load(args.weights, map_location=device)
    base.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state)

    model = MambaTEFull(base, hidden=args.ff_hidden, use_log1p=args.ff_use_log1p).to(device)
    if args.freeze_base:
        for p in model.base.parameters(): p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    # Train
    best_val = float("inf")
    best_path = OUT / "mamba_te_full_best.pt"
    log = {"epochs": [],
           "train_mse": [], "train_mse_scaled": [], "train_pearson_r": [], "train_spearman_r": [],
           "val_mse": [],   "val_mse_scaled": [],   "val_pearson_r": [],   "val_spearman_r": []}

    for ep in range(1, args.epochs + 1):
        tr = run_epoch(model, train_dl, device, trans, optim=optim)
        if val_dl is not None:
            va = run_epoch(model, val_dl, device, trans, optim=None)
            val_mse_raw = va["mse"]
        else:
            va = {"mse": float("nan"), "mse_scaled": float("nan"),
                  "pearson_r": float("nan"), "spearman_r": float("nan")}
            val_mse_raw = tr["mse"]

        log["epochs"].append(ep)
        # store RAW-space MSE under the legacy keys, plus scaled-space too
        log["train_mse"].append(tr["mse"])
        log["train_mse_scaled"].append(tr["mse_scaled"])
        log["train_pearson_r"].append(tr["pearson_r"])
        log["train_spearman_r"].append(tr["spearman_r"])
        log["val_mse"].append(va["mse"])
        log["val_mse_scaled"].append(va["mse_scaled"])
        log["val_pearson_r"].append(va["pearson_r"])
        log["val_spearman_r"].append(va["spearman_r"])

        print(f"[Ep {ep:03d}] "
              f"train MSE(raw)={tr['mse']:.4f}  MSE(scaled)={tr['mse_scaled']:.4f}  "
              f"r={tr['pearson_r']:.3f}  ρ={tr['spearman_r']:.3f} | "
              f"val MSE(raw)={val_mse_raw:.4f}  MSE(scaled)={va['mse_scaled']:.4f}  "
              f"r={va['pearson_r']:.3f}  ρ={va['spearman_r']:.3f}")

        # select best by RAW-space validation MSE
        if val_mse_raw < best_val:
            best_val = val_mse_raw
            torch.save(model.state_dict(), best_path)

    # Load best and evaluate on test
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    te = run_epoch(model, test_dl, device, trans, optim=None)
    print(f"[TEST] MSE(raw)={te['mse']:.4f}  MSE(scaled)={te['mse_scaled']:.4f}  "
          f"r={te['pearson_r']:.3f}  ρ={te['spearman_r']:.3f} (n={len(te['preds'])})")

    # Write logs + artifacts
    with open(OUT / "train_log.json", "w") as fh: json.dump(log, fh, indent=2)

    test_pred_csv = OUT / "mamba_te_full_test_predictions.csv"
    with open(test_pred_csv, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["tx","te_true","te_pred"])  # REAL TE values
        for tx, yt, yp in zip(te["tx"], te["trues"], te["preds"]):
            w.writerow([tx, f"{float(yt):.6f}", f"{float(yp):.6f}"])

    per_tx = per_transcript_mean(te["tx"], te["trues"], te["preds"])
    print(f"[TEST] per-transcript mean over simulations: r={per_tx['pearson_r']:.3f} (n_tx={per_tx['n_tx']})")
    with open(OUT / "test_metrics.json", "w") as fh:
        json.dump({"n": len(te["preds"]),
                   "mse_raw": te["mse"],
                   "mse_scaled": te["mse_scaled"],
                   "pearson_r": te["pearson_r"],
                   "spearman_r": te["spearman_r"],
                   "per_transcript": per_tx}, fh, indent=2)

    torch.save(model.state_dict(), OUT / "mamba_te_full_final.pt")
    print("Saved:")
    print(" ", best_path if best_path.exists() else "(no best saved)")
    print(" ", OUT / "mamba_te_full_final.pt")
    print(" ", test_pred_csv)
    print(" ", OUT / "test_metrics.json")
    print(" ", OUT / "te_transform.json")

if __name__ == "__main__":
    main()
