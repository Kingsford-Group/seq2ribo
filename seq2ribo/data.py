"""Dataset classes and utilities for seq2ribo training and inference."""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import CODON2IDX

# -----------------------------
# Data Loading Helpers
# -----------------------------

def load_geomap(path: Optional[Path]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
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

# -----------------------------
# Dataset Classes
# -----------------------------

class PolishPKLDataset(Dataset):
    """
    Dataset for Polisher training/inference from PKL files.
    """
    def __init__(self, file_list, log_input=True, log_target=True, geomap=None, angle_bins=None, k_angle_bins=4):
        self.files = list(file_list)
        self.log_input = log_input
        self.log_target = log_target
        self.geomap = geomap or {}
        self.angle_bins = np.asarray(angle_bins, dtype=np.float64) if angle_bins is not None else None
        self.k_angle_bins = int(k_angle_bins)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with open(self.files[i], "rb") as fh:
            # Format: tx, cods, obs_counts, sim_vec_raw, sim_vec_scaled, scaling_factor, completed_total
            data = pickle.load(fh)
            tx = data[0]
            cods = data[1]
            obs_counts = data[2]
            sim_vec_scaled = data[4]

        L = len(cods)
        cod_ids = np.array([CODON2IDX.get(c, 64) for c in cods], dtype=np.int64) # 64 is pad, safely handle unknown
        sim = np.asarray(sim_vec_scaled, dtype=np.float32)
        tgt = np.asarray(obs_counts, dtype=np.float32)

        sim_feat = np.log1p(sim) if self.log_input else sim
        tgt_val  = np.log1p(tgt) if self.log_target else tgt

        if (self.angle_bins is not None) and (tx in self.geomap):
            angle_dev_sum, pair_count = self.geomap[tx]
            a = np.asarray(angle_dev_sum, dtype=np.float64)[:L]
            p = np.asarray(pair_count,    dtype=np.int64)[:L]
            kk = np.digitize(a, self.angle_bins) - 1
            kk = np.clip(kk, 0, self.k_angle_bins - 1).astype(np.int64)
            pb = np.clip(p, 0, 3).astype(np.int64)
        else:
            kk = np.zeros(L, dtype=np.int64)
            pb = np.zeros(L, dtype=np.int64)

        bb = np.zeros(L, dtype=np.int64)
        l1 = L // 3
        l2 = (2 * L) // 3
        bb[l1:l2] = 1
        bb[l2:] = 2

        return {
            "tx": tx,
            "cod_ids": torch.tensor(cod_ids, dtype=torch.long),
            "sim_feat": torch.tensor(sim_feat, dtype=torch.float32),
            "target": torch.tensor(tgt_val, dtype=torch.float32),
            "length": int(L),
            "angle_bin": torch.tensor(kk, dtype=torch.long),
            "pair_bin": torch.tensor(pb, dtype=torch.long),
            "bucket_idx": torch.tensor(bb, dtype=torch.long),
        }

def pad_collate(batch):
    batch = sorted(batch, key=lambda x: x["length"], reverse=True)
    B, Lmax = len(batch), batch[0]["length"]

    cod = torch.full((B, Lmax), 64, dtype=torch.long)
    sim = torch.zeros((B, Lmax), dtype=torch.float32)
    tgt = torch.zeros((B, Lmax), dtype=torch.float32)
    msk = torch.zeros((B, Lmax), dtype=torch.bool)
    
    angle_bin = torch.zeros((B, Lmax), dtype=torch.long)
    pair_bin = torch.zeros((B, Lmax), dtype=torch.long)
    bucket_idx = torch.zeros((B, Lmax), dtype=torch.long)

    txs = []
    expr_vals = []
    
    has_expr = "expr" in batch[0]

    for i, ex in enumerate(batch):
        L = ex["length"]
        txs.append(ex["tx"])
        cod[i, :L] = ex["cod_ids"]
        sim[i, :L] = ex["sim_feat"]
        if "target" in ex:
            tgt[i, :L] = ex["target"]
        msk[i, :L] = True

        angle_bin[i, :L] = ex["angle_bin"]
        pair_bin[i, :L] = ex["pair_bin"]
        bucket_idx[i, :L] = ex["bucket_idx"]
        
        if has_expr:
            expr_vals.append(ex["expr"])

    out = {
        "tx": txs,
        "cod_ids": cod,
        "sim_feat": sim,
        "mask": msk,
        "angle_bin": angle_bin,
        "pair_bin": pair_bin,
        "bucket_idx": bucket_idx,
    }
    if "target" in batch[0]:
        out["target"] = tgt
    if has_expr:
        out["expr"] = torch.stack(expr_vals)
        
    return out
