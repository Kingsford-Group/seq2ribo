#!/usr/bin/env python3
# End-to-end fine-tune: Mamba polisher (unfrozen by default) + FULL-counts + UTR -> TE head
# Extends finetune_unscaled.py by adding 5'UTR and 3'UTR nucleotide sequences
# as additional inputs to a Mamba-based TE prediction head.
#
# Architecture:
#   Stage 1: Pretrained RiboPolisherMamba generates per-position ribo counts from CDS
#   Stage 2: 3 separate Mamba blocks for 5'UTR, CDS, 3'UTR + count summary
#            -> pooled outputs concatenated -> MLP -> TE prediction
#
# Trains in a positive space by scaling TE -> [0,1],
# evaluates & saves metrics/CSV back in the original TE space.
#
# Saves:
#   - train_log.json  (per-epoch train/val MSE_raw, MSE_scaled, Pearson r, Spearman rho)
#   - te_transform.json (transform + inverse parameters)
#   - mamba_te_utr_test_predictions.csv  (tx, te_true, te_pred)  # real TE space
#   - test_metrics.json  (n, MSE_raw, MSE_scaled, Pearson r, Spearman rho)
#   - mamba_te_utr_best.pt / mamba_te_utr_final.pt
#
# UTR data:
#   - 5'/3' UTRs of all splits from the TE workbook (--utr_xlsx; --utr_unversioned matches ids without version)
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
import torch.nn.functional as F

K_ANGLE_BINS = 4

CELL_CONFIG = {
    "ipsc": {
        "te_col": "mean_te", "batch_size": 64, "epochs": 10, "te_min": -2.639472099, "te_max": 3.489203289,
    },
    "hek293": {
        "te_col": "TE_HEK293", "batch_size": 64, "epochs": 10, "te_min": -2.666693891, "te_max": 3.401144287,
    },
    "rpe": {
        "te_col": "TE_RPE-1", "batch_size": 64, "epochs": 10, "te_min": -3.625209297, "te_max": 4.258473735,
    },
    "lcl": {
        "te_col": "TE_LCL", "batch_size": 64, "epochs": 10, "te_min": -3.145896817, "te_max": 7.362059739,
    },
}

# ---------------------------
# Nucleotide tokenizer
# ---------------------------
NUC2IDX = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
NUC_PAD_IDX = 4
NUC_VOCAB_SIZE = 5  # A, U, G, C, PAD

def tokenize_nuc_seq(seq: str) -> np.ndarray:
    """Convert nucleotide string to integer array. Unknown chars -> 0 (A)."""
    return np.array([NUC2IDX.get(c, 0) for c in seq.upper().replace('T', 'U')], dtype=np.int64)

# ---------------------------
# UTR data loading
# ---------------------------
# RiboNN input length limits; transcripts beyond them were excluded from the RiboNN baseline and
# from the UTR-aware head so that both see the same transcripts
MAX_UTR5_LEN = 1381
MAX_CDS_UTR3_LEN = 11937


def load_utr_from_xlsx(path: str, unversioned: bool = False) -> Dict[str, Tuple[str, str]]:
    """{tx_id -> (utr5_seq, utr3_seq)} from the Zheng et al. workbook (tx_sequence + region sizes)."""
    import pandas as pd
    df = pd.read_excel(path, converters={"tx_id": str})
    df.columns = df.columns.str.strip()
    utr_map = {}
    for tx_id, seq, u5, cds, u3 in zip(df["tx_id"], df["tx_sequence"], df["utr5_size"], df["cds_size"], df["utr3_size"]):
        u5, cds, u3 = int(u5), int(cds), int(u3)
        if u5 > MAX_UTR5_LEN or cds + u3 > MAX_CDS_UTR3_LEN:
            continue
        seq = str(seq).strip().upper().replace("T", "U")
        if u5 + cds + u3 != len(seq):
            u3 = max(0, len(seq) - (u5 + cds))
        key = str(tx_id).strip()
        utr_map[key.split(".")[0] if unversioned else key] = (seq[:u5], seq[u5 + cds:u5 + cds + u3])
    return utr_map

# ---------------------------
# Angle bins
# ---------------------------
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
    if method == "quantile":
        qs = np.linspace(low_q, high_q, k_bins + 1)
        edges = np.quantile(vals, qs)
        edges = np.maximum.accumulate(edges)
        edges[-1] = max(edges[-1], float(np.max(vals)))
    else:
        lo = float(np.quantile(vals, low_q))
        hi = float(np.quantile(vals, high_q))
        if hi <= lo:
            hi = lo + 1e-6
        edges = np.linspace(lo, hi, k_bins + 1)
        edges[-1] = max(edges[-1], float(np.max(vals)))
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-6
    return edges

# ---------------------------
# Model: Mamba polisher (base, same as finetune_unscaled.py)
# ---------------------------
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
        self.cod_emb  = nn.Embedding(65, d_model)
        self.sim_proj = nn.Linear(1, d_model)
        self.angle_emb  = nn.Embedding(k_angle_bins, d_model)
        self.pair_emb   = nn.Embedding(4, d_model)
        self.bucket_emb = nn.Embedding(3, d_model)
        self.encoder = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        nn.init.xavier_uniform_(self.sim_proj.weight)
        nn.init.zeros_(self.sim_proj.bias)
        nn.init.xavier_uniform_(self.angle_emb.weight)
        nn.init.xavier_uniform_(self.pair_emb.weight)
        nn.init.xavier_uniform_(self.bucket_emb.weight)
        last = self.head[-1]
        if isinstance(last, nn.Linear) and last.bias is not None:
            nn.init.constant_(last.bias, 1.0)

    def forward(self, cod_ids, sim_feat, pad_mask, angle_bin=None, pair_bin=None, bucket_idx=None):
        tok = self.cod_emb(cod_ids)
        sim = self.sim_proj(sim_feat.unsqueeze(-1))
        x = tok + sim
        if (angle_bin is not None) or (pair_bin is not None) or (bucket_idx is not None):
            add = 0
            if angle_bin is not None:
                add = add + self.angle_emb(angle_bin.clamp(0, self.angle_emb.num_embeddings - 1))
            if pair_bin is not None:
                add = add + self.pair_emb(pair_bin.clamp(0, self.pair_emb.num_embeddings - 1))
            if bucket_idx is not None:
                add = add + self.bucket_emb(bucket_idx.clamp(0, self.bucket_emb.num_embeddings - 1))
            x = x + add
        for i, mamba_block in enumerate(self.encoder):
            residual = x
            x = self.norms[i](x)
            x = mamba_block(x)
            x = x + residual
        raw = self.head(x).squeeze(-1)
        logits = F.softplus(raw)
        logits = logits * pad_mask.float()
        return logits

# ---------------------------
# Helper: Mamba sequence block with masked mean pooling
# ---------------------------
class MambaSeqBlock(nn.Module):
    """
    A self-contained Mamba stack that processes a single sequence and returns
    a fixed-size summary via masked mean pooling.
    """
    def __init__(self, d_model, n_layers=2, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.encoder = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Args:
            x:    (B, L, d_model) - input embeddings
            mask: (B, L) bool     - True for real positions
        Returns:
            pooled: (B, d_model) - masked mean pooled output
        """
        # Zero out padded positions
        x = x * mask.unsqueeze(-1).float()

        # Mamba stack with residual connections
        for i, mamba_block in enumerate(self.encoder):
            residual = x
            x = self.norms[i](x)
            x = mamba_block(x)
            x = x + residual

        # Masked mean pooling
        m = mask.float().unsqueeze(-1)         # (B, L, 1)
        denom = m.sum(dim=1).clamp_min(1.0)    # (B, 1)
        pooled = (x * m).sum(dim=1) / denom    # (B, d_model)
        return pooled


# ---------------------------
# NEW: UTR-aware Mamba TE Head (separate blocks per region)
# ---------------------------
class MambaUTRTEHead(nn.Module):
    """
    3 separate Mamba blocks for 5'UTR, CDS, and 3'UTR, plus a count summary.

    Architecture:
      5'UTR nucleotides -> nuc_emb -> MambaSeqBlock -> pooled_5  (d_te)
      CDS counts        -> count_proj -> MambaSeqBlock -> pooled_cds (d_te)
      3'UTR nucleotides -> nuc_emb -> MambaSeqBlock -> pooled_3  (d_te)
      raw counts        -> log1p -> masked mean -> Linear -> count_summary (d_te)

      concat([count_summary, pooled_5, pooled_cds, pooled_3])  => (4 * d_te)
      -> MLP -> sigmoid -> TE in [0,1]
    """
    def __init__(self, d_te=128, n_te_layers=2, d_state=16, d_conv=4,
                 expand=2, dropout=0.1, use_log1p=True):
        super().__init__()
        self.d_te = d_te
        self.use_log1p = use_log1p

        # Shared nucleotide embedding for 5'UTR and 3'UTR
        self.nuc_emb = nn.Embedding(NUC_VOCAB_SIZE, d_te)

        # Count projection for CDS predicted counts -> d_te
        self.count_proj = nn.Linear(1, d_te)

        # 3 independent Mamba blocks
        self.mamba_5utr = MambaSeqBlock(d_te, n_layers=n_te_layers, d_state=d_state,
                                         d_conv=d_conv, expand=expand, dropout=dropout)
        self.mamba_cds  = MambaSeqBlock(d_te, n_layers=n_te_layers, d_state=d_state,
                                         d_conv=d_conv, expand=expand, dropout=dropout)
        self.mamba_3utr = MambaSeqBlock(d_te, n_layers=n_te_layers, d_state=d_state,
                                         d_conv=d_conv, expand=expand, dropout=dropout)

        # Count summary branch: scalar count -> pooled scalar -> linear -> d_te
        self.count_summary = nn.Sequential(
            nn.Linear(1, d_te),
            nn.GELU(),
        )

        # Final MLP:  4 * d_te -> TE
        self.te_mlp = nn.Sequential(
            nn.Linear(4 * d_te, d_te),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_te, d_te // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_te // 2, 1)
        )
        self.final_act = nn.Sigmoid()

        # Initializations
        nn.init.xavier_uniform_(self.nuc_emb.weight)
        nn.init.xavier_uniform_(self.count_proj.weight)
        nn.init.zeros_(self.count_proj.bias)

    def forward(self, utr5_ids, utr3_ids, counts, utr5_mask, cds_mask, utr3_mask):
        """
        Args:
            utr5_ids:  (B, N5) int - nucleotide token IDs for 5'UTR
            utr3_ids:  (B, N3) int - nucleotide token IDs for 3'UTR
            counts:    (B, L_cds) float - predicted ribo counts from base model
            utr5_mask: (B, N5) bool
            cds_mask:  (B, L_cds) bool
            utr3_mask: (B, N3) bool
        Returns:
            te_pred: (B,) float in [0,1]
        """
        # --- 5'UTR Mamba ---
        emb_5 = self.nuc_emb(utr5_ids)                        # (B, N5, d_te)
        pooled_5 = self.mamba_5utr(emb_5, utr5_mask)          # (B, d_te)

        # --- CDS Mamba (over count embeddings) ---
        c = torch.log1p(counts) if self.use_log1p else counts
        emb_cds = self.count_proj(c.unsqueeze(-1))             # (B, L_cds, d_te)
        pooled_cds = self.mamba_cds(emb_cds, cds_mask)         # (B, d_te)

        # --- 3'UTR Mamba ---
        emb_3 = self.nuc_emb(utr3_ids)                        # (B, N3, d_te)
        pooled_3 = self.mamba_3utr(emb_3, utr3_mask)          # (B, d_te)

        # --- Count summary (simple masked mean of log1p counts -> linear) ---
        c_log = torch.log1p(counts)                            # (B, L_cds)
        m = cds_mask.float()                                   # (B, L_cds)
        count_mean = (c_log * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1)
        count_sum = self.count_summary(count_mean)             # (B, d_te)

        # --- Concatenate all 4 summaries ---
        combined = torch.cat([count_sum, pooled_5, pooled_cds, pooled_3], dim=1)  # (B, 4*d_te)

        # --- MLP -> sigmoid ---
        out = self.te_mlp(combined).squeeze(-1)                # (B,)
        out = self.final_act(out)
        return out


class MambaTEFullUTR(nn.Module):
    """
    Top-level model:
      1. base (RiboPolisherMamba) -> per-position ribo counts
      2. te_head (MambaUTRTEHead) -> TE prediction using counts + UTRs
    """
    def __init__(self, base: RiboPolisherMamba,
                 d_te=128, n_te_layers=2, te_d_state=16, te_d_conv=4,
                 te_expand=2, dropout=0.1, use_log1p=True):
        super().__init__()
        self.base = base
        self.te_head = MambaUTRTEHead(
            d_te=d_te, n_te_layers=n_te_layers,
            d_state=te_d_state, d_conv=te_d_conv,
            expand=te_expand, dropout=dropout, use_log1p=use_log1p
        )

    def forward(self, cod_ids, sim_feat, mask,
                utr5_ids, utr3_ids, utr5_mask, utr3_mask,
                angle_bin=None, pair_bin=None, bucket_idx=None):
        # Stage 1: get per-position ribo counts from base model
        logits = self.base(cod_ids, sim_feat, mask,
                           angle_bin=angle_bin, pair_bin=pair_bin, bucket_idx=bucket_idx)
        counts = torch.expm1(torch.clamp(logits, max=20.0)) * mask.float()

        # Stage 2: predict TE using counts + UTRs
        te_pred_scaled = self.te_head(utr5_ids, utr3_ids, counts,
                                       utr5_mask, mask, utr3_mask)
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
def read_te_csv(te_csv: Path, te_col: str = "mean_te") -> Dict[Tuple[str,str], float]:
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
    if path is None: return {}
    p = Path(path)
    if not p.exists(): return {}
    with open(p, "rb") as f:
        gm = pickle.load(f)
    return gm if isinstance(gm, dict) else {}

def compute_geo_bins_for_tx(tx, geomap, angle_bins, L, k_angle_bins):
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
# Dataset with UTR support
# ---------------------------
class PKLTEFineTuneUTR(Dataset):
    def __init__(self,
                 pkl_files: List[str],
                 split_label: str,
                 te_map: Dict[Tuple[str,str], float],
                 geomap: Dict[str, Tuple[np.ndarray, np.ndarray]],
                 angle_bins: np.ndarray,
                 utr_map: Dict[str, Tuple[str, str]],
                 log_input: bool = True):
        self.files = []
        self.labels = []
        self.split = split_label
        self.log_input = log_input
        self.geomap = geomap
        self.angle_bins = np.asarray(angle_bins, dtype=np.float64)
        self.k_angle_bins = max(1, len(self.angle_bins) - 1)
        self.utr_map = utr_map

        matched, unmatched_te, unmatched_utr = 0, 0, 0
        for p in pkl_files:
            base = os.path.basename(p)
            tx = base[: base.rfind("_sample_")]
            key = (split_label, tx)
            if key not in te_map:
                unmatched_te += 1
                continue
            if tx not in utr_map and tx.split(".")[0] not in utr_map:
                unmatched_utr += 1
                continue
            self.files.append(p)
            self.labels.append(te_map[key])
            matched += 1
        
        print(f"  [{split_label}] PKLs matched: {matched}, "
              f"no TE label: {unmatched_te}, no UTR: {unmatched_utr}")

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        with open(path, "rb") as fh:
            tx, cods, obs_counts, sim_vec_raw, sim_vec_scaled, scaling_factor, completed_total = pickle.load(fh)

        L = len(cods)
        cod_ids = np.array([CODON2IDX[c] for c in cods], dtype=np.int64)

        sim = np.asarray(sim_vec_raw, dtype=np.float32) if sim_vec_raw is not None else np.zeros(L, dtype=np.float32)
        sim_feat = np.log1p(sim) if self.log_input else sim

        angle_bin, pair_bin = compute_geo_bins_for_tx(
            tx, self.geomap, self.angle_bins, L, self.k_angle_bins
        )

        bb = np.zeros(L, dtype=np.int64)
        l1 = L // 3
        l2 = (2 * L) // 3
        bb[l1:l2] = 1
        bb[l2:] = 2

        # UTR sequences
        utr5_seq, utr3_seq = self.utr_map.get(tx, self.utr_map.get(tx.split(".")[0], ("", "")))
        utr5_ids = tokenize_nuc_seq(utr5_seq) if utr5_seq else np.array([], dtype=np.int64)
        utr3_ids = tokenize_nuc_seq(utr3_seq) if utr3_seq else np.array([], dtype=np.int64)

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
            "utr5_ids": torch.tensor(utr5_ids, dtype=torch.long),
            "utr3_ids": torch.tensor(utr3_ids, dtype=torch.long),
            "utr5_len": len(utr5_ids),
            "utr3_len": len(utr3_ids),
            "te": torch.tensor(y, dtype=torch.float32),
        }

def pad_collate_utr(batch):
    batch = sorted(batch, key=lambda x: x["length"], reverse=True)
    B = len(batch)
    Lmax = batch[0]["length"]
    N5max = max(ex["utr5_len"] for ex in batch) if batch else 0
    N3max = max(ex["utr3_len"] for ex in batch) if batch else 0
    # Ensure at least 1 for empty UTRs
    N5max = max(N5max, 1)
    N3max = max(N3max, 1)

    cod = torch.full((B, Lmax), 64, dtype=torch.long)
    sim = torch.zeros((B, Lmax), dtype=torch.float32)
    msk = torch.zeros((B, Lmax), dtype=torch.bool)
    ang = torch.zeros((B, Lmax), dtype=torch.long)
    pai = torch.zeros((B, Lmax), dtype=torch.long)
    buc = torch.zeros((B, Lmax), dtype=torch.long)
    u5  = torch.full((B, N5max), NUC_PAD_IDX, dtype=torch.long)
    u3  = torch.full((B, N3max), NUC_PAD_IDX, dtype=torch.long)
    u5m = torch.zeros((B, N5max), dtype=torch.bool)
    u3m = torch.zeros((B, N3max), dtype=torch.bool)
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
        n5 = ex["utr5_len"]
        n3 = ex["utr3_len"]
        if n5 > 0:
            u5[i,:n5] = ex["utr5_ids"]
            u5m[i,:n5] = True
        if n3 > 0:
            u3[i,:n3] = ex["utr3_ids"]
            u3m[i,:n3] = True
        te[i] = ex["te"]

    return {"tx": txs, "cod_ids": cod, "sim_feat": sim, "mask": msk,
            "angle_bin": ang, "pair_bin": pai, "bucket_idx": buc,
            "utr5_ids": u5, "utr3_ids": u3,
            "utr5_mask": u5m, "utr3_mask": u3m, "te": te}

# ---------------------------
# TE transform helpers
# ---------------------------
def make_te_transform(kind, lo, hi, eps):
    if kind != "minmax":
        raise ValueError("Only 'minmax' transform is implemented.")
    if not (hi > lo):
        raise ValueError("te_max must be greater than te_min.")
    return {"name": "minmax", "lo": float(lo), "hi": float(hi), "eps": float(eps)}

def te_forward_torch(y, trans):
    lo, hi = trans["lo"], trans["hi"]
    z = (y - lo) / (hi - lo)
    return z.clamp(0.0, 1.0)

def te_inverse_numpy(z, trans):
    eps = float(trans.get("eps", 1e-6))
    zz = np.clip(np.asarray(z, dtype=np.float64), eps, 1.0 - eps)
    return zz * (trans["hi"] - trans["lo"]) + trans["lo"]

# ---------------------------
# Train / Eval
# ---------------------------
def run_epoch(model, dl, device, trans, optim=None):
    train = optim is not None
    model.train() if train else model.eval()
    total_loss_scaled = 0.0; nobs = 0
    preds_scaled, trues_scaled = [], []
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
            u5  = bo["utr5_ids"].to(device)
            u3  = bo["utr3_ids"].to(device)
            u5m = bo["utr5_mask"].to(device)
            u3m = bo["utr3_mask"].to(device)
            y_real = bo["te"].to(device)

            _, yhat_scaled = model(cod, sim, msk,
                                    u5, u3, u5m, u3m,
                                    angle_bin=ang, pair_bin=pai, bucket_idx=buc)
            y_scaled = te_forward_torch(y_real, trans)

            loss = mse(yhat_scaled, y_scaled)

            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

            bs = y_real.size(0)
            total_loss_scaled += float(loss.item()); nobs += bs

            yhat_s = yhat_scaled.detach().cpu().numpy()
            y_s    = y_scaled.detach().cpu().numpy()
            y_r    = y_real.detach().cpu().numpy()
            yhat_r = te_inverse_numpy(yhat_s, trans)

            preds_scaled.append(yhat_s)
            trues_scaled.append(y_s)
            preds_raw.append(yhat_r)
            trues_raw.append(y_r)
            txs.extend(bo["tx"])

    if nobs == 0:
        return {
            "mse": float("nan"), "mse_scaled": float("nan"),
            "pearson_r": float("nan"), "spearman_r": float("nan"),
            "preds": [], "trues": [], "tx": []
        }

    yhat_scaled_all = np.concatenate(preds_scaled) if preds_scaled else np.array([])
    y_scaled_all    = np.concatenate(trues_scaled) if trues_scaled else np.array([])
    yhat_raw_all    = np.concatenate(preds_raw)    if preds_raw    else np.array([])
    y_raw_all       = np.concatenate(trues_raw)    if trues_raw    else np.array([])

    r  = _pearson_safe(y_raw_all, yhat_raw_all)
    ro = _spearman_safe(y_raw_all, yhat_raw_all)

    mse_raw = float(np.mean((yhat_raw_all - y_raw_all) ** 2)) if y_raw_all.size else float("nan")
    mse_scaled_mean = float(np.mean((yhat_scaled_all - y_scaled_all) ** 2)) if y_scaled_all.size else float("nan")

    return {
        "mse": mse_raw,
        "mse_scaled": mse_scaled_mean,
        "pearson_r": r,
        "spearman_r": ro,
        "preds": yhat_raw_all.tolist(),
        "trues": y_raw_all.tolist(),
        "tx": txs
    }

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


# ---------------------------
# Main
# ---------------------------
def main():
    base_dir = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-line", choices=["ipsc", "hek293", "rpe", "lcl"], required=True)
    ap.add_argument("--pkl_root", default=None)
    ap.add_argument("--te_csv", default=None)
    ap.add_argument("--te_col", default=None, help="TE column name in the CSV")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--geomap_train", type=str, default=None)
    ap.add_argument("--geomap_val", type=str, default=None)
    ap.add_argument("--geomap_test", type=str, default=None)
    ap.add_argument("--utr_xlsx", default=str(Path(__file__).resolve().parent / "41587_2025_2712_MOESM3_ESM.xlsx"),
                    help="workbook with tx_sequence and utr5/cds/utr3 sizes")
    ap.add_argument("--utr_unversioned", action="store_true",
                    help="match UTRs on transcript ids without version suffix (default: exact versioned ids)")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--freeze_base", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d_te", type=int, default=128, help="Hidden dim for UTR Mamba TE head")
    ap.add_argument("--n_te_layers", type=int, default=2, help="Number of Mamba layers in TE head")
    ap.add_argument("--te_d_state", type=int, default=16, help="Mamba d_state for TE head")
    ap.add_argument("--te_d_conv", type=int, default=4, help="Mamba d_conv for TE head")
    ap.add_argument("--te_expand", type=int, default=2, help="Mamba expand factor for TE head")
    ap.add_argument("--te_use_log1p", action="store_true", help="Apply log1p to counts in TE head")
    ap.add_argument("--te_transform", choices=["minmax"], default="minmax")
    ap.add_argument("--te_min", type=float, default=None)
    ap.add_argument("--te_max", type=float, default=None)
    ap.add_argument("--te_eps", type=float, default=1e-6)
    args = ap.parse_args()
    cfg = CELL_CONFIG[args.cell_line]

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
        args.out_dir = str(base_dir / "results" / "TE" / args.cell_line / "finetune_cds_utr")
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
    print(f"Loaded {len(te_map)} TE labels from {args.te_csv}")

    # Geomaps
    geo_train = load_geomap(Path(args.geomap_train) if args.geomap_train else None)
    geo_val   = load_geomap(Path(args.geomap_val)   if args.geomap_val   else None)
    geo_test  = load_geomap(Path(args.geomap_test)  if args.geomap_test  else None)

    # Angle bins
    ANGLE_BINS = derive_angle_bins_from_training(
        geo_train, K_ANGLE_BINS, method="linear", low_q=0.0, high_q=0.995)
    print("Derived ANGLE_BINS:", ANGLE_BINS, "len=", len(ANGLE_BINS))
    angle_bins = ANGLE_BINS
    k_angle_bins = K_ANGLE_BINS

    # TE transform
    trans = make_te_transform(args.te_transform, args.te_min, args.te_max, args.te_eps)
    with open(OUT / "te_transform.json", "w") as fh:
        json.dump(trans, fh, indent=2)

    # Load UTR data
    utr_all = load_utr_from_xlsx(args.utr_xlsx, unversioned=args.utr_unversioned)
    print(f"Loaded {len(utr_all)} UTR entries from {args.utr_xlsx}")
    utr_train = utr_val = utr_test = utr_all

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

    # Datasets
    train_ds = PKLTEFineTuneUTR(train_pkls, "train", te_map, geo_train, angle_bins, utr_train, log_input=True)
    val_ds   = PKLTEFineTuneUTR(val_pkls,   "val",   te_map, geo_val,   angle_bins, utr_val, log_input=True) if val_pkls else None
    test_ds  = PKLTEFineTuneUTR(test_pkls,  "test",  te_map, geo_test,  angle_bins, utr_test, log_input=True)

    # Dataloaders
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=pad_collate_utr, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          collate_fn=pad_collate_utr, pin_memory=True) if val_ds else None
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                          collate_fn=pad_collate_utr, pin_memory=True)

    # Model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base = RiboPolisherMamba(d_model=192, n_layers=4, d_state=16, d_conv=4,
                             dropout=0.1, k_angle_bins=k_angle_bins).to(device)

    # Load base weights
    state = torch.load(args.weights, map_location=device)
    base.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state)

    model = MambaTEFullUTR(
        base,
        d_te=args.d_te,
        n_te_layers=args.n_te_layers,
        te_d_state=args.te_d_state,
        te_d_conv=args.te_d_conv,
        te_expand=args.te_expand,
        dropout=0.1,
        use_log1p=args.te_use_log1p
    ).to(device)

    if args.freeze_base:
        for p in model.base.parameters(): p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    # Train
    best_val = float("inf")
    best_path = OUT / "mamba_te_utr_best.pt"
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

    test_pred_csv = OUT / "mamba_te_utr_test_predictions.csv"
    with open(test_pred_csv, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["tx","te_true","te_pred"])
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

    torch.save(model.state_dict(), OUT / "mamba_te_utr_final.pt")
    print("Saved:")
    print(" ", best_path if best_path.exists() else "(no best saved)")
    print(" ", OUT / "mamba_te_utr_final.pt")
    print(" ", test_pred_csv)
    print(" ", OUT / "test_metrics.json")
    print(" ", OUT / "te_transform.json")

if __name__ == "__main__":
    main()
