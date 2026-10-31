

import torch
import torch.nn as nn
from mamba_ssm import Mamba
import os
import hashlib
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import json
from datetime import datetime
from pathlib import Path

def cds_hash_from_pkl(pkl_path: str) -> str:
    """
    Hash of CDS nucleotide sequence reconstructed from codons in PKL.
    (PKL only has cods, so this is CDS-only identity.)
    """
    try:
        with open(pkl_path, "rb") as fh:
            # Unpack just enough to get cods (item 1)
            data = pickle.load(fh)
            cods = data[1] 
        cds = "".join(cods).upper().replace("U", "T")
        return hashlib.md5(cds.encode("utf-8")).hexdigest()
    except Exception as e:
        print(f"Error processing {pkl_path}: {e}")
        return None

def compute_hashes_parallel(file_list, n_proc=16):
    """
    Compute hashes for a list of files in parallel.
    Returns: dict {filepath: hash}
    """
    with Pool(n_proc) as pool:
        hashes = list(tqdm(pool.imap(cds_hash_from_pkl, file_list), total=len(file_list), desc="Hashing CDS"))
    
    return {fp: h for fp, h in zip(file_list, hashes) if h is not None}

def get_hashes_cached(file_list, cache_path):
    """
    Load hashes from cache if available, otherwise compute and save.
    Only computes for files not in cache.
    """
    cache = {}
    if os.path.exists(cache_path):
        print(f"Loading hash cache from {cache_path}...")
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing.")
            cache = {}
            
    # Key is absolute path or relative? Let's use whatever is in file_list (likely absolute)
    # Check what is missing
    missing = [f for f in file_list if f not in cache]
    
    if missing:
        print(f"Computing hashes for {len(missing)} missing files...")
        new_hashes = compute_hashes_parallel(missing, n_proc=min(32, cpu_count()))
        cache.update(new_hashes)
        
        # Save updated cache
        print(f"Saving updated cache to {cache_path}...")
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
            
    return cache

def filter_target_files_excluding_ref_cds(ref_files, target_files, cache_dir, ref_name="train", target_name="test"):
    """
    Exclude any target PKL whose CDS hash matches any reference PKL CDS hash.
    Use parallel processing and caching.
    """
    os.makedirs(cache_dir, exist_ok=True)
    ref_cache_path    = os.path.join(cache_dir, f"{ref_name}_cds_hashes.pkl")
    target_cache_path = os.path.join(cache_dir, f"{target_name}_cds_hashes.pkl")

    # Reference files: only need sample_0 or representative?
    # Assuming sample_0 exists for all transcripts. If not, fallback to all.
    ref_for_hash = [fp for fp in ref_files if fp.endswith("_sample_0.pkl")]
    if len(ref_for_hash) == 0:
        print(f"Warning: no _sample_0.pkl found in {ref_name}. Hashing ALL files (slow!).")
        ref_for_hash = list(ref_files)
    else:
        print(f"Using {len(ref_for_hash)} representative {ref_name} files (sample_0) for leakage check.")

    # Get hashes
    ref_hash_map = get_hashes_cached(ref_for_hash, ref_cache_path)
    ref_hashes = set(ref_hash_map.values())
    
    # Target files
    target_hash_map = get_hashes_cached(target_files, target_cache_path)
    
    kept, removed = [], []
    for fp in target_files:
        h = target_hash_map.get(fp)
        if h and h in ref_hashes:
            removed.append(fp)
        else:
            kept.append(fp)

    return kept, removed

class RiboPolisherMamba(nn.Module):
    def __init__(
        self,
        d_model=192,
        n_layers=4,
        d_state=16,   # Mamba-specific: state space dimension
        d_conv=4,     # Mamba-specific: local convolution width
        expand=2,     # Mamba-specific: expansion factor
        dropout=0.1,
        k_angle_bins=4  # == K_ANGLE_BINS
    ):
        super().__init__()
        # token & sim
        self.cod_emb  = nn.Embedding(65, d_model)  # 64 codons + padding id 64
        self.sim_proj = nn.Linear(1, d_model)

        # geo feature embeddings (active immediately; no gate)
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

    def forward(self, cod_ids, sim_feat, pad_mask,
                angle_bin=None, pair_bin=None, bucket_idx=None):
        """
        cod_ids   : (B, L) long in [0..64], where 64 is padding id
        sim_feat  : (B, L) float
        pad_mask  : (B, L) bool
        angle_bin : (B, L) long in [0..k_angle_bins-1] (optional)
        pair_bin  : (B, L) long in [0..3]             (optional)
        bucket_idx: (B, L) long in [0..2]             (optional)
        """
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

        logits = self.head(x).squeeze(-1)      # (B, L)
        logits = F.softplus(logits)            # enforce non-negativity
        logits = logits * pad_mask.float()     # keep masked positions zero
        return logits

def derive_angle_bins_from_training(geo_map_train, k_bins, method="linear", low_q=0.0, high_q=0.995):
    if not geo_map_train:
        return np.linspace(0.0, 1.0, k_bins + 1)

    all_angles = []
    for _tx, (angle_dev_sum, _pair_count) in geo_map_train.items():
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
        edges = np.maximum.accumulate(edges)
        edges[-1] = max(edges[-1], vmax)
    else:
        lo = float(np.quantile(vals, low_q))
        hi = float(np.quantile(vals, high_q))
        if hi <= lo:
            hi = lo + 1e-6
        edges = np.linspace(lo, hi, k_bins + 1)
        edges[-1] = max(edges[-1], vmax)

    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6

    # (optional) keep min/max anchored
    edges[0] = min(edges[0], vmin)
    edges[-1] = max(edges[-1], vmax)
    return edges

        
from util import *
import math, glob, pickle, numpy as np, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from multiprocessing import Pool, cpu_count
import pdb




def parse_args():
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Polisher training reproduction")
    parser.add_argument("--cell-line", choices=["ipsc", "hek293", "rpe", "lcl"], required=True)
    parser.add_argument("--data-root", default=str(base_dir / "data"))
    parser.add_argument("--output-root", default=str(base_dir / "results" / "polisher"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--non-zero-weight", type=float, default=None)
    parser.add_argument("--geo-map-dir", default=None)
    return parser.parse_args()


ARGS = parse_args()
POLISHER_ROOT = Path(ARGS.data_root) / ARGS.cell_line / "polisher"
DATA_DIR = POLISHER_ROOT / "data_new"
BATCH_SIZE = ARGS.batch_size
EPOCHS = ARGS.epochs
LR = ARGS.lr
WEIGHT_DECAY = ARGS.weight_decay
GRAD_CLIP = ARGS.grad_clip
DEVICE = ARGS.device
SEED = ARGS.seed
NON_ZERO_WEIGHT = ARGS.non_zero_weight if ARGS.non_zero_weight is not None else (5.0 if ARGS.cell_line == "hek293" else 1.0)
OUT_DIR = Path(ARGS.output_root) / ARGS.cell_line
OUT_DIR.mkdir(parents=True, exist_ok=True)
GEO_MAP_DIR = Path(ARGS.geo_map_dir) if ARGS.geo_map_dir else POLISHER_ROOT
CACHE_DIR = DATA_DIR / "cds_hash_cache"
# ANGLE_BINS = [0.     ,    1.0471990, 2.09439815, 3.14159722, 6.51421177]

# ANGLE_BINS = [0.     ,    1.0471990, 2.09439815, 3.14159722, 6.51421177]


# LAMBDA_PERTRANS = 1.0
# LAMBDA_ELEM     = 0.2434589075
# LAMBDA_CODON    = 0.002944712304
# LAMBDA_BUCKET   = 0.00002370368336
# LAMBDA_PAIR     = 0.00005561812715
# LAMBDA_ANGLE    = 0.00006711538462



torch.manual_seed(SEED); np.random.seed(SEED)


def build_split_filelists(data_dir: Path):
    train_files, val_files, test_files = [], [], []
    train = 0
    val = 0
    test= 0
    pkls = sorted(glob.glob(str(data_dir / "train/*_sample_*.pkl")))
    for p in pkls:
        train_files.append(p)
        train +=1
    pkls = sorted(glob.glob(str(data_dir / "val/*_sample_*.pkl")))
    for p in pkls:
        val_files.append(p)
        val +=1
    pkls = sorted(glob.glob(str(data_dir / "test/*_sample_*.pkl")))
    for p in pkls:
        test_files.append(p)
        test +=1
    print(f"[index] train={len(train_files)} | val={len(val_files)} | test={len(test_files)}")
    return train_files, val_files, test_files

train_files, val_files, test_files = build_split_filelists(DATA_DIR)

cache_dir = str(CACHE_DIR)

# 1. Filter Validation excluding Train
val_files, val_removed = filter_target_files_excluding_ref_cds(
    train_files, val_files, cache_dir, ref_name="train", target_name="val"
)
print(f"[filter] Excluding VAL with identical CDS in TRAIN: removed={len(val_removed)} kept={len(val_files)} (from {len(val_files)+len(val_removed)})")

# 2. Filter Test excluding Train
test_files_no_train, test_removed_train = filter_target_files_excluding_ref_cds(
    train_files, test_files, cache_dir, ref_name="train", target_name="test"
)
print(f"[filter] Excluding TEST with identical CDS in TRAIN: removed={len(test_removed_train)} kept={len(test_files_no_train)} (from {len(test_files)})")

# 3. Filter Test excluding Validation
final_test_files, test_removed_val = filter_target_files_excluding_ref_cds(
    val_files, test_files_no_train, cache_dir, ref_name="val", target_name="test" 
)
print(f"[filter] Excluding TEST with identical CDS in VAL: removed={len(test_removed_val)} kept={len(final_test_files)} (from {len(test_files_no_train)})")

test_files = final_test_files 
# --- Dataset: add geomap + bins ---
class PolishPKLDataset(Dataset):
    def __init__(self, file_list, log_input=True, log_target=True, geomap=None, angle_bins=None):
        self.files = list(file_list)
        self.log_input = log_input
        self.log_target = log_target
        self.geomap = geomap or {}                    # dict: tx -> (angle_dev_sum, pair_count)
        self.angle_bins = np.asarray(angle_bins, dtype=np.float64) if angle_bins is not None else None

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        with open(self.files[i], "rb") as fh:
            tx, cods, obs_counts, sim_vec_raw, sim_vec_scaled, scaling_factor, completed_total = pickle.load(fh)

        L = len(cods)
        cod_ids = np.array([CODON2IDX[c] for c in cods], dtype=np.int64)
        sim = np.asarray(sim_vec_raw, dtype=np.float32)
        tgt = np.asarray(obs_counts,   dtype=np.float32)

        sim_feat = np.log1p(sim) if self.log_input else sim
        tgt_val  = np.log1p(tgt) if self.log_target else tgt

        # --- geo features (bins) ---
        if (self.angle_bins is not None) and (tx in self.geomap):
            angle_dev_sum, pair_count = self.geomap[tx]
            a = np.asarray(angle_dev_sum, dtype=np.float64)[:L]
            p = np.asarray(pair_count,    dtype=np.int64)[:L]
            # angle bins -> 0..K-1
            kk = np.digitize(a, self.angle_bins) - 1
            kk = np.clip(kk, 0, K_ANGLE_BINS - 1).astype(np.int64)
            # pair bins -> 0..3
            pb = np.clip(p, 0, 3).astype(np.int64)
        else:
            kk = np.zeros(L, dtype=np.int64)
            pb = np.zeros(L, dtype=np.int64)

        # bucket bins by thirds -> 0..2
        bb = np.zeros(L, dtype=np.int64)
        l1 = L // 3
        l2 = (2 * L) // 3
        bb[l1:l2] = 1
        bb[l2:]   = 2

        return {
            "tx": tx,
            "cod_ids": torch.tensor(cod_ids, dtype=torch.long),
            "sim_feat": torch.tensor(sim_feat, dtype=torch.float32),
            "target": torch.tensor(tgt_val, dtype=torch.float32),
            "length": int(L),
            "sim_real": torch.tensor(sim_vec_raw.copy(), dtype=torch.float32),
            "obs_real": torch.tensor(obs_counts.copy(),     dtype=torch.float32),
            # NEW:
            "angle_bin": torch.tensor(kk, dtype=torch.long),
            "pair_bin":  torch.tensor(pb, dtype=torch.long),
            "bucket_idx":torch.tensor(bb, dtype=torch.long),
        }

def pad_collate(batch):
    batch = sorted(batch, key=lambda x: x["length"], reverse=True)
    B, Lmax = len(batch), batch[0]["length"]

    cod = torch.full((B, Lmax), 64, dtype=torch.long)
    sim = torch.zeros((B, Lmax), dtype=torch.float32)
    tgt = torch.zeros((B, Lmax), dtype=torch.float32)
    msk = torch.zeros((B, Lmax), dtype=torch.bool)
    sim_real = torch.zeros((B, Lmax), dtype=torch.float32)
    obs_real = torch.zeros((B, Lmax), dtype=torch.float32)

    angle_bin = torch.zeros((B, Lmax), dtype=torch.long)
    pair_bin  = torch.zeros((B, Lmax), dtype=torch.long)
    bucket_idx= torch.zeros((B, Lmax), dtype=torch.long)

    txs = []
    for i, ex in enumerate(batch):
        L = ex["length"]; txs.append(ex["tx"])
        cod[i, :L] = ex["cod_ids"]
        sim[i, :L] = ex["sim_feat"]
        tgt[i, :L] = ex["target"]
        msk[i, :L] = True
        sim_real[i, :L] = ex["sim_real"]
        obs_real[i, :L] = ex["obs_real"]

        angle_bin[i, :L] = ex["angle_bin"]
        pair_bin[i, :L]  = ex["pair_bin"]
        bucket_idx[i,:L] = ex["bucket_idx"]

    return {
        "tx": txs,
        "cod_ids": cod, "sim_feat": sim, "target": tgt, "mask": msk,
        "sim_real": sim_real, "obs_real": obs_real,
        "angle_bin": angle_bin, "pair_bin": pair_bin, "bucket_idx": bucket_idx,
    }


def _accumulate_bins_for_sample(
    cod_ids_np, true_np, pred_np, mask_np,  # vectors length L
    tx, geo_map, ANGLE_BINS,
    cod_obs_accum, cod_pred_accum,          # shape (64,)
    bucket_obs_accum, bucket_pred_accum,    # shape (3,)
    pair_obs_accum, pair_pred_accum,        # shape (4,)
    ang_obs_accum, ang_pred_accum           # shape (K_ANGLE_BINS,)
):
    # keep only valid (masked) positions
    keep = mask_np.astype(bool)
    if not np.any(keep):
        return
    cod_ids = cod_ids_np[keep].astype(np.int64)
    obs = true_np[keep].astype(np.float64)
    pred = pred_np[keep].astype(np.float64)
    L = obs.shape[0]

    # ---- Codon sums (skip STOP codons)
    non_stop = ~np.isin(cod_ids, STOP_IDX)
    if np.any(non_stop):
        cod_ids_ns = cod_ids[non_stop]
        obs_ns = obs[non_stop]
        pred_ns = pred[non_stop]
        cod_obs_accum += np.bincount(cod_ids_ns, weights=obs_ns, minlength=len(CODONS))
        cod_pred_accum += np.bincount(cod_ids_ns, weights=pred_ns, minlength=len(CODONS))

    # ---- Buckets by index thirds
    if L > 0:
        b = np.zeros(L, dtype=np.int64)
        l1 = L // 3
        l2 = (2 * L) // 3
        b[l1:l2] = 1
        b[l2:] = 2
        bucket_obs_accum += np.bincount(b, weights=obs, minlength=3)
        bucket_pred_accum += np.bincount(b, weights=pred, minlength=3)

    # ---- Pair / Angle (needs geo_map[tx])
    if (geo_map is not None) and (tx in geo_map):
        angle_dev_sum, pair_count = geo_map[tx]
        angle_dev_sum = np.asarray(angle_dev_sum)[:L]
        pair_count = np.asarray(pair_count)[:L]

        # Pair bins 0..3 (clip)
        p = np.clip(pair_count.astype(np.int64), 0, 3)
        pair_obs_accum += np.bincount(p, weights=obs, minlength=4)
        pair_pred_accum += np.bincount(p, weights=pred, minlength=4)

        # Angle bins via np.digitize (then clamp)
        K = K_ANGLE_BINS
        kk = np.digitize(angle_dev_sum.astype(np.float64), ANGLE_BINS) - 1
        kk = np.clip(kk, 0, K - 1).astype(np.int64)
        ang_obs_accum += np.bincount(kk, weights=obs, minlength=K)
        ang_pred_accum += np.bincount(kk, weights=pred, minlength=K)





@torch.no_grad()
def evaluate_counts(model, loader, geo_map, ANGLE_BINS):
    """
    Computes metrics using the same 'sum-based MAE' definitions as in codon_loss:
      codonMAE  = mean(|sum_obs_per_codon - sum_pred_per_codon|)
      bucketMAE = mean(|sum_obs_per_bucket - sum_pred_per_bucket|)
      pairMAE   = mean(|sum_obs_per_pair - sum_pred_per_pair|)
      angleMAE  = mean(|sum_obs_per_angle - sum_pred_per_angle|)
    Plus:
      elemwiseMAE (|pred - true| averaged over all positions)
      pertranscriptMAE (mean over transcripts of normalized MAE = sum|diff| / sum obs)
    """
    model.eval()

    # Accumulators for per-position and per-transcript losses
    all_pred, all_true = [], []
    per_transcript_losses = []
    
    total_nll = 0.0
    total_masked_count = 0.0

    # Accumulators for the sum-based MAEs (match codon_loss)
    cod_obs = np.zeros(len(CODONS), dtype=np.float64)
    cod_pred = np.zeros(len(CODONS), dtype=np.float64)
    bucket_obs = np.zeros(3, dtype=np.float64)
    bucket_pred = np.zeros(3, dtype=np.float64)
    pair_obs = np.zeros(4, dtype=np.float64)
    pair_pred = np.zeros(4, dtype=np.float64)
    ang_obs = np.zeros(K_ANGLE_BINS, dtype=np.float64)
    ang_pred = np.zeros(K_ANGLE_BINS, dtype=np.float64)

    for bo in loader:
        txs     = bo["tx"]                         # list[str], length B
        cod_ids = bo["cod_ids"].to(DEVICE)
        sim     = bo["sim_feat"].to(DEVICE)
        tgt_log = bo["target"].to(DEVICE)
        mask    = bo["mask"].to(DEVICE)
        angle_bin  = bo["angle_bin"].to(DEVICE)
        pair_bin   = bo["pair_bin"].to(DEVICE)
        bucket_idx = bo["bucket_idx"].to(DEVICE)

        
        with torch.no_grad():

            pred_log = model(cod_ids, sim, mask,angle_bin=angle_bin, pair_bin=pair_bin, bucket_idx=bucket_idx)
        # counts reported by the evaluation and by seq2ribo inference: expm1 of the output
        pred_cnt = torch.expm1(torch.clamp(pred_log, max=20.0))
        true_cnt = torch.expm1(torch.clamp(tgt_log,  max=20.0))

        # --- NLL Loss ---
        nll = torch.nn.functional.poisson_nll_loss(
            pred_log, true_cnt, log_input=True, full=True, reduction='none'
        )
        masked_nll = nll * mask
        total_nll += masked_nll.sum().item()
        total_masked_count += mask.sum().item()

        # Store for elemwise MAE
        all_pred.append(pred_cnt.masked_select(mask).cpu())
        all_true.append(true_cnt.masked_select(mask).cpu())

        # Per-transcript normalized MAE (sum|diff| / sum obs)
        diffs = torch.abs((pred_cnt - true_cnt) * mask)
        numer = diffs.sum(dim=1)
        ribo_loads = (true_cnt * mask).sum(dim=1)
        per_sample_loss = torch.where(ribo_loads > 0, numer / ribo_loads, torch.zeros_like(numer))
        per_transcript_losses.append(per_sample_loss.cpu())

        # ---- Sum-based accumulations (vectorized per-sample) ----
        B = cod_ids.size(0)
        for i in range(B):
            tx = txs[i]
            mask_i = mask[i].detach().cpu().numpy()
            cod_ids_i = cod_ids[i].detach().cpu().numpy()
            true_i = true_cnt[i].detach().cpu().numpy()
            pred_i = pred_cnt[i].detach().cpu().numpy()

            _accumulate_bins_for_sample(
                cod_ids_i, true_i, pred_i, mask_i, tx, geo_map, ANGLE_BINS,
                cod_obs, cod_pred, bucket_obs, bucket_pred, pair_obs, pair_pred, ang_obs, ang_pred
            )

    if not all_pred:
        return {
            "pertranscriptMAE": float("nan"),
            "elemwiseMAE": float("nan"),
            "codonMAE": float("nan"),
            "bucketMAE": float("nan"),
            "pairMAE": float("nan"),
            "angleMAE": float("nan"),
            "num_samples": 0,
        }

    # Finalize per-transcript/elemwise
    per_transcript_losses = torch.cat(per_transcript_losses)
    pertranscriptMAE = per_transcript_losses.mean().item()

    yhat = torch.cat(all_pred)
    y    = torch.cat(all_true)
    elemwiseMAE = torch.mean(torch.abs(yhat - y)).item() if y.numel() > 0 else float("nan")

    # Element-wise Correlations
    if y.numel() > 1:
        y_np = y.numpy()
        yhat_np = yhat.numpy()
        pearson_elem, _ = pearsonr(y_np, yhat_np)
        spearman_elem, _ = spearmanr(y_np, yhat_np)
    else:
        pearson_elem, spearman_elem = 0.0, 0.0

    # Sum-based MAEs (exactly as at the end of codon_loss)
    codonMAE  = float(np.mean(np.abs(cod_obs - cod_pred)))
    bucketMAE = float(np.mean(np.abs(bucket_obs - bucket_pred)))
    pairMAE   = float(np.mean(np.abs(pair_obs - pair_pred)))
    angleMAE  = float(np.mean(np.abs(ang_obs - ang_pred)))
    
    nll_avg = total_nll / max(total_masked_count, 1.0)

    return {
        "pertranscriptMAE": pertranscriptMAE,
        "elemwiseMAE": elemwiseMAE,
        "pearson_elem": pearson_elem,
        "spearman_elem": spearman_elem,
        "codonMAE": codonMAE,
        "bucketMAE": bucketMAE,
        "pairMAE": pairMAE,
        "angleMAE": angleMAE,
        "nll": nll_avg
    }


@torch.no_grad()
def evaluate_test(loader, geo_map, ANGLE_BINS):
    """
    Like evaluate_counts, but compares sim_real vs obs_real from the PKLs
    (i.e., model-free comparison of simulation vs observed).
    """
    all_pred, all_true = [], []
    per_transcript_losses = []

    # Sum-based accumulators
    cod_obs = np.zeros(len(CODONS), dtype=np.float64)
    cod_pred = np.zeros(len(CODONS), dtype=np.float64)
    bucket_obs = np.zeros(3, dtype=np.float64)
    bucket_pred = np.zeros(3, dtype=np.float64)
    pair_obs = np.zeros(4, dtype=np.float64)
    pair_pred = np.zeros(4, dtype=np.float64)
    ang_obs = np.zeros(K_ANGLE_BINS, dtype=np.float64)
    ang_pred = np.zeros(K_ANGLE_BINS, dtype=np.float64)

    for bo in loader:
        txs      = bo["tx"]
        sim_real = bo["sim_real"].to(DEVICE)
        obs_real = bo["obs_real"].to(DEVICE)
        mask     = bo["mask"].to(DEVICE)

        # store for elemwise
        all_pred.append(sim_real.masked_select(mask).cpu())
        all_true.append(obs_real.masked_select(mask).cpu())

        # per-transcript normalized MAE
        differences = torch.abs((obs_real - sim_real) * mask)
        numer = differences.sum(dim=1)
        ribo_loads = (obs_real * mask).sum(dim=1)
        per_sample_loss = torch.where(ribo_loads > 0, numer / ribo_loads, torch.zeros_like(numer))
        per_transcript_losses.append(per_sample_loss.cpu())

        # sum-based accumulations
        B = sim_real.size(0)
        for i in range(B):
            tx = txs[i]
            mask_i = mask[i].detach().cpu().numpy()
            # here "true" is obs_real; "pred" is sim_real to match "obs vs sim" naming used above
            true_i = obs_real[i].detach().cpu().numpy()
            pred_i = sim_real[i].detach().cpu().numpy()
            cod_ids_i = bo["cod_ids"][i].detach().cpu().numpy()

            _accumulate_bins_for_sample(
                cod_ids_i, true_i, pred_i, mask_i, tx, geo_map, ANGLE_BINS,
                cod_obs, cod_pred, bucket_obs, bucket_pred, pair_obs, pair_pred, ang_obs, ang_pred
            )

    # finalize per-transcript/elemwise
    per_transcript_losses = torch.cat(per_transcript_losses)
    pertranscriptMAE = per_transcript_losses.mean().item()

    all_pred = torch.cat(all_pred)
    all_true = torch.cat(all_true)
    elemwiseMAE = torch.mean(torch.abs(all_pred - all_true)).item()

    # Element-wise Correlations
    if all_true.numel() > 1:
        y_np = all_true.numpy()
        yhat_np = all_pred.numpy()
        pearson_elem, _ = pearsonr(y_np, yhat_np)
        spearman_elem, _ = spearmanr(y_np, yhat_np)
    else:
        pearson_elem, spearman_elem = 0.0, 0.0

    # sum-based MAEs (same as codon_loss ending)
    codonMAE  = float(np.mean(np.abs(cod_obs - cod_pred)))
    bucketMAE = float(np.mean(np.abs(bucket_obs - bucket_pred)))
    pairMAE   = float(np.mean(np.abs(pair_obs - pair_pred)))
    angleMAE  = float(np.mean(np.abs(ang_obs - ang_pred)))

    return {
        "pertranscriptMAE": pertranscriptMAE,
        "elemwiseMAE": elemwiseMAE,
        "pearson_elem": pearson_elem,
        "spearman_elem": spearman_elem,
        "codonMAE": codonMAE,
        "bucketMAE": bucketMAE,
        "pairMAE": pairMAE,
        "angleMAE": angleMAE,
        # NLL is not computed in evaluate_test currently (loss is model-dependent, evaluate_test compares raw sim vs obs)
        "nll": float("nan") 
    }
def _batch_maes_from_bins(bo, pred_cnt, true_cnt, mask, geomap, angle_bins_t, device):
    """
    Differentiable, sum-based MAEs that match the end-of-codon_loss formulas.
    Returns: codonMAE, bucketMAE, pairMAE, angleMAE  (all torch scalars)
    """
    B, Lmax = mask.shape
    dtype = true_cnt.dtype

    # ---- Flatten once
    mask_flat    = mask.reshape(-1)                         # (B*L)
    cod_ids_flat = bo["cod_ids"].to(device).reshape(-1)     # (B*L), includes padding=64
    true_flat    = true_cnt.reshape(-1)
    pred_flat    = pred_cnt.reshape(-1)

    # -----------------------------
    # 1) CODON sums (64 bins, skip STOP)  **SAFE INDEXING**
    # -----------------------------
    num_codons = len(CODONS)  # 64
    cod_obs  = torch.zeros(num_codons, device=device, dtype=dtype)
    cod_pred = torch.zeros_like(cod_obs)

    # Only consider masked positions AND valid codon ids in [0,63]
    mask_cod_valid = mask_flat & (cod_ids_flat >= 0) & (cod_ids_flat < num_codons)
    if mask_cod_valid.any():
        idx_valid = cod_ids_flat[mask_cod_valid].long()  # [0..63]
        # mark STOPs and drop them
        stop_lut = torch.zeros(num_codons, dtype=torch.bool, device=device)
        stop_lut[torch.as_tensor(STOP_IDX, device=device)] = True
        keep = ~stop_lut[idx_valid]
        if keep.any():
            idx_ns = idx_valid[keep]
            vals_true = true_flat[mask_cod_valid][keep]
            vals_pred = pred_flat[mask_cod_valid][keep]
            cod_obs.index_add_(0, idx_ns, vals_true)
            cod_pred.index_add_(0, idx_ns, vals_pred)

    codonMAE = torch.mean(torch.abs(cod_obs - cod_pred))

    # -----------------------------
    # 2) BUCKET sums (3 bins by thirds)  **SAFE**
    # -----------------------------
    lengths = mask.sum(dim=1).tolist()
    bucket_idx = torch.zeros((B, Lmax), device=device, dtype=torch.long)
    for i, Li in enumerate(lengths):
        if Li <= 0: 
            continue
        l1 = Li // 3
        l2 = (2 * Li) // 3
        bucket_idx[i, l1:l2] = 1
        bucket_idx[i, l2:Li] = 2

    bucket_obs  = torch.zeros(3, device=device, dtype=dtype)
    bucket_pred = torch.zeros(3, device=device, dtype=dtype)
    if mask_flat.any():
        bflat = bucket_idx.reshape(-1)[mask_flat]  # values in {0,1,2}
        bucket_obs.index_add_(0, bflat, true_flat[mask_flat])
        bucket_pred.index_add_(0, bflat, pred_flat[mask_flat])

    bucketMAE = torch.mean(torch.abs(bucket_obs - bucket_pred))

    # -----------------------------
    # 3) PAIR & ANGLE sums (need geomap)  **SAFE**
    # -----------------------------
    pair_batch  = torch.zeros((B, Lmax), device=device, dtype=torch.long)
    angle_batch = torch.zeros((B, Lmax), device=device, dtype=dtype)

    txs = bo["tx"]
    for i, tx in enumerate(txs):
        if tx not in geomap:
            continue
        angle_np, pair_np = geomap[tx]
        Li = lengths[i]
        if Li <= 0:
            continue
        # clamp to the valid prefix length
        a = torch.as_tensor(angle_np[:Li], device=device, dtype=dtype)
        p = torch.as_tensor(pair_np[:Li],  device=device, dtype=torch.long)
        angle_batch[i, :Li] = a
        pair_batch[i, :Li]  = p

    # Pair bins 0..3
    pair_bins = pair_batch.clamp(0, 3)
    pair_obs  = torch.zeros(4, device=device, dtype=dtype)
    pair_pred = torch.zeros(4, device=device, dtype=dtype)
    if mask_flat.any():
        pflat = pair_bins.reshape(-1)[mask_flat]            # in [0..3]
        pair_obs.index_add_(0, pflat, true_flat[mask_flat])
        pair_pred.index_add_(0, pflat, pred_flat[mask_flat])
    pairMAE = torch.mean(torch.abs(pair_obs - pair_pred))

    # Angle bins via bucketize on sorted edges; K = K_ANGLE_BINS
    # Ensure angle_bins_t is strictly increasing: [e0 < e1 < ... < eK]
    # and K == K_ANGLE_BINS
    K = K_ANGLE_BINS
    kk = torch.bucketize(angle_batch, angle_bins_t) - 1      # could be -1..len(bins)-1
    kk = kk.clamp(0, K - 1).long()                           # now in [0..K-1]

    ang_obs  = torch.zeros(K, device=device, dtype=dtype)
    ang_pred = torch.zeros(K, device=device, dtype=dtype)
    if mask_flat.any():
        kflat = kk.reshape(-1)[mask_flat]                    # in [0..K-1]
        ang_obs.index_add_(0, kflat, true_flat[mask_flat])
        ang_pred.index_add_(0, kflat, pred_flat[mask_flat])
    angleMAE = torch.mean(torch.abs(ang_obs - ang_pred))

    return codonMAE, bucketMAE, pairMAE, angleMAE


with (GEO_MAP_DIR / "geo_map_train.pkl").open("rb") as f: geo_map_train = pickle.load(f)
with (GEO_MAP_DIR / "geo_map_val.pkl").open("rb") as f:   geo_map_val   = pickle.load(f)
with (GEO_MAP_DIR / "geo_map_test.pkl").open("rb") as f:  geo_map_test  = pickle.load(f)

ANGLE_BINS = derive_angle_bins_from_training(
    geo_map_train,
    k_bins=4
)



print("Derived ANGLE_BINS:", ANGLE_BINS, "len=", len(ANGLE_BINS))

train_ds = PolishPKLDataset(train_files, log_input=True, log_target=True,
                            geomap=geo_map_train, angle_bins=ANGLE_BINS)
val_ds   = PolishPKLDataset(val_files,   log_input=True, log_target=True,
                            geomap=geo_map_val,   angle_bins=ANGLE_BINS)
test_ds  = PolishPKLDataset(test_files,  log_input=True, log_target=True,
                            geomap=geo_map_test,  angle_bins=ANGLE_BINS)


train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=pad_collate, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate, pin_memory=True)


model = RiboPolisherMamba(d_model=192, n_layers=4, d_state=16, d_conv=4, dropout=0.1).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

# --- Check Sparsity ---
print("Checking training data sparsity...")
total_zeros = 0
total_count = 0
for i, bo in enumerate(train_dl):
    tgt = bo["target"]
    mask = bo["mask"]
    # tgt is log1p encoded
    true_cnt = torch.expm1(torch.clamp(tgt, max=20.0))
    valid_cnt = true_cnt[mask]
    total_zeros += (valid_cnt <= 0.1).sum().item() # treat <0.1 as zero
    total_count += valid_cnt.numel()
    if i > 50: break # check first ~50 batches
if total_count > 0:
    print(f"[sparsity] Fraction of zeros in TRAIN (first ~50 batches): {total_zeros/total_count:.4f} ({total_zeros}/{total_count})")
else:
    print("[sparsity] Could not determine sparsity.")







test_metrics = evaluate_test(test_dl, geo_map_test, ANGLE_BINS)
print(test_metrics)



# Angle bins tensor (once)
angle_bins_t = torch.as_tensor(ANGLE_BINS, device=DEVICE, dtype=torch.float32)

best_val, best_path = float("inf"), str(OUT_DIR / "mamba_final_unscaled.pt")
for ep in range(1, EPOCHS + 1):
    model.train()

    run_total, nb = 0.0, 0
    # (Optional) track each loss component to monitor scale
    agg = {
        "pertrans": 0.0, "elem": 0.0,
        "codon": 0.0, "bucket": 0.0, "pair": 0.0, "angle": 0.0, "nll": 0.0,
        "pearson": 0.0, "spearman": 0.0
    }
    zero_pred_count = 0
    nonzero_true_count = 0

    for bo in train_dl:
        cod_ids = bo["cod_ids"].to(DEVICE)
        sim     = bo["sim_feat"].to(DEVICE)
        tgt_log = bo["target"].to(DEVICE)
        mask    = bo["mask"].to(DEVICE)
        angle_bin  = bo["angle_bin"].to(DEVICE)
        pair_bin   = bo["pair_bin"].to(DEVICE)
        bucket_idx = bo["bucket_idx"].to(DEVICE)
        opt.zero_grad(set_to_none=True)

        # Forward
        pred_log = model(cod_ids, sim, mask,angle_bin=angle_bin, pair_bin=pair_bin, bucket_idx=bucket_idx)
        # counts reported by the evaluation and by seq2ribo inference: expm1 of the output
        pred_cnt = torch.expm1(torch.clamp(pred_log, max=20.0))
        true_cnt = torch.expm1(torch.clamp(tgt_log,   max=20.0))

        # --- NLL Loss ---
        # Poisson NLL: loss = exp(pred_log) - true_cnt * pred_log
        nll = torch.nn.functional.poisson_nll_loss(
            pred_log, true_cnt, log_input=True, full=True, reduction='none'
        )
        # Apply mask and WEIGHT for non-zeros
        weight = torch.ones_like(nll)
        is_nonzero = (true_cnt > 0.1) # strictly positive counts
        if NON_ZERO_WEIGHT != 1.0:
            weight[is_nonzero] = NON_ZERO_WEIGHT
            
        masked_nll = nll * mask * weight
        # Normalize by sum of weights (or just mask sum? usually mask sum to keep scale consistent if weight is high)
        # If we normalize by sum(weight*mask), the loss magnitude is comparable.
        # But here we want to PENALIZE errors on non-zeros more. So dividing by mask.sum() keeps the penalty high.
        loss = masked_nll.sum() / mask.sum().clamp(min=1.0)


        # -------- Loss 1: per-transcript normalized MAE --------
        with torch.no_grad():
            diffs = torch.abs((pred_cnt - true_cnt) * mask)       # (B, L)
            numer = diffs.sum(dim=1)                              # (B,)
            ribo_loads = (true_cnt * mask).sum(dim=1)             # (B,)

            per_sample_loss = torch.where(
                ribo_loads > 0,
                numer / ribo_loads,
                torch.zeros_like(numer)
            )
            pertrans_loss = per_sample_loss.mean()

            # -------- Loss 2: elementwise MAE --------
            masked_positions = mask.sum()
            elem_loss = (diffs.sum() / masked_positions) if masked_positions > 0 else torch.tensor(0.0, device=pred_cnt.device)

            # -------- Losses 3–6: sum-based bin MAEs (codon/bucket/pair/angle) --------
            codon_mae_b, bucket_mae_b, pair_mae_b, angle_mae_b = _batch_maes_from_bins(
                bo, pred_cnt, true_cnt, mask, geo_map_train, angle_bins_t, DEVICE
            )
            
            # -------- Correlations (Batch-wise) --------
            # Flatten and mask
            mask_flat = mask.view(-1)
            pred_flat = pred_cnt.view(-1)[mask_flat]
            true_flat = true_cnt.view(-1)[mask_flat]
            
            if pred_flat.numel() > 1:
                p_np = pred_flat.cpu().numpy()
                t_np = true_flat.cpu().numpy()
                if np.std(t_np) > 1e-9 and np.std(p_np) > 1e-9:
                    pc, _ = pearsonr(t_np, p_np)
                    sc, _ = spearmanr(t_np, p_np)
                else:
                    pc, sc = 0.0, 0.0
            else:
                pc, sc = 0.0, 0.0
        
        loss.backward()
        if GRAD_CLIP is not None:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

        # Logging aggregates
        # run_total += loss.item() # loss is now just NLL sum/mask.sum
        agg["nll"]      += loss.item()
        agg["pertrans"] += float(pertrans_loss.item())
        agg["elem"]     += float(elem_loss.item())
        agg["codon"]    += float(codon_mae_b.item())
        agg["bucket"]   += float(bucket_mae_b.item())
        agg["pair"]     += float(pair_mae_b.item())
        agg["angle"]    += float(angle_mae_b.item())
        agg["pearson"]  += pc
        agg["spearman"] += sc
        nb += 1

        # Track "false zeros" (where pred is effectively 0 but true > 0)
        # Using 0.1 as threshold for "zero" prediction since these are counts
        with torch.no_grad():
            is_nonzero_true = (true_cnt > 0) & mask
            is_zero_pred = (pred_cnt < 0.1) & is_nonzero_true
            
            zero_pred_count += is_zero_pred.sum().item()
            nonzero_true_count += is_nonzero_true.sum().item()

    # Epoch means
    # tr_loss = run_total / max(nb, 1) # This was mixed loss before
    for k in agg: agg[k] /= max(nb, 1)

    # ===== Validation (uses your new evaluator with all 6 metrics) =====
    val_metrics = evaluate_counts(model, val_dl, geo_map_val, ANGLE_BINS)
    # val_loss_total is now NLL? Or mix? Scheduler usually tracks invalid loss.
    # Let's track NLL for scheduler
    val_loss_total = val_metrics["nll"]
    sched.step(val_loss_total)  

    print(f"[Ep {ep:02d}")
    print(f"TRAIN: NLL:{agg['nll']:.4f} P:{agg['pearson']:.4f} S:{agg['spearman']:.4f} "
          f"PT:{agg['pertrans']:.4f} EL:{agg['elem']:.4f} "
          f"Cod:{agg['codon']:.4f} Buc:{agg['bucket']:.4f} Pr:{agg['pair']:.4f} Ang:{agg['angle']:.4f}")

    print(f"VAL  : NLL:{val_metrics['nll']:.4f} P:{val_metrics['pearson_elem']:.4f} S:{val_metrics['spearman_elem']:.4f} "
          f"PT:{val_metrics['pertranscriptMAE']:.4f} EL:{val_metrics['elemwiseMAE']:.4f} "
          f"Cod:{val_metrics['codonMAE']:.4f} Buc:{val_metrics['bucketMAE']:.4f} Pr:{val_metrics['pairMAE']:.4f} Ang:{val_metrics['angleMAE']:.4f}")

    if nonzero_true_count > 0:
        zero_pred_pct = 100.0 * zero_pred_count / nonzero_true_count
        print(f"TRAIN False Zeros: {zero_pred_pct:.2f}% ({zero_pred_count}/{nonzero_true_count})")
    else:
        print(f"TRAIN False Zeros: N/A")

    # Save by your chosen criterion (using Val NLL now)
    if val_loss_total < best_val:
        best_val = val_loss_total
        torch.save(model.state_dict(), best_path)



    print("VAL:", val_metrics)
    print("")


model = RiboPolisherMamba(d_model=192, n_layers=4, d_state=16, d_conv=4, dropout=0.1).to(DEVICE)
best_path = str(OUT_DIR / "mamba_final_unscaled.pt")
model.load_state_dict(torch.load(best_path, weights_only=True))

model.eval()
test_metrics = evaluate_counts(model, test_dl, geo_map_test, ANGLE_BINS)
print(test_metrics)


run_meta = {
    "cell_line": ARGS.cell_line,
    "data_dir": str(DATA_DIR),
    "geo_map_dir": str(GEO_MAP_DIR),
    "cache_dir": str(CACHE_DIR),
    "device": DEVICE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "weight_decay": WEIGHT_DECAY,
    "grad_clip": GRAD_CLIP,
    "non_zero_weight": NON_ZERO_WEIGHT,
    "best_checkpoint": best_path,
    "final_test_metrics": test_metrics,
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
}
with (OUT_DIR / "run_metrics.json").open("w", encoding="utf-8") as f:
    json.dump(run_meta, f, indent=2, default=float)
print(f"Saved polisher outputs to: {OUT_DIR}")

