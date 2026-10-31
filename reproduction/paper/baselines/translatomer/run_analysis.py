import torch
import torch.nn as nn
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from util import * 
from scipy.stats import pearsonr, spearmanr

BATCH_SIZE = 192
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
K_ANGLE_BINS = 4

class InMemoryDataset(Dataset):
    def __init__(self, pred_counts_dict, true_counts_dict, test_data_pkl_path, geomap=None, angle_bins=None):
        self.tx_list = [tx for tx in pred_counts_dict if tx in true_counts_dict]
        self.pred_counts = pred_counts_dict
        self.true_counts = true_counts_dict
        self.geomap = geomap or {}
        self.angle_bins = angle_bins
        
        with open(test_data_pkl_path, 'rb') as f:
            raw_data = pickle.load(f)
        self.cod_ids_map = {}
        for item in raw_data:
            tx = item[0]
            # Codon IDs are in Item 4[1] if Item 4 is tuple
            if len(item) >= 5 and isinstance(item[4], tuple) and len(item[4]) >= 2:
                self.cod_ids_map[tx] = item[4][1]
            else:
                # Fallback or error logic
                pass

    def __len__(self):
        return len(self.tx_list)

    def __getitem__(self, i):
        tx = self.tx_list[i]
        tgt = self.true_counts[tx].astype(np.float32)
        sim = self.pred_counts[tx].astype(np.float32)
        # Determine L
        L = len(tgt)
        if len(sim) > L:
            sim = sim[:L]
        elif len(sim) < L:
            sim = np.pad(sim, (0, L - len(sim)))

        
        # Get Codon IDs
        # If mapping logic worked, L should roughly match codon IDs len
        cods = self.cod_ids_map.get(tx, np.full(L, -1, dtype=np.int32))
        if len(cods) > L: cods = cods[:L]
        elif len(cods) < L:
             # This happens if we have more NT counts // 3 than codon ids?
             # Pad with 64 (empty)
             pad = np.full(L - len(cods), 64, dtype=np.int32)
             cods = np.concatenate([cods, pad])
             
        # Geo/Angle bins
        kk = np.zeros(L, dtype=np.int64)
        pb = np.zeros(L, dtype=np.int64)
        
        if (self.angle_bins is not None) and (tx in self.geomap):
            # geomap[tx] is (angle_dev_sum, pair_count)
            tup = self.geomap[tx]
            if len(tup) >= 2:
                angle_dev_sum, pair_count = tup[0], tup[1]
                # Ensure length matches L
                if len(angle_dev_sum) >= L:
                    a = np.asarray(angle_dev_sum, dtype=np.float64)[:L]
                    p = np.asarray(pair_count,    dtype=np.int64)[:L]
                    
                    kk = np.digitize(a, self.angle_bins) - 1
                    kk = np.clip(kk, 0, K_ANGLE_BINS - 1).astype(np.int64)
                    
                    pb = np.clip(p, 0, 3).astype(np.int64)
        
        # bucket bins
        bb = np.zeros(L, dtype=np.int64)
        l1 = L // 3
        l2 = (2 * L) // 3
        bb[l1:l2] = 1
        bb[l2:]   = 2
        
        return {
            "tx": tx,
            "cod_ids": torch.tensor(cods, dtype=torch.long),
            "sim_feat": torch.tensor(sim, dtype=torch.float32),
            "target": torch.tensor(np.log1p(tgt), dtype=torch.float32),
            "length": int(L),
            "sim_real": torch.tensor(sim, dtype=torch.float32),
            "obs_real": torch.tensor(tgt, dtype=torch.float32),
            "angle_bin": torch.tensor(kk, dtype=torch.long),
            "pair_bin":  torch.tensor(pb, dtype=torch.long),
            "bucket_idx":torch.tensor(bb, dtype=torch.long),
            "mask": torch.ones(L, dtype=torch.bool)
        }

def pad_collate(batch):
    batch = sorted(batch, key=lambda x: x["length"], reverse=True)
    B, Lmax = len(batch), batch[0]["length"]

    cod = torch.full((B, Lmax), 64, dtype=torch.long)
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
        tgt[i, :L] = ex["target"]
        msk[i, :L] = True
        sim_real[i, :L] = ex["sim_real"]
        obs_real[i, :L] = ex["obs_real"]

        angle_bin[i, :L] = ex["angle_bin"]
        pair_bin[i, :L]  = ex["pair_bin"]
        bucket_idx[i,:L] = ex["bucket_idx"]

    return {
        "tx": txs,
        "cod_ids": cod, "target": tgt, "mask": msk,
        "sim_real": sim_real, "obs_real": obs_real,
        "angle_bin": angle_bin, "pair_bin": pair_bin, "bucket_idx": bucket_idx,
    }

def _corr_safe(a, b):
    """Compute Pearson and Spearman, returning 0.0 on failure."""
    pr, sr = 0.0, 0.0
    if len(a) > 1 and np.std(a) > 1e-9 and np.std(b) > 1e-9:
        try:
            r, _ = pearsonr(a, b)
            if np.isfinite(r): pr = float(r)
        except Exception: pass
        try:
            rho, _ = spearmanr(a, b)
            if np.isfinite(rho): sr = float(rho)
        except Exception: pass
    return float(pr), float(sr)


def _compute_all_correlations(tx_grouped):
    """
    From tx_grouped: {tx -> {'true': np.array, 'pred': np.array}}
    Compute shape, txlvl, and elemwise correlations.
    """
    shape_pears, shape_spears = [], []
    tx_obs_sums, tx_pred_sums = [], []
    all_true, all_pred = [], []

    for tx, data in tx_grouped.items():
        t, p = data['true'], data['pred']

        pr, sr = _corr_safe(t, p)
        shape_pears.append(pr); shape_spears.append(sr)

        tx_obs_sums.append(float(np.sum(t)))
        tx_pred_sums.append(float(np.sum(p)))

        all_true.append(t)
        all_pred.append(p)

    avg_shape_pear = float(np.nanmean(shape_pears)) if shape_pears else 0.0
    avg_shape_spear = float(np.nanmean(shape_spears)) if shape_spears else 0.0

    txlvl_pear, txlvl_spear = _corr_safe(
        np.array(tx_obs_sums), np.array(tx_pred_sums)) if tx_obs_sums else (0.0, 0.0)

    elem_pear, elem_spear = 0.0, 0.0
    if all_true:
        cat_true = np.concatenate(all_true)
        cat_pred = np.concatenate(all_pred)
        elem_pear, elem_spear = _corr_safe(cat_true, cat_pred)

    return {
        "shape_pear": avg_shape_pear,
        "shape_spear": avg_shape_spear,
        "txlvl_pear": txlvl_pear,
        "txlvl_spear": txlvl_spear,
        "elemwise_pear": elem_pear,
        "elemwise_spear": elem_spear,
    }


def _accumulate_bins(obs, pred, cods, bucket_idx, pair_bin, angle_bin,
                     cod_obs, cod_pred, bucket_obs, bucket_pred,
                     pair_obs, pair_pred, ang_obs, ang_pred):
    """Accumulate bin-level sums for MAE computation."""
    non_stop = ~np.isin(cods, STOP_IDX)
    if np.any(non_stop):
        c_ns = cods[non_stop]
        valid_ids = (c_ns >= 0) & (c_ns < len(CODONS))
        if np.any(valid_ids):
            cod_obs += np.bincount(c_ns[valid_ids], weights=obs[non_stop][valid_ids], minlength=len(CODONS))
            cod_pred += np.bincount(c_ns[valid_ids], weights=pred[non_stop][valid_ids], minlength=len(CODONS))

    bucket_obs += np.bincount(bucket_idx, weights=obs, minlength=3)
    bucket_pred += np.bincount(bucket_idx, weights=pred, minlength=3)

    pair_obs += np.bincount(pair_bin, weights=obs, minlength=4)
    pair_pred += np.bincount(pair_bin, weights=pred, minlength=4)

    ang_obs += np.bincount(angle_bin, weights=obs, minlength=K_ANGLE_BINS)
    ang_pred += np.bincount(angle_bin, weights=pred, minlength=K_ANGLE_BINS)


@torch.no_grad()
def evaluate_test(loader, match_scale=False):
    """
    Evaluate predictions. Reports BOTH unscaled and per-transcript-scaled metrics.
    The match_scale parameter is kept for backward compatibility but no longer needed;
    both versions are always computed.

    Returns dict with unscaled metrics and sc_ prefixed scaled metrics.
    """
    # ---- Unscaled accumulators ----
    u_per_tx_losses = []
    u_cod_obs = np.zeros(len(CODONS), dtype=np.float64); u_cod_pred = np.zeros(len(CODONS), dtype=np.float64)
    u_bucket_obs = np.zeros(3, dtype=np.float64); u_bucket_pred = np.zeros(3, dtype=np.float64)
    u_pair_obs = np.zeros(4, dtype=np.float64); u_pair_pred = np.zeros(4, dtype=np.float64)
    u_ang_obs = np.zeros(K_ANGLE_BINS, dtype=np.float64); u_ang_pred = np.zeros(K_ANGLE_BINS, dtype=np.float64)
    u_tx_grouped = {}

    # ---- Scaled accumulators ----
    s_per_tx_losses = []
    s_cod_pred = np.zeros(len(CODONS), dtype=np.float64)
    s_bucket_obs = np.zeros(3, dtype=np.float64); s_bucket_pred = np.zeros(3, dtype=np.float64)
    s_pair_obs = np.zeros(4, dtype=np.float64); s_pair_pred = np.zeros(4, dtype=np.float64)
    s_ang_obs = np.zeros(K_ANGLE_BINS, dtype=np.float64); s_ang_pred = np.zeros(K_ANGLE_BINS, dtype=np.float64)
    s_tx_grouped = {}

    u_total_abs_diff = 0.0; s_total_abs_diff = 0.0
    total_positions = 0

    for bo in loader:
        mask = bo["mask"].to(DEVICE)
        sim_raw = bo["sim_real"].to(DEVICE)
        obs_real = bo["obs_real"].to(DEVICE)

        B = mask.shape[0]

        # Batch-level accumulation data
        mask_np = bo["mask"].detach().cpu().numpy()
        true_np = bo["obs_real"].detach().cpu().numpy()
        pred_raw_np = sim_raw.detach().cpu().numpy()
        cod_ids_np = bo["cod_ids"].detach().cpu().numpy()
        bucket_np = bo["bucket_idx"].detach().cpu().numpy()
        pair_np = bo["pair_bin"].detach().cpu().numpy()
        angle_np = bo["angle_bin"].detach().cpu().numpy()

        for i in range(B):
            tx = bo["tx"][i]
            keep = mask_np[i].astype(bool)
            if not np.any(keep):
                continue

            obs_f = true_np[i][keep].astype(np.float64)
            raw_f = pred_raw_np[i][keep].astype(np.float64)
            cods_i = cod_ids_np[i][keep].astype(np.int64)
            buck_i = bucket_np[i][keep].astype(np.int64)
            pair_i = pair_np[i][keep].astype(np.int64)
            ang_i = angle_np[i][keep].astype(np.int64)

            obs_s = float(np.sum(obs_f))
            pred_s = float(np.sum(raw_f))

            # Per-transcript scaling
            if pred_s > 0:
                sf = obs_s / pred_s
                sc_f = raw_f * sf
            else:
                sc_f = np.zeros_like(raw_f)

            # ---- Unscaled ----
            u_absdiff = np.abs(raw_f - obs_f)
            u_numer = float(np.sum(u_absdiff))
            u_per_tx_losses.append(u_numer / obs_s if obs_s > 0 else 0.0)
            u_total_abs_diff += u_numer
            total_positions += u_absdiff.size

            u_tx_grouped[tx] = {'true': obs_f, 'pred': raw_f}

            _accumulate_bins(obs_f, raw_f, cods_i, buck_i, pair_i, ang_i,
                             u_cod_obs, u_cod_pred, u_bucket_obs, u_bucket_pred,
                             u_pair_obs, u_pair_pred, u_ang_obs, u_ang_pred)

            # ---- Scaled ----
            s_absdiff = np.abs(sc_f - obs_f)
            s_numer = float(np.sum(s_absdiff))
            s_per_tx_losses.append(s_numer / obs_s if obs_s > 0 else 0.0)
            s_total_abs_diff += s_numer

            s_tx_grouped[tx] = {'true': obs_f, 'pred': sc_f}

            _accumulate_bins(obs_f, sc_f, cods_i, buck_i, pair_i, ang_i,
                             np.zeros(len(CODONS)),  # dummy obs (already in u_cod_obs)
                             s_cod_pred, s_bucket_obs, s_bucket_pred,
                             s_pair_obs, s_pair_pred, s_ang_obs, s_ang_pred)

    # ---- Correlations ----
    u_corr = _compute_all_correlations(u_tx_grouped)
    s_corr = _compute_all_correlations(s_tx_grouped)

    # ---- Final MAEs: Unscaled ----
    u_pertxMAE = float(np.mean(u_per_tx_losses)) if u_per_tx_losses else float("nan")
    u_elemMAE = float(u_total_abs_diff / max(total_positions, 1))
    u_codonMAE = float(np.mean(np.abs(u_cod_obs - u_cod_pred)))
    u_bucketMAE = float(np.mean(np.abs(u_bucket_obs - u_bucket_pred)))
    u_pairMAE = float(np.mean(np.abs(u_pair_obs - u_pair_pred)))
    u_angleMAE = float(np.mean(np.abs(u_ang_obs - u_ang_pred)))

    # ---- Final MAEs: Scaled ----
    s_pertxMAE = float(np.mean(s_per_tx_losses)) if s_per_tx_losses else float("nan")
    s_elemMAE = float(s_total_abs_diff / max(total_positions, 1))
    s_codonMAE = float(np.mean(np.abs(u_cod_obs - s_cod_pred)))
    s_bucketMAE = float(np.mean(np.abs(s_bucket_obs - s_bucket_pred)))
    s_pairMAE = float(np.mean(np.abs(s_pair_obs - s_pair_pred)))
    s_angleMAE = float(np.mean(np.abs(s_ang_obs - s_ang_pred)))

    result = {
        # ---- Unscaled ----
        "pertranscriptMAE": u_pertxMAE,
        "elemwiseMAE": u_elemMAE,
        "codonMAE": u_codonMAE,
        "bucketMAE": u_bucketMAE,
        "pairMAE": u_pairMAE,
        "angleMAE": u_angleMAE,
    }
    result.update(u_corr)

    # ---- Scaled ----
    result["sc_pertranscriptMAE"] = s_pertxMAE
    result["sc_elemwiseMAE"] = s_elemMAE
    result["sc_codonMAE"] = s_codonMAE
    result["sc_bucketMAE"] = s_bucketMAE
    result["sc_pairMAE"] = s_pairMAE
    result["sc_angleMAE"] = s_angleMAE
    for k, v in s_corr.items():
        result[f"sc_{k}"] = v

    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_counts', required=True)
    parser.add_argument('--true_counts', required=True)
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--geo_map', required=False)
    parser.add_argument('--match_scale', action='store_true', help='Scale prediction sum to match ground truth sum per transcript')
    args = parser.parse_args()
    
    print("Loading dictionaries...")
    with open(args.pred_counts, 'rb') as f:
        pred_counts = pickle.load(f)
    with open(args.true_counts, 'rb') as f:
        true_counts = pickle.load(f)
        
    geomap = {}
    if args.geo_map:
        print(f"Loading geomap from {args.geo_map}...")
        with open(args.geo_map, 'rb') as f:
            geomap = pickle.load(f)
            
    # Define linear Angle Bins for 0-2pi
    ANGLE_BINS = np.linspace(0, 2*np.pi, K_ANGLE_BINS + 1)
        
    print("Creating dataset...")
    ds = InMemoryDataset(pred_counts, true_counts, args.test_data, geomap=geomap, angle_bins=ANGLE_BINS)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)
    
    print(f"Running evaluation (match_scale={args.match_scale})...")
    metrics = evaluate_test(dl, match_scale=args.match_scale)
    print("Results:")
    print(metrics)
