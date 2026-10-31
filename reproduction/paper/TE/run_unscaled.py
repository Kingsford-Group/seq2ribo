# -*- coding: utf-8 -*-
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import pickle
import random
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_data import load_data
from simulation import simulate_transcript_parallel
from util import K_ANGLE_BINS, NONSTOP_IDX, build_full_rate

try:
    from scipy.stats import pearsonr, spearmanr
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

TE_COL_BY_CELL = {
    "ipsc": "mean_te",
    "hek293": "TE_HEK293",
    "rpe": "TE_RPE-1",
    "lcl": "TE_LCL",
}


def parse_args():
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="TE run simulation (canonical)")
    parser.add_argument("--cell-line", required=True, choices=["ipsc", "hek293", "rpe", "lcl"])
    parser.add_argument("--subset", choices=["full", "01", "001", "0001"], default="full")
    parser.add_argument("--times", type=int, default=10)
    parser.add_argument("--sheet-name", default=0)
    parser.add_argument("--xlsx-path", default=str(Path(__file__).resolve().parent / "41587_2025_2712_MOESM3_ESM.xlsx"))
    parser.add_argument("--te-col", default=None)
    parser.add_argument("--data-root", default=str(base_dir / "data"))
    parser.add_argument("--output-root", default=str(base_dir / "results" / "TE"))
    parser.add_argument("--params", default=None,
                        help="fit output JSON (sTASEP); default: seq2ribo/constants.py for --cell-line")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="replace existing samples")
    return parser.parse_args()


def _extract_cds_seq(cds_val):
    if isinstance(cds_val, str):
        return cds_val
    if isinstance(cds_val, (list, tuple)) and len(cds_val) >= 1 and isinstance(cds_val[0], str):
        return cds_val[0]
    return str(cds_val)


def _normalize_seq(seq: str) -> str:
    return seq.upper().replace("U", "T").strip()


def cds_hash_from_tx(cds_seqs_val) -> str:
    seq = _normalize_seq(_extract_cds_seq(cds_seqs_val))
    return hashlib.md5(seq.encode("utf-8")).hexdigest()


def filter_split_dicts(ref_cds_seqs, target_cds_seqs, target_asite, target_geo, ref_name="Ref", target_name="Target", verbose=True):
    ref_hashes = {cds_hash_from_tx(cds_val) for _tx, cds_val in ref_cds_seqs.items()}
    kept, removed = [], []
    for tx, cds_val in target_cds_seqs.items():
        if cds_hash_from_tx(cds_val) in ref_hashes:
            removed.append(tx)
        else:
            kept.append(tx)

    cds_seqs_f = {tx: target_cds_seqs[tx] for tx in kept}
    asite_f = {tx: target_asite[tx] for tx in kept if tx in target_asite}
    geo_f = {tx: target_geo[tx] for tx in kept if tx in target_geo}

    if verbose:
        print(f"[filter] {target_name} vs {ref_name}: kept={len(kept)} removed={len(removed)}")
    return cds_seqs_f, asite_f, geo_f, removed


def derive_angle_bins_from_training(geo_map_train, k_bins, method, low_q, high_q):
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
    return edges


def _normalized_rate(rate_fit: np.ndarray) -> np.ndarray:
    full = build_full_rate(rate_fit)
    mu = np.mean(full[NONSTOP_IDX]) if np.any(full[NONSTOP_IDX]) else 1.0
    full[NONSTOP_IDX] = full[NONSTOP_IDX] / (mu + 1e-9)
    return full


def load_stasep_params(cell_line, params_json=None):
    """Fitted sTASEP wait times: from a fit output JSON if given, else seq2ribo/constants.py.

    alpha indexes the pair count, beta the angle bin, buckets the CDS third.
    """
    if params_json:
        with open(params_json) as f:
            p = json.load(f)["params"]
        return (np.asarray(p["rates"], dtype=np.float64), np.asarray(p["alpha"], dtype=np.float64),
                np.asarray(p["beta"], dtype=np.float64), np.asarray(p["buckets"], dtype=np.float64))
    spec = importlib.util.spec_from_file_location(
        "seq2ribo_constants", Path(__file__).resolve().parents[3] / "seq2ribo" / "constants.py")
    const = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(const)
    key = {"ipsc": "IPSC", "hek293": "HEK293", "rpe": "RPE", "lcl": "LCL"}[cell_line]
    return tuple(np.asarray(getattr(const, f"{key}_{name}"), dtype=np.float64)
                 for name in ("RATES", "ALPHA", "BETA", "BUCKETS"))


def _seeded_simulation(job):
    seed, args = job
    random.seed(seed)
    return simulate_transcript_parallel(args)


def _to_scalar_completed_total(ct):
    try:
        arr = np.asarray(ct, dtype=np.float64)
        return float(arr.item()) if arr.size == 1 else float(np.nanmean(arr))
    except Exception:
        try:
            return float(ct)
        except Exception:
            return np.nan


def _to_total_ribo(sim_scaled, sim_raw):
    arr = None
    try:
        if sim_raw is not None:
            arr = np.asarray(sim_raw, dtype=np.float64)
        elif sim_scaled is not None:
            arr = np.asarray(sim_scaled, dtype=np.float64)
    except Exception:
        pass
    if arr is None:
        return np.nan
    arr = arr[np.isfinite(arr)]
    return float(arr.sum()) if arr.size else np.nan


def _scatter_with_fit(x, y, xlabel, ylabel, title, outpath, r_txt=None):
    fig = plt.figure(dpi=300)
    plt.scatter(x, y, alpha=1, s=25)
    try:
        if len(x) >= 2:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(np.min(x), np.max(x), 200)
            plt.plot(xs, m * xs + b, linewidth=2)
    except Exception:
        pass
    if r_txt:
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))
        xpad = 0.03 * (xmax - xmin) if np.isfinite(xmin) and np.isfinite(xmax) else 0.03
        ypad = 0.05 * (ymax - ymin) if np.isfinite(ymin) and np.isfinite(ymax) else 0.05
        plt.text(xmin + xpad, ymax - ypad, r_txt)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main():
    args = parse_args()
    te_col = args.te_col or TE_COL_BY_CELL[args.cell_line]
    out_dir = Path(args.output_root) / args.cell_line / "run_unscaled"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_root) / args.cell_line
    (cds_seqs_train, asite_train, geo_map_train), (cds_seqs_val, asite_val, geo_map_val), (cds_seqs_test, asite_test, geo_map_test) = load_data(args.subset, str(data_dir))
    print(f"Train: {len(cds_seqs_train)}, Val: {len(cds_seqs_val)}, Test: {len(cds_seqs_test)}")

    cds_seqs_val_f, asite_val_f, geo_map_val_f, removed_val = filter_split_dicts(cds_seqs_train, cds_seqs_val, asite_val, geo_map_val, ref_name="Train", target_name="Val")
    cds_seqs_test_f1, asite_test_f1, geo_map_test_f1, removed_test_train = filter_split_dicts(cds_seqs_train, cds_seqs_test, asite_test, geo_map_test, ref_name="Train", target_name="Test")
    cds_seqs_test_f, asite_test_f, geo_map_test_f, removed_test_val = filter_split_dicts(cds_seqs_val_f, cds_seqs_test_f1, asite_test_f1, geo_map_test_f1, ref_name="Val", target_name="Test")
    print(f"[Final] Filtered Test: {len(cds_seqs_test_f)} (removed {len(removed_test_train)} vs Train, {len(removed_test_val)} vs Val)")

    angle_bins = derive_angle_bins_from_training(geo_map_train, K_ANGLE_BINS, "linear", 0.0, 0.995)
    rates, alpha_vec, beta_vec, bucket_vec = load_stasep_params(args.cell_line, args.params)
    rate = _normalized_rate(rates)

    df = pd.read_excel(args.xlsx_path, sheet_name=args.sheet_name, converters={"gene_id": str, "tx_id": str})
    df.columns = df.columns.str.strip()
    if te_col not in df.columns:
        raise ValueError(f"Expected TE column '{te_col}' in workbook")

    def _normalize_tx_id(tx: str) -> str:
        s = str(tx) if pd.notna(tx) else ""
        return s.split(".")[0] if s else s

    def _match_to_split(df_in, asite_dict, split_label):
        norm_to_full = {_normalize_tx_id(tx): tx for tx in asite_dict.keys()}
        tx_exact = df_in["tx_id"].astype(str)
        tx_norm = tx_exact.map(_normalize_tx_id)
        match_tx = tx_exact.where(tx_exact.isin(asite_dict.keys()), tx_norm.map(norm_to_full))
        out = df_in[~match_tx.isna()].copy()
        out["match_tx"] = match_tx[~match_tx.isna()].astype(str)
        out["split"] = split_label
        return out[["gene_id", "tx_id", "match_tx", te_col, "split"]]

    matched_train = _match_to_split(df, asite_train, "train")
    matched_val = _match_to_split(df, asite_val_f, "val")
    matched_test = _match_to_split(df, asite_test_f, "test")
    matched_all = pd.concat([matched_train, matched_val, matched_test], ignore_index=True)

    def _get_split_dicts(split_label):
        if split_label == "train":
            return asite_train, cds_seqs_train, geo_map_train
        if split_label == "val":
            return asite_val_f, cds_seqs_val_f, geo_map_val_f
        if split_label == "test":
            return asite_test_f, cds_seqs_test_f, geo_map_test_f
        raise ValueError("Unknown split")

    def _get_matched_df(split_label):
        return {"train": matched_train, "val": matched_val, "test": matched_test}[split_label]

    def _build_job_args(split_label):
        a_dict, c_dict, g_dict = _get_split_dicts(split_label)
        matched_txs = set(_get_matched_df(split_label)["match_tx"].tolist())
        txs = sorted(matched_txs & set(a_dict.keys()) & set(c_dict.keys()) & set(g_dict.keys()))
        jobs = []
        for tx in txs:
            seq, a_cnts = a_dict[tx]
            angle_dev_sum, pair_count = g_dict[tx]
            jobs.append((tx, seq, a_cnts, angle_dev_sum, pair_count, rate, alpha_vec, beta_vec, bucket_vec, angle_bins))
        return jobs

    completed_total_by_key = {}
    total_ribo_by_key = {}
    for split in ["train", "val", "test"]:
        (out_dir / split).mkdir(parents=True, exist_ok=True)
        stale = list((out_dir / split).glob("*_sample_*.pkl"))
        if stale:
            if not args.overwrite:
                raise SystemExit(f"{out_dir / split} already has {len(stale)} samples; pass --overwrite to replace them")
            for f in stale:
                f.unlink()

    for split_label in ["train", "val", "test"]:
        jobs = _build_job_args(split_label)
        print(f"[{split_label}] transcripts to simulate: {len(jobs)}, TIMES={args.times}")
        for rep in range(args.times):
            seeded = [(zlib.crc32(f"{args.seed}:{split_label}:{rep}:{j[0]}".encode()), j) for j in jobs]
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
                results = list(ex.map(_seeded_simulation, seeded, chunksize=8))
            for res in results:
                tx, cods, obs_counts, sim_vec_raw, sim_vec_scaled, scaling_factor, completed_total = res
                out_pkl = out_dir / split_label / f"{tx}_sample_{rep}.pkl"
                with out_pkl.open("wb") as fh:
                    pickle.dump((tx, cods, obs_counts, sim_vec_raw, sim_vec_scaled, scaling_factor, completed_total), fh)
                key = (split_label, tx)
                completed_total_by_key.setdefault(key, []).append(_to_scalar_completed_total(completed_total))
                total_ribo_by_key.setdefault(key, []).append(_to_total_ribo(sim_vec_scaled, sim_vec_raw))
            print(f"  [{split_label}] replica {rep + 1}/{args.times} done")

    def _agg_metric(metric_dict, name):
        rows = []
        for (split_label, tx), vals in metric_dict.items():
            vals = [v for v in vals if pd.notna(v)]
            if not vals:
                continue
            rows.append({
                "split": split_label,
                "match_tx": tx,
                name: float(np.mean(vals)),
                f"{name}_std": float(np.std(vals, ddof=0)),
                "replicas": int(len(vals)),
            })
        return pd.DataFrame(rows)

    completed_df = _agg_metric(completed_total_by_key, "completed_total")
    totalribo_df = _agg_metric(total_ribo_by_key, "total_ribo_count")
    joined = matched_all.merge(completed_df, on=["split", "match_tx"], how="left").merge(totalribo_df, on=["split", "match_tx", "replicas"], how="left")

    out_xlsx = out_dir / "matched_mean_te_with_completed_total_train_val_test.xlsx"
    out_csv = out_dir / "matched_mean_te_with_completed_total_train_val_test.csv"
    joined.to_excel(out_xlsx, index=False)
    joined.to_csv(out_csv, index=False)

    with (out_dir / "mean_te_summary.txt").open("w", encoding="utf-8") as fh:
        for split_label in ["train", "val", "test"]:
            sub = joined.loc[joined["split"] == split_label, te_col]
            m = pd.to_numeric(sub, errors="coerce").dropna().mean()
            fh.write(f"{split_label}: mean TE over matched = {m} (N={sub.notna().sum()})\n")
        all_m = pd.to_numeric(joined[te_col], errors="coerce").dropna().mean()
        fh.write(f"all: mean TE over matched = {all_m} (N={joined[te_col].notna().sum()})\n")

    def _corr_and_plot(df_sub, x_col, y_col, base_name, label):
        x = pd.to_numeric(df_sub[x_col], errors="coerce")
        y = pd.to_numeric(df_sub[y_col], errors="coerce")
        mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        n = int(mask.sum())

        pearson_r = float("nan")
        pearson_p = float("nan")
        spearman_r = float("nan")
        spearman_p = float("nan")
        if n >= 2 and _HAS_SCIPY:
            pearson_r, pearson_p = pearsonr(x[mask], y[mask])
            spearman_r, spearman_p = spearmanr(x[mask], y[mask])

        out_png = out_dir / f"{base_name}_{label}.png"
        if n > 0:
            r_txt = f"n={n}; r={pearson_r:.3f}" if np.isfinite(pearson_r) else f"n={n}"
            _scatter_with_fit(
                x[mask].to_numpy(),
                y[mask].to_numpy(),
                f"Translation efficiency ({te_col})",
                y_col.replace("_", " ").title(),
                f"{y_col.replace('_',' ').title()} vs TE ({label})",
                str(out_png),
                r_txt=r_txt,
            )
        return n, pearson_r, pearson_p, spearman_r, spearman_p, out_png

    with (out_dir / "correlation_summary.txt").open("w", encoding="utf-8") as fh:
        for label, df_sub in [("train", joined[joined["split"] == "train"]), ("val", joined[joined["split"] == "val"]), ("test", joined[joined["split"] == "test"]), ("all", joined)]:
            fh.write(f"=== Split: {label} ===\n")
            for ycol, base in [("completed_total", "corr_completed_vs_TE"), ("total_ribo_count", "corr_totalribo_vs_TE")]:
                n, pr, pp, sr, sp, png = _corr_and_plot(df_sub, te_col, ycol, base, label)
                fh.write(f"{ycol} vs TE: rows={n}, Pearson r={pr}")
                if _HAS_SCIPY:
                    fh.write(f", p={pp}")
                fh.write("\n")
                if _HAS_SCIPY:
                    fh.write(f"Spearman r={sr}, p={sp}\n")
                fh.write(f"plot: {png.name}\n")
            fh.write("\n")

    print("Saved:")
    print(f"  PKLs under {out_dir}/train, {out_dir}/val, {out_dir}/test (TIMES={args.times})")
    print(f"  {out_csv}")
    print(f"  {out_xlsx}")


if __name__ == "__main__":
    main()
