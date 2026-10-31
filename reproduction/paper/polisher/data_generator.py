import argparse
import concurrent.futures
import importlib.util
import json
import pickle
import random
import zlib
from pathlib import Path

import numpy as np

from load_data import load_data
from simulation import simulate_transcript_parallel
from util import K_ANGLE_BINS, NONSTOP_IDX, build_full_rate


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


def simulate_epoch(rate_fit, alpha_vec, beta_vec, bucket_vec, cds_seqs, asite, geo_map, times, split, polisher_root, angle_bins, seed, workers, overwrite):
    full = build_full_rate(rate_fit)
    mu = np.mean(full[NONSTOP_IDX]) if np.any(full[NONSTOP_IDX]) else 1.0
    full[NONSTOP_IDX] = full[NONSTOP_IDX] / (mu + 1e-9)
    rate = full

    split_dir = polisher_root / "data_new" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    stale = list(split_dir.glob("*_sample_*.pkl"))
    if stale:
        if not overwrite:
            raise SystemExit(f"{split_dir} already has {len(stale)} shards; pass --overwrite to replace them")
        for f in stale:
            f.unlink()

    job_args = []
    for tx, (seq, a_cnts) in asite.items():
        if tx not in cds_seqs or tx not in geo_map:
            continue
        angle_dev_sum, pair_count = geo_map[tx]
        job_args.append((tx, seq, a_cnts, angle_dev_sum, pair_count, rate, alpha_vec, beta_vec, bucket_vec, angle_bins))

    for count in range(times):
        jobs = [(zlib.crc32(f"{seed}:{split}:{count}:{a[0]}".encode()), a) for a in job_args]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_seeded_simulation, jobs, chunksize=8))
        for tx, cods, obs_counts, sim_vec_raw, sim_vec_scaled, scaling_factor, completed_total in results:
            with (split_dir / f"{tx}_sample_{count}.pkl").open("wb") as f:
                pickle.dump((tx, cods, obs_counts, sim_vec_raw, sim_vec_scaled, scaling_factor, completed_total), f)
        print(f"[{split}] sample iteration {count + 1}/{times}")


def parse_args():
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate polisher data from fitted sTASEP params")
    parser.add_argument("--cell-line", required=True, choices=["ipsc", "hek293", "rpe", "lcl"])
    parser.add_argument("--subset", default="full", choices=["full", "01", "001", "0001"])
    parser.add_argument("--times", type=int, default=32)
    parser.add_argument("--data-root", default=str(base_dir / "data"))
    parser.add_argument("--output-root", default=str(base_dir / "data"))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--params", default=None,
                        help="fit output JSON (TASEP/sTASEP); default: seq2ribo/constants.py for --cell-line")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="replace existing shards of a split")
    return parser.parse_args()


def main():
    args = parse_args()
    base_data_dir = Path(args.data_root) / args.cell_line
    polisher_root = Path(args.output_root) / args.cell_line / "polisher"

    (cds_seqs_train, asite_train, geo_map_train), (cds_seqs_val, asite_val, geo_map_val), (cds_seqs_test, asite_test, geo_map_test) = load_data(
        args.subset, str(base_data_dir)
    )
    print(f"Train: {len(cds_seqs_train)} transcripts, Val: {len(cds_seqs_val)}, Test: {len(cds_seqs_test)}")
    print(f"Generating {args.times} samples per transcript into: {polisher_root / 'data_new'}")

    angle_bins = derive_angle_bins_from_training(geo_map_train, K_ANGLE_BINS, "linear", 0.0, 0.995)

    rates, alpha_vec, beta_vec, bucket_vec = load_stasep_params(args.cell_line, args.params)

    if not args.skip_train:
        print("train samples started")
        simulate_epoch(rates, alpha_vec, beta_vec, bucket_vec, cds_seqs_train, asite_train, geo_map_train, args.times, "train", polisher_root, angle_bins, args.seed, args.workers, args.overwrite)
    if not args.skip_val:
        print("val samples started")
        simulate_epoch(rates, alpha_vec, beta_vec, bucket_vec, cds_seqs_val, asite_val, geo_map_val, args.times, "val", polisher_root, angle_bins, args.seed, args.workers, args.overwrite)
    if not args.skip_test:
        print("test samples started")
        simulate_epoch(rates, alpha_vec, beta_vec, bucket_vec, cds_seqs_test, asite_test, geo_map_test, args.times, "test", polisher_root, angle_bins, args.seed, args.workers, args.overwrite)


if __name__ == "__main__":
    main()
