import argparse
import json
from datetime import datetime
from pathlib import Path

import concurrent.futures
import multiprocessing
import numpy as np

from load_data import load_data
from simulation import codon_loss, simulate_transcript_parallel
from util import CODON2IDX, CODONS, K_ANGLE_BINS, NONSTOP_IDX, STOP_IDX, build_full_rate


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


def fit_three_phase(
    angle_bins,
    params_init,
    cds_seqs,
    asite,
    geo_map,
    cds_seqs_test,
    asite_test,
    geo_map_test,
    cds_seqs_val,
    asite_val,
    geo_map_val,
    nonphased_iters=100,
    step_bucket=0.01,
    step_angle=0.01,
    step_pair=0.01,
    step_rate=0.01,
    normalize_by_occ=True,
):
    eps = 1e-9

    def simulate_epoch(rate_fit, alpha_vec, beta_vec, bucket_vec, cds_seqs, asite, geo_map, exclude_zero_completed: bool):
        full = build_full_rate(rate_fit)
        mu = np.mean(full[NONSTOP_IDX]) if np.any(full[NONSTOP_IDX]) else 1.0
        full[NONSTOP_IDX] = full[NONSTOP_IDX] / (mu + 1e-9)
        rate = full

        job_args = []
        for tx, (seq, a_cnts) in asite.items():
            if tx not in cds_seqs or tx not in geo_map:
                continue
            angle_dev_sum, pair_count = geo_map[tx]
            job_args.append((tx, seq, a_cnts, angle_dev_sum, pair_count, rate, alpha_vec, beta_vec, bucket_vec, angle_bins))

        obs_cod_global = np.zeros(len(CODONS), dtype=np.float64)
        sim_cod_global_raw = np.zeros(len(CODONS), dtype=np.float64)
        sim_cod_global_scaled = np.zeros(len(CODONS), dtype=np.float64)
        cod_occ_global = np.zeros(len(CODONS), dtype=np.int64)

        pair_obs = np.zeros(4, dtype=np.float64)
        pair_sim = np.zeros(4, dtype=np.float64)
        pair_occ = np.zeros(4, dtype=np.int64)
        ang_obs = np.zeros(K_ANGLE_BINS, dtype=np.float64)
        ang_sim = np.zeros(K_ANGLE_BINS, dtype=np.float64)
        ang_occ = np.zeros(K_ANGLE_BINS, dtype=np.int64)
        bucket_obs = np.zeros(3, dtype=np.float64)
        bucket_sim = np.zeros(3, dtype=np.float64)
        bucket_occ = np.zeros(3, dtype=np.int64)

        per_sample_losses = []
        total_abs_diff = 0.0
        total_positions = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as ex:
            results = list(ex.map(simulate_transcript_parallel, job_args))

        for res in results:
            if res is None:
                continue
            tx, cods, obs_counts, sim_vec_raw, sim_vec_scaled, _scale, completed_total = res
            if completed_total == 0 and exclude_zero_completed:
                continue

            absdiff = np.abs(sim_vec_scaled.astype(np.float64) - obs_counts.astype(np.float64))
            numer = float(np.sum(absdiff))
            ribo_loads = float(np.sum(obs_counts.astype(np.float64)))
            per_sample_losses.append(numer / ribo_loads if ribo_loads > 0 else 0.0)
            total_abs_diff += numer
            total_positions += absdiff.size

            angle_dev_sum, pair_count = geo_map[tx]
            for i, cod in enumerate(cods):
                k = CODON2IDX.get(cod, None)
                if k is None or k in STOP_IDX:
                    continue

                obs_cod_global[k] += obs_counts[i]
                sim_cod_global_raw[k] += sim_vec_raw[i]
                sim_cod_global_scaled[k] += sim_vec_scaled[i]
                cod_occ_global[k] += 1

                a = float(angle_dev_sum[i])
                p = int(pair_count[i])

                if 0 <= p <= 3:
                    pair_obs[p] += obs_counts[i]
                    pair_sim[p] += sim_vec_scaled[i]
                    pair_occ[p] += 1

                kk = np.digitize(a, angle_bins) - 1
                if kk < 0:
                    kk = 0
                elif kk >= K_ANGLE_BINS:
                    kk = K_ANGLE_BINS - 1
                ang_obs[kk] += obs_counts[i]
                ang_sim[kk] += sim_vec_scaled[i]
                ang_occ[kk] += 1

                if i < len(cods) // 3:
                    bucket_obs[0] += obs_counts[i]
                    bucket_occ[0] += 1
                    bucket_sim[0] += sim_vec_scaled[i]
                elif i < 2 * len(cods) // 3:
                    bucket_obs[1] += obs_counts[i]
                    bucket_occ[1] += 1
                    bucket_sim[1] += sim_vec_scaled[i]
                else:
                    bucket_obs[2] += obs_counts[i]
                    bucket_occ[2] += 1
                    bucket_sim[2] += sim_vec_scaled[i]

        per_transcript_mae = float(np.mean(per_sample_losses)) if per_sample_losses else float("nan")
        elemwise_mae = float(total_abs_diff / max(total_positions, 1))
        cod_mse = np.mean(np.abs(obs_cod_global - sim_cod_global_scaled))

        return (
            cod_mse,
            obs_cod_global,
            sim_cod_global_raw,
            sim_cod_global_scaled,
            cod_occ_global,
            pair_obs,
            pair_sim,
            pair_occ,
            ang_obs,
            ang_sim,
            ang_occ,
            bucket_obs,
            bucket_sim,
            bucket_occ,
            per_transcript_mae,
            elemwise_mae,
        )

    n_nonstop = len(NONSTOP_IDX)
    rate_init_fit = params_init[:n_nonstop].copy()
    alpha_vec = np.array(params_init[n_nonstop:n_nonstop + 4], dtype=np.float64)
    beta_vec = np.array(params_init[n_nonstop + 4:n_nonstop + 4 + K_ANGLE_BINS], dtype=np.float64)
    bucket_vec = np.array(params_init[n_nonstop + K_ANGLE_BINS + 4:n_nonstop + K_ANGLE_BINS + 7], dtype=np.float64)

    alpha_bias = 0.0
    beta_bias = 0.0
    bucket_bias = 0.0

    print("\n=== Fit ===")
    for it in range(1, nonphased_iters + 1):
        effective_alpha = alpha_vec + alpha_bias
        effective_beta = beta_vec + beta_bias
        effective_bucket = bucket_vec + bucket_bias

        tr = simulate_epoch(
            rate_init_fit,
            effective_alpha,
            effective_beta,
            effective_bucket,
            cds_seqs,
            asite,
            geo_map,
            exclude_zero_completed=True,
        )

        (
            train_cod_mse,
            obs_g,
            _sim_g,
            sim_g_scaled,
            occ_g,
            pair_obs,
            pair_sim,
            pair_occ,
            ang_obs,
            ang_sim,
            ang_occ,
            bucket_obs,
            bucket_sim,
            bucket_occ,
            train_per_tx_mae,
            train_elem_mae,
        ) = tr

        angle_mse_v, pair_mse_v, bucket_mse_v, codon_mse_v, per_tx_mae_v, elem_mae_v = codon_loss(
            rate_init_fit,
            effective_alpha,
            effective_beta,
            effective_bucket,
            cds_seqs_val,
            asite_val,
            geo_map_val,
            angle_bins,
        )
        angle_mse_t, pair_mse_t, bucket_mse_t, codon_mse_t, per_tx_mae_t, elem_mae_t = codon_loss(
            rate_init_fit,
            effective_alpha,
            effective_beta,
            effective_bucket,
            cds_seqs_test,
            asite_test,
            geo_map_test,
            angle_bins,
        )

        train_bucket = float(np.mean(np.abs(bucket_obs - bucket_sim)))
        train_angle = float(np.mean(np.abs(ang_obs - ang_sim)))
        train_pair = float(np.mean(np.abs(pair_obs - pair_sim)))

        print(
            f"[P1 iter {it:02d}] "
            f"codonMSE: train={train_cod_mse} | bucketMSE: train={train_bucket} | "
            f"angleMSE: train={train_angle} | pairMSE: train={train_pair} | "
            f"perTransMAE: train={train_per_tx_mae} | elemwiseMAE: train={train_elem_mae}"
        )
        print(
            f"[P1 iter {it:02d}] "
            f"codonMSE: val={codon_mse_v} | bucketMSE: val={bucket_mse_v} | "
            f"angleMSE: val={angle_mse_v} | pairMSE: val={pair_mse_v} | "
            f"perTransMAE: val={per_tx_mae_v} | elemwiseMAE: val={elem_mae_v}"
        )
        print(
            f"[P1 iter {it:02d}] "
            f"codonMSE: test={codon_mse_t} | bucketMSE: test={bucket_mse_t} | "
            f"angleMSE: test={angle_mse_t} | pairMSE: test={pair_mse_t} | "
            f"perTransMAE: test={per_tx_mae_t} | elemwiseMAE: test={elem_mae_t}"
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            if normalize_by_occ:
                bucket_diff_avg = (bucket_sim - bucket_obs) / (bucket_occ + 1.0)
                ang_diff_avg = (ang_sim - ang_obs) / (ang_occ + 1.0)
                pair_diff_avg = (pair_sim - pair_obs) / (pair_occ + 1.0)
            else:
                bucket_diff_avg = bucket_sim - bucket_obs
                ang_diff_avg = ang_sim - ang_obs
                pair_diff_avg = pair_sim - pair_obs

            bucket_mean = float(np.mean(bucket_diff_avg))
            ang_mean = float(np.mean(ang_diff_avg))
            pair_mean = float(np.mean(pair_diff_avg))

            bucket_vec = bucket_vec - step_bucket * (bucket_diff_avg - bucket_mean)
            alpha_vec = alpha_vec - step_pair * (pair_diff_avg - pair_mean)
            beta_vec = beta_vec - step_angle * (ang_diff_avg - ang_mean)

            bias_frac = 0.5
            bucket_bias = bucket_bias - step_bucket * (bias_frac * bucket_mean)
            alpha_bias = alpha_bias - step_pair * (bias_frac * pair_mean)
            beta_bias = beta_bias - step_angle * (bias_frac * ang_mean)

        raw_diff = sim_g_scaled - obs_g
        raw_diff[STOP_IDX] = 0.0
        scale = (1.0 / (occ_g + 1.0)) if normalize_by_occ else np.ones_like(raw_diff)
        delta_fit = step_rate * raw_diff[NONSTOP_IDX] * scale[NONSTOP_IDX]
        rate_init_fit = np.clip(rate_init_fit - delta_fit, 1e-6, None)

        print("rate head:", rate_init_fit[:8], "...")

    final_alpha = alpha_vec + alpha_bias
    final_beta = beta_vec + beta_bias
    final_bucket = bucket_vec + bucket_bias

    print("Fit is done")
    # fitted wait times of the 61 sense codons (the simulator rescales them to unit mean);
    # alpha indexes the pair count, beta the angle bin, bucket the CDS third
    return rate_init_fit.copy(), (final_alpha, final_beta, final_bucket)


def parse_args():
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Reproduction sTASEP fitter")
    parser.add_argument("--cell-line", choices=["ipsc", "hek293", "rpe", "lcl"], required=True)
    parser.add_argument("--subset", choices=["full", "01", "001", "0001"], default="full")
    parser.add_argument("--data-root", default=str(base_dir / "data"))
    parser.add_argument("--output-root", default=str(base_dir / "results"))
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--step-rate", type=float, default=0.01)
    parser.add_argument("--step-angle", type=float, default=0.01)
    parser.add_argument("--step-pair", type=float, default=0.01)
    parser.add_argument("--step-bucket", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_root) / args.cell_line
    out_dir = Path(args.output_root) / "sTASEP" / args.cell_line
    out_dir.mkdir(parents=True, exist_ok=True)

    (cds_seqs_train, asite_train, geo_map_train), (cds_seqs_val, asite_val, geo_map_val), (cds_seqs_test, asite_test, geo_map_test) = load_data(
        args.subset, str(data_dir)
    )
    print(f"Train: {len(cds_seqs_train)} transcripts, Val: {len(cds_seqs_val)}, Test: {len(cds_seqs_test)}")
    angle_bins = derive_angle_bins_from_training(geo_map_train, K_ANGLE_BINS, "linear", 0.0, 0.995)

    rate_init = np.ones(len(CODONS), dtype=np.float64) / 2.0
    rate_init_fit = rate_init[NONSTOP_IDX].copy()
    alpha_init_vec = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    beta_init_vec = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    bucket_init_vec = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    params_init = np.concatenate([rate_init_fit, alpha_init_vec, beta_init_vec, bucket_init_vec])

    initial_metrics = codon_loss(
        rate_init_fit,
        alpha_init_vec,
        beta_init_vec,
        bucket_init_vec,
        cds_seqs_test,
        asite_test,
        geo_map_test,
        angle_bins,
    )
    print("Initial test metrics (angle, pair, bucket, codon, per_tx_mae, elem_mae):", initial_metrics)

    rate_opt, (alpha_opt, beta_opt, bucket_opt) = fit_three_phase(
        angle_bins=angle_bins,
        params_init=params_init,
        cds_seqs=cds_seqs_train,
        asite=asite_train,
        geo_map=geo_map_train,
        cds_seqs_test=cds_seqs_test,
        asite_test=asite_test,
        geo_map_test=geo_map_test,
        cds_seqs_val=cds_seqs_val,
        asite_val=asite_val,
        geo_map_val=geo_map_val,
        nonphased_iters=args.iters,
        step_bucket=args.step_bucket,
        step_angle=args.step_angle,
        step_pair=args.step_pair,
        step_rate=args.step_rate,
    )

    final_metrics = codon_loss(
        rate_opt,
        alpha_opt,
        beta_opt,
        bucket_opt,
        cds_seqs_test,
        asite_test,
        geo_map_test,
        angle_bins,
    )
    print("Final test metrics (angle, pair, bucket, codon, per_tx_mae, elem_mae):", final_metrics)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.savez(
        out_dir / f"stasep_fit_{args.subset}_{stamp}.npz",
        rates=rate_opt,
        alpha_opt=alpha_opt,
        beta_opt=beta_opt,
        bucket_opt=bucket_opt,
        angle_bins=angle_bins,
    )
    with (out_dir / f"stasep_fit_{args.subset}_{stamp}.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "sTASEP",
                "cell_line": args.cell_line,
                "subset": args.subset,
                "iters": args.iters,
                "step_rate": args.step_rate,
                "step_angle": args.step_angle,
                "step_pair": args.step_pair,
                "step_bucket": args.step_bucket,
                "data_dir": str(data_dir),
                # same layout as seq2ribo/constants.py
                "params": {
                    "rates": [float(x) for x in rate_opt],
                    "alpha": [float(x) for x in alpha_opt],
                    "beta": [float(x) for x in beta_opt],
                    "buckets": [float(x) for x in bucket_opt],
                },
                "initial_test_metrics": {
                    "angle": float(initial_metrics[0]),
                    "pair": float(initial_metrics[1]),
                    "bucket": float(initial_metrics[2]),
                    "codon": float(initial_metrics[3]),
                    "per_transcript_mae": float(initial_metrics[4]),
                    "elemwise_mae": float(initial_metrics[5]),
                },
                "final_test_metrics": {
                    "angle": float(final_metrics[0]),
                    "pair": float(final_metrics[1]),
                    "bucket": float(final_metrics[2]),
                    "codon": float(final_metrics[3]),
                    "per_transcript_mae": float(final_metrics[4]),
                    "elemwise_mae": float(final_metrics[5]),
                },
                "timestamp": stamp,
            },
            f,
            indent=2,
        )
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
