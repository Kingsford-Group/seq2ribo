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
    p95  = float(np.quantile(vals, 0.95))
    p99  = float(np.quantile(vals, 0.99))

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
    cds_seqs, asite, geo_map,
    cds_seqs_val, asite_val, geo_map_val,
    cds_seqs_test, asite_test, geo_map_test,
    phase4_iters=100,
    step_rate=0.01,
    normalize_by_occ=True,
):
    def simulate_epoch(rate_fit,  cds_seqs, asite, geo_map, exclude_zero_completed: bool):
        # Build full RATE array and normalize to unit mean over NONSTOP
        full = build_full_rate(rate_fit)
        mu = np.mean(full[NONSTOP_IDX]) if np.any(full[NONSTOP_IDX]) else 1.0
        full[NONSTOP_IDX] = full[NONSTOP_IDX] / (mu + 1e-9)
        rate = full

        job_args = []
        for tx, (seq, a_cnts) in asite.items():
            if tx not in cds_seqs or tx not in geo_map: continue
            angle_dev_sum, pair_count = geo_map[tx]
            job_args.append((tx, seq, a_cnts, angle_dev_sum, pair_count, rate, angle_bins))

        obs_cod_global = np.zeros(len(CODONS), dtype=np.float64)
        sim_cod_global_raw = np.zeros(len(CODONS), dtype=np.float64)
        sim_cod_global_scaled = np.zeros(len(CODONS), dtype=np.float64)
        cod_occ_global = np.zeros(len(CODONS), dtype=np.int64)

        pair_obs = np.zeros(4, dtype=np.float64); pair_sim = np.zeros(4, dtype=np.float64); pair_occ = np.zeros(4, dtype=np.int64)
        ang_obs  = np.zeros(K_ANGLE_BINS, dtype=np.float64); ang_sim  = np.zeros(K_ANGLE_BINS, dtype=np.float64); ang_occ = np.zeros(K_ANGLE_BINS, dtype=np.int64)
        bucket_obs = np.zeros(3, dtype=np.float64); bucket_sim = np.zeros(3, dtype=np.float64); bucket_occ = np.zeros(3, dtype=np.int64)

        sse_generic = 0.0
        n_generic   = 0

        zeros = []; used = 0

        # NEW accumulators for the two MAE-style losses
        per_sample_losses = []
        total_abs_diff = 0.0
        total_positions = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as ex:
            results = list(ex.map(simulate_transcript_parallel, job_args))

        for res in results:
            if res is None: continue
            tx, cods, obs_counts, sim_vec_raw, sim_vec_scaled, _, completed_total = res
            if completed_total == 0:
                zeros.append(tx)
                if exclude_zero_completed: continue
            used += 1

            # accumulate for existing generic MSE (kept for backward-compatibility)
            diff = obs_counts.astype(np.float64) - sim_vec_scaled.astype(np.float64)
            sse_generic += float(np.dot(diff, diff))
            n_generic   += diff.size

            # ---- NEW: per-transcript normalized MAE and elementwise MAE accumulators ----
            absdiff = np.abs(sim_vec_scaled.astype(np.float64) - obs_counts.astype(np.float64))
            numer = float(np.sum(absdiff))
            ribo_loads = float(np.sum(obs_counts.astype(np.float64)))
            if ribo_loads > 0:
                per_sample_loss = numer / ribo_loads
            else:
                per_sample_loss = 0.0
            per_sample_losses.append(per_sample_loss)

            total_abs_diff += numer
            total_positions += absdiff.size

            # previous global codon/pair/angle/bucket accumulations
            angle_dev_sum, pair_count = geo_map[tx]
            for i, cod in enumerate(cods):
                k = CODON2IDX.get(cod, None)
                if k is None or k in STOP_IDX: continue

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
                if kk < 0: kk = 0
                elif kk >= K_ANGLE_BINS: kk = K_ANGLE_BINS - 1
                ang_obs[kk] += obs_counts[i]
                ang_sim[kk] += sim_vec_scaled[i]
                ang_occ[kk] += 1

                #bucket
                if i < len(cods)//3:
                    bucket_obs[0] += obs_counts[i]
                    bucket_occ[0] += 1
                    bucket_sim[0] += sim_vec_scaled[i]
                elif i < 2*len(cods)//3:
                    bucket_obs[1] += obs_counts[i]
                    bucket_occ[1] += 1
                    bucket_sim[1] += sim_vec_scaled[i]
                else:
                    bucket_obs[2] += obs_counts[i]
                    bucket_occ[2] += 1
                    bucket_sim[2] += sim_vec_scaled[i]

        # per-codon reference loss (MSE over codon IDs)
        eps = 1e-9
        valid = (cod_occ_global > 0)
        for si in STOP_IDX: valid[si] = False
        obs_avg = np.zeros_like(obs_cod_global)
        sim_avg = np.zeros_like(sim_cod_global_scaled)
        obs_avg[valid] = obs_cod_global[valid] / (cod_occ_global[valid] + eps)
        sim_avg[valid] = sim_cod_global_scaled[valid] / (cod_occ_global[valid] + eps)
        cod_mse = np.mean((obs_avg[valid] - sim_avg[valid])**2) if np.any(valid) else np.inf

        # NEW: generic MSE across positions (kept for reference; still computed)
        generic_mse = (sse_generic / max(n_generic, 1))

        # ---- NEW: the two MAE-style losses ----
        if per_sample_losses:
            per_transcript_mae = float(np.mean(per_sample_losses))
        else:
            per_transcript_mae = float("nan")

        elemwise_mae = float(total_abs_diff / max(total_positions, 1))
        cod_mse = np.mean(np.abs(obs_cod_global - sim_cod_global_scaled))

        return (cod_mse, obs_cod_global, sim_cod_global_raw, sim_cod_global_scaled, cod_occ_global,
                pair_obs, pair_sim, pair_occ,
                ang_obs, ang_sim, ang_occ, bucket_obs, bucket_sim, bucket_occ,
                per_transcript_mae, elemwise_mae)



    # ==== PARAM INIT ====
    n_nonstop = len(NONSTOP_IDX) 
    rate_init_fit = params_init[:n_nonstop].copy()  

    # ------------------ PHASE 4: learn codon rates ------------------
    print("\n=== Fit codon rates  ===")
    #rate_fit = np.ones(len(NONSTOP_IDX), dtype=np.float64)
    #best_rate_fit = rate_fit.copy(); best_val_mse = float("inf")

    for it in range(1, phase4_iters + 1):
        if it % 50 == 0: step_rate *= 0.1

        # TRAIN (drop zero-completion)
        tr = simulate_epoch(
        rate_init_fit,
        cds_seqs, asite, geo_map,
        exclude_zero_completed=True)

        (train_codMSE, obsG, simG, simG_scaled, occG,
        pair_obs, pair_sim, pair_occ,
        ang_obs, ang_sim, ang_occ, bucket_obs, bucket_sim, bucket_occ,
        train_perTrans_MAE, train_elemwise_MAE) = tr






        eps = 1e-9
        with np.errstate(divide='ignore', invalid='ignore'):
            # per/occ for angle (train/val)
            obs_ang_per_tr = np.divide(ang_obs,     ang_occ + eps)
            sim_ang_per_tr = np.divide(ang_sim,     ang_occ + eps)


            # per/occ for pair (train/val)
            obs_pair_per_tr = np.divide(pair_obs,     pair_occ + eps)
            sim_pair_per_tr = np.divide(pair_sim,     pair_occ + eps)

            #bucket 
            obs_bucket_per_tr = np.divide(bucket_obs, bucket_occ + eps)
            sim_bucket_per_tr = np.divide(bucket_sim, bucket_occ + eps)
            obs_bucket_per_tr = np.where(np.isfinite(obs_bucket_per_tr), obs_bucket_per_tr, 0.0)
            sim_bucket_per_tr = np.where(np.isfinite(sim_bucket_per_tr), sim_bucket_per_tr, 0.0)


            # clean NaNs/Infs
            for arr in (obs_ang_per_tr, sim_ang_per_tr,
                        obs_pair_per_tr, sim_pair_per_tr):
                arr[...] = np.where(np.isfinite(arr), arr, 0.0)

        # losses
        train_bucketMSE = float(np.mean((sim_bucket_per_tr - obs_bucket_per_tr)**2))


        train_angleMSE = float(np.mean((sim_ang_per_tr - obs_ang_per_tr)**2))

        train_pairMSE  = float(np.mean((sim_pair_per_tr - obs_pair_per_tr)**2))

        # per-codon obs/sim per-occ (global, optional prints you already had)
        obs_per_occ = np.divide(obsG, np.where(occG>0, occG, 1), where=occG>0)
        sim_per_occ = np.divide(simG_scaled, np.where(occG>0, occG, 1), where=occG>0)
        
        angleMSE_v, pairMSE_v, bucketMSE_v, codon_mse_v, per_transcript_mae_v, elemwise_mae_v = codon_loss(
            rate_init_fit, cds_seqs_val, asite_val, geo_map_val, angle_bins
        )
        angleMSE_t, pairMSE_t, bucketMSE_t, codon_mse_t, per_transcript_mae_t, elemwise_mae_t = codon_loss(
            rate_init_fit, cds_seqs_test, asite_test, geo_map_test, angle_bins
        )

        train_bucketMSE = float(np.mean(np.abs(bucket_obs - bucket_sim)))
        train_angleMSE = float(np.mean(np.abs(ang_obs - ang_sim)))
        train_pairMSE  = float(np.mean(np.abs(pair_obs - pair_sim)))
        print(
            f"[P1 iter {it:02d}] "
            f"codonMSE: train={train_codMSE}| "
            f"bucketMSE: train={train_bucketMSE} | "
            f"angleMSE: train={train_angleMSE} | "
            f"pairMSE: train={train_pairMSE} | "
            f"perTransMAE: train={train_perTrans_MAE} | "
            f"elemwiseMAE: train={train_elemwise_MAE} | "
        )
        print(
            f"[P1 iter {it:02d}] "
            f"codonMSE: val={codon_mse_v}| "
            f"bucketMSE: val={bucketMSE_v} | "
            f"angleMSE: val={angleMSE_v} | "
            f"pairMSE: val={pairMSE_v} | "
            f"perTransMAE: val={per_transcript_mae_v} | "
            f"elemwiseMAE: val={elemwise_mae_v} | "
        )
        print(
            f"[P1 iter {it:02d}] "
            f"codonMSE: test={codon_mse_t}| "
            f"bucketMSE: test={bucketMSE_t} | "
            f"angleMSE: test={angleMSE_t} | "
            f"pairMSE: test={pairMSE_t} | "
            f"perTransMAE: test={per_transcript_mae_t} | "
            f"elemwiseMAE: test={elemwise_mae_t} | "
        )
        #print("  obs  :", np.round(obs_per_occ, 3))
        #print("  sim  :", np.round(sim_per_occ, 3))
        # residual-driven update (sim>obs ⇒ decrease rate)
        raw_diff = (simG_scaled - obsG)  # sim-obs
        raw_diff[STOP_IDX] = 0.0
        scale = (1.0 / (occG + 1.0)) if normalize_by_occ else np.ones_like(raw_diff)
        delta_fit = step_rate * raw_diff[NONSTOP_IDX] * scale[NONSTOP_IDX]

        rate_init_fit = np.clip(rate_init_fit - delta_fit, 1e-6, None)
        #rate_init_fit = np.clip(rate_init_fit + delta_fit, 1e-6, None)
        print("rate head:", rate_init_fit[:8], "...")
        #idx_sorted = np.argsort(-np.abs(raw_diff[NONSTOP_IDX]))[:10]
        #offenders = [(CODONS[NONSTOP_IDX[i]], raw_diff[NONSTOP_IDX[i]], occG[NONSTOP_IDX[i]]) for i in idx_sorted]
        #print("  top diffs (sim-obs, n):", ", ".join([f"{c}:{d:.0f} (n={n})" for c,d,n in offenders]))

        # selection: keep best by validation codonMSE (same as before)
        #if val_codMSE < best_val_mse:
        #    best_val_mse = val_codMSE
        #    best_rate_fit = rate_fit.copy()
    print("Fit is done")
    # build full arrays to return
    # fitted wait times of the 61 sense codons; the simulator rescales them to unit mean
    return rate_init_fit.copy()


def parse_args():
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Reproduction TASEP fitter")
    parser.add_argument("--cell-line", choices=["ipsc", "hek293", "rpe", "lcl"], required=True)
    parser.add_argument("--subset", choices=["full", "01", "001", "0001"], default="full")
    parser.add_argument("--data-root", default=str(base_dir / "data"))
    parser.add_argument("--output-root", default=str(base_dir / "results"))
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--step-rate", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_root) / args.cell_line
    out_dir = Path(args.output_root) / "TASEP" / args.cell_line
    out_dir.mkdir(parents=True, exist_ok=True)

    (cds_seqs_train, asite_train, geo_map_train), (cds_seqs_val, asite_val, geo_map_val), (cds_seqs_test, asite_test, geo_map_test) = load_data(
        args.subset, str(data_dir)
    )
    print(f"Train: {len(cds_seqs_train)} transcripts, Val: {len(cds_seqs_val)}, Test: {len(cds_seqs_test)}")
    angle_bins = derive_angle_bins_from_training(geo_map_train, K_ANGLE_BINS, "linear", 0.0, 0.995)

    rate_init = np.ones(len(CODONS), dtype=np.float64) / 2.0
    rate_init_fit = rate_init[NONSTOP_IDX].copy()
    params_init = np.concatenate([rate_init_fit])

    initial_metrics = codon_loss(rate_init_fit, cds_seqs_test, asite_test, geo_map_test, angle_bins)
    print("Initial test metrics (angle, pair, bucket, codon, per_tx_mae, elem_mae):", initial_metrics)

    rate_opt = fit_three_phase(
        angle_bins=angle_bins,
        params_init=params_init,
        cds_seqs=cds_seqs_train,
        asite=asite_train,
        geo_map=geo_map_train,
        cds_seqs_val=cds_seqs_val,
        asite_val=asite_val,
        geo_map_val=geo_map_val,
        cds_seqs_test=cds_seqs_test,
        asite_test=asite_test,
        geo_map_test=geo_map_test,
        phase4_iters=args.iters,
        step_rate=args.step_rate,
    )
    final_metrics = codon_loss(rate_opt, cds_seqs_test, asite_test, geo_map_test, angle_bins)
    print("Final test metrics (angle, pair, bucket, codon, per_tx_mae, elem_mae):", final_metrics)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.savez(
        out_dir / f"tasep_fit_{args.subset}_{stamp}.npz",
        rates=rate_opt,
        angle_bins=angle_bins,
    )
    with (out_dir / f"tasep_fit_{args.subset}_{stamp}.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "TASEP",
                "cell_line": args.cell_line,
                "subset": args.subset,
                "iters": args.iters,
                "step_rate": args.step_rate,
                "data_dir": str(data_dir),
                # same layout as seq2ribo/constants.py; TASEP has no structural terms
                "params": {
                    "rates": [float(x) for x in rate_opt],
                    "alpha": [0.0] * 4,
                    "beta": [0.0] * K_ANGLE_BINS,
                    "buckets": [0.0] * 3,
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
