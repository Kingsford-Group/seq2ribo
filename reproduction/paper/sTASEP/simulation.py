from util import *
import numpy as np
import random
from collections import Counter
import concurrent.futures
import multiprocessing

# initiation probability shared by all transcripts (the simulator input is sequence-only)
INIT_P = 0.01

EXCLUSION = 3


def simulate_once(codons, angle_dev_sum, pair_count, init_p, rate, alpha_vec, beta_vec,bucket_vec,ANGLE_BINS):
    def ahead_clear(start_idx: int, gap: int) -> bool:
        end = min(L, start_idx + gap)
        if start_idx >= L:
            return False
        return all(not occ[i] for i in range(start_idx, end))
    L = len(codons)
    occ = [False]*L
    ribos = []
    completed = 0
    eps = 1e-12
    times = 200
    ribos_general = []
    while times > 0:


        if completed > 10 and random.random() < 0.1:
            
            ribos_general.extend(ribos)
            times -= 1
            #ribos = []
        #initiate new ribosome
        if (random.random() < init_p and codons[0] not in STOP_CODONS and ahead_clear(0, EXCLUSION+1)):  
            ribos.append(0); occ[0] = True

        for idx in reversed(range(len(ribos))):
            pos = ribos[idx]
            
            #finish
            if codons[pos] in STOP_CODONS:
                occ[pos] = False; ribos.pop(idx); completed += 1; continue
            
            if pos == L-1:
                occ[pos] = False; ribos.pop(idx); completed += 1; continue

            #move
            nxt = pos + 1
            if codons[nxt] in STOP_CODONS:
                occ[pos] = False; ribos.pop(idx); completed += 1; continue

            k = CODON2IDX[codons[pos]]

            # angle bin index
            a = float(angle_dev_sum[pos])
            kk = np.digitize(a, ANGLE_BINS) - 1
            if kk < 0: kk = 0
            elif kk >= K_ANGLE_BINS: kk = K_ANGLE_BINS - 1

            # pair bin index (0..3)
            b = int(pair_count[pos])
            if b < 0: b = 0
            elif b > 3: b = 3

            #bucket index 
            if pos < L//3:
                bucket = 0
            elif pos < 2*L//3:
                bucket = 1
            else:
                bucket = 2

            #base_time = 1.0 / max(float(rate[k]), eps)
            base_time = rate[k]
            #print(rate)
            wait = base_time + float(alpha_vec[b]) + float(beta_vec[kk]) + float(bucket_vec[bucket])
            #wait = base_time
            eff_rate = 1.0 / max(wait, eps)

            if random.random() < eff_rate and ahead_clear(nxt, EXCLUSION):
                ribos[idx] = nxt
                occ[pos] = False
                occ[nxt] = True
    
    

    return Counter(ribos_general), completed

def simulate_many(codons, angle_dev_sum, pair_count, init_p, rate, alpha_vec, beta_vec, bucket_vec, n_runs, ANGLE_BINS):
    L = len(codons)
    acc = np.zeros(L, dtype=np.int32)
    completed_total = 0
    for _ in range(n_runs):
        snap, comp = simulate_once(codons, angle_dev_sum, pair_count, init_p, rate, alpha_vec, beta_vec, bucket_vec, ANGLE_BINS)
        completed_total += comp
        for p, c in snap.items():
            acc[p] += c
    return acc, completed_total


def simulate_transcript_parallel(args):
    """
    args: (tx, seq, a_cnts, angle_dev_sum, pair_count, rate, alpha_vec, beta_vec, bucket_vec, ANGLE_BINS)
    """
    (tx, seq, a_cnts, angle_dev_sum, pair_count, rate, alpha_vec, beta_vec, bucket_vec, ANGLE_BINS) = args

    init_p = INIT_P

    n_runs_tx = 1
    if n_runs_tx == 0:
        return None

    cods = [seq[i:i+3] for i in range(0, len(seq), 3)]
    sim_vec_raw, completed_total = simulate_many(
        cods, angle_dev_sum, pair_count, init_p, rate, alpha_vec, beta_vec, bucket_vec, n_runs_tx, ANGLE_BINS
    )
    obs_counts = np.array([a_cnts[i*3:(i+1)*3].sum() for i in range(len(cods))], dtype=np.float32)

    sim_sum = sim_vec_raw.sum()
    obs_sum = obs_counts.sum()
    scale = (obs_sum / sim_sum) if sim_sum > 0 else 0.0
    sim_vec_scaled = sim_vec_raw * scale

    return tx, cods, obs_counts, sim_vec_raw, sim_vec_scaled, scale, completed_total


def codon_loss(rate_fit, alpha_vec, beta_vec, bucket_vec, cds_seqs, asite, geo_map,ANGLE_BINS):
    # Build full RATE array and normalize to unit mean over NONSTOP
    full = build_full_rate(rate_fit)
    mu = np.mean(full[NONSTOP_IDX]) if np.any(full[NONSTOP_IDX]) else 1.0
    full[NONSTOP_IDX] = full[NONSTOP_IDX] / (mu + 1e-9)
    rate = full

    job_args = []
    for tx, (seq, a_cnts) in asite.items():
        if tx not in cds_seqs or tx not in geo_map: continue
        angle_dev_sum, pair_count = geo_map[tx]
        job_args.append((tx, seq, a_cnts, angle_dev_sum, pair_count, rate, alpha_vec, beta_vec,bucket_vec,ANGLE_BINS))

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

            kk = np.digitize(a, ANGLE_BINS) - 1
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

    #bucketMSE = float(np.mean((sim_bucket_per_tr - obs_bucket_per_tr)**2))
    #angleMSE = float(np.mean((sim_ang_per_tr - obs_ang_per_tr)**2))
    #pairMSE  = float(np.mean((sim_pair_per_tr - obs_pair_per_tr)**2))


    cod_mse = np.mean(np.abs(obs_cod_global - sim_cod_global_scaled))
    bucketMSE = float(np.mean(np.abs(bucket_obs - bucket_sim)))
    angleMSE = float(np.mean(np.abs(ang_obs - ang_sim)))
    pairMSE  = float(np.mean(np.abs(pair_obs - pair_sim)))
    return angleMSE, pairMSE, bucketMSE, cod_mse, per_transcript_mae, elemwise_mae