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
