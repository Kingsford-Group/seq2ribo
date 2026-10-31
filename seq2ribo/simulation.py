"""sTASEP / TASEP simulation of ribosome traffic along a CDS.

Discrete-time exclusion process on the codon lattice. Each step:
  * a snapshot of all ribosome positions is taken with probability SNAPSHOT_P once more than
    BURN_IN ribosomes have terminated (a run ends after SNAPSHOTS snapshots);
  * initiation: a ribosome enters codon 0 with probability init_p * dt if codons 0..EXCLUSION are free;
  * elongation, ribosomes visited 3'->5': a ribosome on the last codon, or whose next codon is a stop
    codon, terminates; otherwise it moves from codon j to j+1 with probability dt / w_j if codons j+1..j+EXCLUSION are free.
Wait time of codon j:
    w_j = tau[codon_j] + alpha[pair_j] + beta[angle_j] + gamma[bucket_j]   (sTASEP)
    w_j = tau[codon_j]                                                     (TASEP: alpha = beta = gamma = 0)
with the 61 sense-codon wait times tau rescaled to mean 1. The time step dt = min(1, min_j w_j) keeps
every move probability dt / w_j <= 1 (dt = 1 whenever all waits are >= 1).
A run returns, for every codon, the number of snapshots in which it held a ribosome (A-site counts);
if no ribosome can initiate (stop codon first, or init_p = 0) the profile is all zero.
"""
import random

import numpy as np

from .utils import CODON2IDX, NONSTOP_IDX, STOP_CODONS

EXCLUSION = 3
BURN_IN = 10
SNAPSHOT_P = 0.1
SNAPSHOTS = 200


def full_wait_times(tau_sense):
    """64-codon wait-time array from the 61 sense-codon wait times, rescaled to mean 1."""
    tau = np.ones(64, dtype=np.float64)
    tau[NONSTOP_IDX] = np.asarray(tau_sense, dtype=np.float64)
    tau[NONSTOP_IDX] /= tau[NONSTOP_IDX].mean()
    return tau


def bucket_index(L):
    """Position bucket of every codon: first, middle and last third of the CDS."""
    b = np.zeros(L, dtype=np.int64)
    b[L // 3:(2 * L) // 3] = 1
    b[(2 * L) // 3:] = 2
    return b


def angle_bin_index(angle_sum, angle_edges):
    k = len(angle_edges) - 1
    return np.clip(np.digitize(np.asarray(angle_sum, dtype=np.float64), angle_edges) - 1, 0, k - 1)


def codon_waits(codons, angle_bin, pair_count, tau_full, alpha, beta, gamma):
    """Wait time of every codon position (+inf for stop codons, which are never entered)."""
    L = len(codons)
    pair = np.clip(np.asarray(pair_count, dtype=np.int64), 0, 3)
    bk = bucket_index(L)
    w = np.empty(L, dtype=np.float64)
    for j, c in enumerate(codons):
        if c in STOP_CODONS:
            w[j] = np.inf
        else:
            w[j] = tau_full[CODON2IDX[c]] + alpha[pair[j]] + beta[angle_bin[j]] + gamma[bk[j]]
    return np.maximum(w, 1e-12)


def simulate_run(codons, waits, init_p, rng=random):
    """One run; returns A-site counts (int64 array, one entry per codon)."""
    L = len(codons)
    finite = waits[np.isfinite(waits)]
    dt = min(1.0, float(finite.min())) if finite.size else 1.0
    p_move = np.where(np.isfinite(waits), dt / waits, 0.0).tolist()
    terminates = [(j + 1 >= L) or (codons[j + 1] in STOP_CODONS) for j in range(L)]
    can_initiate = codons[0] not in STOP_CODONS
    p_init = init_p * dt
    if not can_initiate or p_init <= 0:   # no ribosome can enter: empty profile
        return np.zeros(L, dtype=np.int64)
    occ = [False] * L
    ribos = []
    counts = [0] * L
    completed = snaps = 0
    while snaps < SNAPSHOTS:
        if completed > BURN_IN and rng.random() < SNAPSHOT_P:
            for pos in ribos:
                counts[pos] += 1
            snaps += 1
        if can_initiate and rng.random() < p_init and not any(occ[:EXCLUSION + 1]):
            ribos.append(0)
            occ[0] = True
        for idx in range(len(ribos) - 1, -1, -1):
            pos = ribos[idx]
            if terminates[pos]:
                occ[pos] = False
                ribos.pop(idx)
                completed += 1
                continue
            if rng.random() < p_move[pos]:
                nxt = pos + 1
                if not any(occ[nxt:nxt + EXCLUSION]):
                    occ[pos] = False
                    occ[nxt] = True
                    ribos[idx] = nxt
    return np.asarray(counts, dtype=np.int64)


def simulate_runs(seq, angle_sum, pair_count, angle_edges, tau_sense, alpha, beta, gamma,
                  n_runs, init_p, seed=None):
    """n_runs independent runs of one CDS -> (codons, counts of shape (n_runs, n_codons))."""
    codons = [seq[i:i + 3] for i in range(0, len(seq), 3)]
    waits = codon_waits(codons, angle_bin_index(angle_sum, angle_edges), pair_count,
                        full_wait_times(tau_sense), np.asarray(alpha, float), np.asarray(beta, float),
                        np.asarray(gamma, float))
    rng = random.Random(seed) if seed is not None else random
    return codons, np.stack([simulate_run(codons, waits, init_p, rng) for _ in range(n_runs)])
