#!/usr/bin/env python
"""Fit sTASEP (codon, pair, angle and bucket wait times) or TASEP (codon wait times only).

Every outer iteration simulates each training transcript once with the current wait times,
rescales the simulated counts of each transcript to its observed total over sense codons (the
simulator never occupies the stop codon), and aggregates scaled
simulated and observed counts by codon, pair count, angle bin and position bucket. With
Delta_g = (sum of scaled simulated - sum of observed) / (number of positions in group g):
  tau_c   <- max(tau_c - eta * Delta_c, eps)                      (61 sense codons)
  alpha_r <- alpha_r - eta * (Delta_r - mean_r Delta_r)           (pair counts 0..3)
  bias    <- bias - eta * lambda * mean_r Delta_r;  alpha_r <- alpha_r + bias   (bias folded each iteration)
and the same centred update for beta (angle bins) and gamma (buckets). TASEP keeps alpha = beta = gamma = 0.
All wait times start at FIT_INIT. Train and validation metrics are recorded every iteration,
test metrics before and after fitting. Output: results/fit/<model>/<cell>/{params.json, history.json}.
"""
import argparse
import json
import pickle
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from seq2ribo import simulation as S  # noqa: E402
from seq2ribo.metrics import profile_metrics  # noqa: E402
from seq2ribo.utils import CODON2IDX, NONSTOP_IDX  # noqa: E402

DATA = {}  # split -> list of prepared transcripts (filled before the worker pool forks)


def prepare(records, edges):
    out = []
    for tx, seq, counts, ang, pair in records:
        L = len(seq) // 3
        codons = [seq[3 * j:3 * j + 3] for j in range(L)]
        out.append({
            "tx": tx, "codons": codons,
            "obs": np.add.reduceat(counts[:3 * L].astype(np.float64), np.arange(0, 3 * L, 3)),
            "cod": np.array([CODON2IDX[c] for c in codons], dtype=np.int64),
            "pair": np.clip(np.asarray(pair, np.int64), 0, 3),
            "angle": S.angle_bin_index(ang, edges),
            "bucket": S.bucket_index(L),
            "sense": np.array([c not in S.STOP_CODONS for c in codons]),
        })
    return out


def simulate_one(job):
    split, i, tau_full, alpha, beta, gamma, seed = job
    r = DATA[split][i]
    waits = S.codon_waits(r["codons"], r["angle"], r["pair"], tau_full, alpha, beta, gamma)
    import random
    return S.simulate_run(r["codons"], waits, config.INIT_P, random.Random(seed)).astype(np.float64)


def run_split(pool, split, params, it, seed):
    tau_full = S.full_wait_times(params["tau"])
    jobs = [(split, i, tau_full, params["alpha"], params["beta"], params["gamma"],
             zlib.crc32(f"{seed}:{split}:{it}:{r['tx']}".encode())) for i, r in enumerate(DATA[split])]
    return list(pool.map(simulate_one, jobs, chunksize=16))


def group_deltas(split, sims):
    """Delta per group from per-transcript scaled simulated vs observed counts."""
    agg = {k: [np.zeros(n), np.zeros(n), np.zeros(n)] for k, n in (("cod", 64), ("pair", 4), ("angle", config.ANGLE_BINS_K), ("bucket", 3))}
    for r, sim in zip(DATA[split], sims):
        # the simulator never places a ribosome on a stop codon: compare sense codons only
        m = r["sense"]
        obs, sim = r["obs"][m], sim[m]
        s = sim.sum()
        sim_sc = sim * (obs.sum() / s) if s > 0 else np.zeros_like(sim)
        for k in agg:
            idx, n = r[k][m], len(agg[k][0])
            agg[k][0] += np.bincount(idx, weights=sim_sc, minlength=n)
            agg[k][1] += np.bincount(idx, weights=obs, minlength=n)
            agg[k][2] += np.bincount(idx, minlength=n)
    return {k: (v[0] - v[1]) / np.maximum(v[2], 1) for k, v in agg.items()}


def metrics(split, sims, scaled_only):
    d = DATA[split]
    return profile_metrics([r["obs"] for r in d], sims, [r["cod"] for r in d], [r["pair"] for r in d],
                           [r["angle"] for r in d], config.ANGLE_BINS_K, scaled_only=scaled_only)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["stasep", "tasep"])
    p.add_argument("--cell-line", required=True, choices=config.CELL_LINES)
    p.add_argument("--data-dir", type=Path, default=config.DATA)
    p.add_argument("--subset", default="", choices=["", "_01"], help="'_01' = 10%% subsets for quick runs")
    p.add_argument("--iters", type=int, default=config.FIT_ITERS)
    p.add_argument("--lr", type=float, default=config.FIT_LR)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main():
    a = parse_args()
    out = a.out_dir or config.RESULTS / "fit" / a.model / a.cell_line
    out.mkdir(parents=True, exist_ok=True)
    edges = np.asarray(json.load(open(a.data_dir / "angle_bins.json"))["edges"])
    for split in ("train", "val", "test"):
        with open(a.data_dir / a.cell_line / f"{split}{a.subset}.pkl", "rb") as f:
            DATA[split] = prepare(pickle.load(f), edges)
    print(f"[{a.model} {a.cell_line}] train {len(DATA['train'])}  val {len(DATA['val'])}  test {len(DATA['test'])}", flush=True)

    structural = a.model == "stasep"
    zeros = lambda n: np.zeros(n)
    params = {"tau": np.full(61, config.FIT_INIT),
              "alpha": np.full(4, config.FIT_INIT) if structural else zeros(4),
              "beta": np.full(config.ANGLE_BINS_K, config.FIT_INIT) if structural else zeros(config.ANGLE_BINS_K),
              "gamma": np.full(3, config.FIT_INIT) if structural else zeros(3)}
    history = []
    with ProcessPoolExecutor(a.workers) as pool:
        test_initial = metrics("test", run_split(pool, "test", params, -1, a.seed), scaled_only=False)
        for it in range(1, a.iters + 1):
            t0 = time.time()
            sims = run_split(pool, "train", params, it, a.seed)
            train_m = metrics("train", sims, scaled_only=True)
            val_m = metrics("val", run_split(pool, "val", params, it, a.seed), scaled_only=True)
            d = group_deltas("train", sims)
            params["tau"] = np.maximum(params["tau"] - a.lr * d["cod"][NONSTOP_IDX], config.WAIT_EPS)
            if structural:
                for name, key in (("alpha", "pair"), ("beta", "angle"), ("gamma", "bucket")):
                    mean = d[key].mean()
                    bias = -a.lr * config.FIT_BIAS_LAMBDA * mean
                    params[name] = params[name] - a.lr * (d[key] - mean) + bias
            history.append({"iter": it, "train": train_m, "val": val_m, "seconds": time.time() - t0})
            print(f"[it {it:03d}] train sc_codonMAE {train_m['sc_codonMAE']:.1f} sc_ptxMAE {train_m['sc_pertranscriptMAE']:.4f} "
                  f"| val sc_codonMAE {val_m['sc_codonMAE']:.1f} sc_ptxMAE {val_m['sc_pertranscriptMAE']:.4f} "
                  f"| {time.time() - t0:.0f}s", flush=True)
            with open(out / "history.json", "w") as f:
                json.dump(history, f)
        test_final = metrics("test", run_split(pool, "test", params, a.iters + 1, a.seed), scaled_only=False)
    with open(out / "params.json", "w") as f:
        json.dump({"model": a.model, "cell_line": a.cell_line, "iters": a.iters, "lr": a.lr, "seed": a.seed,
                   "init_p": config.INIT_P, "angle_edges": edges.tolist(),
                   # same layout as seq2ribo/constants.py: alpha = pair counts, beta = angle bins, gamma = buckets
                   "params": {"rates": params["tau"].tolist(), "alpha": params["alpha"].tolist(),
                              "beta": params["beta"].tolist(), "buckets": params["gamma"].tolist()},
                   "test_initial": test_initial, "test_final": test_final}, f, indent=1)
    print(f"test before: {test_initial}\ntest after: {test_final}\nwrote {out}")


if __name__ == "__main__":
    main()
