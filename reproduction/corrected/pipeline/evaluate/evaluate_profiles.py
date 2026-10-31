#!/usr/bin/env python
"""Test-set metrics of ribosome profiles: a simulator (from its runs) and/or a polisher.

For every test transcript, each simulation run is polished separately; predictions are averaged
over runs per transcript and scored with seq2ribo.metrics.profile_metrics. With --polisher omitted
only the simulator's runs (averaged) are scored. Optional:
  --bootstrap B     percentile 95% CIs from B resamples of transcripts (runs of a transcript stay together)
  --run-variance    metrics of each single run, and of the mean of n random runs for n in 1,2,4,...,K
Output: JSON with "simulator", optionally "polisher", "*_bootstrap", "*_run_variance".
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from common import Shards, angle_edges, codon_counts, features, load_records  # noqa: E402
from seq2ribo.metrics import profile_metrics  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell-line", required=True, choices=config.CELL_LINES)
    p.add_argument("--shards", type=Path, required=True, help="directory containing test.npy/test.json")
    p.add_argument("--split", default="test")
    p.add_argument("--polisher", type=Path, default=None, help="polisher.pt (omit to score the simulator only)")
    p.add_argument("--inputs", nargs="+", default=["sequence", "simulation", "structure"])
    p.add_argument("--data-dir", type=Path, default=config.DATA)
    p.add_argument("--runs", type=int, default=None, help="use the first n runs (default: all)")
    p.add_argument("--bootstrap", type=int, default=0)
    p.add_argument("--run-variance", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


@torch.no_grad()
def polish(model, feats, sims, device, batch_size):
    cod, ab, pb, bk = (torch.as_tensor(x) for x in feats)
    out = []
    for s in range(0, len(sims), batch_size):
        sim = torch.as_tensor(np.log1p(sims[s:s + batch_size]), dtype=torch.float32, device=device)
        B, L = sim.shape
        rep = lambda t: t.unsqueeze(0).expand(B, -1).to(device)
        out.append(model(rep(cod), sim, torch.ones(B, L, dtype=torch.bool, device=device),
                         rep(ab), rep(pb), rep(bk)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def score(entries, runs_pred, idx=None, run_sel=None):
    """Metrics of the run-averaged predictions over transcripts `idx` and runs `run_sel`."""
    idx = range(len(entries)) if idx is None else idx
    obs, pred, cod, pair, ang = [], [], [], [], []
    for i in idx:
        e = entries[i]
        r = runs_pred[i] if run_sel is None else runs_pred[i][run_sel]
        obs.append(e["obs"]); pred.append(r.mean(axis=0)); cod.append(e["cod"]); pair.append(e["pair"]); ang.append(e["angle"])
    return profile_metrics(obs, pred, cod, pair, ang, k_angle=config.ANGLE_BINS_K)


def bootstrap(entries, runs_pred, B, seed=42):
    rng = np.random.RandomState(seed)
    point = score(entries, runs_pred)
    names = [k for k, v in point.items() if isinstance(v, float)]
    vals = {k: [] for k in names}
    n = len(entries)
    for _ in range(B):
        m = score(entries, runs_pred, rng.randint(0, n, size=n))
        for k in names:
            vals[k].append(m[k])
    return {k: {"point": point[k], "ci_lo": float(np.nanpercentile(vals[k], 2.5)),
                "ci_hi": float(np.nanpercentile(vals[k], 97.5))} for k in names}


def run_variance(entries, runs_pred, K, repeats=20, seed=42):
    single = [score(entries, runs_pred, run_sel=[k]) for k in range(K)]
    names = [k for k, v in single[0].items() if isinstance(v, float)]
    out = {"single_run": {k: {"mean": float(np.nanmean([m[k] for m in single])),
                              "std": float(np.nanstd([m[k] for m in single]))} for k in names}, "subsets": {}}
    rng = np.random.RandomState(seed)
    n = 1
    while n <= K:
        ms = [score(entries, runs_pred, run_sel=sorted(rng.choice(K, n, replace=False))) for _ in range(repeats if n < K else 1)]
        out["subsets"][str(n)] = {k: {"mean": float(np.nanmean([m[k] for m in ms])),
                                      "std": float(np.nanstd([m[k] for m in ms]))} for k in names}
        n *= 2
    return out


def main():
    a = parse_args()
    edges = angle_edges(a.data_dir)
    shards = Shards(a.shards / a.split)
    K = a.runs or shards.runs
    recs = [r for r in load_records(a.cell_line, a.split, a.data_dir) if r[0] in shards.index]
    entries, sim_runs = [], []
    for tx, seq, counts, ang, pair in recs:
        cod, ab, pb, bk = features(seq, ang, pair, edges)
        entries.append({"tx": tx, "obs": codon_counts(counts), "cod": cod, "pair": pb, "angle": ab, "feats": (cod, ab, pb, bk)})
        sim_runs.append(shards.all_runs(tx)[:K].astype(np.float64))
    res = {"cell_line": a.cell_line, "split": a.split, "n_transcripts": len(entries), "runs": K,
           "shards": str(a.shards), "simulator_model": shards.meta.get("model")}
    res["simulator"] = score(entries, sim_runs)
    if a.bootstrap:
        res["simulator_bootstrap"] = bootstrap(entries, sim_runs, a.bootstrap)
    if a.run_variance:
        res["simulator_run_variance"] = run_variance(entries, sim_runs, K)
    if a.polisher:
        from seq2ribo.models import RiboPolisherMamba
        from seq2ribo.utils import load_state_dict_safely
        P = config.POLISHER
        device = torch.device(a.device)
        model = RiboPolisherMamba(d_model=P["d_model"], n_layers=P["n_layers"], d_state=P["d_state"], d_conv=P["d_conv"],
                                  expand=P["expand"], dropout=P["dropout"], k_angle_bins=len(edges) - 1, inputs=a.inputs)
        model.load_state_dict(load_state_dict_safely(a.polisher, device))
        model.to(device).eval()
        pol_runs = [polish(model, e["feats"], s, device, a.batch_size) for e, s in zip(entries, sim_runs)]
        res["polisher"] = score(entries, pol_runs)
        res["polisher_checkpoint"] = str(a.polisher)
        if a.bootstrap:
            res["polisher_bootstrap"] = bootstrap(entries, pol_runs, a.bootstrap)
        if a.run_variance:
            res["polisher_run_variance"] = run_variance(entries, pol_runs, K)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    for k in ("simulator", "polisher"):
        if k in res:
            m = res[k]
            print(f"{k:9s} n={m['n_transcripts']} Tx r={m['txlevel_r']:.3f} Shape r={m['shape_r']:.3f} "
                  f"Elem r={m['elemwise_r']:.3f} | scaled ptxMAE {m['sc_pertranscriptMAE']:.3f} elemMAE {m['sc_elemwiseMAE']:.3f} "
                  f"codon {m['sc_codonMAE']:.0f} pair {m['sc_pairMAE']:.0f} angle {m['sc_angleMAE']:.0f} bucket {m['sc_bucketMAE']:.0f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
