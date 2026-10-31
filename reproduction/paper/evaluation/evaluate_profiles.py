#!/usr/bin/env python
"""Test-set evaluation of ribosome profiles (sTASEP input and polished seq2ribo output).

For every test transcript whose CDS is not identical to a training or validation CDS, each
simulation shard (<shards>/test/<tx>_sample_<k>.pkl, written by polisher/data_generator.py)
is polished independently; predictions are averaged per transcript and scored with
evaluation/metrics.py. The same shards, unpolished, give the sTASEP metrics.

Example
  python reproduction/paper/evaluation/evaluate_profiles.py --cell-line hek293 --bootstrap 1000
"""
import argparse
import glob
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from seq2ribo.models import RiboPolisherMamba  # noqa: E402
from seq2ribo.utils import CODON2IDX, load_state_dict_safely  # noqa: E402
import metrics as M  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell-line", required=True, choices=["ipsc", "hek293", "rpe", "lcl"])
    p.add_argument("--data-root", default=str(HERE.parent / "data"))
    p.add_argument("--shards", default=None, help="default: <data-root>/<cell>/polisher/data_new")
    p.add_argument("--checkpoint", default=None, help="default: weights/paper/<cell>_mamba_final_unscaled.pt")
    p.add_argument("--max-samples", type=int, default=None, help="use at most this many shards per transcript")
    p.add_argument("--bootstrap", type=int, default=0, help="number of bootstrap resamples for 95%% CIs")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="output JSON (default: results/evaluation/<cell>_profiles.json)")
    return p.parse_args()


def cds_md5(seq):
    return hashlib.md5(seq.upper().replace("U", "T").encode()).hexdigest()


def derive_angle_bins(geo_map_train, k=4, low_q=0.0, high_q=0.995):
    vals = np.concatenate([np.asarray(a, dtype=np.float64) for a, _ in geo_map_train.values() if a is not None])
    vals = vals[np.isfinite(vals)]
    lo, hi = float(np.quantile(vals, low_q)), float(np.quantile(vals, high_q))
    edges = np.linspace(lo, hi if hi > lo else lo + 1e-6, k + 1)
    edges[-1] = max(edges[-1], float(vals.max()))
    return edges


@torch.no_grad()
def polish(model, feats, sims, device, batch_size):
    """feats: (cod_ids, angle_bin, pair_bin, bucket) of one transcript; sims: list of raw count vectors."""
    cod, ang, pair, bkt = (torch.as_tensor(x, dtype=torch.long) for x in feats)
    out = []
    for s in range(0, len(sims), batch_size):
        chunk = sims[s:s + batch_size]
        B = len(chunk)
        sim = torch.as_tensor(np.log1p(np.stack(chunk)), dtype=torch.float32).to(device)
        rep = lambda t: t.unsqueeze(0).expand(B, -1).to(device)
        mask = torch.ones_like(sim, dtype=torch.bool)
        y = model(rep(cod), sim, mask, angle_bin=rep(ang), pair_bin=rep(pair), bucket_idx=rep(bkt))
        out.extend(torch.expm1(y).cpu().numpy().astype(np.float64))
    return out


def main():
    a = parse_args()
    data = Path(a.data_root) / a.cell_line
    shards = Path(a.shards) if a.shards else data / "polisher" / "data_new"
    ckpt = Path(a.checkpoint) if a.checkpoint else REPO / "weights" / "paper" / f"{a.cell_line}_mamba_final_unscaled.pt"
    out = Path(a.out) if a.out else HERE.parent / "results" / "evaluation" / f"{a.cell_line}_profiles.json"

    with open(data / "polisher" / "geo_map_train.pkl", "rb") as f:
        angle_bins = derive_angle_bins(pickle.load(f))
    with open(data / "polisher" / "geo_map_test.pkl", "rb") as f:
        geo_test = pickle.load(f)
    seen = set()
    for split in ("train", "val"):
        with open(data / f"{split}_data.pkl", "rb") as f:
            seen.update(cds_md5(r[1]) for r in pickle.load(f))
    with open(data / "test_data.pkl", "rb") as f:
        test = [r for r in pickle.load(f) if cds_md5(r[1]) not in seen and r[0] in geo_test]
    print(f"[{a.cell_line}] test transcripts after removing CDSs shared with train/val: {len(test)}")

    model = RiboPolisherMamba()
    model.load_state_dict(load_state_dict_safely(ckpt, a.device))
    model.to(a.device).eval()

    ent_model, ent_sim, missing = [], [], 0
    for i, r in enumerate(test):
        tx = r[0]
        files = sorted(glob.glob(str(shards / "test" / f"{tx}_sample_*.pkl")))[:a.max_samples]
        if not files:
            missing += 1
            continue
        recs = [pickle.load(open(f, "rb")) for f in files]
        cods = recs[0][1]
        obs = np.asarray(recs[0][2], dtype=np.float64)
        sims = [np.asarray(x[3], dtype=np.float32) for x in recs]
        ang, pc = geo_test[tx]
        L = len(cods)
        feats = (np.array([CODON2IDX.get(c, 64) for c in cods]), M.angle_bin_index(np.asarray(ang)[:L], angle_bins),
                 np.clip(np.asarray(pc)[:L], 0, 3), M.bucket_index(L))
        preds = polish(model, feats, sims, a.device, a.batch_size)
        ent_model.append(M.make_entry(obs, preds, feats[0], feats[2], feats[1]))
        ent_sim.append(M.make_entry(obs, sims, feats[0], feats[2], feats[1]))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(test)}", flush=True)
    if missing:
        print(f"  {missing} test transcripts have no shards under {shards / 'test'}")

    res = {"cell_line": a.cell_line, "checkpoint": str(ckpt), "shards": str(shards),
           "angle_bins": [float(x) for x in angle_bins],
           "seq2ribo": M.compute_metrics(ent_model), "sTASEP": M.compute_metrics(ent_sim)}
    if a.bootstrap:
        res["seq2ribo_ci"] = M.bootstrap_ci(ent_model, a.bootstrap)
        res["sTASEP_ci"] = M.bootstrap_ci(ent_sim, a.bootstrap)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    for name in ("sTASEP", "seq2ribo"):
        m = res[name]
        print(f"{name:9s} n={m['n_transcripts']}  Tx-level r={m['txlvl_pearson']:.3f}  Shape r={m['shape_pearson']:.3f}  "
              f"Elemwise r={m['elemwise_pearson']:.3f} | scaled MAEs: pertranscript={m['sc_pertranscriptMAE']:.3f} "
              f"elemwise={m['sc_elemwiseMAE']:.3f} codon={m['sc_codonMAE']:.0f} pair={m['sc_pairMAE']:.0f} "
              f"angle={m['sc_angleMAE']:.0f} bucket={m['sc_bucketMAE']:.0f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
