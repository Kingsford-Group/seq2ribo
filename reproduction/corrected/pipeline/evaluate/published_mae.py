#!/usr/bin/env python
"""pertranscriptMAE and codonMAE as computed by the published evaluation code
(revision/uncertainty/uncertainty_lib.py: collect_*_data + compute_metrics_from_collected).

Per transcript i and simulation run k (one "run" for Translatomer):
  scaled prediction   p_sc = p * (sum y / sum p)  (zeros if sum p = 0)
  pertranscript loss  sum_j |p - y| / sum_j y       (unscaled) and the same with p_sc (scaled)
  codon sums          per codon id over non-stop positions, of y and of p / p_sc
Losses are averaged over runs per transcript, then over transcripts; codon sums of the predictions
are averaged over runs per transcript and summed over transcripts; codonMAE is the mean of
|Y_c - Yhat_c| over all 64 codon slots (stop slots are 0 in both). These two are the values reported
in the profile table; the other metrics are those of evaluate_profiles.py (seq2ribo.metrics).

Methods: TASEP and sTASEP (results/sims/<model>/<cell>/test.*, all runs), Translatomer
(results/baselines/translatomer/<cell>/profile_predictions.pkl, mapping "translatomer") and seq2ribo
(every sTASEP run polished by results/polisher/seq2ribo/<cell>/polisher.pt, once that cell line's
polisher is trained and evaluated, i.e. results/eval/profiles/<cell>/seq2ribo_stasep.json exists).
Output: results/eval/profiles/published_mae.json and .md
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from common import Shards, angle_edges, codon_counts, features, load_records  # noqa: E402
from seq2ribo.utils import CODON2IDX, STOP_CODONS  # noqa: E402

STOP_IDX = np.array([CODON2IDX[c] for c in STOP_CODONS])


def published(records, runs_of):
    """runs_of(tx) -> array (K, L) of predictions. Returns the four published-code metrics."""
    ptx, ptx_sc = [], []
    cod_o, cod_p, cod_psc = np.zeros(64), np.zeros(64), np.zeros(64)
    for tx, seq, counts, *_ in records:
        y = codon_counts(counts)
        cid = np.array([CODON2IDX[seq[3 * j:3 * j + 3]] for j in range(len(y))])
        ns = ~np.isin(cid, STOP_IDX)
        P = np.atleast_2d(np.asarray(runs_of(tx), np.float64))
        ys = y.sum()
        s = P.sum(axis=1, keepdims=True)
        Psc = np.where(s > 0, P * (ys / np.where(s > 0, s, 1)), 0.0)
        ptx.append(np.mean(np.abs(P - y).sum(axis=1) / ys) if ys > 0 else 0.0)
        ptx_sc.append(np.mean(np.abs(Psc - y).sum(axis=1) / ys) if ys > 0 else 0.0)
        cod_o += np.bincount(cid[ns], weights=y[ns], minlength=64)
        cod_p += np.mean([np.bincount(cid[ns], weights=p[ns], minlength=64) for p in P], axis=0)
        cod_psc += np.mean([np.bincount(cid[ns], weights=p[ns], minlength=64) for p in Psc], axis=0)
    return {"pertranscriptMAE": float(np.mean(ptx)), "sc_pertranscriptMAE": float(np.mean(ptx_sc)),
            "codonMAE": float(np.mean(np.abs(cod_o - cod_p))), "sc_codonMAE": float(np.mean(np.abs(cod_o - cod_psc)))}


def polisher_runs(cell, recs, device):
    """{tx: (K, L) polished sTASEP runs} or None if the cell line has no polisher yet."""
    path = config.RESULTS / "polisher" / "seq2ribo" / cell / "polisher.pt"
    # only finished polishers (training done and evaluated by jobs/gpu_04_polisher.sh)
    if not (config.RESULTS / "eval" / "profiles" / cell / "seq2ribo_stasep.json").exists():
        return None
    from evaluate_profiles import polish
    from seq2ribo.models import RiboPolisherMamba
    from seq2ribo.utils import load_state_dict_safely
    P, edges = config.POLISHER, angle_edges(config.DATA)
    model = RiboPolisherMamba(d_model=P["d_model"], n_layers=P["n_layers"], d_state=P["d_state"], d_conv=P["d_conv"],
                              expand=P["expand"], dropout=P["dropout"], k_angle_bins=len(edges) - 1).to(device)
    model.load_state_dict(load_state_dict_safely(path, device))
    model.eval()
    sh = Shards(config.RESULTS / "sims" / "stasep" / cell / "test")
    return {tx: polish(model, features(seq, ang, pair, edges), sh.all_runs(tx), device, 64)
            for tx, seq, _, ang, pair in recs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0", help="for the polisher")
    a = ap.parse_args()
    d = config.RESULTS / "eval" / "profiles"
    out = {}
    for c in config.CELL_LINES:
        recs = load_records(c, "test", config.DATA)
        out[c] = {}
        for name, model in (("TASEP", "tasep"), ("sTASEP", "stasep")):
            sh = Shards(config.RESULTS / "sims" / model / c / "test")
            out[c][name] = published(recs, sh.all_runs)
        tp = config.RESULTS / "baselines" / "translatomer" / c / "profile_predictions.pkl"
        if tp.exists():
            pred = pickle.load(open(tp, "rb"))["pred"]
            out[c]["Translatomer (seq.)"] = published(recs, lambda t: pred[t]["translatomer"])
        pol = polisher_runs(c, recs, a.device)
        if pol is not None:
            out[c]["seq2ribo"] = published(recs, pol.__getitem__)
        print(c, json.dumps(out[c]), flush=True)
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / "published_mae.json", "w"), indent=1)
    methods = ["TASEP", "sTASEP", "Translatomer (seq.)", "seq2ribo"]
    md = ["| Cell line | Metric | " + " | ".join(methods) + " |", "|---|---|" + "---|" * len(methods)]
    for c in out:
        for i, (lab, k, f) in enumerate([("pertranscriptMAE", "sc_pertranscriptMAE", "{:.3f}"),
                                         ("codonMAE", "sc_codonMAE", "{:,.0f}")]):
            md.append(f"| {c if i == 0 else ''} | {lab} | " +
                      " | ".join(f.format(out[c][m][k]) if m in out[c] else "-" for m in methods) + " |")
    (d / "published_mae.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
