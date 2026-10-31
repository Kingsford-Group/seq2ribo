#!/usr/bin/env python
"""Evaluate a trained sequence-only Translatomer (model_best.pt) on profiles, TE and expression.

profiles    data/<cell>/test.pkl: windows tiled exactly as in training (windows.py), predictions
            mapped bins -> CDS nucleotides (transcript orientation) -> codons, scored with
            seq2ribo.metrics.profile_metrics against codon_counts(asite_counts) (codon ids from the
            sequence, pair counts clipped 0..3, angle bins from data/angle_bins.json, k_angle=4).
            "translatomer": every nucleotide gets the value of the 64-bp bin covering it divided by 64,
            and a codon the sum over its three nucleotides (averaged over overlapping windows);
            "translatomer_cds_spread" divides each bin over the CDS nucleotides in it (for reference).
te          data/te/<cell>/test.pkl + labels.tsv: total predicted count over the CDS (sum of the
            codon-level prediction) vs TE; Pearson and Spearman. No RNA-seq or observed counts
            are used as inputs.
expression  data/expression/test.pkl + labels.tsv (mRFP): each CDS (DNA, sense) centred in an
            all-N 65,536-bp window, total predicted count over the CDS vs expression.
Correlations of constant predictions are reported as NaN (with "constant_predictions": true).

Outputs in --out-dir: profiles.json, te_metrics.json, expression_metrics.json and the per-transcript
predictions (profile_predictions.pkl, te_predictions.tsv, expression_predictions.tsv).
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4]))   # repo root (seq2ribo package)
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import config  # noqa: E402
from common import angle_edges, codon_counts, features, load_records  # noqa: E402
from seq2ribo.metrics import profile_metrics  # noqa: E402
from translatomer_model import TransModelSeqOnly  # noqa: E402
from window_datasets import InferenceWindows, check_record, load_cds, predict_bins  # noqa: E402
from windows import (SEQLEN, Genome, aggregate_windows, centered_placement, codon_sum,  # noqa: E402
                     transcript_windows)

MODES = {"translatomer": "uniform", "translatomer_cds_spread": "cds"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell-line", required=True, choices=config.CELL_LINES)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", default=["profiles", "te", "expression"],
                   choices=["profiles", "te", "expression"])
    p.add_argument("--split", default="test")
    p.add_argument("--subset", default="", help="profiles only: '' or '_01'")
    p.add_argument("--max-tx", type=int, default=None, help="smoke tests: first n transcripts/sequences per task")
    p.add_argument("--data-dir", type=Path, default=config.DATA)
    p.add_argument("--cds", type=Path, default=config.DATA / "cds.pkl")
    p.add_argument("--fasta", type=Path, default=config.GENOME_FASTA)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    model = TransModelSeqOnly(**ck.get("model_cfg", {}))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), {"checkpoint": str(path), "epoch": ck.get("epoch"), "val_loss": ck.get("val_loss")}


def predict_genomic(model, txs, cds, genome, a, device):
    """{tx: {mode: codon-level prediction}} for transcripts tiled with genomic windows."""
    wins = [w for tx in txs for w in transcript_windows(tx, cds[tx])]
    bins = predict_bins(model, InferenceWindows([("genomic", c, s, ws) for _, c, s, ws, _ in wins], genome),
                        device, a.batch_size, a.num_workers)
    by_tx = {}
    for w, b in zip(wins, bins):
        by_tx.setdefault(w[0], ([], []))
        by_tx[w[0]][0].append(b); by_tx[w[0]][1].append(w[4])
    return {tx: {m: codon_sum(aggregate_windows(*by_tx[tx], len(cds[tx]["seq"]), mode)) for m, mode in MODES.items()}
            for tx in txs}, len(wins)


def correlations(pred, label):
    pred, label = np.asarray(pred, np.float64), np.asarray(label, np.float64)
    out = {"n": int(len(pred)), "pred_mean": float(pred.mean()), "pred_std": float(pred.std()),
           "constant_predictions": bool(len(pred) < 2 or np.ptp(pred) == 0)}
    if out["constant_predictions"]:
        out.update(pearson_r=float("nan"), pearson_p=float("nan"), spearman_r=float("nan"), spearman_p=float("nan"))
    else:
        r, pr = pearsonr(pred, label); s, ps = spearmanr(pred, label)
        out.update(pearson_r=float(r), pearson_p=float(pr), spearman_r=float(s), spearman_p=float(ps))
    return out


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=1))  # NaN is written as NaN (json allow_nan)
    print(f"wrote {path}", flush=True)


def main():
    a = parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    model, ck_info = load_model(a.checkpoint, device)
    cds, genome = load_cds(a.cds), Genome(a.fasta)
    preds = {}

    if "profiles" in a.tasks:
        t0 = time.time()
        edges = angle_edges(a.data_dir)
        recs = load_records(a.cell_line, a.split, a.data_dir, a.subset)[:a.max_tx]
        for r in recs:
            check_record(r, cds[r[0]])
        preds, n_win = predict_genomic(model, [r[0] for r in recs], cds, genome, a, device)
        obs, cod, pair, ang = [], [], [], []
        for tx, seq, counts, angle_sum, pair_count in recs:
            c, ab, pb, _ = features(seq, angle_sum, pair_count, edges)
            o = codon_counts(counts)
            assert len(o) == len(c) == len(preds[tx]["translatomer"]), tx
            obs.append(o); cod.append(c); pair.append(pb); ang.append(ab)
        res = {"cell_line": a.cell_line, "split": f"{a.split}{a.subset}", "n_transcripts": len(recs), "n_windows": n_win,
               **ck_info, "nt_mapping": {"translatomer": "each nucleotide gets its bin's value / 64; codon = sum of its 3 nt",
                                         "translatomer_cds_spread": "bin value / #CDS nt in bin"}}
        for m in MODES:
            res[m] = profile_metrics(obs, [preds[r[0]][m] for r in recs], cod, pair, ang, k_angle=config.ANGLE_BINS_K)
        res["seconds"] = time.time() - t0
        write_json(a.out_dir / "profiles.json", res)
        with open(a.out_dir / "profile_predictions.pkl", "wb") as f:
            pickle.dump({"obs": dict(zip([r[0] for r in recs], obs)), "pred": preds}, f)
        m = res["translatomer"]
        print(f"[profiles] n={m['n_transcripts']} Tx r={m['txlevel_r']:.3f} Shape r={m['shape_r']:.3f} "
              f"Elem r={m['elemwise_r']:.3f} | ptxMAE {m['pertranscriptMAE']:.3f} sc_ptxMAE {m['sc_pertranscriptMAE']:.3f} "
              f"codon {m['codonMAE']:.0f} sc_codon {m['sc_codonMAE']:.0f}", flush=True)

    if "te" in a.tasks:
        lab = pd.read_csv(a.data_dir / "te" / a.cell_line / "labels.tsv", sep="\t")
        lab = lab[lab["split"] == a.split].set_index("tx_id")["te"]
        recs = load_records(a.cell_line, a.split, a.data_dir / "te")[:a.max_tx]   # data/te/<cell>/test.pkl
        txs = [r[0] for r in recs]
        missing = [t for t in txs if t not in preds]
        if missing:
            preds.update(predict_genomic(model, missing, cds, genome, a, device)[0])
        res = {"cell_line": a.cell_line, "split": a.split, "label": config.TE_COLUMNS[a.cell_line], **ck_info,
               "predictor": "total predicted A-site count over the CDS (sequence-only model)"}
        tab = pd.DataFrame({"tx_id": txs, "te": lab.loc[txs].values})
        for m in MODES:
            tab[f"total_{m}"] = [preds[t][m].sum() for t in txs]
            res[m] = correlations(tab[f"total_{m}"], tab["te"])
        tab.to_csv(a.out_dir / "te_predictions.tsv", sep="\t", index=False)
        write_json(a.out_dir / "te_metrics.json", res)
        print(f"[te] n={res['translatomer']['n']} r={res['translatomer']['pearson_r']:.4f} "
              f"rho={res['translatomer']['spearman_r']:.4f}", flush=True)

    if "expression" in a.tasks:
        lab = pd.read_csv(a.data_dir / "expression" / "labels.tsv", sep="\t")
        lab = lab[lab["split"] == a.split].set_index("seq_id")["expression"]
        with open(a.data_dir / "expression" / f"{a.split}.pkl", "rb") as f:
            recs = pickle.load(f)[:a.max_tx]
        specs, widxs = [], []
        for sid, seq, *_ in recs:
            pad, widx = centered_placement(len(seq))
            dna = seq.replace("U", "T")
            specs.append(("sequence", "N" * pad + dna + "N" * (SEQLEN - pad - len(dna))))
            widxs.append(widx)
        bins = predict_bins(model, InferenceWindows(specs), device, a.batch_size, a.num_workers)
        ids = [r[0] for r in recs]
        tab = pd.DataFrame({"seq_id": ids, "expression": lab.loc[ids].values})
        res = {"split": a.split, "cell_line_model": a.cell_line, **ck_info,
               "predictor": "total predicted A-site count over the CDS centred in an all-N window"}
        for m, mode in MODES.items():
            tab[f"total_{m}"] = [aggregate_windows([b], [w], len(w), mode).sum() for b, w in zip(bins, widxs)]
            res[m] = correlations(tab[f"total_{m}"], tab["expression"])
        tab.to_csv(a.out_dir / "expression_predictions.tsv", sep="\t", index=False)
        write_json(a.out_dir / "expression_metrics.json", res)
        e = res["translatomer"]
        print(f"[expression] n={e['n']} r={e['pearson_r']:.4f} rho={e['spearman_r']:.4f} "
              f"constant={e['constant_predictions']}", flush=True)


if __name__ == "__main__":
    main()
