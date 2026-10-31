#!/usr/bin/env python
"""Per-cell-line datasets on the published train/val/test transcript sets, with corrected counts.

1. Transcripts and their split come from the published split lists (config.PROFILE_SPLITS,
   reproduction/splits/<cell>.tsv.gz). For CDSs with an incomplete 5' end the published sequence
   starts cds_trim_5p (1-2) nt after the annotated CDS start (reading frame of the annotation); the
   same nucleotides are removed from the CDS, the A-site counts and the CDS blocks. Every CDS is
   checked against the published one (cds_md5: first 12 hex digits of the MD5 of the RNA sequence).
2. A-site counts come from the corrected mapping (map_asite.py: transcript orientation, every
   base of multi-base bedGraph records, stop codon included); transcripts whose corrected total
   is zero are dropped (published filter).
3. As in the published training and evaluation code, validation transcripts whose CDS also occurs
   in training, and test transcripts whose CDS occurs in training or validation, are removed.
4. Structure features (seq2ribo.geometry, shared cache); angle-bin edges: K linear bins between the
   training quantiles ANGLE_BINS_QUANTILES of the per-codon angle sum over the training
   transcripts of all cell lines, last edge raised to the training maximum.

Outputs in config.DATA: cds.pkl (CDS models of the published sequences), split.tsv
(cell_line, tx_id, gene_id, split), angle_bins.json, summary.json and
<cell>/{train,val,test}.pkl (+ *_01.pkl 10% subsets) with records
(tx_id, cds_rna, asite_counts_per_nt, angle_sum_per_codon, pair_count_per_codon).
"""
import argparse
import copy
import hashlib
import json
import pickle
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from seq2ribo.geometry import compute_geometry_features  # noqa: E402

GEO_CACHE = config.SHARED / "geometry_cache"
SPLITS = ("train", "val", "test")


def md5(seq):
    return hashlib.md5(seq.encode()).hexdigest()


def fold_one(seq):
    g = compute_geometry_features(seq, cache_dir=GEO_CACHE)
    return md5(seq), (np.asarray(g[0], np.float64), np.asarray(g[1], np.int64))


def trim_5prime(entry, off):
    """CDS model without its first `off` nucleotides (transcript orientation)."""
    e = copy.deepcopy(entry)
    starts, ends = list(e["starts"]), list(e["ends"])
    left = off
    while left:
        if e["strand"] == "+":
            n = min(left, ends[0] - starts[0])
            starts[0] += n
            if starts[0] == ends[0]:
                starts.pop(0); ends.pop(0)
        else:
            n = min(left, ends[-1] - starts[-1])
            ends[-1] -= n
            if starts[-1] == ends[-1]:
                starts.pop(); ends.pop()
        left -= n
    e["starts"], e["ends"], e["seq"] = starts, ends, e["seq"][off:]
    return e


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell-lines", nargs="+", default=config.CELL_LINES)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--out-dir", type=Path, default=config.DATA)
    return p.parse_args()


def main():
    a = parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with open(config.SHARED / "cds.pkl", "rb") as f:
        cds_src = pickle.load(f)
    cds_out, summary = {}, {"source_splits": {c: str(p) for c, p in config.PROFILE_SPLITS.items()}, "cell_lines": {}}
    data = {}
    for cl in a.cell_lines:
        with open(config.SHARED / cl / "asite_counts.pkl", "rb") as f:
            counts = pickle.load(f)
        st = {"published": {}, "trimmed_5prime": 0, "zero_total": 0}
        data[cl] = {}
        split_tab = pd.read_csv(config.PROFILE_SPLITS[cl], sep="\t")
        bad = [tx for tx, off, h in zip(split_tab["tx_id"], split_tab["cds_trim_5p"].astype(int), split_tab["cds_md5"])
               if tx not in cds_src or md5(cds_src[tx]["seq"][off:])[:12] != h]
        if bad:
            raise ValueError(f"{cl}: {len(bad)} transcripts are missing or have a CDS different from the published one "
                             f"(e.g. {bad[:3]}); use the GENCODE v48 annotation and the GRCh38 genome")
        for s in SPLITS:
            pub = split_tab[split_tab["split"] == s]
            st["published"][s] = len(pub)
            rows = []
            for tx, off in zip(pub["tx_id"], pub["cds_trim_5p"].astype(int)):
                seq = cds_src[tx]["seq"][off:]
                if tx in cds_out and cds_out[tx]["seq"] != seq:
                    raise ValueError(f"{tx}: different CDS across cell lines")
                cds_out[tx] = trim_5prime(cds_src[tx], off) if off else cds_src[tx]
                st["trimmed_5prime"] += int(off > 0)
                c = np.asarray(counts[tx])[off:]
                if len(c) != len(seq):
                    raise ValueError(f"{cl} {tx}: counts/CDS length mismatch")
                if c.sum() == 0:   # published filter: non-zero total A-site count
                    st["zero_total"] += 1
                    continue
                rows.append((tx, seq, c))
            data[cl][s] = rows
        # duplicate filter of the published code: val vs train, test vs train and val
        train_cds = {r[1] for r in data[cl]["train"]}
        val = [r for r in data[cl]["val"] if r[1] not in train_cds]
        val_cds = {r[1] for r in val}
        test = [r for r in data[cl]["test"] if r[1] not in train_cds and r[1] not in val_cds]
        st["removed_identical_cds"] = {"val": len(data[cl]["val"]) - len(val), "test": len(data[cl]["test"]) - len(test)}
        data[cl]["val"], data[cl]["test"] = val, test
        st["kept"] = {s: len(data[cl][s]) for s in SPLITS}
        summary["cell_lines"][cl] = st
        print(f"[{cl}] published {st['published']}  removed identical CDS {st['removed_identical_cds']}  "
              f"kept {st['kept']}  5'-trimmed {st['trimmed_5prime']}  dropped (zero total) {st['zero_total']}", flush=True)

    with open(a.out_dir / "cds.pkl", "wb") as f:
        pickle.dump(cds_out, f)

    seqs = sorted({r[1] for cl in data for s in SPLITS for r in data[cl][s]}, key=len, reverse=True)
    with ProcessPoolExecutor(a.workers) as ex:
        geometry = dict(ex.map(fold_one, seqs, chunksize=4))
    print(f"[geometry] {len(geometry)} distinct CDSs", flush=True)

    with open(a.out_dir / "split.tsv", "w") as f:
        f.write("cell_line\ttx_id\tgene_id\tsplit\n")
        for cl in data:
            for s in SPLITS:
                for r in data[cl][s]:
                    f.write(f"{cl}\t{r[0]}\t{cds_out[r[0]]['gene_id']}\t{s}\n")

    def edges_from(vals):
        lo, hi = (float(np.quantile(vals, q)) for q in config.ANGLE_BINS_QUANTILES)
        e = np.linspace(lo, hi if hi > lo else lo + 1e-6, config.ANGLE_BINS_K + 1)
        e[-1] = max(e[-1], float(vals.max()))
        return e
    per_cell = {cl: edges_from(np.concatenate([geometry[md5(r[1])][0] for r in data[cl]["train"]])) for cl in data}
    train_seqs = {r[1] for cl in data for r in data[cl]["train"]}
    e = edges_from(np.concatenate([geometry[md5(s)][0] for s in train_seqs]))
    for cl, ec in per_cell.items():
        print(f"[angle bins] {cl} training set alone: {np.round(ec, 6).tolist()}")
    print(f"[angle bins] all training transcripts: {np.round(e, 6).tolist()}")
    with open(a.out_dir / "angle_bins.json", "w") as f:
        json.dump({"edges": [float(x) for x in e], "k": config.ANGLE_BINS_K,
                   "per_cell_line_training_edges": {cl: [float(x) for x in ec] for cl, ec in per_cell.items()}}, f, indent=1)

    sub_rng = random.Random(1)
    for cl in data:
        d = a.out_dir / cl
        d.mkdir(parents=True, exist_ok=True)
        for s in SPLITS:
            recs = [(tx, seq, c, *geometry[md5(seq)]) for tx, seq, c in data[cl][s]]
            with open(d / f"{s}.pkl", "wb") as f:
                pickle.dump(recs, f)
            k = int(0.1 * len(recs))
            with open(d / f"{s}_01.pkl", "wb") as f:
                pickle.dump(sub_rng.sample(recs, k) if k else [], f)
    with open(a.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {a.out_dir}")


if __name__ == "__main__":
    main()
