#!/usr/bin/env python
"""Build the published train/val/test split pickles and geometry maps from a CDS A-site counts file.

Input: the output of `asite_by_cds.py --published-mapping` (header, CDS sequence, per-nucleotide counts).
Transcripts and their split come from the published split lists (reproduction/splits/<cell>.tsv.gz);
the original split cannot be regenerated (it depended on the iteration order of a Python set). Every
CDS is checked against the published one (cds_md5: first 12 hex digits of the MD5 of the RNA sequence).
Structure features: seq2ribo.geometry with version="paper" (features of the published models).

Writes {train,val,test}_data.pkl, 10% subsets {train,val,test}_data_01.pkl and
polisher/geo_map_{train,val,test}.pkl. Each record is
(tx_id, cds_rna, asite_counts_per_nt, zeros_like_counts, (angle_dev_sum, pair_count)).
"""
import argparse
import hashlib
import importlib.util
import pickle
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
# load seq2ribo/geometry.py directly (no torch / mamba-ssm needed)
_spec = importlib.util.spec_from_file_location("seq2ribo_geometry", REPO / "seq2ribo" / "geometry.py")
_geometry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_geometry)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--counts", required=True, type=Path, help="output of asite_by_cds.py --published-mapping")
    p.add_argument("--cell-line", required=True, choices=["ipsc", "hek293", "lcl", "rpe"])
    p.add_argument("--out-dir", required=True, type=Path, help="e.g. reproduction/paper/data/<cell_line>")
    p.add_argument("--splits-dir", type=Path, default=REPO / "reproduction" / "splits")
    p.add_argument("--subset-seed", type=int, default=1, help="seed of the 10%% subsets")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def read_counts(path):
    data = {}
    with open(path) as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            seq = fh.readline().strip().upper().replace("T", "U")
            counts = np.array(fh.readline().split(), dtype=np.int32)
            data[header[1:].split("|")[0].split()[0]] = (seq, counts)
    return data


def geometry(item):
    tx, seq = item
    try:
        a, p = _geometry.fold_features(seq, "paper")
    except RuntimeError:  # published behaviour: zero features when folding or the layout fails
        a, p = np.zeros(len(seq) // 3), np.zeros(len(seq) // 3)
    return tx, (np.asarray(a, np.float64), np.asarray(p, np.int32))


def main():
    a = parse_args()
    data = read_counts(a.counts)
    split = pd.read_csv(a.splits_dir / f"{a.cell_line}.tsv.gz", sep="\t")
    missing = sorted(set(split["tx_id"]) - set(data))
    if missing:
        sys.exit(f"{len(missing)} transcripts of the published split are missing from {a.counts}, e.g. {missing[:3]}")
    bad = [tx for tx, h in zip(split["tx_id"], split["cds_md5"]) if hashlib.md5(data[tx][0].encode()).hexdigest()[:12] != h]
    if bad:
        sys.exit(f"{len(bad)} transcripts have a CDS different from the published one, e.g. {bad[:3]}; "
                 "use the GENCODE v48 annotation and the GRCh38 genome")
    with ProcessPoolExecutor(a.workers) as ex:
        geo = dict(ex.map(geometry, [(tx, data[tx][0]) for tx in split["tx_id"]], chunksize=16))
    out = a.out_dir
    (out / "polisher").mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.subset_seed)
    for name in ("train", "val", "test"):
        txs = split.loc[split["split"] == name, "tx_id"].tolist()
        recs = [(tx, data[tx][0], data[tx][1], np.zeros_like(data[tx][1]), geo[tx]) for tx in txs]
        with open(out / f"{name}_data.pkl", "wb") as f:
            pickle.dump(recs, f)
        k = int(0.1 * len(recs))
        with open(out / f"{name}_data_01.pkl", "wb") as f:
            pickle.dump(rng.sample(recs, k) if k > 0 else [], f)
        with open(out / "polisher" / f"geo_map_{name}.pkl", "wb") as f:
            pickle.dump({tx: geo[tx] for tx in txs}, f)
        print(f"[split] {name}: {len(recs)}")


if __name__ == "__main__":
    main()
