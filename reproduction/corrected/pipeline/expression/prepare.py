#!/usr/bin/env python
"""Protein expression dataset (mRFP, Nieuwkoop et al. 2023; split as in CodonBERT).

Folds every mRFP CDS (seq2ribo.geometry, shared cache) and writes data/expression/{train,val,test}.pkl
with records (seq_id, cds_rna, zeros, angle_sum, pair_count) and labels.tsv (seq_id, split, expression).
"""
import argparse
import pickle
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


def fold(seq):
    return compute_geometry_features(seq, cache_dir=GEO_CACHE)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=config.RAW / "mRFP_Expression.csv")
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()
    df = pd.read_csv(a.csv)
    df["seq"] = df["Sequence"].str.upper().str.replace("T", "U")
    df["seq_id"] = [f"mrfp_{i:05d}" for i in range(len(df))]
    bad = df[~df["seq"].map(lambda s: len(s) % 3 == 0 and not set(s) - set("ACGU"))]
    if len(bad):
        raise SystemExit(f"{len(bad)} mRFP sequences are not valid CDSs")
    with ProcessPoolExecutor(a.workers) as ex:
        geo = list(ex.map(fold, df["seq"], chunksize=8))
    out = config.DATA / "expression"
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        idx = np.flatnonzero(df["Split"].values == split)
        recs = [(df["seq_id"].iat[i], df["seq"].iat[i], np.zeros(len(df["seq"].iat[i]), np.int32), *geo[i]) for i in idx]
        with open(out / f"{split}.pkl", "wb") as f:
            pickle.dump(recs, f)
        print(f"[expression] {split}: {len(recs)}")
    df.rename(columns={"Split": "split", "Value": "expression"})[["seq_id", "split", "expression"]].to_csv(
        out / "labels.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
