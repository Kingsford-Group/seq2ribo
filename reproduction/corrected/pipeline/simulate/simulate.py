#!/usr/bin/env python
"""Simulation runs (sTASEP or TASEP) for a set of transcripts with fitted parameters.

For every transcript, K independent runs (seeded per transcript and run) are stored as
  <out>/<name>.npy   uint8 array, shape (K, total codons); transcript i occupies columns
                     offsets[i]:offsets[i+1]  (counts <= SNAPSHOTS = 200 fit in uint8)
  <out>/<name>.json  {"tx": [...], "offsets": [...], "runs": K, "params": <fit json path>, ...}
Input transcripts: the {train,val,test} records of a cell line (default) or any pickle of
records (tx_id, cds_rna, ..., angle_sum, pair_count) via --records.
"""
import argparse
import json
import pickle
import random
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from seq2ribo import simulation as S  # noqa: E402

JOB = {}


def run_tx(i):
    tx, seq, ang, pair = JOB["records"][i]
    codons = [seq[3 * j:3 * j + 3] for j in range(len(seq) // 3)]
    p = JOB["params"]
    waits = S.codon_waits(codons, S.angle_bin_index(ang, JOB["edges"]), pair, S.full_wait_times(p["rates"]),
                          np.asarray(p["alpha"], float), np.asarray(p["beta"], float), np.asarray(p["buckets"], float))
    out = np.empty((JOB["runs"], len(codons)), dtype=np.uint8)
    for k in range(JOB["runs"]):
        rng = random.Random(zlib.crc32(f"{JOB['seed']}:{tx}:{k}".encode()))
        out[k] = S.simulate_run(codons, waits, config.INIT_P, rng)
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params", type=Path, required=True, help="params.json from pipeline/fit/fit.py")
    p.add_argument("--runs", type=int, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cell-line", choices=config.CELL_LINES, help="simulate this cell line's split records")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--records", type=Path, help="pickle of records instead of --cell-line splits")
    p.add_argument("--name", default=None, help="output name with --records")
    p.add_argument("--data-dir", type=Path, default=config.DATA)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    fit = json.load(open(a.params))
    edges = np.asarray(json.load(open(a.data_dir / "angle_bins.json"))["edges"])
    if not np.allclose(edges, fit["angle_edges"]):
        raise SystemExit(f"angle edges of {a.params} differ from {a.data_dir / 'angle_bins.json'}")
    if a.records:
        inputs = {a.name or a.records.stem: pickle.load(open(a.records, "rb"))}
    else:
        inputs = {s: pickle.load(open(a.data_dir / a.cell_line / f"{s}.pkl", "rb")) for s in a.splits}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in inputs.items():
        JOB.update(records=[(r[0], r[1], r[-2], r[-1]) for r in recs], params=fit["params"],
                   edges=edges, runs=a.runs, seed=a.seed)
        with ProcessPoolExecutor(a.workers) as ex:
            blocks = list(ex.map(run_tx, range(len(JOB["records"])), chunksize=8))
        offsets = np.concatenate([[0], np.cumsum([b.shape[1] for b in blocks])]).tolist()
        np.save(a.out_dir / f"{name}.npy", np.concatenate(blocks, axis=1) if blocks else np.zeros((a.runs, 0), np.uint8))
        with open(a.out_dir / f"{name}.json", "w") as f:
            json.dump({"tx": [r[0] for r in JOB["records"]], "offsets": offsets, "runs": a.runs,
                       "params": str(a.params), "model": fit["model"], "seed": a.seed, "init_p": config.INIT_P}, f)
        print(f"[{name}] {len(blocks)} transcripts x {a.runs} runs -> {a.out_dir / (name + '.npy')}", flush=True)


if __name__ == "__main__":
    main()
