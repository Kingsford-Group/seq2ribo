"""Data access shared by the training and evaluation scripts."""
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import config
from seq2ribo.simulation import angle_bin_index, bucket_index
from seq2ribo.utils import CODON2IDX


def load_records(cell_line, split, data_dir=config.DATA, subset=""):
    """[(tx_id, cds_rna, asite_counts_per_nt, angle_sum, pair_count), ...]"""
    with open(Path(data_dir) / cell_line / f"{split}{subset}.pkl", "rb") as f:
        return pickle.load(f)


def angle_edges(data_dir=config.DATA):
    return np.asarray(json.load(open(Path(data_dir) / "angle_bins.json"))["edges"], dtype=np.float64)


def codon_counts(counts_nt):
    L = len(counts_nt) // 3
    return np.add.reduceat(np.asarray(counts_nt[:3 * L], np.float64), np.arange(0, 3 * L, 3))


class Shards:
    """Simulation runs written by pipeline/simulate/simulate.py (memory-mapped)."""

    def __init__(self, path_stem):
        path_stem = Path(path_stem)
        meta = json.load(open(str(path_stem) + ".json"))
        self.tx = meta["tx"]
        self.offsets = np.asarray(meta["offsets"], dtype=np.int64)
        self.runs = meta["runs"]
        self.meta = meta
        self.array = np.load(str(path_stem) + ".npy", mmap_mode="r")
        self.index = {t: i for i, t in enumerate(self.tx)}

    def get(self, tx, run):
        i = self.index[tx]
        return np.asarray(self.array[run, self.offsets[i]:self.offsets[i + 1]], dtype=np.float32)

    def all_runs(self, tx):
        i = self.index[tx]
        return np.asarray(self.array[:, self.offsets[i]:self.offsets[i + 1]], dtype=np.float32)


def features(seq, angle_sum, pair_count, edges):
    L = len(seq) // 3
    return (np.array([CODON2IDX[seq[3 * j:3 * j + 3]] for j in range(L)], dtype=np.int64),
            angle_bin_index(angle_sum, edges).astype(np.int64),
            np.clip(np.asarray(pair_count, np.int64), 0, 3),
            bucket_index(L))


class ProfileDataset(Dataset):
    """One item per (transcript, simulation run); target = observed codon-level A-site counts."""

    def __init__(self, records, shards, edges, runs=None):
        self.recs = [r for r in records if r[0] in shards.index]
        self.shards = shards
        self.edges = edges
        runs = list(range(shards.runs)) if runs is None else list(runs)
        self.items = [(i, k) for i in range(len(self.recs)) for k in runs]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, n):
        i, k = self.items[n]
        tx, seq, counts, ang, pair = self.recs[i]
        cod, ab, pb, bk = features(seq, ang, pair, self.edges)
        return {"tx": tx, "run": k, "length": len(cod),
                "cod_ids": torch.from_numpy(cod), "angle_bin": torch.from_numpy(ab),
                "pair_bin": torch.from_numpy(pb), "bucket_idx": torch.from_numpy(bk),
                "sim_feat": torch.from_numpy(np.log1p(self.shards.get(tx, k))),
                "target": torch.from_numpy(codon_counts(counts).astype(np.float32))}


def pad_collate(batch):
    """Pad to the longest item; keeps the input order."""
    B, Lmax = len(batch), max(x["length"] for x in batch)
    out = {"tx": [x["tx"] for x in batch], "run": [x["run"] for x in batch],
           "length": torch.tensor([x["length"] for x in batch]),
           "cod_ids": torch.full((B, Lmax), 64, dtype=torch.long),
           "mask": torch.zeros((B, Lmax), dtype=torch.bool)}
    for k in ("angle_bin", "pair_bin", "bucket_idx"):
        out[k] = torch.zeros((B, Lmax), dtype=torch.long)
    for k in ("sim_feat", "target"):
        if k in batch[0]:
            out[k] = torch.zeros((B, Lmax), dtype=torch.float32)
    for i, x in enumerate(batch):
        L = x["length"]
        out["mask"][i, :L] = True
        for k in ("cod_ids", "angle_bin", "pair_bin", "bucket_idx", "sim_feat", "target"):
            if k in x:
                out[k][i, :L] = x[k]
    return out
