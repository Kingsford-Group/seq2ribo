"""Datasets of 65,536-bp windows for the sequence-only Translatomer (see windows.py for conventions)."""
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from windows import (Genome, cds_genomic_positions, onehot, target_bins, window_index, window_starts)


def load_cds(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def check_record(rec, entry):
    """The record's CDS and counts must match the CDS blocks of data/cds.pkl."""
    tx, seq, counts = rec[0], rec[1], rec[2]
    n_blocks = int(np.sum(np.asarray(entry["ends"]) - np.asarray(entry["starts"])))
    if entry["seq"] != seq or len(counts) != len(seq) or n_blocks != len(seq):
        raise ValueError(f"{tx}: record CDS/counts do not match data/cds.pkl")


class TrainWindows(Dataset):
    """One item per (transcript, window); target = this transcript's CDS A-site counts in 64-bp bins.

    Items are built lazily from (record index, window start) so that per-nt index arrays are not
    held in memory for ~10^5 windows.
    """

    def __init__(self, records, cds, genome: Genome):
        self.recs, self.entries, self.genome = [], [], genome
        self.samples = []
        for r in records:
            e = cds[r[0]]
            check_record(r, e)
            i = len(self.recs)
            self.recs.append(np.asarray(r[2], np.float32))
            self.entries.append(e)
            self.samples += [(i, ws) for ws in window_starts(e)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, n):
        i, ws = self.samples[n]
        e = self.entries[i]
        widx = window_index(cds_genomic_positions(e), e["strand"], ws)
        x = onehot(self.genome.window(e["chrom"], e["strand"], ws))
        return torch.from_numpy(x), torch.from_numpy(target_bins(widx, self.recs[i]))


class InferenceWindows(Dataset):
    """Window inputs only. specs: ("genomic", chrom, strand, ws) or ("sequence", dna_string)."""

    def __init__(self, specs, genome: Genome = None):
        self.specs, self.genome = specs, genome

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, n):
        s = self.specs[n]
        seq = self.genome.window(s[1], s[2], s[3]) if s[0] == "genomic" else s[1]
        return torch.from_numpy(onehot(seq))


@torch.no_grad()
def predict_bins(model, dataset, device, batch_size=32, num_workers=8):
    """[n_windows, 1024] predicted binned counts in dataset order (the model predicts log(1 + count))."""
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=True, persistent_workers=False)
    model.eval()
    out = [torch.expm1(model(x.to(device, non_blocking=True))).float().cpu().numpy() for x in dl]
    return np.concatenate(out) if out else np.zeros((0, 1024), np.float32)
