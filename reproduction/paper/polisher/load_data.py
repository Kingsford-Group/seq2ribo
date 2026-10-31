import pickle
from pathlib import Path

import numpy as np


def _subset_suffix(which: str) -> tuple[str, bool, float]:
    if which == "full":
        return "data.pkl", False, 1.0
    if which == "01":
        return "data_01.pkl", False, 1.0
    if which == "001":
        return "data_01.pkl", True, 10.0
    if which == "0001":
        return "data_01.pkl", True, 100.0
    raise ValueError("subset must be one of: full, 01, 001, 0001")


def _load_split(data_dir: Path, split: str, suffix: str):
    with (data_dir / f"{split}_{suffix}").open("rb") as f:
        split_data = pickle.load(f)
    cds_seqs = {tx: (seq, zeros) for tx, seq, asite_counts, zeros, geo_map in split_data}
    asite = {tx: (seq, asite_counts) for tx, seq, asite_counts, zeros, geo_map in split_data}
    geo_map = {tx: geo_map for tx, seq, asite_counts, zeros, geo_map in split_data}
    return cds_seqs, asite, geo_map


def _subsample(cds_seqs, asite, geo_map, factor: float):
    keys = list(cds_seqs.keys())
    np.random.seed(42)
    selected_keys = np.random.choice(keys, size=max(1, int(len(keys) / factor)), replace=False)
    cds_seqs = {k: cds_seqs[k] for k in selected_keys}
    asite = {k: asite[k] for k in selected_keys}
    geo_map = {k: geo_map[k] for k in selected_keys}
    return cds_seqs, asite, geo_map


def load_data(which: str, data_dir: str):
    suffix, divide, factor = _subset_suffix(which)
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"data dir not found: {root}")

    cds_seqs_train, asite_train, geo_map_train = _load_split(root, "train", suffix)
    cds_seqs_val, asite_val, geo_map_val = _load_split(root, "val", suffix)
    cds_seqs_test, asite_test, geo_map_test = _load_split(root, "test", suffix)

    if divide:
        cds_seqs_train, asite_train, geo_map_train = _subsample(cds_seqs_train, asite_train, geo_map_train, factor)
        cds_seqs_val, asite_val, geo_map_val = _subsample(cds_seqs_val, asite_val, geo_map_val, factor)
        cds_seqs_test, asite_test, geo_map_test = _subsample(cds_seqs_test, asite_test, geo_map_test, factor)

    return (
        (cds_seqs_train, asite_train, geo_map_train),
        (cds_seqs_val, asite_val, geo_map_val),
        (cds_seqs_test, asite_test, geo_map_test),
    )
