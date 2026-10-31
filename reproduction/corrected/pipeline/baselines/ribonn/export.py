#!/usr/bin/env python
"""Write RiboNN-format input tables for the clean RiboNN TE baseline.

Source: data/te/<cell>/labels.tsv (tx_id, split, te, utr5, utr3, utr_ok) and data/te/<cell>/{train,val,test}.pkl
(the CDS, start..stop). The transcript is utr5 + CDS + utr3. cds_only uses every transcript of a task,
utr_cds the transcripts with utr_ok (workbook UTRs matching the CDS, within RiboNN's length limits);
nothing else is filtered, and every transcript of a task is written and later predicted.

Output: <results-dir>/inputs/<setting>/<task>.tsv for setting in {utr_cds, cds_only} and task in
{ipsc, hek293, lcl, rpe, multitask}, with the columns upstream RiboNNDataModule reads
  tx_id, split (train/valid/test), utr5_size, cds_size, utr3_size, tx_size, tx_sequence (DNA alphabet),
  TE_<cell> (one column per individual task; four columns for multitask, NaN where a cell has no label)
  utr_cds : tx_sequence = 5'UTR + CDS + 3'UTR
  cds_only: tx_sequence = CDS, utr5_size = utr3_size = 0
The multitask table is the union of the four tasks' transcripts. Splits may differ between cell lines
(published per-cell-line splits), so every transcript gets the most restrictive of its splits
(test > val > train), extended to all transcripts with the same CDS; a cell line's label is kept only
where that cell line's split equals the row's split (rows left without labels are dropped). No CDS of
any cell line's val or test set is then in multitask training, and each cell line's multitask test
rows are exactly its own test set. The sequence of a transcript must agree across tasks (checked).
<results-dir>/inputs/summary.json holds counts.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ribonn_lib as rl  # noqa: E402
from ribonn_lib import config  # noqa: E402

COLUMNS = ["tx_id", "split", "utr5_size", "cds_size", "utr3_size", "tx_size", "tx_sequence"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=config.DATA)
    p.add_argument("--results-dir", type=Path, default=rl.RESULTS_DIR)
    return p.parse_args()


def load_task(data_dir, cell):
    """One row per transcript: tx_id, split, utr5, cds, utr3 (RNA alphabet), te."""
    lab = pd.read_csv(data_dir / "te" / cell / "labels.tsv", sep="\t", keep_default_na=False,
                      dtype={"tx_id": str, "split": str, "utr5": str, "utr3": str, "te": float})
    lab["utr_ok"] = lab["utr_ok"].astype(str).str.lower() == "true"
    cds = {}
    for split in ("train", "val", "test"):
        with open(data_dir / "te" / cell / f"{split}.pkl", "rb") as f:
            for r in pickle.load(f):
                cds[r[0]] = (split, r[1])
    if set(cds) != set(lab.tx_id) or lab.tx_id.duplicated().any():
        raise ValueError(f"{cell}: labels.tsv and the pkl records do not hold the same transcripts")
    if any(cds[t][0] != s for t, s in zip(lab.tx_id, lab.split)):
        raise ValueError(f"{cell}: split in labels.tsv disagrees with the pkl file")
    lab["cds"] = lab.tx_id.map(lambda t: cds[t][1])
    if lab.te.isna().any():
        raise ValueError(f"{cell}: missing TE labels")
    return lab


def ribonn_rows(df, setting):
    """RiboNN columns for one setting (sequence converted to the DNA alphabet RiboNN encodes)."""
    u5 = df.utr5 if setting == "utr_cds" else pd.Series("", index=df.index)
    u3 = df.utr3 if setting == "utr_cds" else pd.Series("", index=df.index)
    seq = (u5 + df.cds + u3).str.upper().str.replace("U", "T")
    out = pd.DataFrame({
        "tx_id": df.tx_id,
        "split": df.split.map(rl.SPLIT_NAMES),
        "utr5_size": u5.str.len(),
        "cds_size": df.cds.str.len(),
        "utr3_size": u3.str.len(),
        "tx_size": seq.str.len(),
        "tx_sequence": seq,
    })
    # the checks upstream RiboNNDataModule makes, plus the alphabet DataFrameDataset needs
    assert (out.cds_size % 3 == 0).all(), "CDS length not a multiple of 3"
    cds_dna = df.cds.str.replace("U", "T")
    assert cds_dna.str[-3:].isin(["TAA", "TGA", "TAG"]).all(), "CDS without a stop codon"
    assert seq.str.fullmatch("[ACGT]+").all(), "non-ACGT characters"
    rl.check_geometry(out, setting)
    return out


def multitask_union(tasks):
    """Union over tasks with leakage-free splits (see module docstring) and per-cell labels."""
    allrows = pd.concat([d.assign(cell=c) for c, d in tasks.items()], ignore_index=True)
    per_tx = allrows.groupby("tx_id")[["utr5", "cds", "utr3"]].nunique()
    if (per_tx > 1).any().any():
        raise ValueError("a transcript has different sequences across tasks")
    rank = {"train": 0, "val": 1, "test": 2}
    allrows["rank"] = allrows.split.map(rank)
    tx_rank = allrows.groupby("tx_id")["rank"].max()
    union = allrows.drop_duplicates("tx_id")[["tx_id", "utr5", "cds", "utr3"]].copy()
    union["rank"] = union.tx_id.map(tx_rank)
    union["rank"] = union.groupby("cds")["rank"].transform("max")       # same CDS -> same split
    union["split"] = union["rank"].map({v: k for k, v in rank.items()})
    for c, d in tasks.items():
        d = d.set_index("tx_id")
        keep = union.tx_id.map(d.split) == union.split
        union[rl.LABEL_COLUMNS[c]] = union.tx_id.map(d.te).where(keep)
    labels = [rl.LABEL_COLUMNS[c] for c in rl.TASKS]
    n_unlabeled = int(union[labels].isna().all(axis=1).sum())
    union = union[union[labels].notna().any(axis=1)]
    union = union.drop(columns="rank").sort_values(["split", "tx_id"]).reset_index(drop=True)
    for c, d in tasks.items():   # each cell line's multitask test rows are its own test set
        got = set(union.tx_id[(union.split == "test") & union[rl.LABEL_COLUMNS[c]].notna()])
        if got != set(d.tx_id[d.split == "test"]):
            raise ValueError(f"{c}: multitask test rows differ from the task's test set")
    cross = union.groupby("cds").split.nunique()   # CDS shared across splits would leak
    if (cross > 1).any():
        raise ValueError(f"{int((cross > 1).sum())} CDS occur in more than one split")
    return union, labels, n_unlabeled


def main():
    a = parse_args()
    all_tasks = {c: load_task(a.data_dir, c) for c in rl.TASKS}
    summary = {"source": str(a.data_dir / "te"), "tasks": {}, "multitask": {}}
    for setting in rl.SETTINGS:
        # CDS+UTR uses the transcripts with workbook UTRs that match the CDS (utr_ok), CDS-only all
        tasks = {c: (d[d.utr_ok] if setting == "utr_cds" else d) for c, d in all_tasks.items()}
        union, labels, n_unlabeled = multitask_union(tasks)
        out_dir = a.results_dir / "inputs" / setting
        out_dir.mkdir(parents=True, exist_ok=True)
        for c, d in tasks.items():
            d = d.sort_values(["split", "tx_id"]).reset_index(drop=True)
            t = ribonn_rows(d, setting)
            t[rl.LABEL_COLUMNS[c]] = d.te.values
            t.to_csv(rl.table_path(a.results_dir, setting, c), sep="\t", index=False)
            n = t.split.value_counts().to_dict()
            dup = {s: int(g.tx_sequence.duplicated().sum()) for s, g in t.groupby("split")}
            summary["tasks"].setdefault(c, {})[setting] = {
                "n": n, "duplicate_sequences_within_split": dup,
                "max_utr5": int(t.utr5_size.max()), "max_cds_utr3": int((t.tx_size - t.utr5_size).max())}
            print(f"[{setting} {c}] {n}  duplicate sequences within split {dup}")
        t = ribonn_rows(union, setting)
        t[labels] = union[labels].values
        t.to_csv(rl.table_path(a.results_dir, setting, "multitask"), sep="\t", index=False)
        n = t.split.value_counts().to_dict()
        obs = {s: {c: int(g[rl.LABEL_COLUMNS[c]].notna().sum()) for c in rl.TASKS}
               for s, g in t.groupby("split")}
        missing = float(np.isnan(t[labels].to_numpy()).mean())
        summary["multitask"][setting] = {"n": n, "labels_observed": obs, "fraction_missing": missing,
                                         "rows_without_label_dropped": n_unlabeled}
        print(f"[{setting} multitask] {n}  missing label fraction {missing:.3f}")
    json.dump(summary, open(a.results_dir / "inputs" / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
