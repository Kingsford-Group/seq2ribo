#!/usr/bin/env python3
"""
Merge per-cell-line RiboNN training files into a single multitask training file.

Produces two files:
  - data/training_multitask.txt        (full sequence: UTR5 + CDS + UTR3)
  - data/training_multitask_cds_only.txt (CDS only)

Each output has columns:
  tx_id, tx_size, utr5_size, utr3_size, cds_size, SYMBOL, tx_sequence,
  TE_HEK293, TE_LCL, TE_RPE-1, TE_IPSC
  (+ cds_hash for the CDS-only file)

Transcripts present in one cell line but absent in another get NaN for
the missing TE columns, which masked_mse_loss handles during training.
"""

import os
import argparse
import pandas as pd


DATASETS = {
    "HEK293": {"te_col": "TE_HEK293", "rename": None},
    "LCL":    {"te_col": "TE_LCL",    "rename": None},
    "RPE":    {"te_col": "TE_RPE-1",  "rename": None},
    "IPSC":   {"te_col": "mean_te",   "rename": "TE_IPSC"},
}

META_COLS = ["tx_id", "tx_size", "utr5_size", "utr3_size", "cds_size", "SYMBOL", "tx_sequence"]


def load_and_prepare(base_dir, cell_line, file_name, extra_drop=None):
    """Load a per-cell-line training file, rename TE column if needed,
    and return (df_meta, df_te) where df_te has only tx_id + canonical TE col."""
    info = DATASETS[cell_line]
    path = os.path.join(base_dir, cell_line, file_name)
    df = pd.read_csv(path, sep="\t")

    te_col = info["te_col"]
    target_col = info["rename"] if info["rename"] else te_col

    if info["rename"]:
        df = df.rename(columns={te_col: target_col})

    if extra_drop:
        df = df.drop(columns=[c for c in extra_drop if c in df.columns], errors="ignore")

    te_df = df[["tx_id", target_col]].copy()
    meta_df = df[[c for c in META_COLS if c in df.columns]].copy()
    meta_df = meta_df.drop_duplicates(subset=["tx_id"])

    return meta_df, te_df


def merge_all(base_dir, file_name, extra_drop=None):
    """Outer-join all cell lines on tx_id, combining TE columns."""
    all_meta = []
    te_frames = []

    for cell_line in DATASETS:
        meta, te = load_and_prepare(base_dir, cell_line, file_name, extra_drop=extra_drop)
        all_meta.append(meta)
        te_frames.append(te)

    meta_merged = pd.concat(all_meta, ignore_index=True).drop_duplicates(subset=["tx_id"])

    merged = meta_merged.copy()
    for te_df in te_frames:
        te_col = [c for c in te_df.columns if c != "tx_id"][0]
        merged = merged.merge(te_df[["tx_id", te_col]], on="tx_id", how="outer")

    # For transcripts that appeared only in TE frames but not in meta
    # (shouldn't happen, but be safe), fill meta from any available source
    merged = merged.drop_duplicates(subset=["tx_id"])

    return merged


def main():
    default_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Prepare multitask training data")
    parser.add_argument("--base_dir", default=default_base,
                        help="RiboNN project root (default: auto-detected)")
    args = parser.parse_args()

    base_dir = args.base_dir
    out_dir = os.path.join(base_dir, "data")
    os.makedirs(out_dir, exist_ok=True)

    # --- Full-sequence version ---
    print("Merging full-sequence training files...")
    full_df = merge_all(base_dir, "training.txt")
    full_out = os.path.join(out_dir, "training_multitask.txt")
    full_df.to_csv(full_out, sep="\t", index=False)
    te_cols = [c for c in full_df.columns if c.startswith("TE_")]
    non_null_counts = {c: full_df[c].notna().sum() for c in te_cols}
    print(f"  Wrote {full_out}")
    print(f"  Total transcripts: {len(full_df)}")
    print(f"  Non-null counts per target: {non_null_counts}")

    # --- CDS-only version ---
    print("\nMerging CDS-only training files...")
    cds_df = merge_all(base_dir, "training_cds_only.txt", extra_drop=["cds_hash"])
    cds_out = os.path.join(out_dir, "training_multitask_cds_only.txt")
    cds_df.to_csv(cds_out, sep="\t", index=False)
    te_cols = [c for c in cds_df.columns if c.startswith("TE_")]
    non_null_counts = {c: cds_df[c].notna().sum() for c in te_cols}
    print(f"  Wrote {cds_out}")
    print(f"  Total transcripts: {len(cds_df)}")
    print(f"  Non-null counts per target: {non_null_counts}")

    print("\nDone.")


if __name__ == "__main__":
    main()
