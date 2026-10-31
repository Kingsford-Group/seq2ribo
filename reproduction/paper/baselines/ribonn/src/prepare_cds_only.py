#!/usr/bin/env python3
"""
Prepare CDS-only versions of RiboNN training and prediction input files.

For training files (tx_sequence column = UTR5+CDS+UTR3):
  - Extract CDS portion using utr5_size and cds_size
  - Set tx_sequence = CDS only, utr5_size=0, utr3_size=0, tx_size=cds_size

For prediction input files (separate utr5_sequence, cds_sequence, utr3_sequence):
  - Keep only tx_id and cds_sequence (renamed to cds_sequence)
  - Set utr5_sequence="" (empty), utr3_sequence="" (empty)

Also applies train/val/test CDS-hash filtering to remove leaking transcripts.
"""
import os
import sys
import hashlib
import pandas as pd
import numpy as np


def cds_hash(cds_seq: str) -> str:
    """MD5 hash of a CDS sequence (uppercased, U->T)."""
    cds = cds_seq.strip().upper().replace("U", "T")
    return hashlib.md5(cds.encode("utf-8")).hexdigest()


def make_cds_only_training(input_path: str, output_path: str):
    """Convert a training.txt (with tx_sequence) to CDS-only format."""
    df = pd.read_csv(input_path, sep="\t")
    
    # Extract CDS from tx_sequence
    df["tx_sequence"] = df["tx_sequence"].str.strip().str.upper().str.replace("U", "T")
    df["cds_only"] = df.apply(
        lambda row: row["tx_sequence"][int(row["utr5_size"]):int(row["utr5_size"]) + int(row["cds_size"])],
        axis=1
    )
    
    # Overwrite columns
    df["tx_sequence"] = df["cds_only"]
    df["tx_size"] = df["cds_size"]
    df["utr5_size"] = 0
    df["utr3_size"] = 0
    df.drop(columns=["cds_only"], inplace=True)
    
    # Add CDS hash for filtering
    df["cds_hash"] = df["tx_sequence"].apply(cds_hash)
    
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Created CDS-only training file: {output_path} ({len(df)} transcripts)")
    return df


def make_cds_only_prediction(input_path: str, output_path: str):
    """Convert prediction_input.txt (with separate UTR/CDS columns) to CDS-only."""
    df = pd.read_csv(input_path, sep="\t")
    
    # Keep cds_sequence only
    df["cds_sequence"] = df["cds_sequence"].str.strip().str.upper().str.replace("U", "T")
    df["utr5_sequence"] = ""
    df["utr3_sequence"] = ""
    
    # Add CDS hash for filtering
    df["cds_hash"] = df["cds_sequence"].apply(cds_hash)
    
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Created CDS-only prediction file: {output_path} ({len(df)} transcripts)")
    return df


def filter_test_by_cds_leakage(train_df, test_tsv_path, pred_df, output_test_path, output_pred_path, gt_col):
    """
    Filter test samples removing any whose CDS hash matches training CDS hashes.
    
    train_df: training DataFrame with 'cds_hash' column
    test_tsv_path: path to test_mean_te.tsv (tx_id + TE column)
    pred_df: prediction input DataFrame with 'cds_hash' and 'tx_id'
    output_test_path: filtered test TSV output
    output_pred_path: filtered prediction input output
    gt_col: name of the ground truth TE column
    """
    train_hashes = set(train_df["cds_hash"].unique())
    
    # Load test ground truth
    test_gt = pd.read_csv(test_tsv_path, sep="\t")
    
    # Strip version from tx_ids for matching
    def strip_ver(tx):
        s = str(tx)
        if "." in s:
            base, maybe = s.rsplit(".", 1)
            return base if maybe.isdigit() else s
        return s
    
    # pred_df has tx_id and cds_hash
    pred_df = pred_df.copy()
    pred_df["tx_id_base"] = pred_df["tx_id"].apply(strip_ver)
    test_gt["tx_id_base"] = test_gt["tx_id"].apply(strip_ver)
    
    # Mark pred samples that leak into train
    pred_df["in_train"] = pred_df["cds_hash"].isin(train_hashes)
    leaking_tx_bases = set(pred_df.loc[pred_df["in_train"], "tx_id_base"])
    
    # Filter test GT
    test_gt_filtered = test_gt[~test_gt["tx_id_base"].isin(leaking_tx_bases)].drop(columns=["tx_id_base"])
    
    # Filter prediction input
    pred_filtered = pred_df[~pred_df["in_train"]].drop(columns=["tx_id_base", "in_train"])
    
    n_removed = len(test_gt) - len(test_gt_filtered)
    print(f"  Leakage filter: removed {n_removed} test samples (from {len(test_gt)} to {len(test_gt_filtered)})")
    
    test_gt_filtered.to_csv(output_test_path, sep="\t", index=False)
    pred_filtered.to_csv(output_pred_path, sep="\t", index=False)
    
    return test_gt_filtered, pred_filtered


def main():
    base_dir = os.environ.get("RIBONN_ROOT", ".")
    
    datasets = {
        "HEK293": {"target": "TE_HEK293"},
        "LCL":    {"target": "TE_LCL"},
        "RPE":    {"target": "TE_RPE-1"},
        "IPSC":   {"target": "mean_te"},
    }
    
    for dataset, info in datasets.items():
        print(f"\n{'='*60}")
        print(f"Processing {dataset}...")
        print(f"{'='*60}")
        
        ds_dir = os.path.join(base_dir, dataset)
        
        # 1. Create CDS-only training file
        train_path = os.path.join(ds_dir, "training.txt")
        train_cds_path = os.path.join(ds_dir, "training_cds_only.txt")
        train_df = make_cds_only_training(train_path, train_cds_path)
        
        # 2. Create CDS-only prediction input file
        pred_path = os.path.join(ds_dir, "prediction_input.txt")
        pred_cds_path = os.path.join(ds_dir, "prediction_input_cds_only.txt")
        pred_df = make_cds_only_prediction(pred_path, pred_cds_path)
        
        # 3. Filter test samples by CDS leakage
        test_path = os.path.join(ds_dir, "test_mean_te.tsv")
        if os.path.exists(test_path):
            filtered_test_path = os.path.join(ds_dir, "test_mean_te_cds_filtered.tsv")
            filtered_pred_path = os.path.join(ds_dir, "prediction_input_cds_only_filtered.txt")
            filter_test_by_cds_leakage(
                train_df, test_path, pred_df,
                filtered_test_path, filtered_pred_path,
                info["target"]
            )
        else:
            print(f"  No test file found at {test_path}, skipping leakage filter.")
    
    print("\nDone! CDS-only files created for all datasets.")


if __name__ == "__main__":
    main()
