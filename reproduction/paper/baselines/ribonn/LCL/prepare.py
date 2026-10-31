#!/usr/bin/env python3
import argparse, pickle, re, sys
from pathlib import Path
import pandas as pd
import numpy as np

LIMIT_5UTR = 1381          # RiboNN will drop inputs with 5'UTR > 1381
LIMIT_CDS_UTR3 = 11937     # RiboNN will drop inputs with CDS+3'UTR > 11937

PAPER = Path(__file__).resolve().parents[3]  # reproduction/paper


def load_test_keys(pkl_path, strip_version=True):
    with open(pkl_path, "rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, dict):
        keys = list(obj.keys())
    elif isinstance(obj, (list, set, tuple)):
        keys = list(obj)
    else:
        raise ValueError("Unsupported pickle type for asite_tests; expected dict or list-like of IDs")
    keys = [str(k) for k in keys]
    if strip_version:
        stripped = {re.sub(r"\.\d+$", "", k) for k in keys}
        return set(keys), stripped
    return set(keys), set()

def normalize_tx_id(tx_id, strip_version=True):
    tx_id = str(tx_id)
    return re.sub(r"\.\d+$", "", tx_id) if strip_version else tx_id

def split_sequences(df, to_rna=True):
    # Expect columns: tx_sequence, utr5_size, cds_size, utr3_size
    def _split(row):
        seq = str(row["tx_sequence"]).upper().replace(" ", "")
        if to_rna:
            seq = seq.replace("T", "U")
        u5, cds, u3 = int(row["utr5_size"]), int(row["cds_size"]), int(row["utr3_size"])
        # If sizes do not add up, try to fix utr3_size from total length
        if u5 + cds + u3 != len(seq):
            u3 = max(0, len(seq) - (u5 + cds))
        utr5_seq = seq[:u5]
        cds_seq  = seq[u5:u5+cds]
        utr3_seq = seq[u5+cds:u5+cds+u3]
        return pd.Series({"utr5_sequence": utr5_seq, "cds_sequence": cds_seq, "utr3_sequence": utr3_seq,
                          "len_utr5": len(utr5_seq), "len_cds": len(cds_seq), "len_utr3": len(utr3_seq)})
    return df.apply(_split, axis=1)

def main():
    ap = argparse.ArgumentParser(description="Prepare RiboNN prediction_input.txt from Excel + asite_tests.pkl")
    ap.add_argument("--xlsx", default=str(PAPER / "TE" / "41587_2025_2712_MOESM3_ESM.xlsx"))
    ap.add_argument("--asite_tests", default=str(PAPER / "data" / "lcl" / "test_data.pkl"))
    ap.add_argument("--sheet", default="Human")
    ap.add_argument("--out_dir", default=".", help="Output directory")
    ap.add_argument("--no_to_rna", action="store_true", help="Do not convert T->U")
    ap.add_argument("--keep_version", action="store_true", help="Match tx_id with version suffix exactly")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load test keys
    # test_keys_raw, test_keys_stripped = load_test_keys(args.asite_tests, strip_version=not args.keep_version)

    with open(args.asite_tests, "rb") as fh:
        obj = pickle.load(fh)

    test_dict = {}

    def _collect_ndarrays(x):
        out = []
        if isinstance(x, np.ndarray):
            out.append(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                out.extend(_collect_ndarrays(y))
        return out

    strip_version = not args.keep_version

    if isinstance(obj, dict):
        for k, v in obj.items():
            txid = str(k)
            # try to extract sequence and arrays from the value
            seq = None
            arrays = []
            if isinstance(v, tuple) or isinstance(v, list):
                # first string-like element is candidate for seq
                for part in v:
                    if seq is None and isinstance(part, str):
                        seq = part
                    # collect arrays from remaining parts
                    arrays.extend(_collect_ndarrays(part))
            elif isinstance(v, str):
                seq = v
            else:
                # try stringify fallback
                try:
                    seq = str(v)
                except Exception:
                    seq = None

            if seq is None:
                # skip entries without a usable sequence
                continue

            seq = seq.upper().replace(" ", "")
            if not args.no_to_rna:
                seq = seq.replace("T", "U")

            # choose best array: prefer int 1D arrays of same length
            sel = None
            for a in arrays:
                if a.ndim == 1 and a.shape[0] == len(seq) and np.issubdtype(a.dtype, np.integer):
                    sel = a
                    break
            if sel is None:
                for a in arrays:
                    if a.ndim == 1 and a.shape[0] == len(seq):
                        sel = a
                        break
            if sel is None:
                sel = np.zeros(len(seq), dtype=np.int32)

            test_dict[txid] = (seq, sel.astype(np.int32))

    else:
        # assume iterable of records like (txid, seq, arr1, arr2, ...)
        for rec in obj:
            try:
                txid = str(rec[0])
                seq = str(rec[1])
            except Exception:
                continue
            seq = seq.upper().replace(" ", "")
            if not args.no_to_rna:
                seq = seq.replace("T", "U")

            arrays = []
            for part in rec[2:]:
                arrays.extend(_collect_ndarrays(part))

            sel = None
            for a in arrays:
                if a.ndim == 1 and a.shape[0] == len(seq) and np.issubdtype(a.dtype, np.integer):
                    sel = a
                    break
            if sel is None:
                for a in arrays:
                    if a.ndim == 1 and a.shape[0] == len(seq):
                        sel = a
                        break
            if sel is None:
                sel = np.zeros(len(seq), dtype=np.int32)

            test_dict[txid] = (seq, sel.astype(np.int32))

    # create the two variables the rest of the script expects
    test_keys_raw = set(test_dict.keys())
    test_keys_stripped = {re.sub(r"\.\d+$", "", k) for k in test_keys_raw}

    print(f"Loaded {len(test_dict)} records from {args.asite_tests}")
    # optional quick sanity check
    if test_dict:
        example_k = next(iter(test_dict))
        print("Example key:", example_k, "seq_len:", len(test_dict[example_k][0]))
    # Load Excel
    usecols = ["tx_id","tx_sequence","utr5_size","cds_size","utr3_size","TE_LCL"]
    df = pd.read_excel(args.xlsx, sheet_name=args.sheet, dtype=str)
    # coerce numeric for sizes
    for c in ["utr5_size","cds_size","utr3_size"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    missing = [c for c in usecols if c not in df.columns]
    if missing:
        sys.exit(f"Missing required columns in Excel: {missing}")

    # Filter to test set
    df["tx_id"] = df["tx_id"].astype(str)
    df["tx_id_novers"] = df["tx_id"].apply(lambda x: normalize_tx_id(x, strip_version=True))
    if args.keep_version:
        mask = df["tx_id"].isin(test_keys_raw)
    else:
        mask = df["tx_id"].isin(test_keys_raw) | df["tx_id_novers"].isin(test_keys_stripped)
    df = df.loc[mask].copy()
    if df.empty:
        sys.exit("After filtering, zero transcripts matched the test set keys")

    # Drop rows with missing sizes or sequences
    df = df.dropna(subset=["tx_sequence","utr5_size","cds_size","utr3_size"]).copy()

    # Split sequences
    pieces = split_sequences(df, to_rna=not args.no_to_rna)
    df = pd.concat([df.reset_index(drop=True), pieces.reset_index(drop=True)], axis=1)

    # Enforce RiboNN limits up front to avoid surprises
    df["limit_5utr"] = df["len_utr5"] > LIMIT_5UTR
    df["limit_cds_utr3"] = (df["len_cds"] + df["len_utr3"]) > LIMIT_CDS_UTR3
    excluded = df.loc[df["limit_5utr"] | df["limit_cds_utr3"], ["tx_id","len_utr5","len_cds","len_utr3"]]
    if not excluded.empty:
        excluded.to_csv(out_dir / "excluded_by_ribonn_limits.tsv", sep="\t", index=False)
    df_ok = df.loc[~(df["limit_5utr"] | df["limit_cds_utr3"])].copy()

    # Deduplicate by tx_id keeping first occurrence
    df_ok = df_ok.drop_duplicates(subset=["tx_id"], keep="first")

    # Write RiboNN prediction input
    pred_path = out_dir / "prediction_input.txt"
    df_ok[["tx_id","utr5_sequence","cds_sequence","utr3_sequence"]].to_csv(
        pred_path, sep="\t", index=False
    )

    # Also write ground truth for convenience
    gt_path = out_dir / "test_mean_te.tsv"
    df_ok[["tx_id","TE_LCL"]].to_csv(gt_path, sep="\t", index=False)

    # Log summary
    print(f"Matched test set transcripts: {len(df_ok)}")
    if not excluded.empty:
        print(f"Excluded by RiboNN limits: {len(excluded)}  -> saved to excluded_by_ribonn_limits.tsv")
    print(f"Wrote RiboNN input: {pred_path}")
    print(f"Wrote ground truth: {gt_path}")

if __name__ == "__main__":
    main()
