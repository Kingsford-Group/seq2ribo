#!/usr/bin/env python3
import argparse, pickle, re, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

# RiboNN length gates (mirrors their filtering in prediction)
LIMIT_5UTR = 1381
LIMIT_CDS_UTR3 = 11937

PAPER = Path(__file__).resolve().parents[3]  # reproduction/paper


def load_keys_from_pkl(pkl_path, strip_version=True):
    with open(pkl_path, "rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, dict):
        keys = list(obj.keys())
    elif isinstance(obj, (list, set, tuple)):
        keys = list(obj)
    else:
        raise ValueError("Unsupported pickle content; expected dict or list-like of transcript IDs")
    keys = [str(k) for k in keys]
    if strip_version:
        novers = {re.sub(r"\.\d+$", "", k) for k in keys}
        return set(keys), novers
    return set(keys), set()

def strip_ver(txid: str) -> str:
    s = str(txid)
    if "." in s:
        base, maybe = s.rsplit(".", 1)
        return base if maybe.isdigit() else s
    return s

def to_rna(seq: str) -> str:
    return str(seq).upper().replace(" ", "").replace("T", "U")



def main():
    ap = argparse.ArgumentParser(description="Prepare RiboNN val.txt (single TE column = TE_RPE-1)")
    ap.add_argument("--xlsx", default=str(PAPER / "TE" / "41587_2025_2712_MOESM3_ESM.xlsx"))
    ap.add_argument("--val_pkl", default=str(PAPER / "data" / "rpe" / "val_data.pkl"))
    ap.add_argument("--sheet", default="Human")
    ap.add_argument("--out_dir", default=".", help="Where to write val.txt")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load training keys
    with open(args.val_pkl, "rb") as fh:
        obj = pickle.load(fh)

    # Build a dict mapping txid -> (tx_sequence, per-position int array)
    val_dict = {}

    def _collect_ndarrays(x):
        """Recursively collect numpy arrays from nested tuples/lists."""
        out = []
        if isinstance(x, np.ndarray):
            out.append(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                out.extend(_collect_ndarrays(y))
        return out

    if isinstance(obj, dict):
        # already in desired form (assume values are (seq, arr) or similar)
        for k, v in obj.items():
            # attempt to normalize to (seq, int-array)
            if isinstance(v, tuple) and len(v) >= 2 and isinstance(v[0], str):
                seq = to_rna(v[0])
                # find first ndarray of length == len(seq) and integer dtype
                arrays = _collect_ndarrays(v[1:])
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
                val_dict[str(k)] = (seq, sel.astype(np.int32))
            else:
                # fallback: store sequence-like string if possible, else skip
                try:
                    seq = to_rna(str(v))
                    val_dict[str(k)] = (seq, np.zeros(len(seq), dtype=np.int32))
                except Exception:
                    continue
    else:
        # assume obj is an iterable of records like (txid, seq, arr1, arr2, ...)
        for rec in obj:
            try:
                txid = str(rec[0])
                seq = to_rna(rec[1])
            except Exception:
                # unexpected record format; skip
                continue

            # collect arrays from the rest of the record (handles nested tuples)
            arrays = []
            for part in rec[2:]:
                arrays.extend(_collect_ndarrays(part))

            # prefer an integer 1D array with same length as sequence
            sel = None
            for a in arrays:
                if a.ndim == 1 and a.shape[0] == len(seq) and np.issubdtype(a.dtype, np.integer):
                    sel = a
                    break
            # otherwise accept any 1D array with matching length
            if sel is None:
                for a in arrays:
                    if a.ndim == 1 and a.shape[0] == len(seq):
                        sel = a
                        break
            # fallback: zeros
            if sel is None:
                sel = np.zeros(len(seq), dtype=np.int32)

            val_dict[txid] = (seq, sel.astype(np.int32))

    # Provide the same two variables downstream code expects
    val_keys_raw = set(val_dict.keys())
    val_keys_novers = {re.sub(r"\.\d+$", "", k) for k in val_keys_raw}

    print(f"Loaded {len(val_dict)} records from {args.val_pkl}")
    # ---- Load Excel
    df = pd.read_excel(args.xlsx, sheet_name=args.sheet, dtype=str)

    # Coerce numeric for sizes & TE_RPE-1
    for c in ["utr5_size", "cds_size", "utr3_size", "tx_size", "TE_RPE-1"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Required columns (we will compute tx_size if missing)
    req = ["tx_id", "SYMBOL", "tx_sequence", "utr5_size", "cds_size", "utr3_size", "TE_RPE-1"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        sys.exit(f"Missing required columns in Excel: {missing}")

    # ---- Filter to training set keys (version-insensitive)
    df["tx_id"] = df["tx_id"].astype(str)
    df["key"] = df["tx_id"].map(strip_ver)
    mask = df["tx_id"].isin(val_keys_raw) | df["key"].isin(val_keys_novers)
    df = df.loc[mask].copy()
    if df.empty:
        sys.exit("No val transcripts matched between Excel and val_pkl keys.")

    # ---- Clean sequences (T->U) and recompute tx_size if needed
    df["tx_sequence"] = df["tx_sequence"].map(to_rna)
    # If sizes don't sum to len(seq), fix utr3_size from total length
    def _fix_sizes(row):
        u5 = int(row["utr5_size"]) if pd.notna(row["utr5_size"]) else 0
        cds = int(row["cds_size"]) if pd.notna(row["cds_size"]) else 0
        u3 = int(row["utr3_size"]) if pd.notna(row["utr3_size"]) else 0
        L = len(row["tx_sequence"])
        if u5 + cds + u3 != L:
            # try to fix utr3 as the residual; keep u5 and cds as given
            u3 = max(0, L - (u5 + cds))
        tx_size = L
        return pd.Series({"utr5_size": u5, "cds_size": cds, "utr3_size": u3, "tx_size": tx_size})

    fixed = df.apply(_fix_sizes, axis=1)
    for c in ["utr5_size", "cds_size", "utr3_size", "tx_size"]:
        df[c] = fixed[c].astype(int)

    # ---- Enforce RiboNN length limits; save excluded
    df["limit_5utr"] = df["utr5_size"] > LIMIT_5UTR
    df["limit_cds_utr3"] = (df["cds_size"] + df["utr3_size"]) > LIMIT_CDS_UTR3
    excluded = df.loc[df["limit_5utr"] | df["limit_cds_utr3"],
                      ["tx_id", "utr5_size", "cds_size", "utr3_size", "tx_size"]]
    if not excluded.empty:
        excluded.to_csv(out_dir / "excluded_by_ribonn_limits.tsv", sep="\t", index=False)
    df = df.loc[~(df["limit_5utr"] | df["limit_cds_utr3"])].copy()

    # ---- Drop rows without numeric TE_RPE-1 or empty sequences
    df = df.dropna(subset=["TE_RPE-1", "tx_sequence"]).copy()



    # ---- Deduplicate by tx_id (keep first)
    df = df.drop_duplicates(subset=["tx_id"], keep="first")

    # ---- Final column order (single TE column = TE_RPE-1)
    out_cols = [
        "tx_id", "tx_size", "utr5_size", "utr3_size", "cds_size",
        "SYMBOL", "tx_sequence", "TE_RPE-1"
    ]
    missing_final = [c for c in out_cols if c not in df.columns]
    if missing_final:
        sys.exit(f"Unexpected missing columns just before write: {missing_final}")

    out_path = out_dir / "val.txt"
    df[out_cols].to_csv(out_path, sep="\t", index=False)

    # ---- Summary
    kept = len(df)
    print(f"Wrote {kept} val rows to {out_path}")
    if not excluded.empty:
        print(f"Excluded by RiboNN limits: {len(excluded)} (saved to excluded_by_ribonn_limits.tsv)")

if __name__ == "__main__":
    main()


