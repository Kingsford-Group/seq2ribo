#!/usr/bin/env python
"""TE datasets: the published TE sets (Zheng et al. 2025 labels) with corrected counts.

Task per cell line: ipsc -> mean_te, hek293 -> TE_HEK293, lcl -> TE_LCL, rpe -> TE_RPE-1.
Transcripts and train/val/test come from the published TE sets (config.TE_SPLITS,
reproduction/splits/te_<cell>.tsv.gz: tx_id = the profile transcript, workbook_tx_id = the workbook
transcript, which may differ in its version suffix); labels come from the workbook. Rows without a
label are dropped. As for the profile evaluation, validation
transcripts whose CDS also occurs in TE training, and test transcripts whose CDS occurs in TE
training or validation, are removed.
CDS, A-site counts (corrected mapping) and structure features as in build_dataset.py; 5'/3'UTRs
from the workbook. utr_ok marks transcripts usable with UTRs: the workbook CDS equals the CDS and
the transcript is within RiboNN's input limits (5'UTR <= 1381 nt, CDS + 3'UTR <= 11937 nt); the
CDS+UTR head and RiboNN's CDS+UTR setting use these, the CDS-only methods use all transcripts.
Outputs in data/te/<cell>/:
  {train,val,test}.pkl  records (tx_id, cds_rna, asite_counts, angle_sum, pair_count)
  labels.tsv            tx_id, split, te, utr5, utr3, utr_ok
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from seq2ribo.geometry import compute_geometry_features  # noqa: E402

MAX_UTR5 = 1381
MAX_CDS_UTR3 = 11937
SPLITS = ("train", "val", "test")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell-lines", nargs="+", default=config.CELL_LINES)
    p.add_argument("--data-dir", type=Path, default=config.DATA)
    return p.parse_args()


def main():
    a = parse_args()
    cds = pickle.load(open(a.data_dir / "cds.pkl", "rb"))              # published CDS models
    cds_src = pickle.load(open(config.SHARED / "cds.pkl", "rb"))       # annotated CDS (counts are indexed on it)
    wb = pd.read_csv(config.RAW / "zheng2025_te.tsv", sep="\t", dtype={"tx_id": str})
    wb["key"] = wb["tx_id"].str.split(".").str[0]
    wb = wb.drop_duplicates("key").set_index("key")
    for cl in a.cell_lines:
        col = config.TE_COLUMNS[cl]
        counts = pickle.load(open(config.SHARED / cl / "asite_counts.pkl", "rb"))
        pub = pd.read_csv(config.TE_SPLITS[cl], sep="\t")
        lab_col = wb[col]
        pub[col] = pub["workbook_tx_id"].str.split(".").str[0].map(lab_col)
        st = {"published": pub.split.value_counts().to_dict(), "no_label": int(pub[col].isna().sum())}
        pub = pub[pub[col].notna()].assign(label=lambda d: d[col].astype(float))
        rows = {s: pub[pub.split == s] for s in SPLITS}
        seq_of = lambda t: cds[t]["seq"]
        train_cds = {seq_of(t) for t in rows["train"].tx_id}
        rows["val"] = rows["val"][~rows["val"].tx_id.map(seq_of).isin(train_cds)]
        val_cds = {seq_of(t) for t in rows["val"].tx_id}
        rows["test"] = rows["test"][~rows["test"].tx_id.map(seq_of).isin(train_cds | val_cds)]
        st["removed_identical_cds"] = {s: int((pub.split == s).sum() - len(rows[s])) for s in ("val", "test")}
        out = a.data_dir / "te" / cl
        out.mkdir(parents=True, exist_ok=True)
        labels = []
        for s in SPLITS:
            recs = []
            for r in rows[s].itertuples():
                tx, seq = r.tx_id, seq_of(r.tx_id)
                off = len(cds_src[tx]["seq"]) - len(seq)
                c = np.asarray(counts[tx])[off:]
                ang, pair = compute_geometry_features(seq, cache_dir=config.SHARED / "geometry_cache")
                recs.append((tx, seq, c, np.asarray(ang, np.float64), np.asarray(pair, np.int64)))
                w = wb.loc[str(r.workbook_tx_id).split(".")[0]]
                u5, cs, u3 = int(w["utr5_size"]), int(w["cds_size"]), int(w["utr3_size"])
                full = str(w["tx_sequence"]).upper().replace("T", "U")
                ok = full[u5:u5 + cs] == seq and u5 <= MAX_UTR5 and cs + u3 <= MAX_CDS_UTR3
                labels.append({"tx_id": tx, "split": s, "te": r.label,
                               "utr5": full[:u5] if ok else "", "utr3": full[u5 + cs:u5 + cs + u3] if ok else "",
                               "utr_ok": bool(ok)})
            with open(out / f"{s}.pkl", "wb") as f:
                pickle.dump(recs, f)
        df = pd.DataFrame(labels)
        df.to_csv(out / "labels.tsv", sep="\t", index=False)
        n = df.split.value_counts().to_dict()
        nu = df[df.utr_ok].split.value_counts().to_dict()
        print(f"[{cl} {col}] published {st['published']}  no label {st['no_label']}  removed identical CDS "
              f"{st['removed_identical_cds']}  -> train {n.get('train', 0)} val {n.get('val', 0)} test {n.get('test', 0)}  "
              f"(with UTRs: train {nu.get('train', 0)} val {nu.get('val', 0)} test {nu.get('test', 0)})", flush=True)


if __name__ == "__main__":
    main()
