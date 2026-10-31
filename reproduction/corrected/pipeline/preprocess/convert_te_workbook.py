#!/usr/bin/env python
"""Convert the Zheng et al. 2025 TE workbook (Supplementary Table, xlsx) to a TSV (requires openpyxl)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

src = config.RAW / "41587_2025_2712_MOESM3_ESM.xlsx"
df = pd.read_excel(src, converters={"tx_id": str, "gene_id": str})
df.columns = df.columns.str.strip()
keep = ["tx_id", "gene_id", "SYMBOL", "tx_size", "utr5_size", "cds_size", "utr3_size", "tx_sequence"] + list(config.TE_COLUMNS.values())
df[keep].to_csv(config.RAW / "zheng2025_te.tsv", sep="\t", index=False)
print(f"{len(df)} rows -> {config.RAW / 'zheng2025_te.tsv'}")
