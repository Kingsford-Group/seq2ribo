#!/usr/bin/env python
"""Test-set profile metrics per cell line in the layout of the paper's profile table.

Columns: TASEP, sTASEP (simulator runs averaged), Translatomer (sequence only), seq2ribo (polisher),
each where its result file exists:
  results/eval/profiles/<cell>/tasep.json            "simulator"
  results/eval/profiles/<cell>/seq2ribo_stasep.json "simulator"
  results/baselines/translatomer/<cell>/profiles.json "translatomer"
  results/eval/profiles/<cell>/seq2ribo_stasep.json  "polisher"
Rows: Tx-level r, Shape r, Elemwise r and the six MAE metrics after per-transcript scaling.
Output: results/eval/profiles/table.tsv and table.md
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

CELL_NAMES = {"ipsc": "iPSC", "hek293": "HEK293", "lcl": "LCL", "rpe": "RPE-1"}
ROWS = [("Tx-level r", "txlevel_r"), ("Shape r", "shape_r"), ("Elemwise r", "elemwise_r"),
        ("pertranscriptMAE", "sc_pertranscriptMAE"), ("elemwiseMAE", "sc_elemwiseMAE"),
        ("codonMAE", "sc_codonMAE"), ("pairMAE", "sc_pairMAE"), ("angleMAE", "sc_angleMAE"),
        ("bucketMAE", "sc_bucketMAE")]


def load(path, key):
    return json.load(open(path))[key] if path.exists() else None


def fmt(v, key):
    if v is None:
        return "-"
    return f"{v:.3f}" if key.endswith("_r") or key in ("sc_pertranscriptMAE", "sc_elemwiseMAE") else f"{v:,.0f}"


def main():
    ev, res = config.RESULTS / "eval" / "profiles", config.RESULTS
    methods = [("TASEP", lambda c: load(ev / c / "tasep.json", "simulator")),
               ("sTASEP", lambda c: load(ev / c / "seq2ribo_stasep.json", "simulator")),
               ("Translatomer (seq.)", lambda c: load(res / "baselines" / "translatomer" / c / "profiles.json", "translatomer")),
               ("seq2ribo", lambda c: load(ev / c / "seq2ribo_stasep.json", "polisher"))]
    tsv = ["cell_line\tmetric\t" + "\t".join(m for m, _ in methods)]
    md = ["| Cell line | Metric | " + " | ".join(m for m, _ in methods) + " |", "|---|---|" + "---|" * len(methods)]
    for c in config.CELL_LINES:
        got = [f(c) for _, f in methods]
        for i, (name, key) in enumerate(ROWS):
            vals = [g.get(key) if g else None for g in got]
            tsv.append(f"{c}\t{name}\t" + "\t".join("" if v is None else repr(v) for v in vals))
            md.append(f"| {CELL_NAMES[c] if i == 0 else ''} | {name} | " + " | ".join(fmt(v, key) for v in vals) + " |")
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "table.tsv").write_text("\n".join(tsv) + "\n")
    (ev / "table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
