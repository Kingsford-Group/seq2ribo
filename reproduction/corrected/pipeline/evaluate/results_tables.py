#!/usr/bin/env python
"""Result tables of UPDATE.md (test sets) in the layout of the paper's main-text tables.

Profiles   TASEP, sTASEP, Translatomer (seq.), seq2ribo: Tx-level r, Shape r, Elemwise r (unscaled
           predictions) and the six MAEs after per-transcript scaling; pertranscriptMAE and codonMAE as
           computed by the paper's evaluation code (results/eval/profiles/published_mae.json)
TE         Translatomer (seq.), RiboNN individual and multitask, seq2ribo
           (results/heads/te_cds|te_utr/s<seed>/<cell>); Pearson r
Expression Translatomer (seq.), seq2ribo (results/heads/expression/s<seed>/<cell>); Pearson r
Single model = seed with the lowest validation MSE; ensemble = mean test prediction of all seeds.
Cells read "paper -> corrected" where the paper reports the value (Tables 1-3 of the paper).
Bold marks the best corrected value per row (TE: per input setting).
Output: results/eval/results_tables.md
"""
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

CELLS = config.CELL_LINES
NAMES = {"ipsc": "iPSC", "hek293": "HEK293", "lcl": "LCL", "rpe": "RPE-1"}
TE_NAMES = dict(NAMES, ipsc="Mean")
R = config.RESULTS
PROFILE_ROWS = [("Tx-level r", "txlevel_r", ".3f", max), ("Shape r", "shape_r", ".3f", max),
                ("Elemwise r", "elemwise_r", ".3f", max), ("pertranscriptMAE", "sc_pertranscriptMAE", ".3f", min),
                ("elemwiseMAE", "sc_elemwiseMAE", ".3f", min), ("codonMAE", "sc_codonMAE", ",.0f", min),
                ("pairMAE", "sc_pairMAE", ",.0f", min), ("angleMAE", "sc_angleMAE", ",.0f", min),
                ("bucketMAE", "sc_bucketMAE", ",.0f", min)]
PUBLISHED_CODE = ("sc_pertranscriptMAE", "sc_codonMAE")
# published values (paper Tables 1-3); profile rows in PROFILE_ROWS order
PAPER_PROFILES = {
    "ipsc": {"TASEP": [0.200, -0.015, 0.056, 1.339, 4.895, 40894, 40181, 16182, 43280],
             "sTASEP": [0.210, 0.039, 0.078, 1.308, 4.561, 3725, 15120, 12608, 27123],
             "Translatomer (seq.)": [0.051, 0.001, -0.002, 1.532, 6.638, 5916, 94648, 94648, 311640],
             "seq2ribo": [0.920, 0.162, 0.598, 1.386, 4.135, 12028, 44153, 85032, 287231]},
    "hek293": {"TASEP": [0.006, 0.002, -0.004, 1.811, 0.225, 244, 300, 431, 7403],
               "sTASEP": [0.013, 0.007, -0.001, 1.807, 0.222, 183, 318, 259, 6685],
               "Translatomer (seq.)": [0.023, 0.000, 0.000, 1.001, 0.150, 4859, 77744, 77744, 103658],
               "seq2ribo": [0.657, 0.054, 0.396, 1.909, 0.247, 3627, 15164, 23899, 66934]},
    "lcl": {"TASEP": [0.079, -0.012, -0.006, 1.642, 32.927, 800628, 298006, 119977, 464613],
            "sTASEP": [0.101, 0.046, 0.014, 1.518, 28.956, 35579, 248121, 155088, 299512],
            "Translatomer (seq.)": [0.055, 0.000, 0.000, 1.599, 32.164, 33724, 539576, 562607, 789398],
            "seq2ribo": [0.901, 0.186, 0.612, 1.471, 22.418, 38902, 187622, 84351, 920437]},
    "rpe": {"TASEP": [0.137, -0.005, 0.003, 1.411, 3.750, 23034, 27098, 17323, 15733],
            "sTASEP": [0.130, 0.041, 0.017, 1.369, 3.506, 2383, 6858, 4286, 22854],
            "Translatomer (seq.)": [0.035, 0.003, 0.003, 1.590, 5.100, 4740, 75832, 76336, 497259],
            "seq2ribo": [0.771, 0.139, 0.610, 1.503, 3.362, 12404, 24184, 15336, 356474]},
}
# TE: Translatomer, RiboNN ind., RiboNN mul., seq2ribo (CDS only) | RiboNN ind., RiboNN mul., seq2ribo (CDS + UTR)
PAPER_TE = {"ipsc": [0.372, 0.523, 0.529, 0.688, 0.698, 0.682, 0.708],
            "hek293": [0.209, 0.419, 0.442, 0.570, 0.602, 0.587, 0.652],
            "lcl": [0.301, 0.604, 0.590, 0.697, 0.711, 0.707, 0.732],
            "rpe": [0.339, 0.584, 0.523, 0.616, 0.667, 0.652, 0.624]}
# expression: Translatomer, seq2ribo
PAPER_EXPRESSION = {"ipsc": [0.052, 0.903], "hek293": [0.000, 0.883], "lcl": [0.261, 0.873], "rpe": [0.027, 0.830]}


def heads(task, cell, key):
    """(single, ensemble, n) over the seeds of a seq2ribo head."""
    dirs = [Path(d) for d in sorted(glob.glob(str(R / "heads" / task / "s*" / cell)))
            if (Path(d) / "test_predictions.tsv").exists()]
    val = [min(h["val_mse"] for h in json.load(open(d / "history.json"))) for d in dirs]
    single = json.load(open(dirs[int(np.argmin(val))] / "test_metrics.json"))["pearson_r"]
    preds = [pd.read_csv(d / "test_predictions.tsv", sep="\t").set_index(key) for d in dirs]
    y = preds[0]["true"]
    ens = float(pearsonr(np.mean([p.loc[y.index, "pred"].values for p in preds], axis=0), y.values)[0])
    return single, ens, len(dirs)


def ribonn(setting, mode, cell):
    """(single, ensemble, n) of RiboNN (individual or multitask)."""
    base = R / "baselines" / "ribonn"
    m = json.load(open(base / setting / mode / cell / "test_metrics.json"))
    runs = base / "runs" / setting / mode / (cell if mode == "individual" else "multitask")
    val = {}
    for f in runs.glob("seed*/member.json"):
        j = json.load(open(f))
        val[str(j["seed"])] = (next(iter(j["val_metrics"].values())) if mode == "individual"
                               else j["val_metrics"][f"TE_{cell}"])["mse"]
    return m["member_test_pearson_r"][min(val, key=val.get)], m["pearson_r"], len(val)


def translatomer(cell, name):
    r = json.load(open(R / "baselines" / "translatomer" / cell / name))["translatomer"]["pearson_r"]
    return None if r is None or math.isnan(r) else r


def row(label, values, spec, best, paper=None):
    """Markdown row; None -> '-' (constant predictions); best value(s) in bold; paper values as 'paper -> '."""
    ok = [v for v in values if v is not None]
    b = best(ok) if ok else None
    cells = ["-" if v is None else (f"**{v:{spec}}**" if f"{v:{spec}}" == f"{b:{spec}}" else f"{v:{spec}}")
             for v in values]
    if paper is not None:
        cells = [c if p is None else f"{p:{spec}} → {c}" for p, c in zip(paper, cells)]
    return "| " + " | ".join(label + cells) + " |"


def main():
    ev = R / "eval" / "profiles"
    pub = json.load(open(ev / "published_mae.json"))
    L = ["**Ribosome profiles**", "",
         "| Cell line | Metric | TASEP | sTASEP | Translatomer (seq.) | seq2ribo |", "|---|---|---|---|---|---|"]
    for c in CELLS:
        cols = {"TASEP": json.load(open(ev / c / "tasep.json"))["simulator"],
                "sTASEP": json.load(open(ev / c / "seq2ribo_stasep.json"))["simulator"],
                "Translatomer (seq.)": json.load(open(R / "baselines" / "translatomer" / c / "profiles.json"))["translatomer"],
                "seq2ribo": json.load(open(ev / c / "seq2ribo_stasep.json"))["polisher"]}
        for i, (lab, key, spec, best) in enumerate(PROFILE_ROWS):
            vals = [pub[c][m][key] if key in PUBLISHED_CODE else cols[m][key] for m in cols]
            L.append(row([NAMES[c] if i == 0 else "", lab], vals, spec, best,
                         [PAPER_PROFILES[c][m][i] for m in cols]))

    te = {c: {"ts": translatomer(c, "te_metrics.json"),
              **{f"{s}_{m}": ribonn(s, m, c) for s in ("cds_only", "utr_cds") for m in ("individual", "multitask")},
              "cds_only_s2r": heads("te_cds", c, "tx_id"), "utr_cds_s2r": heads("te_utr", c, "tx_id")} for c in CELLS}
    n = {k: te[CELLS[0]][k][2] for k in ("cds_only_individual", "cds_only_s2r")}
    group = "| | CDS only | | | | CDS + UTR | | |"
    L += ["", "**Translation efficiency**, single models (Pearson r)", "", group, "|---|---|---|---|---|---|---|---|",
          "| **Cell line** | **Translatomer (seq.)** | **RiboNN (ind.)** | **RiboNN (mul.)** | **seq2ribo** "
          "| **RiboNN (ind.)** | **RiboNN (mul.)** | **seq2ribo** |"]
    for c in CELLS:
        t = te[c]
        L.append(row([TE_NAMES[c]], [t["ts"]] + [t[f"cds_only_{m}"][0] for m in ("individual", "multitask", "s2r")],
                     ".3f", max, PAPER_TE[c][:4])
                 + row([], [t[f"utr_cds_{m}"][0] for m in ("individual", "multitask", "s2r")], ".3f", max,
                       PAPER_TE[c][4:])[1:])
    L += ["", f"**Translation efficiency**, ensembles (Pearson r; RiboNN {n['cds_only_individual']} seeds, "
              f"seq2ribo {n['cds_only_s2r']} seeds)", "", "| | CDS only | | | CDS + UTR | | |", "|---|---|---|---|---|---|---|",
          "| **Cell line** | **RiboNN (ind.)** | **RiboNN (mul.)** | **seq2ribo** "
          "| **RiboNN (ind.)** | **RiboNN (mul.)** | **seq2ribo** |"]
    for c in CELLS:
        t = te[c]
        L.append(row([TE_NAMES[c]], [t[f"cds_only_{m}"][1] for m in ("individual", "multitask", "s2r")], ".3f", max)
                 + row([], [t[f"utr_cds_{m}"][1] for m in ("individual", "multitask", "s2r")], ".3f", max)[1:])

    ex = {c: heads("expression", c, "seq_id") for c in CELLS}
    L += ["", f"**Protein expression** (mRFP, Pearson r; ensemble of {ex[CELLS[0]][2]} seeds)", "",
          "| Cell line | Translatomer (seq.) | seq2ribo | seq2ribo, ensemble |", "|---|---|---|---|"]
    for c in CELLS:
        L.append(row([NAMES[c]], [translatomer(c, "expression_metrics.json"), ex[c][0], ex[c][1]], ".3f", max,
                     PAPER_EXPRESSION[c] + [None]))
    out = R / "eval" / "results_tables.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
