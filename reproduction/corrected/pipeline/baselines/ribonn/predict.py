#!/usr/bin/env python
"""Ensemble prediction and evaluation of the clean RiboNN baseline on the shared test split.

For each task the members trained by train.py (seeds base_seed .. base_seed + n_seeds - 1, all required,
all with the same configuration) are loaded from the checkpoint recorded in their member.json; the run id
stored inside the checkpoint must match. Each member's validation metrics are recomputed and must agree
with those written at training time (guards against loading a different checkpoint). The test
prediction of a transcript is the mean over members that were not flagged as not learning for that task
(--keep-flagged averages all members). Every test transcript of data/te/<cell>/labels.tsv is predicted;
nothing is filtered, and the set is checked against labels.tsv.

Output: <results-dir>/<setting>/<mode>/<cell>/{test_predictions.tsv, test_metrics.json}
        <results-dir>/<setting>/<mode>/summary.tsv
  test_predictions.tsv: tx_id, true, pred (ensemble mean of used members), pred_seed<k> (every member)

  python pipeline/baselines/ribonn/predict.py --setting utr_cds --mode individual --device cuda:0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ribonn_lib as rl  # noqa: E402
from ribonn_lib import config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setting", choices=rl.SETTINGS, required=True)
    p.add_argument("--mode", choices=rl.MODES, required=True)
    p.add_argument("--tasks", nargs="+", choices=rl.TASKS, default=rl.TASKS, help="cell lines to evaluate")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--keep-flagged", action="store_true", help="also average members flagged as not learning")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-workers", type=int, default=8, help="workers for the one-off input encoding")
    p.add_argument("--data-dir", type=Path, default=config.DATA)
    p.add_argument("--results-dir", type=Path, default=rl.RESULTS_DIR)
    p.add_argument("--ribonn-root", default=None)
    return p.parse_args()


def load_members(run_dir, seeds):
    members = []
    for s in seeds:
        mj = run_dir / f"seed{s}" / "member.json"
        if not mj.exists():
            raise FileNotFoundError(f"{mj} missing: member seed {s} not trained")
        rec = json.load(open(mj))
        if not Path(rec["checkpoint"]).is_absolute():   # released runs store paths relative to member.json
            rec["checkpoint"] = str(mj.parent / rec["checkpoint"])
        if rec.get("status") != "done":
            raise RuntimeError(f"{mj}: status {rec.get('status')}")
        members.append(rec)
    if len({m["config_hash"] for m in members}) != 1:
        raise RuntimeError(f"members in {run_dir} were trained with different configurations")
    return members


def run_members(members, RiboNN, x_val, x_test, device):
    """Per member: (val predictions, test predictions), after verifying checkpoint identity."""
    out = []
    for rec in members:
        model = RiboNN.load_from_checkpoint(rec["checkpoint"], map_location="cpu")
        if model.hparams.get("clean_run_uid") != rec["run_uid"]:
            raise RuntimeError(f"{rec['checkpoint']} does not carry run id {rec['run_uid']}")
        out.append((rl.predict(model, x_val, device), rl.predict(model, x_test, device)))
        del model
        torch.cuda.empty_cache()
    return out


def main():
    a = parse_args()
    RiboNN, DataFrameDataset, _, provenance = rl.import_ribonn(a.ribonn_root)
    seeds = [a.base_seed + k for k in range(a.n_seeds)]
    # one model group per individual task, one shared group for the multitask model
    groups = [(c, [c]) for c in a.tasks] if a.mode == "individual" else [("multitask", a.tasks)]
    rows = []
    for task_name, cells in groups:
        run_dir = a.results_dir / "runs" / a.setting / a.mode / task_name
        members = load_members(run_dir, seeds)
        label_names = members[0]["label_columns"]
        df = rl.read_table(a.results_dir, a.setting, task_name)
        val = df[df.split == "valid"].reset_index(drop=True)
        test = df[df.split == "test"].reset_index(drop=True)
        x_val = rl.encode(val, a.setting, DataFrameDataset, a.num_workers)
        x_test = rl.encode(test, a.setting, DataFrameDataset, a.num_workers)
        preds = run_members(members, RiboNN, x_val, x_test, a.device)

        for cell in cells:
            name = rl.LABEL_COLUMNS[cell]
            j = label_names.index(name)
            # identity check: recomputed val metrics equal those logged at training time
            for rec, (pv, _) in zip(members, preds):
                logged = rec["val_metrics"][name]["pearson_r"]
                now = rl.metrics(pv[:, j], val[name].to_numpy())["pearson_r"]
                if not (np.isnan(logged) and np.isnan(now)) and abs(logged - now) > 1e-3:
                    raise RuntimeError(f"seed {rec['seed']} {name}: val r {now:.5f} != logged {logged:.5f}")
            flagged = [rec["seed"] for rec in members if rec["flags"][name]]
            used = [i for i, rec in enumerate(members) if a.keep_flagged or not rec["flags"][name]]
            if not used:
                raise RuntimeError(f"{task_name}/{name}: every member is flagged as not learning")

            keep = test[name].notna().to_numpy()   # multitask union -> this cell's test transcripts
            t = test[keep].reset_index(drop=True)
            expected = pd.read_csv(a.data_dir / "te" / cell / "labels.tsv", sep="\t", usecols=["tx_id", "split", "utr_ok"])
            if a.setting == "utr_cds":   # CDS+UTR: only transcripts with matching workbook UTRs (export.py)
                expected = expected[expected.utr_ok.astype(str).str.lower() == "true"]
            expected = set(expected.tx_id[expected.split == "test"])
            if set(t.tx_id) != expected or len(t) != len(expected):
                raise RuntimeError(f"{cell}: test transcripts differ from data/te/{cell}/labels.tsv")

            member_pred = np.stack([pt[keep, j] for _, pt in preds], axis=1)
            ens = member_pred[:, used].mean(axis=1)
            true = t[name].to_numpy()
            out = pd.DataFrame({"tx_id": t.tx_id, "true": true, "pred": ens})
            for i, rec in enumerate(members):
                out[f"pred_seed{rec['seed']}"] = member_pred[:, i]
            m = rl.metrics(ens, true)
            vkeep = val[name].notna().to_numpy()
            val_ens = np.stack([pv[vkeep, j] for pv, _ in preds], axis=1)[:, used].mean(axis=1)
            res = {
                "setting": a.setting, "mode": a.mode, "cell_line": cell, "label": name,
                "n": m["n"], "pearson_r": m["pearson_r"], "spearman_r": m["spearman_r"], "mse": m["mse"],
                "members_used": [members[i]["seed"] for i in used], "members_flagged": flagged,
                "keep_flagged": a.keep_flagged,
                "member_test_pearson_r": {members[i]["seed"]: rl.metrics(member_pred[:, i], true)["pearson_r"]
                                          for i in range(len(members))},
                "member_val": {rec["seed"]: {"best_epoch": rec["best_epoch"], "epochs_run": rec["epochs_run"],
                                             **rec["val_metrics"][name], "flags": rec["flags"][name],
                                             "warnings": rec.get("warnings", {}).get(name, [])}
                               for rec in members},
                "ensemble_val": rl.metrics(val_ens, val[name].to_numpy()[vkeep]),
                "checkpoints": [rec["checkpoint"] for rec in members],
                "config_hash": members[0]["config_hash"], "train_config": members[0]["train_config"],
                "model_config": members[0]["model_config"], "ribonn": provenance,
            }
            od = a.results_dir / a.setting / a.mode / cell
            od.mkdir(parents=True, exist_ok=True)
            out.to_csv(od / "test_predictions.tsv", sep="\t", index=False)
            json.dump(res, open(od / "test_metrics.json", "w"), indent=1)
            rows.append({"setting": a.setting, "mode": a.mode, "cell_line": cell, "n": m["n"],
                         "pearson_r": m["pearson_r"], "spearman_r": m["spearman_r"],
                         "members_used": len(used), "members_flagged": len(flagged)})
            print(f"[TEST {a.setting} {a.mode} {cell}] n={m['n']} r={m['pearson_r']:.4f} "
                  f"rho={m['spearman_r']:.4f}  members {res['members_used']} flagged {flagged}", flush=True)

    summary = a.results_dir / a.setting / a.mode / "summary.tsv"
    new = pd.DataFrame(rows)
    if summary.exists():  # keep rows of cells not evaluated in this call
        old = pd.read_csv(summary, sep="\t")
        new = pd.concat([old[~old.cell_line.isin(new.cell_line)], new], ignore_index=True)
    new.to_csv(summary, sep="\t", index=False)


if __name__ == "__main__":
    main()
