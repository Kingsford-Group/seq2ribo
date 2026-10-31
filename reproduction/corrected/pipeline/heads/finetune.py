#!/usr/bin/env python
"""Finetune a polisher end to end with a task head: TE (CDS-only or CDS+UTR) or protein expression.

Every (transcript, simulation run) pair is a training example with the transcript's label.
  te_cds      per-codon FFN (256, 128, 1; GELU) on log1p(counts), masked mean, linear, sigmoid
  te_utr      5'UTR / log1p(counts) / 3'UTR Mamba branches (2 layers, d=128) + count summary, FFN, sigmoid
  expression  per-codon FFN (128, 64, 64, 1; GELU) on log1p(counts), masked mean, linear
te_utr uses the transcripts with utr_ok in labels.tsv. TE labels are min-max scaled with the training range; the loss is MSE in the scaled space (TE) or on
the label (expression). Adam, gradient clipping; the epoch with the lowest validation MSE (per-transcript
mean prediction, original scale) is kept. Test predictions are the mean over all runs of a transcript.
Output: <out>/{model.pt, transform.json, history.json, test_predictions.tsv, test_metrics.json}
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from common import ProfileDataset, Shards, angle_edges, pad_collate  # noqa: E402
from seq2ribo.models import MambaExprFull, MambaTEFull, MambaTEFullUTR, RiboPolisherMamba  # noqa: E402
from seq2ribo.utils import load_state_dict_safely  # noqa: E402

NUC2IDX = {"A": 0, "U": 1, "G": 2, "C": 3}
NUC_PAD = 4


class LabeledDataset(ProfileDataset):
    def __init__(self, records, shards, edges, labels, utrs=None, runs=None):
        super().__init__(records, shards, edges, runs)
        self.labels, self.utrs = labels, utrs

    def __getitem__(self, n):
        item = super().__getitem__(n)
        item["label"] = self.labels[item["tx"]]
        if self.utrs is not None:
            u5, u3 = self.utrs[item["tx"]]
            item["utr5"] = torch.tensor([NUC2IDX.get(c, 0) for c in u5], dtype=torch.long)
            item["utr3"] = torch.tensor([NUC2IDX.get(c, 0) for c in u3], dtype=torch.long)
        return item


def collate(batch):
    out = pad_collate(batch)
    out["label"] = torch.tensor([x["label"] for x in batch], dtype=torch.float32)
    if "utr5" in batch[0]:
        for k in ("utr5", "utr3"):
            L = max(1, max(len(x[k]) for x in batch))
            ids = torch.full((len(batch), L), NUC_PAD, dtype=torch.long)
            mask = torch.zeros((len(batch), L), dtype=torch.bool)
            for i, x in enumerate(batch):
                ids[i, :len(x[k])] = x[k]; mask[i, :len(x[k])] = True
            out[k], out[k + "_mask"] = ids, mask
    return out


def forward(model, task, b, device):
    args = [b[k].to(device) for k in ("cod_ids", "sim_feat", "mask")]
    kw = {k: b[k].to(device) for k in ("angle_bin", "pair_bin", "bucket_idx")}
    if task == "te_utr":
        args += [b[k].to(device) for k in ("utr5", "utr3", "utr5_mask", "utr3_mask")]
    return model(*args, **kw)[1]


def predict(model, task, dl, device):
    model.eval()
    txs, preds = [], []
    with torch.no_grad():
        for b in dl:
            preds.append(forward(model, task, b, device).float().cpu().numpy()); txs += b["tx"]
    df = pd.DataFrame({"tx_id": txs, "pred": np.concatenate(preds)})
    return df.groupby("tx_id", sort=False)["pred"].mean()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=["te_cds", "te_utr", "expression"])
    p.add_argument("--cell-line", required=True, choices=config.CELL_LINES)
    p.add_argument("--polisher", type=Path, required=True)
    p.add_argument("--polisher-inputs", nargs="+", default=list(RiboPolisherMamba.ALL_INPUTS))
    p.add_argument("--shards", type=Path, required=True, help="simulation runs of the task transcripts ({split}.npy)")
    p.add_argument("--data-dir", type=Path, default=config.DATA)
    p.add_argument("--runs", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, required=True)
    # regularization (defaults = paper settings: Adam, one lr for all parameters, all runs every epoch)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    p.add_argument("--backbone-lr-mult", type=float, default=1.0, help="lr of the polisher = lr x this")
    p.add_argument("--utr-lr-mult", type=float, default=1.0,
                   help="te_utr: lr of the UTR branches (nucleotide embedding, 5'/3' UTR Mamba blocks) = lr x this")
    p.add_argument("--freeze-backbone-epochs", type=int, default=0,
                   help="keep the polisher frozen (lr 0) for the first k epochs, then train it at lr x backbone-lr-mult")
    p.add_argument("--train-runs-per-epoch", type=int, default=None,
                   help="draw this many of the runs (same draw for every transcript, new each epoch)")
    p.add_argument("--val-runs", type=int, default=None, help="runs per transcript for validation (default: all)")
    p.add_argument("--patience", type=int, default=None, help="early stop after this many epochs without val MSE gain")
    p.add_argument("--no-log1p", action="store_true",
                   help="heads read the counts directly instead of log1p(counts) (published expression runs)")
    p.add_argument("--eval-only", action="store_true",
                   help="skip training: score <out-dir>/model.pt (best checkpoint so far) on the test set")
    return p.parse_args()


def main():
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    H = config.EXPRESSION if a.task == "expression" else config.TE
    runs = a.runs or (H["sims_per_seq"] if a.task == "expression" else H["sims_per_tx"])
    epochs, bs = a.epochs or H["epochs"], a.batch_size or H["batch_size"]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    task_dir = a.data_dir / ("expression" if a.task == "expression" else f"te/{a.cell_line}")
    lab = pd.read_csv(task_dir / "labels.tsv", sep="\t")
    if a.task == "te_utr":   # transcripts whose workbook UTRs match the CDS (te/prepare.py)
        lab = lab[lab["utr_ok"].astype(str).str.lower() == "true"]
    key = "seq_id" if a.task == "expression" else "tx_id"
    ycol = "expression" if a.task == "expression" else "te"
    y = dict(zip(lab[key], lab[ycol].astype(float)))
    utrs = dict(zip(lab[key], zip(lab["utr5"].fillna(""), lab["utr3"].fillna("")))) if a.task == "te_utr" else None

    # label transform: min-max over the training labels for TE, identity for expression
    if a.task == "expression":
        tr = {"name": "identity"}
        fwd, inv = (lambda v: v), (lambda v: v)
    else:
        lo, hi = float(lab.loc[lab["split"] == "train", "te"].min()), float(lab.loc[lab["split"] == "train", "te"].max())
        tr = {"name": "minmax", "lo": lo, "hi": hi, "eps": 1e-6}
        fwd = lambda v: (v - lo) / (hi - lo)
        inv = lambda s: lo + (hi - lo) * s
    json.dump(tr, open(a.out_dir / "transform.json", "w"), indent=1)

    edges = angle_edges(a.data_dir)
    import pickle
    ds, src = {}, {}
    for s in ("train", "val", "test"):
        recs = [r for r in pickle.load(open(task_dir / f"{s}.pkl", "rb")) if r[0] in y]
        labels = {r[0]: fwd(y[r[0]]) for r in recs}
        src[s] = (recs, Shards(a.shards / s), labels)
        n_runs = a.val_runs or runs if s == "val" else runs
        ds[s] = LabeledDataset(recs, src[s][1], edges, labels, utrs, runs=range(n_runs))
    loader = lambda d, shuffle: DataLoader(d, batch_size=bs, shuffle=shuffle, collate_fn=collate, num_workers=a.workers,
                                           pin_memory=True, persistent_workers=a.workers > 0 and not a.train_runs_per_epoch)
    dl = {s: loader(ds[s], s == "train") for s in ds}
    run_rng = random.Random(a.seed)
    print(f"[{a.task} {a.cell_line}] examples train {len(ds['train'])} val {len(ds['val'])} test {len(ds['test'])} "
          f"({runs} runs/transcript)", flush=True)

    device = torch.device(a.device)
    P = config.POLISHER
    base = RiboPolisherMamba(d_model=P["d_model"], n_layers=P["n_layers"], d_state=P["d_state"], d_conv=P["d_conv"],
                             expand=P["expand"], dropout=P["dropout"], k_angle_bins=len(edges) - 1,
                             inputs=a.polisher_inputs)
    base.load_state_dict(load_state_dict_safely(a.polisher, "cpu"))
    if a.task == "te_cds":
        model = MambaTEFull(base, hidden=256, use_log1p=not a.no_log1p)
    elif a.task == "te_utr":
        model = MambaTEFullUTR(base, d_te=128, n_te_layers=2, te_d_state=16, te_d_conv=4, te_expand=2, dropout=0.1,
                               use_log1p=not a.no_log1p)
    else:
        model = MambaExprFull(base, hidden=128, use_log1p=not a.no_log1p)
    model.to(device)
    lr, wd = a.lr or H["lr"], H["weight_decay"] if a.weight_decay is None else a.weight_decay
    base_ids = {id(p) for p in model.base.parameters()}
    utr_ids = set()
    if a.task == "te_utr":
        for name in ("nuc_emb", "mamba_5utr", "mamba_3utr"):
            utr_ids |= {id(p) for p in getattr(model.te_head, name).parameters()}
    base_lr = lr * a.backbone_lr_mult
    groups = [{"params": [p for p in model.parameters() if id(p) in base_ids], "lr": base_lr},
              {"params": [p for p in model.parameters() if id(p) not in base_ids | utr_ids], "lr": lr}]
    if utr_ids:
        groups.append({"params": [p for p in model.parameters() if id(p) in utr_ids], "lr": lr * a.utr_lr_mult})
    opt = (torch.optim.AdamW if a.optimizer == "adamw" else torch.optim.Adam)(groups, lr=lr, weight_decay=wd)
    cfg = {k: getattr(a, k) for k in ("optimizer", "backbone_lr_mult", "utr_lr_mult", "freeze_backbone_epochs",
                                      "train_runs_per_epoch", "val_runs", "patience", "no_log1p")}
    cfg.update(lr=lr, weight_decay=wd, epochs=epochs, batch_size=bs, runs=runs)
    if not a.eval_only:
        json.dump(cfg, open(a.out_dir / "train_config.json", "w"), indent=1)
    y_val = pd.Series({t: y[t] for t in ds["val"].shards.tx if t in y})
    best, history = float("inf"), []
    if a.eval_only:
        history = json.load(open(a.out_dir / "history.json"))
        best, epochs = min(h["val_mse"] for h in history), 0
        print(f"[eval only] {a.out_dir / 'model.pt'}: best of {len(history)} epochs, val MSE {best:.4f}", flush=True)
    best_ep = 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        opt.param_groups[0]["lr"] = 0.0 if ep <= a.freeze_backbone_epochs else base_lr   # polisher group
        if a.train_runs_per_epoch:
            recs, sh, labels = src["train"]
            sel = sorted(run_rng.sample(range(runs), a.train_runs_per_epoch))
            dl["train"] = loader(LabeledDataset(recs, sh, edges, labels, utrs, runs=sel), True)
        model.train()
        tot, n = 0.0, 0
        for b in dl["train"]:
            pred = forward(model, a.task, b, device)
            loss = torch.nn.functional.mse_loss(pred, b["label"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), H["grad_clip"])
            opt.step()
            tot += loss.item() * len(b["tx"]); n += len(b["tx"])
        pv = inv(predict(model, a.task, dl["val"], device))
        val_mse = float(np.mean((pv.values - y_val[pv.index].values) ** 2))
        val_r = float(pearsonr(pv.values, y_val[pv.index].values)[0])
        history.append({"epoch": ep, "train_loss": tot / n, "val_mse": val_mse, "val_pearson": val_r, "seconds": time.time() - t0})
        if val_mse < best:
            best, best_ep = val_mse, ep
            torch.save(model.state_dict(), a.out_dir / "model.pt")
        json.dump(history, open(a.out_dir / "history.json", "w"), indent=1)
        print(f"[ep {ep:02d}] train {tot / n:.5f}  val MSE {val_mse:.4f}  val r {val_r:.3f}  {time.time() - t0:.0f}s", flush=True)
        if a.patience and ep - best_ep >= a.patience:
            print(f"early stop: no val MSE improvement since epoch {best_ep}", flush=True)
            break

    model.load_state_dict(load_state_dict_safely(a.out_dir / "model.pt", device))
    pt = inv(predict(model, a.task, dl["test"], device))
    yt = np.array([y[t] for t in pt.index])
    pd.DataFrame({key: pt.index, "true": yt, "pred": pt.values}).to_csv(a.out_dir / "test_predictions.tsv", sep="\t", index=False)
    m = {"task": a.task, "cell_line": a.cell_line, "n": int(len(yt)), "runs": runs,
         "pearson_r": float(pearsonr(pt.values, yt)[0]), "spearman_r": float(spearmanr(pt.values, yt)[0]),
         "mse": float(np.mean((pt.values - yt) ** 2)), "best_val_mse": best, "polisher": str(a.polisher),
         "epochs_trained": len(history), "best_epoch": int(np.argmin([h["val_mse"] for h in history])) + 1}
    json.dump(m, open(a.out_dir / "test_metrics.json", "w"), indent=1)
    print(f"[TEST] n={m['n']} r={m['pearson_r']:.4f} rho={m['spearman_r']:.4f} MSE={m['mse']:.4f}")


if __name__ == "__main__":
    main()
