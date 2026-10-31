#!/usr/bin/env python
"""Train an ensemble of upstream RiboNN models on the shared train split (one task or the multitask).

Protocol (replaces the published nested-CV runs):
  * train on split == train, early stopping and checkpoint selection on OUR split == valid
    (upstream criterion: val_r2, patience 20, at most 200 epochs); no nested CV, no test data touched;
  * N members (default 5) that differ only in the random seed;
  * every member trains in a fresh directory runs/<setting>/<mode>/<task>/seed<k>/attempt-<time>-<uid>/;
    the checkpoint path and a random run id are written to seed<k>/member.json, the run id is also stored
    inside the checkpoint's hyperparameters, and predict.py refuses a checkpoint whose id differs
    (no fixed checkpoint names, no globbing);
  * per-member validation metrics are recomputed from the selected checkpoint; a member is flagged as
    not learning for a task (and left out of that task's ensemble by predict.py) when its val predictions
    are constant or its val Pearson r is below --min-val-pearson. A val MSE not below that of predicting
    the training mean is only a warning (the evaluation is by correlation, so calibration is not required).
Multitask: the union of the four tasks' transcripts per split, NaN for missing labels; upstream's
`with_NAs=True` masked MSE (mean over observed entries) is used for training and validation loss.
Hyperparameters are listed in ribonn_lib.HPARAMS / ribonn_lib.TRAIN and are the same for both modes.

  python pipeline/baselines/ribonn/train.py --setting utr_cds --mode individual --task ipsc --device cuda:0
  python pipeline/baselines/ribonn/train.py --setting cds_only --mode multitask --device cuda:0
"""
import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ribonn_lib as rl  # noqa: E402

torch.set_float32_matmul_precision("high")  # as upstream src/train.py


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setting", choices=rl.SETTINGS, required=True)
    p.add_argument("--mode", choices=rl.MODES, required=True)
    p.add_argument("--task", choices=rl.TASKS, help="cell line (individual mode only)")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--base-seed", type=int, default=0, help="member k uses seed base_seed + k")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-epochs", type=int, default=rl.TRAIN["max_epochs"])
    p.add_argument("--patience", type=int, default=rl.TRAIN["patience"])
    p.add_argument("--limit-train-batches", type=float, default=1.0,
                   help="fraction (<=1) or number (>1) of training batches per epoch; smoke tests only")
    p.add_argument("--min-val-pearson", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=8, help="workers for the one-off input encoding")
    p.add_argument("--retrain", action="store_true",
                   help="train a new attempt even if member.json exists (old attempts are kept)")
    p.add_argument("--results-dir", type=Path, default=rl.RESULTS_DIR)
    p.add_argument("--ribonn-root", default=None)
    a = p.parse_args()
    if (a.mode == "individual") != (a.task is not None):
        p.error("--task is required for --mode individual and not allowed for --mode multitask")
    return a


class EpochLog(Callback):
    """Print and keep one line of validation metrics per epoch."""

    def __init__(self):
        self.history, self.t0 = [], time.time()

    def on_train_epoch_end(self, trainer, module):  # runs after this epoch's validation
        m = {k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith("val_")}
        m["epoch"] = trainer.current_epoch
        m["lr"] = trainer.optimizers[0].param_groups[0]["lr"]
        m["seconds"] = round(time.time() - self.t0, 1)
        self.history.append(m)
        print(f"  [ep {m['epoch']:03d}] val_loss {m.get('val_loss', float('nan')):.4f}  "
              f"val_r2 {m.get('val_r2', float('nan')):.4f}  val_pearson {m.get('val_pearson', float('nan')):.4f}  "
              f"lr {m['lr']:.1e}  {m['seconds']:.0f}s", flush=True)


def trainer_device(device):
    if device == "cpu":
        return dict(accelerator="cpu", devices=1)
    if not device.startswith("cuda"):
        raise ValueError(f"unsupported device {device}")
    return dict(accelerator="gpu", devices=[int(device.split(":")[1]) if ":" in device else 0])


def config_hash(d):
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def train_member(a, seed, cfg, x, y, split, label_names, RiboNN, provenance):
    """Train one member in a fresh attempt directory; returns the member record."""
    seed_dir = a.run_dir / f"seed{seed}"
    uid = uuid.uuid4().hex
    attempt = seed_dir / f"attempt-{time.strftime('%Y%m%d-%H%M%S')}-{uid[:8]}"
    attempt.mkdir(parents=True, exist_ok=False)
    pl.seed_everything(seed, workers=True)

    tr, va = np.flatnonzero(split == "train"), np.flatnonzero(split == "valid")
    dl_train = torch.utils.data.DataLoader(rl.EncodedDataset(x[tr], y[tr]), batch_size=rl.TRAIN["train_batch_size"],
                                           shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    dl_val = torch.utils.data.DataLoader(rl.EncodedDataset(x[va], y[va]), batch_size=rl.TRAIN["eval_batch_size"],
                                         shuffle=False, num_workers=2, pin_memory=True)

    ckpt = ModelCheckpoint(dirpath=attempt / "checkpoints", filename="best-{epoch:03d}", save_top_k=1,
                           monitor=rl.TRAIN["monitor"], mode=rl.TRAIN["monitor_mode"])
    stop = EarlyStopping(monitor=rl.TRAIN["monitor"], mode=rl.TRAIN["monitor_mode"], patience=a.patience)
    log = EpochLog()
    trainer = pl.Trainer(max_epochs=a.max_epochs, gradient_clip_val=rl.TRAIN["gradient_clip_val"],
                         callbacks=[ckpt, stop, LearningRateMonitor(logging_interval="epoch"), log],
                         logger=CSVLogger(str(attempt), name="", version=""), log_every_n_steps=20,
                         enable_progress_bar=False, enable_model_summary=False,
                         limit_train_batches=a.limit_train_batches if a.limit_train_batches <= 1
                         else int(a.limit_train_batches), **trainer_device(a.device))
    model = RiboNN(**dict(cfg, clean_run_uid=uid, clean_seed=seed))
    t0 = time.time()
    trainer.fit(model, dl_train, dl_val)
    seconds = time.time() - t0

    # reload the selected checkpoint and verify it belongs to this run
    best = RiboNN.load_from_checkpoint(ckpt.best_model_path, map_location="cpu")
    if best.hparams.get("clean_run_uid") != uid:
        raise RuntimeError(f"checkpoint {ckpt.best_model_path} does not belong to run {uid}")
    pred = rl.predict(best, x[va], a.device)
    train_mean = np.nanmean(y[tr], axis=0)
    val, flags, warnings = {}, {}, {}
    for j, name in enumerate(label_names):
        m = val[name] = rl.metrics(pred[:, j], y[va, j], train_mean[j])
        flags[name] = (["constant_predictions"] if np.isnan(m["pearson_r"]) else
                       [f"val_pearson_below_{a.min_val_pearson}"] if m["pearson_r"] < a.min_val_pearson else [])
        warnings[name] = [] if m["r2_vs_train_mean"] > 0 else ["val_mse_not_below_train_mean_predictor"]
    vp = pd.DataFrame({"tx_id": a.tx_ids[va]})
    for j, name in enumerate(label_names):
        vp[f"true_{name}"], vp[f"pred_{name}"] = y[va, j], pred[:, j]
    vp.to_csv(attempt / "val_predictions.tsv", sep="\t", index=False)

    best_epoch = int(Path(ckpt.best_model_path).stem.split("epoch=")[-1])
    record = {
        "status": "done", "seed": seed, "run_uid": uid, "attempt_dir": str(attempt),
        "checkpoint": str(Path(ckpt.best_model_path).resolve()), "config_hash": a.config_hash,
        "setting": a.setting, "mode": a.mode, "task": a.task_name, "label_columns": label_names,
        "best_epoch": best_epoch, "best_monitor": {rl.TRAIN["monitor"]: float(ckpt.best_model_score)},
        "epochs_run": trainer.current_epoch, "early_stopped": bool(stop.stopped_epoch > 0),
        "train_seconds": round(seconds, 1), "n_train": int(len(tr)), "n_val": int(len(va)),
        "val_metrics": val, "flags": flags, "warnings": warnings, "history": log.history,
        "train_config": a.train_config, "model_config": cfg, "ribonn": provenance,
    }
    json.dump(record, open(attempt / "member.json", "w"), indent=1)
    json.dump(record, open(seed_dir / "member.json", "w"), indent=1)
    return record


def main():
    a = parse_args()
    RiboNN, DataFrameDataset, RiboNNDataModule, provenance = rl.import_ribonn(a.ribonn_root)
    a.task_name = a.task if a.mode == "individual" else "multitask"
    df = rl.read_table(a.results_dir, a.setting, a.task_name)
    df = df[df.split.isin(["train", "valid"])].reset_index(drop=True)
    label_names = [rl.LABEL_COLUMNS[a.task]] if a.mode == "individual" else [rl.LABEL_COLUMNS[c] for c in rl.TASKS]
    y = df[label_names].to_numpy(np.float32)
    if a.mode == "individual" and np.isnan(y).any():
        raise ValueError("missing labels in an individual task")
    if np.isnan(y).all(axis=1).any():
        raise ValueError("rows without any label")

    cfg = rl.model_config(a.setting, a.mode, len(label_names), RiboNNDataModule)
    a.train_config = {"max_epochs": a.max_epochs, "patience": a.patience, "limit_train_batches": a.limit_train_batches,
                      "min_val_pearson": a.min_val_pearson, **{k: v for k, v in rl.TRAIN.items()
                                                               if k not in ("max_epochs", "patience")}}
    a.config_hash = config_hash({"model": cfg, "train": a.train_config, "labels": label_names})
    a.run_dir = a.results_dir / "runs" / a.setting / a.mode / a.task_name
    a.tx_ids = df.tx_id.to_numpy()
    print(f"[{a.setting} {a.mode} {a.task_name}] train {int((df.split == 'train').sum())} "
          f"valid {int((df.split == 'valid').sum())}  targets {label_names}  input length "
          f"{cfg['max_utr5_len'] + cfg['max_cds_utr3_len']}  len_after_conv {cfg['len_after_conv']}  "
          f"config {a.config_hash}", flush=True)
    print(f"RiboNN: {provenance['root']} @ {provenance['commit']}  modified: {provenance['modified_files'] or 'none'}")

    seeds = [a.base_seed + k for k in range(a.n_seeds)]
    todo = []
    for s in seeds:
        mj = a.run_dir / f"seed{s}" / "member.json"
        if mj.exists() and not a.retrain:
            rec = json.load(open(mj))
            if rec.get("status") == "done" and rec.get("config_hash") == a.config_hash:
                print(f"seed {s}: done already ({rec['checkpoint']}), skipped")
                continue
            raise RuntimeError(f"{mj} exists with another configuration; use --retrain or another --results-dir")
        todo.append(s)
    if not todo:
        return

    t0 = time.time()
    x = rl.encode(df, a.setting, DataFrameDataset, num_workers=a.num_workers)
    print(f"encoded {tuple(x.shape)} in {time.time() - t0:.0f}s", flush=True)
    split = df.split.to_numpy()
    for s in todo:
        print(f"seed {s}: training", flush=True)
        rec = train_member(a, s, cfg, x, y, split, label_names, RiboNN, provenance)
        for name in label_names:
            m = rec["val_metrics"][name]
            warn = (f"  WARNING not learning, excluded: {rec['flags'][name]}" if rec["flags"][name] else "") + \
                   (f"  note: {rec['warnings'][name]}" if rec["warnings"][name] else "")
            print(f"seed {s} {name}: best epoch {rec['best_epoch']}/{rec['epochs_run']}  val n={m['n']} "
                  f"r={m['pearson_r']:.4f} rho={m['spearman_r']:.4f} mse={m['mse']:.4f} "
                  f"r2_vs_mean={m['r2_vs_train_mean']:.3f}  {rec['train_seconds']:.0f}s{warn}", flush=True)


if __name__ == "__main__":
    main()
