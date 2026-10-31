#!/usr/bin/env python
"""Train the sequence-only Translatomer baseline on one cell line.

Data: data/<cell>/train.pkl (the seq2ribo training split; 90% for training, 10% held out for
checkpoint selection as in the supplement), CDS blocks from data/cds.pkl, hg38 windows (windows.py:
65,536 bp, 1,024 x 64-bp bins, transcript-oriented, identical tiling for training and inference).

Settings (seq2ribo supplement where it is explicit, the released Translatomer code otherwise):
  model      released Translatomer AttnModule, 5-channel one-hot DNA input (translatomer_model.py)
  loss       MSE on log(1 + binned counts) (Translatomer's target transform)
  optimizer  Adam, lr 1e-5; warm-up of 2,000 steps then cosine decay (Translatomer's schedule)
  batch      32 windows, shuffled, drop_last; dropout 0.1; fp32; 10 epochs; seed 42
  selection  a random 10% of the training transcripts is held out; the checkpoint with the lowest
             loss on it is kept

Outputs in --out-dir: model_best.pt, model_last.pt (resumable), history.json, train_config.json.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4]))   # repo root (seq2ribo package)
sys.path.insert(0, str(HERE.parents[1]))   # pipeline/
sys.path.insert(0, str(HERE))
import config  # noqa: E402
from common import load_records  # noqa: E402
from window_datasets import TrainWindows, load_cds  # noqa: E402
from translatomer_model import TRANSLATOMER_ROOT, TransModelSeqOnly, translatomer_commit  # noqa: E402
from windows import BIN, NBINS, SEQLEN, STRIDE, Genome  # noqa: E402

DEFAULTS = dict(lr=1e-5, weight_decay=0.0, warmup_steps=2000, batch_size=32, epochs=10, seed=42, val_fraction=0.1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell-line", required=True, choices=config.CELL_LINES)
    p.add_argument("--data-dir", type=Path, default=config.DATA)
    p.add_argument("--subset", default="", help="'' (full) or '_01' (10%% subsets)")
    p.add_argument("--max-train-tx", type=int, default=None, help="smoke tests: first n train transcripts")
    p.add_argument("--max-val-tx", type=int, default=None)
    p.add_argument("--cds", type=Path, default=config.DATA / "cds.pkl")
    p.add_argument("--fasta", type=Path, default=config.GENOME_FASTA)
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--warmup-steps", type=int, default=DEFAULTS["warmup_steps"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--val-fraction", type=float, default=DEFAULTS["val_fraction"],
                   help="random fraction of the training transcripts held out for checkpoint selection")
    p.add_argument("--tf32", action="store_true", help="allow TF32 matmuls (faster on A100; default off = fp32)")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume", action="store_true", help="continue from <out-dir>/model_last.pt")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def cosine_with_warmup(optimizer, warmup, total):
    """Same schedule as transformers.get_cosine_schedule_with_warmup (num_cycles=0.5)."""
    def f(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


@torch.no_grad()
def validate(model, dl, device):
    """Mean per-window MSE (the training loss) and the Pearson r over all val bins."""
    model.eval()
    se, n, sp, st, spp, stt, spt, nb = 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    for x, y in dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        p = model(x).double(); y = torch.log1p(y.double())
        se += ((p - y) ** 2).mean(dim=1).sum().item(); n += len(y)
        sp += p.sum().item(); st += y.sum().item(); spp += (p * p).sum().item()
        stt += (y * y).sum().item(); spt += (p * y).sum().item(); nb += y.numel()
    cov = spt - sp * st / nb
    var = math.sqrt(max(spp - sp * sp / nb, 0) * max(stt - st * st / nb, 0))
    return se / max(n, 1), (cov / var if var > 0 else float("nan"))


def main():
    a = parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = a.tf32

    cds = load_cds(a.cds)
    genome = Genome(a.fasta)
    recs = load_records(a.cell_line, "train", a.data_dir, a.subset)[:a.max_train_tx]
    order = list(range(len(recs)))
    random.Random(a.seed).shuffle(order)
    n_val = int(round(a.val_fraction * len(recs)))
    va = [recs[i] for i in sorted(order[:n_val])][:a.max_val_tx]
    tr = [recs[i] for i in sorted(order[n_val:])]
    ds_tr, ds_va = TrainWindows(tr, cds, genome), TrainWindows(va, cds, genome)
    print(f"[{a.cell_line}] train {len(tr)} tx / {len(ds_tr)} windows, val {len(va)} tx / {len(ds_va)} windows",
          flush=True)
    g = torch.Generator().manual_seed(a.seed)
    dl_tr = DataLoader(ds_tr, batch_size=a.batch_size, shuffle=True, drop_last=True, generator=g,
                       num_workers=a.num_workers, pin_memory=True, persistent_workers=a.num_workers > 0)
    dl_va = DataLoader(ds_va, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
                       pin_memory=True, persistent_workers=a.num_workers > 0)
    if len(dl_tr) == 0:
        raise SystemExit("fewer training windows than one batch")

    model_cfg = {}
    model = TransModelSeqOnly(**model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    total_steps = a.epochs * len(dl_tr)
    sched = cosine_with_warmup(opt, a.warmup_steps, total_steps)
    crit = nn.MSELoss()

    cfg = {"cell_line": a.cell_line, "train_split": f"train{a.subset}", "val_split": f"{a.val_fraction:g} of train{a.subset}",
           "n_train_tx": len(tr), "n_val_tx": len(va), "n_train_windows": len(ds_tr), "n_val_windows": len(ds_va),
           "seqlen": SEQLEN, "n_bins": NBINS, "bin_bp": BIN, "stride": STRIDE, "model": model_cfg,
           "translatomer_root": TRANSLATOMER_ROOT, "translatomer_commit": translatomer_commit(),
           "n_params": sum(p.numel() for p in model.parameters()),
           "loss": "MSE on log(1 + binned counts)", "optimizer": "Adam",
           "validation": f"random {a.val_fraction:.0%} of the training transcripts (seed {a.seed})", "lr": a.lr, "weight_decay": a.weight_decay,
           "schedule": f"cosine with {a.warmup_steps} linear warm-up steps over {total_steps} steps",
           "batch_size": a.batch_size, "epochs": a.epochs, "steps_per_epoch": len(dl_tr), "grad_clip": None,
           "precision": "fp32 matmuls (TF32 allowed)" if a.tf32 else "fp32 matmuls", "seed": a.seed,
           "selection": "lowest val loss (mean per-window MSE)",
           "cds": str(a.cds), "fasta": str(a.fasta), "device": a.device}
    (a.out_dir / "train_config.json").write_text(json.dumps(cfg, indent=1))

    start_ep, best, history = 0, float("inf"), []
    last = a.out_dir / "model_last.pt"
    if a.resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        start_ep, best, history = ck["epoch"] + 1, ck["best_val_loss"], ck["history"]
        g.manual_seed(a.seed + start_ep)
        print(f"resumed from epoch {ck['epoch']} (best val {best:.4f})", flush=True)

    for ep in range(start_ep, a.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for step, (x, y) in enumerate(dl_tr):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = crit(model(x), torch.log1p(y))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            tot += loss.item(); n += 1
            if a.log_every and (step + 1) % a.log_every == 0:
                print(f"  ep {ep} step {step + 1}/{len(dl_tr)} loss {tot / n:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{(time.time() - t0) / (step + 1):.3f}s/step", flush=True)
        train_s = time.time() - t0
        val_loss, val_r = validate(model, dl_va, device)
        rec = {"epoch": ep, "train_loss": tot / max(n, 1), "val_loss": val_loss, "val_bin_pearson": val_r,
               "train_seconds": train_s, "seconds": time.time() - t0}
        history.append(rec)
        state = {"model": model.state_dict(), "model_cfg": model_cfg, "epoch": ep, "val_loss": val_loss}
        if val_loss < best:
            best = val_loss
            torch.save(state, a.out_dir / "model_best.pt")
        torch.save({**state, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "best_val_loss": best, "history": history}, last)
        (a.out_dir / "history.json").write_text(json.dumps(history, indent=1))
        print(f"[ep {ep:02d}] train {rec['train_loss']:.4f} val {val_loss:.4f} (bin r {val_r:.3f}) "
              f"best {best:.4f} {rec['seconds']:.0f}s", flush=True)
    print(f"done; best val loss {best:.4f} -> {a.out_dir / 'model_best.pt'}")


if __name__ == "__main__":
    main()
