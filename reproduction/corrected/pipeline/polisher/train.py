#!/usr/bin/env python
"""Train the Mamba polisher (or an input ablation of it) on sTASEP simulation runs.

Each (transcript, run) pair of the training split is one example. Loss: Poisson negative
log-likelihood with the Softplus output as the rate, averaged over codon positions; for cell
lines in POLISHER["nonzero_weight"] positions with a non-zero observed count are weighted.
Adam (lr, weight decay), gradient clipping, ReduceLROnPlateau on the validation loss; the
checkpoint with the lowest validation loss is kept, and training stops early after
POLISHER["early_stop_patience"] epochs without improvement (at most --epochs). --accum-steps n splits every batch into n
micro-batches (same padding, loss normalised by the positions of the whole batch), so the update
equals that of the full batch while the activation memory is ~1/n.
Output: results/polisher/<name>/<cell>/{polisher.pt, history.json}
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from common import ProfileDataset, Shards, angle_edges, load_records, pad_collate  # noqa: E402
from seq2ribo.models import RiboPolisherMamba  # noqa: E402

P = config.POLISHER


TENSORS = ("cod_ids", "sim_feat", "mask", "target", "angle_bin", "pair_bin", "bucket_idx")


def poisson_loss(pred, target, mask, nonzero_weight=1.0, denom=None):
    nll = torch.nn.functional.poisson_nll_loss(pred, target, log_input=False, full=True, eps=1e-8, reduction="none")
    w = mask.float()
    if nonzero_weight != 1.0:
        w = w * torch.where(target > 0, nonzero_weight, 1.0)
    return (nll * w).sum() / (mask.sum().clamp(min=1) if denom is None else denom)


def run_epoch(model, dl, device, nonzero_weight, opt=None, accum_steps=1):
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for b in dl:
            B = len(b["tx"])
            denom = b["mask"].sum().clamp(min=1).to(device)
            if train:
                opt.zero_grad(set_to_none=True)
            loss_b = 0.0
            for idx in torch.arange(B).chunk(min(accum_steps, B)):
                m = {k: b[k][idx].to(device, non_blocking=True) for k in TENSORS}
                pred = model(m["cod_ids"], m["sim_feat"], m["mask"],
                             angle_bin=m["angle_bin"], pair_bin=m["pair_bin"], bucket_idx=m["bucket_idx"])
                loss = poisson_loss(pred, m["target"], m["mask"], nonzero_weight, denom)
                if train:
                    loss.backward()
                loss_b += loss.item()
            if train:
                torch.nn.utils.clip_grad_norm_(model.parameters(), P["grad_clip"])
                opt.step()
            total += loss_b * B; n += B
    return total / max(n, 1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell-line", required=True, choices=config.CELL_LINES)
    p.add_argument("--shards", type=Path, required=True, help="directory with {train,val}.npy/.json from simulate.py")
    p.add_argument("--inputs", nargs="+", default=list(RiboPolisherMamba.ALL_INPUTS),
                   choices=list(RiboPolisherMamba.ALL_INPUTS))
    p.add_argument("--name", default="seq2ribo", help="output subdirectory (e.g. ablation name)")
    p.add_argument("--data-dir", type=Path, default=config.DATA)
    p.add_argument("--epochs", type=int, default=P["epochs"], help="maximum number of epochs")
    p.add_argument("--early-stop-patience", type=int, default=P["early_stop_patience"])
    p.add_argument("--batch-size", type=int, default=P["batch_size"])
    p.add_argument("--runs", type=int, default=P["sims_per_tx"], help="simulation runs per transcript used")
    p.add_argument("--accum-steps", type=int, default=1, help="micro-batches per batch (memory only)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main():
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = a.out_dir or config.RESULTS / "polisher" / a.name / a.cell_line
    out.mkdir(parents=True, exist_ok=True)
    edges = angle_edges(a.data_dir)
    ds = {s: ProfileDataset(load_records(a.cell_line, s, a.data_dir), Shards(a.shards / s), edges, runs=range(a.runs))
          for s in ("train", "val")}
    dl = {s: DataLoader(ds[s], batch_size=a.batch_size, shuffle=(s == "train"), collate_fn=pad_collate,
                        num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
          for s in ds}
    print(f"[{a.name} {a.cell_line}] inputs={a.inputs} train {len(ds['train'])} val {len(ds['val'])} examples", flush=True)

    device = torch.device(a.device)
    model = RiboPolisherMamba(d_model=P["d_model"], n_layers=P["n_layers"], d_state=P["d_state"], d_conv=P["d_conv"],
                              expand=P["expand"], dropout=P["dropout"], k_angle_bins=len(edges) - 1,
                              inputs=a.inputs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=P["plateau_factor"], patience=P["plateau_patience"])
    nzw = P["nonzero_weight"].get(a.cell_line, 1.0)
    best, best_ep, history = float("inf"), 0, []
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, dl["train"], device, nzw, opt, a.accum_steps)
        va = run_epoch(model, dl["val"], device, nzw, accum_steps=a.accum_steps)
        sched.step(va)
        history.append({"epoch": ep, "train_loss": tr, "val_loss": va, "lr": opt.param_groups[0]["lr"],
                        "seconds": time.time() - t0})
        if va < best:
            best, best_ep = va, ep
            torch.save(model.state_dict(), out / "polisher.pt")
        with open(out / "history.json", "w") as f:
            json.dump({"inputs": a.inputs, "nonzero_weight": nzw, "batch_size": a.batch_size, "best_val_loss": best,
                       "best_epoch": best_ep, "epochs": history}, f, indent=1)
        print(f"[ep {ep:03d}] train {tr:.5f}  val {va:.5f}  best {best:.5f}  lr {opt.param_groups[0]['lr']:.1e}  "
              f"{time.time() - t0:.0f}s", flush=True)
        if ep - best_ep >= a.early_stop_patience:
            print(f"early stop: no improvement since epoch {best_ep}", flush=True)
            break


if __name__ == "__main__":
    main()
