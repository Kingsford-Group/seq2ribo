#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from scipy.stats import spearmanr as _scipy_spearmanr

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from mamba_ssm import Mamba
from util import CODON2IDX

ANGLE_BINS = np.array([0.0, 1.0471990, 2.09439815, 3.14159722, 6.51421177], dtype=np.float64)
K_ANGLE_BINS = len(ANGLE_BINS) - 1



class RiboPolisherMamba(nn.Module):
    def __init__(self, d_model=192, n_layers=4, d_state=16, d_conv=4, expand=2, dropout=0.1, k_angle_bins=4):
        super().__init__()
        self.cod_emb = nn.Embedding(65, d_model)
        self.sim_proj = nn.Linear(1, d_model)
        self.angle_emb = nn.Embedding(k_angle_bins, d_model)
        self.pair_emb = nn.Embedding(4, d_model)
        self.bucket_emb = nn.Embedding(3, d_model)
        self.encoder = nn.ModuleList(
            [Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
        nn.init.xavier_uniform_(self.sim_proj.weight)
        nn.init.zeros_(self.sim_proj.bias)
        nn.init.xavier_uniform_(self.angle_emb.weight)
        nn.init.xavier_uniform_(self.pair_emb.weight)
        nn.init.xavier_uniform_(self.bucket_emb.weight)

    def forward(self, cod_ids, sim_feat, pad_mask, angle_bin=None, pair_bin=None, bucket_idx=None):
        tok = self.cod_emb(cod_ids)
        sim = self.sim_proj(sim_feat.unsqueeze(-1))
        x = tok + sim
        add = 0
        if angle_bin is not None:
            add = add + self.angle_emb(angle_bin.clamp(0, self.angle_emb.num_embeddings - 1))
        if pair_bin is not None:
            add = add + self.pair_emb(pair_bin.clamp(0, self.pair_emb.num_embeddings - 1))
        if bucket_idx is not None:
            add = add + self.bucket_emb(bucket_idx.clamp(0, self.bucket_emb.num_embeddings - 1))
        x = x + add
        for i, blk in enumerate(self.encoder):
            res = x
            x = self.norms[i](x)
            x = blk(x)
            x = x + res
        logits = self.head(x).squeeze(-1)
        logits = F.softplus(logits)
        logits = logits * pad_mask.float()
        return logits


class ExprHeadFullCounts(nn.Module):
    def __init__(self, hidden=64, use_log1p=True):
        super().__init__()
        self.use_log1p = use_log1p
        self.ff = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self.post = nn.Linear(1, 1)

    def forward(self, counts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.log1p(counts) if self.use_log1p else counts
        x = self.ff(x.unsqueeze(-1)).squeeze(-1)
        m = mask.float()
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (x * m).sum(dim=1, keepdim=True) / denom
        return self.post(pooled).squeeze(1)


class MambaExprFull(nn.Module):
    def __init__(self, base: RiboPolisherMamba, hidden=64, use_log1p=True):
        super().__init__()
        self.base = base
        self.expr_head = ExprHeadFullCounts(hidden=hidden, use_log1p=use_log1p)

    def forward(self, cod_ids, sim_feat, mask, angle_bin=None, pair_bin=None, bucket_idx=None):
        logits = self.base(cod_ids, sim_feat, mask, angle_bin=angle_bin, pair_bin=pair_bin, bucket_idx=bucket_idx)
        counts = torch.exp(torch.clamp(logits, max=20.0)) * mask.float()
        expr_pred = self.expr_head(counts, mask)
        return counts, expr_pred


def _pearson_safe(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm * xm).sum()) * np.sqrt((ym * ym).sum())
    return float((xm * ym).sum() / denom) if denom != 0 else float("nan")


def _rankdata_avg_ties(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    arr = a[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and arr[j] == arr[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    return ranks


def _spearman_safe(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    if _HAS_SCIPY:
        try:
            r = _scipy_spearmanr(x, y, nan_policy="omit").correlation
            return float(r) if np.isfinite(r) else float("nan")
        except Exception:
            pass
    return _pearson_safe(_rankdata_avg_ties(x), _rankdata_avg_ties(y))


def _tx_to_geom_name(tx: str) -> str:
    parts = tx.split("|")
    if len(parts) != 3:
        return tx.replace("|", "__")
    return f"{parts[0]}__{parts[1]}__{parts[2]}"


def _choose_input(sim_vec_scaled, sim_vec_raw, length):
    chosen = sim_vec_raw
    if chosen is None:
        chosen = np.zeros(length, dtype=np.float32)
    return np.asarray(chosen, dtype=np.float32)


def _load_geometry(cache_dir: Path, tx: str, length: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    gfile = cache_dir / "geometry" / f"{_tx_to_geom_name(tx)}.geo.pkl"
    if gfile.exists():
        try:
            obj = pickle.loads(gfile.read_bytes())
            ang = obj.get("angle_dev_sum")
            pc = obj.get("pair_count")
            if isinstance(ang, np.ndarray) and isinstance(pc, np.ndarray) and len(ang) == length and len(pc) == length:
                return ang.astype(np.float64), pc.astype(np.int64), True
        except Exception:
            pass
    return np.zeros(length, dtype=np.float64), np.zeros(length, dtype=np.int64), False


def _parse_sample_id(path_str: str) -> int:
    m = re.search(r"_sample_(\d+)", os.path.basename(path_str))
    return int(m.group(1)) if m else -1


class CSVPKLExprDataset(Dataset):
    def __init__(
        self,
        pkl_files: List[str],
        cache_dir: Path,
        expr_map: Dict[str, float],
        log_input: bool = True,
        require_geometry: bool = False,
    ):
        self.cache_dir = cache_dir
        self.log_input = log_input
        self.expr_map = expr_map
        self.require_geometry = require_geometry
        self.files: List[str] = []
        self._n_with_geo = 0
        self._n_without_geo = 0
        for p in pkl_files:
            try:
                with open(p, "rb") as fh:
                    tx, cods, *_rest = pickle.load(fh)
                length = len(cods)
                _, _, ok = _load_geometry(self.cache_dir, tx, length)
                if require_geometry and not ok:
                    self._n_without_geo += 1
                    continue
                if ok:
                    self._n_with_geo += 1
                else:
                    self._n_without_geo += 1
                self.files.append(p)
            except Exception:
                continue
        print(
            f"[DATASET] kept {len(self.files)} PKLs "
            f"(geometry ok={self._n_with_geo}, require_geometry={self.require_geometry})"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        with open(path, "rb") as fh:
            tx, cods, obs_counts, sim_vec_raw, sim_vec_scaled, scaling_factor, completed_total = pickle.load(fh)
        length = len(cods)
        cod_ids = np.array([CODON2IDX[c] for c in cods], dtype=np.int64)
        sim = _choose_input(sim_vec_scaled, sim_vec_raw, length)
        sim_feat = np.log1p(sim) if self.log_input else sim
        ang, pair, ok = _load_geometry(self.cache_dir, tx, length)
        if self.require_geometry and not ok:
            raise RuntimeError(f"Missing geometry for {tx} (len={length}) under {self.cache_dir}")
        angle_idx = np.clip(np.digitize(ang, ANGLE_BINS) - 1, 0, K_ANGLE_BINS - 1).astype(np.int64)
        pair_idx = np.clip(pair, 0, 3).astype(np.int64)
        bb = np.zeros(length, dtype=np.int64)
        l1 = length // 3
        l2 = (2 * length) // 3
        bb[l1:l2] = 1
        bb[l2:] = 2
        expr = float(self.expr_map.get(tx, float("nan")))
        return {
            "tx": tx,
            "length": length,
            "sample_id": _parse_sample_id(path),
            "cod_ids": torch.tensor(cod_ids, dtype=torch.long),
            "sim_feat": torch.tensor(sim_feat, dtype=torch.float32),
            "mask": torch.ones(length, dtype=torch.bool),
            "angle_bin": torch.tensor(angle_idx, dtype=torch.long),
            "pair_bin": torch.tensor(pair_idx, dtype=torch.long),
            "bucket_idx": torch.tensor(bb, dtype=torch.long),
            "expr": torch.tensor(expr, dtype=torch.float32),
        }


def pad_collate(batch):
    batch = sorted(batch, key=lambda x: x["length"], reverse=True)
    b = len(batch)
    lmax = batch[0]["length"]
    cod = torch.full((b, lmax), 64, dtype=torch.long)
    sim = torch.zeros((b, lmax), dtype=torch.float32)
    msk = torch.zeros((b, lmax), dtype=torch.bool)
    angle = torch.zeros((b, lmax), dtype=torch.long)
    pair = torch.zeros((b, lmax), dtype=torch.long)
    buck = torch.zeros((b, lmax), dtype=torch.long)
    expr = torch.zeros((b,), dtype=torch.float32)
    txs, sids = [], []
    for i, ex in enumerate(batch):
        length = ex["length"]
        txs.append(ex["tx"])
        sids.append(ex["sample_id"])
        cod[i, :length] = ex["cod_ids"]
        sim[i, :length] = ex["sim_feat"]
        msk[i, :length] = True
        angle[i, :length] = ex["angle_bin"]
        pair[i, :length] = ex["pair_bin"]
        buck[i, :length] = ex["bucket_idx"]
        expr[i] = ex["expr"]
    return {
        "tx": txs,
        "sample_id": sids,
        "cod_ids": cod,
        "sim_feat": sim,
        "mask": msk,
        "angle_bin": angle,
        "pair_bin": pair,
        "bucket_idx": buck,
        "expr": expr,
    }


def read_stasep_csv(stasep_csv: Path) -> Tuple[Dict[str, float], Dict[str, str]]:
    expr_map, split_map = {}, {}
    with stasep_csv.open("r", newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            tx = r["tx"]
            try:
                expr_map[tx] = float(r["expression_value"])
            except Exception:
                expr_map[tx] = float("nan")
            split_map[tx] = r.get("split", "").strip().lower()
    return expr_map, split_map


def pick_pkls_for_split(pkl_dir: Path, split_map: Dict[str, str], desired_split: str) -> List[str]:
    files = []
    for p in sorted(glob.glob(str(pkl_dir / "*_sample_*.pkl"))):
        base = os.path.basename(p)
        tx = base[: base.rfind("_sample_")]
        if split_map.get(tx, "") == desired_split:
            files.append(p)
    return files


def run_epoch(model, dl, device, optim=None):
    is_train = optim is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    nobs = 0
    preds, trues, txs = [], [], []
    mse = nn.MSELoss(reduction="sum")
    with torch.set_grad_enabled(is_train):
        for bo in dl:
            cod = bo["cod_ids"].to(device)
            sim = bo["sim_feat"].to(device)
            msk = bo["mask"].to(device)
            ang = bo["angle_bin"].to(device)
            pai = bo["pair_bin"].to(device)
            buc = bo["bucket_idx"].to(device)
            y = bo["expr"].to(device)
            _, yhat = model(cod, sim, msk, angle_bin=ang, pair_bin=pai, bucket_idx=buc)
            loss = mse(yhat, y)
            if is_train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            bs = y.size(0)
            total_loss += float(loss.item())
            nobs += bs
            preds.append(yhat.detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())
            txs.extend(bo["tx"])
    if nobs == 0:
        return {"mse": float("nan"), "pearson": float("nan"), "spearman": float("nan"), "preds": [], "trues": [], "tx": []}
    yhat = np.concatenate(preds) if preds else np.array([])
    ytru = np.concatenate(trues) if trues else np.array([])
    return {
        "mse": total_loss / nobs,
        "pearson": _pearson_safe(ytru, yhat),
        "spearman": _spearman_safe(ytru, yhat),
        "preds": yhat.tolist(),
        "trues": ytru.tolist(),
        "tx": txs,
    }


@torch.no_grad()
def _mc_predict_batch(model, bo, device, t: int):
    cod = bo["cod_ids"].to(device)
    sim = bo["sim_feat"].to(device)
    msk = bo["mask"].to(device)
    ang = bo["angle_bin"].to(device)
    pai = bo["pair_bin"].to(device)
    buc = bo["bucket_idx"].to(device)
    preds = []
    model.train()
    for _ in range(max(1, t)):
        _, yhat = model(cod, sim, msk, angle_bin=ang, pair_bin=pai, bucket_idx=buc)
        preds.append(yhat.detach().cpu().numpy())
    model.eval()
    arr = np.stack(preds, axis=0)
    mean = arr.mean(axis=0)
    var = arr.var(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
    return mean, var


def eval_select_confident(model, dl, device, mc_t: int = 8):
    samples = []
    for bo in dl:
        yt = bo["expr"].numpy()
        txs = bo["tx"]
        sids = bo["sample_id"]
        mean, var = _mc_predict_batch(model, bo, device, t=mc_t)
        for i in range(len(txs)):
            samples.append(
                {
                    "tx": txs[i],
                    "true": float(yt[i]),
                    "pred": float(mean[i]),
                    "var": float(var[i]),
                    "sample_id": int(sids[i]),
                }
            )
    best_by_tx: Dict[str, Dict] = {}
    for rec in samples:
        tx = rec["tx"]
        if tx not in best_by_tx or rec["var"] < best_by_tx[tx]["var"]:
            best_by_tx[tx] = rec
    chosen = list(best_by_tx.values())
    ytru = np.array([r["true"] for r in chosen], dtype=np.float64)
    yhat = np.array([r["pred"] for r in chosen], dtype=np.float64)
    mse = float(np.mean((yhat - ytru) ** 2)) if ytru.size else float("nan")
    return {
        "mse": mse,
        "pearson": _pearson_safe(ytru, yhat),
        "spearman": _spearman_safe(ytru, yhat),
        "preds": yhat.tolist(),
        "trues": ytru.tolist(),
        "tx": [r["tx"] for r in chosen],
        "picked_sample_id": [r["sample_id"] for r in chosen],
        "picked_var": [r["var"] for r in chosen],
    }


def _parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-line", choices=["ipsc", "hek293", "rpe", "lcl"], default="ipsc")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--pkl-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--stasep-csv", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--ff-hidden", type=int, default=128)
    parser.add_argument("--ff-use-log1p", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-geometry", action="store_true")
    parser.add_argument("--test-conf-select", action="store_true")
    parser.add_argument("--mc-passes", type=int, default=32)
    args = parser.parse_args()

    run_root = base_dir / "results" / "expression" / args.cell_line / "run"
    if args.weights is None:
        args.weights = base_dir / "results" / "polisher" / args.cell_line / "mamba_final_unscaled.pt"
        if not args.weights.exists():
            args.weights = base_dir.parents[1] / "weights" / "paper" / f"{args.cell_line}_mamba_final_unscaled.pt"
    print("Base polisher weights:", args.weights)
    if args.out_dir is None:
        args.out_dir = base_dir / "results" / "expression" / args.cell_line / "finetune"
    if args.pkl_dir is None:
        args.pkl_dir = run_root / "pkls"
    if args.cache_dir is None:
        args.cache_dir = run_root / "fold_cache"
    if args.stasep_csv is None:
        args.stasep_csv = run_root / "stasep_predicted_totals_vs_expression.csv"
    if args.device is None:
        args.device = "cuda"
    return args


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.pkl_dir.exists():
        raise RuntimeError(f"Missing PKL dir: {args.pkl_dir}")
    if not args.stasep_csv.exists():
        raise RuntimeError(f"Missing CSV: {args.stasep_csv}")
    (args.cache_dir / "geometry").mkdir(parents=True, exist_ok=True)

    expr_map, split_map = read_stasep_csv(args.stasep_csv)
    train_pkls = pick_pkls_for_split(args.pkl_dir, split_map, "train")
    val_pkls = pick_pkls_for_split(args.pkl_dir, split_map, "val")
    test_pkls = pick_pkls_for_split(args.pkl_dir, split_map, "test")
    if not val_pkls and len(train_pkls) >= 10:
        k = max(1, int(0.1 * len(train_pkls)))
        val_pkls = train_pkls[:k]
        train_pkls = train_pkls[k:]
    print(f"PKLs -> train={len(train_pkls)}  val={len(val_pkls)}  test={len(test_pkls)}")

    train_ds = CSVPKLExprDataset(train_pkls, args.cache_dir, expr_map, log_input=True, require_geometry=args.require_geometry)
    val_ds = (
        CSVPKLExprDataset(val_pkls, args.cache_dir, expr_map, log_input=True, require_geometry=args.require_geometry)
        if val_pkls
        else None
    )
    test_ds = CSVPKLExprDataset(test_pkls, args.cache_dir, expr_map, log_input=True, require_geometry=args.require_geometry)
    if len(train_ds) == 0:
        raise RuntimeError("No training samples after geometry filtering.")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate, pin_memory=True)
    val_dl = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate, pin_memory=True)
        if val_ds
        else None
    )
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate, pin_memory=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base = RiboPolisherMamba(d_model=192, n_layers=4, d_state=16, d_conv=4, dropout=0.1, k_angle_bins=K_ANGLE_BINS).to(
        device
    )
    state = torch.load(args.weights, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        base.load_state_dict(state["state_dict"])
    else:
        base.load_state_dict(state)

    model = MambaExprFull(base, hidden=args.ff_hidden, use_log1p=args.ff_use_log1p).to(device)
    if args.freeze_base:
        for p in model.base.parameters():
            p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")
    best_path = out_dir / "mamba_expr_full_best.pt"
    log = {
        "epochs": [],
        "train_mse": [],
        "train_pearson": [],
        "train_spearman": [],
        "val_mse": [],
        "val_pearson": [],
        "val_spearman": [],
    }

    for ep in range(1, args.epochs + 1):
        tr = run_epoch(model, train_dl, device, optim=optim)
        if val_dl is not None and len(val_ds) > 0:
            va = run_epoch(model, val_dl, device, optim=None)
            val_mse = va["mse"]
        else:
            va = {"mse": float("nan"), "pearson": float("nan"), "spearman": float("nan")}
            val_mse = tr["mse"]
        log["epochs"].append(ep)
        log["train_mse"].append(tr["mse"])
        log["train_pearson"].append(tr["pearson"])
        log["train_spearman"].append(tr["spearman"])
        log["val_mse"].append(va["mse"])
        log["val_pearson"].append(va["pearson"])
        log["val_spearman"].append(va["spearman"])
        print(
            f"[Ep {ep:03d}] train MSE={tr['mse']:.4f} rP={tr['pearson']:.3f} rS={tr['spearman']:.3f} | "
            f"val MSE={val_mse:.4f} rP={va['pearson']:.3f} rS={va['spearman']:.3f}"
        )
        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), best_path)

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    if args.test_conf_select:
        te = eval_select_confident(model, test_dl, device, mc_t=args.mc_passes)
        print(f"[TEST (conf-pick)] MSE={te['mse']:.4f}  rP={te['pearson']:.3f}  rS={te['spearman']:.3f}")
    else:
        te = run_epoch(model, test_dl, device, optim=None)
        print(f"[TEST] MSE={te['mse']:.4f}  rP={te['pearson']:.3f}  rS={te['spearman']:.3f}")

    with (out_dir / "train_log.json").open("w") as fh:
        json.dump(log, fh, indent=2)

    test_pred_csv = out_dir / "mamba_expr_full_test_predictions.csv"
    with test_pred_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        if args.test_conf_select:
            w.writerow(["tx", "expr_true", "expr_pred", "picked_sample_id", "picked_variance"])
            for tx, yt, yp, sid, vv in zip(te["tx"], te["trues"], te["preds"], te["picked_sample_id"], te["picked_var"]):
                w.writerow([tx, f"{float(yt):.6f}", f"{float(yp):.6f}", int(sid), f"{float(vv):.6e}"])
        else:
            w.writerow(["tx", "expr_true", "expr_pred"])
            for tx, yt, yp in zip(te["tx"], te["trues"], te["preds"]):
                w.writerow([tx, f"{float(yt):.6f}", f"{float(yp):.6f}"])

    acc = {}
    for tx, yt, yp in zip(te["tx"], te["trues"], te["preds"]):
        acc.setdefault(tx, [float(yt), []])[1].append(float(yp))
    y_tx = np.array([v[0] for v in acc.values()])
    p_tx = np.array([np.mean(v[1]) for v in acc.values()])
    per_tx = {"n_tx": int(len(y_tx)), "pearson_r": _pearson_safe(p_tx, y_tx), "spearman_rho": _spearman_safe(p_tx, y_tx)}
    print(f"[TEST] per-transcript mean over simulations: r={per_tx['pearson_r']:.3f} (n_tx={per_tx['n_tx']})")

    with (out_dir / "test_metrics.json").open("w") as fh:
        json.dump(
            {
                "per_transcript": per_tx,
                "n": len(te["preds"]),
                "mse": te["mse"],
                "pearson_r": te["pearson"],
                "spearman_rho": te["spearman"],
                "conf_pick": bool(args.test_conf_select),
                "mc_passes": int(args.mc_passes) if args.test_conf_select else 0,
            },
            fh,
            indent=2,
        )

    torch.save(model.state_dict(), out_dir / "mamba_expr_full_final.pt")
    print("Saved:")
    print(" ", best_path if best_path.exists() else "(no best saved)")
    print(" ", out_dir / "mamba_expr_full_final.pt")
    print(" ", test_pred_csv)
    print(" ", out_dir / "test_metrics.json")


if __name__ == "__main__":
    main()
