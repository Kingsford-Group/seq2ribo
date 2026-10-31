"""Shared pieces of the clean RiboNN TE baseline (export, train, predict).

RiboNN (Zheng et al. 2025, https://github.com/Sanofi-Public/RiboNN) is used as-is: the model class
(`src.model.RiboNN`) and the input encoder (`src.data.DataFrameDataset`) are imported from an upstream
checkout instead of being copied. The upstream code is under Sanofi's non-commercial license, which
allows academic copying only with its notice attached; importing keeps third-party code out of this
repository. The checkout is located by `--ribonn-root` or the RIBONN_ROOT environment variable and its
git commit and file hashes are written next to every trained model.

Architecture actually used by upstream RiboNN (and therefore here): one-hot nucleotides (4 channels) plus
one channel marking the first nucleotide of every codon, transcripts aligned at the start codon
(5' padding to a fixed 5'UTR length), an initial Conv1d followed by 10 [LayerNorm, ReLU, Conv1d, Dropout,
MaxPool] blocks, and a flatten -> dense -> BatchNorm -> linear head with one output per task.
There is no codon tokenization and no recurrent layer in `RiboNN` (the GRU lives in the unused `Saluki`).
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

DEFAULT_RIBONN_ROOT = None   # set --ribonn-root or RIBONN_ROOT
RESULTS_DIR = config.RESULTS / "baselines" / "ribonn"

SETTINGS = ("utr_cds", "cds_only")
MODES = ("individual", "multitask")
TASKS = list(config.CELL_LINES)                  # one individual task per cell line
LABEL_COLUMNS = {c: f"TE_{c}" for c in TASKS}    # RiboNN picks targets with the regex '^TE_'
SPLIT_NAMES = {"train": "train", "val": "valid", "test": "test"}  # ours -> RiboNN's split names

# Input geometry (fixed, so that a model's input length never depends on which transcripts are loaded).
# utr_cds : upstream human values; every transcript is aligned at its start codon.
# cds_only: no 5'UTR, the CDS starts at position 0; same pad_5_prime code path, shorter input.
GEOMETRY = {
    "utr_cds": dict(max_utr5_len=1381, max_cds_utr3_len=11937),
    "cds_only": dict(max_utr5_len=0, max_cds_utr3_len=11937),
}

# Hyperparameters: upstream config/conf.yml (commit 318e758) + the overrides that upstream src/main.py
# uses for the human models (pad_5_prime, AdamW lr 1e-4 / l2 1e-3, 10 conv layers, no shift
# augmentation, ReLU). Identical for individual and multitask models except `with_NAs`, which selects
# upstream's masked MSE (mean over observed labels) for the multitask model, whose union of transcripts
# has missing labels. With no missing labels the masked MSE equals F.mse_loss.
HPARAMS = dict(
    pad_5_prime=True,
    split_utr5_cds_utr3_channels=False,
    label_codons=True,
    label_3rd_nt_of_codons=False,
    label_utr5=False,
    label_utr3=False,
    label_splice_sites=False,
    label_up_probs=False,
    max_shift=0,
    symmetric_shift=True,
    residual=False,
    filters=64,
    kernel_size=5,
    conv_stride=1,
    conv_dilation=1,
    conv_padding=0,
    num_conv_layers=10,
    dropout=0.3,
    ln_epsilon=0.007,
    bn_momentum=0.9,
    activation="relu",
    optimizer="AdamW",
    lr=1e-4,
    l2_scale=1e-3,
    min_lr=1e-7,
    adam_beta1=0.9,
    adam_beta2=0.998,
)
# Training schedule: upstream src/train.py defaults (max_epochs 200, early stopping patience 20 and
# checkpoint selection on val_r2, gradient clipping 0.5) and conf.yml batch sizes (64 train, 128 eval).
# The LR schedule (ReduceLROnPlateau on val_loss, factor 0.5, patience 8) is inside upstream RiboNN.
TRAIN = dict(max_epochs=200, patience=20, monitor="val_r2", monitor_mode="max", gradient_clip_val=0.5,
             train_batch_size=64, eval_batch_size=128)


# ----------------------------------------------------------------------------------------- upstream
def import_ribonn(root=None):
    """Import upstream RiboNN; returns (RiboNN, DataFrameDataset, RiboNNDataModule, provenance)."""
    root = root or os.environ.get("RIBONN_ROOT", DEFAULT_RIBONN_ROOT)
    if not root:
        raise FileNotFoundError("no RiboNN checkout given (set --ribonn-root or RIBONN_ROOT)")
    root = Path(root).resolve()
    if not (root / "src" / "model.py").is_file():
        raise FileNotFoundError(f"no RiboNN checkout at {root} (set --ribonn-root or RIBONN_ROOT)")
    sys.path.insert(0, str(root))
    import src  # noqa: F401  (the upstream package is literally called `src`)
    if Path(src.__file__).resolve().parent != root / "src":
        raise ImportError(f"`src` resolved to {src.__file__}, not to the RiboNN checkout {root}")
    from src.data import DataFrameDataset, RiboNNDataModule
    from src.model import RiboNN
    return RiboNN, DataFrameDataset, RiboNNDataModule, ribonn_provenance(root)


def ribonn_provenance(root):
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    files = ["src/model.py", "src/data.py", "src/utils/helpers.py"]
    return {
        "root": str(root),
        "commit": git("rev-parse", "HEAD"),
        "modified_files": git("diff", "--name-only", "HEAD", "--", *files),
        "md5": {f: hashlib.md5((root / f).read_bytes()).hexdigest() for f in files},
    }


def len_after_conv(data_module_cls, setting):
    """Flattened length after the conv tower, via upstream RiboNNDataModule's own formula."""
    ns = SimpleNamespace(pad_5_prime=True, **GEOMETRY[setting],
                         **{k: HPARAMS[k] for k in ("conv_padding", "conv_dilation", "conv_stride",
                                                    "kernel_size", "num_conv_layers")})
    return data_module_cls.get_sequence_length_after_ConvBlocks(ns)


def model_config(setting, mode, num_targets, data_module_cls):
    cfg = dict(HPARAMS, **GEOMETRY[setting])
    cfg.update(num_targets=num_targets, with_NAs=(mode == "multitask"),
               len_after_conv=len_after_conv(data_module_cls, setting))
    return cfg


# ----------------------------------------------------------------------------------------- tables
def table_path(results_dir, setting, task):
    """Exported RiboNN-format table; task in TASKS or 'multitask'."""
    return Path(results_dir) / "inputs" / setting / f"{task}.tsv"


def read_table(results_dir, setting, task):
    df = pd.read_csv(table_path(results_dir, setting, task), sep="\t",
                     dtype={"tx_id": str, "split": str, "tx_sequence": str})
    return df


def check_geometry(df, setting):
    g = GEOMETRY[setting]
    bad = (df.utr5_size > g["max_utr5_len"]) | (df.tx_size - df.utr5_size > g["max_cds_utr3_len"])
    if bad.any():
        raise ValueError(f"{int(bad.sum())} transcripts exceed the {setting} input geometry {g}; "
                         "they would be dropped by RiboNN, which this baseline does not allow")


def encode(df, setting, dataset_cls, num_workers=4, batch_size=64):
    """One-hot encode every transcript once with upstream DataFrameDataset; uint8 (N, C, L) tensor."""
    check_geometry(df, setting)
    g = GEOMETRY[setting]
    ds = dataset_cls(df, None, g["max_utr5_len"], g["max_cds_utr3_len"],
                     g["max_utr5_len"] + g["max_cds_utr3_len"],
                     *[HPARAMS[k] for k in ("pad_5_prime", "split_utr5_cds_utr3_channels", "label_codons",
                                            "label_3rd_nt_of_codons", "label_utr5", "label_utr3",
                                            "label_splice_sites", "label_up_probs")])
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    out = torch.cat([x.to(torch.uint8) for x in dl])
    assert len(out) == len(df)
    return out


class EncodedDataset(torch.utils.data.Dataset):
    """(x float32, y float32) pairs from a pre-encoded uint8 tensor; y may contain NaN (multitask)."""

    def __init__(self, x, y):
        self.x, self.y = x, torch.as_tensor(np.asarray(y, np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i].float(), self.y[i]


@torch.no_grad()
def predict(model, x, device, batch_size=TRAIN["eval_batch_size"]):
    """(N, num_targets) predictions in eval mode, using upstream's _compute_output (shift averaging)."""
    model.eval().to(device)
    out = []
    for i in range(0, len(x), batch_size):
        out.append(model._compute_output(x[i:i + batch_size].float().to(device)).float().cpu())
    return torch.cat(out).numpy()


def metrics(pred, true, train_mean=None):
    """Pearson/Spearman/MSE over non-missing labels; r2_vs_train_mean compares to a constant predictor."""
    pred, true = np.asarray(pred, np.float64), np.asarray(true, np.float64)
    ok = ~np.isnan(true)
    pred, true = pred[ok], true[ok]
    const = np.ptp(pred) == 0
    m = {"n": int(ok.sum()),
         "pearson_r": float("nan") if const else float(pearsonr(pred, true)[0]),
         "spearman_r": float("nan") if const else float(spearmanr(pred, true)[0]),
         "mse": float(np.mean((pred - true) ** 2)),
         "pred_std": float(np.std(pred))}
    if train_mean is not None:
        base = float(np.mean((train_mean - true) ** 2))
        m["r2_vs_train_mean"] = 1.0 - m["mse"] / base
    return m
