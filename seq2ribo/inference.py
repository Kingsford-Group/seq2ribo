"""Main inference API for seq2ribo predictions.

Two model versions:
  "corrected" (default)  checkpoints of the corrected pipeline, weights/<cell>/ (see UPDATE.md)
  "paper"                checkpoints of the published paper, weights/paper/ (release asset, downloaded
                         automatically on first use)
"""

import contextlib
import hashlib
import json
import os
import tarfile
import tempfile
import urllib.request
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from . import constants as CONST
from .data import pad_collate
from .geometry import compute_geometry_features
from .models import MambaExprFull, MambaTEFull, MambaTEFullUTR, RiboPolisherMamba, mamba_reference_path
from .simulation import angle_bin_index, bucket_index, simulate_runs
from .utils import CODON2IDX, DEFAULT_ANGLE_BINS, load_state_dict_safely

NUC2IDX = {"A": 0, "U": 1, "G": 2, "C": 3}
NUC_PAD_IDX = 4
CELL_LINES = ("hek293", "lcl", "rpe", "ipsc")
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "weights"
# published checkpoints (release asset); SEQ2RIBO_PAPER_WEIGHTS_URL overrides the URL (e.g. a local mirror)
PAPER_WEIGHTS_URL = "https://github.com/Kingsford-Group/seq2ribo/releases/download/v0.2.0/seq2ribo-paper-weights.tar.gz"
PAPER_WEIGHTS_SHA256 = "924a193517bfc95a75f6dbad83e2af219709a141268833502af6c20ea0ed578d"
POLISHER = dict(d_model=192, n_layers=4, d_state=16, d_conv=4, expand=2, dropout=0.1)

# published checkpoints: file names, sTASEP parameters and head settings
PAPER_FILES = {
    "riboseq": "{c}_mamba_final_unscaled.pt",
    "te_cds": "{c}_mamba_te_full_final_cds.pt",
    "te_utr": "{c}_mamba_te_utr_final.pt",
    "protein": "{c}_mamba_expr_full_final.pt",
    "te_cds_transform": "{c}_te_transform_cds.json",
    "te_utr_transform": "{c}_te_transform_utr.json",
}
PAPER_TE_CDS_LOG1P = {"hek293": False, "lcl": True, "rpe": True, "ipsc": True}

# corrected checkpoints: file names in weights/<cell>/
CORRECTED_FILES = {
    "riboseq": "polisher.pt",
    "te_cds": "te_cds.pt",
    "te_utr": "te_utr.pt",
    "protein": "expression.pt",
    "te_cds_transform": "te_cds_transform.json",
    "te_utr_transform": "te_utr_transform.json",
}


def download_paper_weights(weights_dir: Optional[str] = None) -> Path:
    """Download the published checkpoints into <weights_dir>/paper/ (once) and return that folder."""
    root = Path(weights_dir) if weights_dir else DEFAULT_WEIGHTS
    target = root / "paper"
    if target.is_dir() and any(target.iterdir()):
        return target
    url = os.environ.get("SEQ2RIBO_PAPER_WEIGHTS_URL", PAPER_WEIGHTS_URL)
    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading the published checkpoints (75 MB) from {url} ...")
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        archive = Path(tmp) / "paper.tar.gz"
        try:
            urllib.request.urlretrieve(url, archive)
        except OSError as e:
            raise RuntimeError(f"Could not download {url} ({e}). Download it manually and run "
                               f"`tar -xzf seq2ribo-paper-weights.tar.gz -C {root}`.") from e
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != PAPER_WEIGHTS_SHA256:
            raise RuntimeError(f"Checksum mismatch for {url}: {digest}")
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers()
                       if m.isfile() and Path(m.name).parts[0] == "paper" and ".." not in Path(m.name).parts]
            try:
                tar.extractall(tmp, members=members, filter="data")
            except TypeError:   # Python without extraction filters
                tar.extractall(tmp, members=members)
        Path(tmp, "paper").rename(target)
    print(f"Published checkpoints saved to {target}")
    return target


def _paper_stasep(cell):
    p = {"hek293": "HEK293", "lcl": "LCL", "rpe": "RPE", "ipsc": "IPSC"}[cell]
    return {"tau": getattr(CONST, f"{p}_RATES"), "alpha": getattr(CONST, f"{p}_ALPHA"),
            "beta": getattr(CONST, f"{p}_BETA"), "gamma": getattr(CONST, f"{p}_BUCKETS")}


class Seq2Ribo:
    def __init__(self,
                 cell_line: str = "hek293",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 weights_dir: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 version: str = "corrected"):
        self.cell_line = cell_line.lower()
        if self.cell_line not in CELL_LINES:
            raise ValueError(f"Unknown cell line: {cell_line}. Supported: {', '.join(CELL_LINES)}")
        if version not in ("corrected", "paper"):
            raise ValueError("version must be 'corrected' or 'paper'")
        self.version = version
        self.device = torch.device(device)
        self.models = {}
        self.te_transforms = {}
        root = Path(weights_dir) if weights_dir else DEFAULT_WEIGHTS
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/geometry")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if version == "corrected":
            self.weights_dir = root / self.cell_line
            cfg_path = self.weights_dir / "config.json"
            if not cfg_path.exists():
                raise FileNotFoundError(f"{cfg_path} not found; pass weights_dir pointing to the weights/ folder.")
            cfg = json.load(open(cfg_path))
            s = cfg["stasep"]
            self.stasep = {"tau": s["tau"], "alpha": s["alpha_pair"], "beta": s["beta_angle"], "gamma": s["gamma_bucket"]}
            self.default_init_p = float(s["init_p"])
            self.angle_edges = np.asarray(cfg["angle_edges"], dtype=np.float64)
            self.heads = cfg["heads"]
            self.files = {k: v for k, v in CORRECTED_FILES.items()}
        else:
            if (root / PAPER_FILES["riboseq"].format(c=self.cell_line)).exists():
                self.weights_dir = root          # weights_dir points at the published files themselves
            else:
                self.weights_dir = download_paper_weights(root)
            self.stasep = _paper_stasep(self.cell_line)
            self.default_init_p = 0.01
            self.angle_edges = DEFAULT_ANGLE_BINS
            self.heads = {"te_cds": {"hidden": 256, "log1p": PAPER_TE_CDS_LOG1P[self.cell_line]},
                          "te_utr": {"d_te": 128, "n_te_layers": 2, "log1p": False},
                          "expression": {"hidden": 128, "log1p": False}}
            self.files = {k: v.format(c=self.cell_line) for k, v in PAPER_FILES.items()}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _normalize_rna(seq: str) -> str:
        return seq.replace("T", "U").replace("t", "u").upper()

    @staticmethod
    def _tokenize_nucleotides(seq: str) -> np.ndarray:
        return np.array([NUC2IDX.get(ch, NUC2IDX["A"]) for ch in Seq2Ribo._normalize_rna(seq)], dtype=np.int64)

    @staticmethod
    def _ensure_sequence_list(seq_input: Union[str, List[str], tuple], name: str) -> List[str]:
        if isinstance(seq_input, str):
            return [seq_input]
        if isinstance(seq_input, (list, tuple)) and all(isinstance(x, str) for x in seq_input):
            return list(seq_input)
        raise ValueError(f"'{name}' must be a string or list/tuple of strings.")

    @staticmethod
    def _pad_collate_te_utr(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        out = pad_collate(batch)
        for key in ("utr5", "utr3"):
            n = max(1, max(x[f"{key}_len"] for x in batch))
            ids = torch.full((len(batch), n), NUC_PAD_IDX, dtype=torch.long)
            mask = torch.zeros((len(batch), n), dtype=torch.bool)
            for i, x in enumerate(batch):
                if x[f"{key}_len"]:
                    ids[i, :x[f"{key}_len"]] = x[f"{key}_ids"]
                    mask[i, :x[f"{key}_len"]] = True
            out[f"{key}_ids"], out[f"{key}_mask"] = ids, mask
        return out

    def _path(self, key: str) -> Path:
        p = self.weights_dir / self.files[key]
        if not p.exists():
            raise FileNotFoundError(f"{p} not found (version='{self.version}'). "
                                    + ("Download the paper weights from the release assets into weights/paper/."
                                       if self.version == "paper" else "Pass weights_dir pointing to the weights/ folder."))
        return p

    def _load_te_transform(self, use_utr: bool = False):
        key = "te_utr_transform" if use_utr else "te_cds_transform"
        if key not in self.te_transforms:
            t = json.load(open(self._path(key)))
            if t.get("name") != "minmax" or float(t["hi"]) <= float(t["lo"]):
                raise ValueError(f"invalid TE transform in {self._path(key)}")
            self.te_transforms[key] = t
        return self.te_transforms[key]

    @staticmethod
    def _inverse_minmax_te(te_scaled_np: np.ndarray, t: Dict[str, float]) -> np.ndarray:
        lo, hi, eps = float(t["lo"]), float(t["hi"]), float(t["eps"])
        return lo + (hi - lo) * np.clip(te_scaled_np, eps, 1.0 - eps)

    def _load_model(self, task: str, use_utr: bool = False):
        key = "te_utr" if task == "te" and use_utr else "te_cds" if task == "te" else task
        if key in self.models:
            return self.models[key]
        paper = self.version == "paper"
        base = RiboPolisherMamba(**POLISHER, k_angle_bins=len(self.angle_edges) - 1)
        if key == "riboseq":
            model = base
        elif key == "te_cds":
            h = self.heads["te_cds"]
            model = MambaTEFull(base, hidden=h["hidden"], use_log1p=h["log1p"],
                                base_transform="expm1" if paper else "none")
        elif key == "te_utr":
            h = self.heads["te_utr"]
            model = MambaTEFullUTR(base, d_te=h["d_te"], n_te_layers=h["n_te_layers"], te_d_state=16, te_d_conv=4,
                                   te_expand=2, dropout=0.1, use_log1p=h["log1p"],
                                   base_transform="expm1" if paper else "none")
        elif key == "protein":
            h = self.heads["expression"]
            model = MambaExprFull(base, hidden=h["hidden"], use_log1p=h["log1p"],
                                  base_transform="exp" if paper else "none")
        else:
            raise ValueError(f"Unknown task: {task}")
        path = self._path(key)
        print(f"Loading {self.version} weights from {path}...")
        state = load_state_dict_safely(path, self.device)
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        model.to(self.device).eval()
        self.models[key] = model
        return model

    # ------------------------------------------------------------------ prediction
    def predict(
        self,
        sequences: Optional[Union[str, List[str]]] = None,
        task: str = "riboseq",
        geomap: Optional[Dict] = None,
        use_polisher: bool = True,
        n_stasep_runs: int = 100,
        init_p: Optional[float] = None,
        return_scaled_te: bool = False,
        use_utr: bool = False,
        utr5_list: Optional[Union[str, List[str]]] = None,
        cds_list: Optional[Union[str, List[str]]] = None,
        utr3_list: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        batch_size: int = 64,
    ):
        """
        Run prediction pipeline:
        1. n_stasep_runs independent sTASEP simulations per sequence
        2. polish every run separately (if use_polisher=True) and average the predictions
           per sequence; with use_polisher=False the simulated counts are averaged instead

        task: "riboseq" (per-codon profile), "te" (translation efficiency; use_utr=True with
        utr5_list/cds_list/utr3_list for the CDS+UTR head) or "protein" (expression).
        n_stasep_runs: simulations per sequence (default 100, as used to evaluate the models).
        init_p: sTASEP initiation probability (default: the value used for training, 0.01).
        geomap: optional precomputed structure features {"seq_<i>": (angle_dev_sum, pair_count)},
        keyed by the position i of the sequence in the input, one value per codon of each array.
        seed: if given, the simulations of each sequence are seeded from (seed, sequence), so a
        sequence gets the same prediction whether it is predicted alone or in a batch.
        Runs on CPU too (slower): the Mamba layers then use their pure-PyTorch implementation.
        """
        if use_utr and task != "te":
            raise ValueError("use_utr=True is only supported for task='te'.")
        if use_utr:
            if sequences is not None:
                raise ValueError("For TE+UTR, pass utr5_list, cds_list and utr3_list instead of sequences.")
            if utr5_list is None or cds_list is None or utr3_list is None:
                raise ValueError("TE+UTR requires utr5_list, cds_list, and utr3_list.")
            utr5_values = self._ensure_sequence_list(utr5_list, "utr5_list")
            sequence_values = self._ensure_sequence_list(cds_list, "cds_list")
            utr3_values = self._ensure_sequence_list(utr3_list, "utr3_list")
            if not (len(utr5_values) == len(sequence_values) == len(utr3_values)):
                raise ValueError("utr5_list, cds_list, and utr3_list must have equal lengths.")
        else:
            if sequences is None:
                raise ValueError("Please provide 'sequences' for this prediction mode.")
            sequence_values = self._ensure_sequence_list(sequences, "sequences")
        if n_stasep_runs < 1:
            raise ValueError("n_stasep_runs must be >= 1.")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if not use_polisher and task != "riboseq":
            raise ValueError("use_polisher=False returns the simulated sTASEP profiles; it applies to task='riboseq' only.")
        init_p = self.default_init_p if init_p is None else init_p
        geomap = geomap or {}
        if use_polisher:
            self._load_model(task, use_utr=use_utr)   # unknown task or missing weights fail before the simulations
        print(f"Running sTASEP simulation for {len(sequence_values)} sequences with {n_stasep_runs} runs "
              f"per sequence (init_p={init_p}, version={self.version})...")

        # 1. sTASEP simulations and per-sequence model inputs
        items = []
        for i, seq_raw in enumerate(sequence_values):
            seq = self._normalize_rna(seq_raw)
            if len(seq) % 3 != 0:
                raise ValueError(f"Sequence {i} has length {len(seq)} which is not divisible by 3.")
            if not set(seq) <= set("ACGU"):
                raise ValueError(f"Sequence {i} contains characters other than A, C, G, U/T: "
                                 f"{''.join(sorted(set(seq) - set('ACGU')))}")
            tx_id = f"seq_{i}"
            if tx_id in geomap:
                angle_sum, pair_count = geomap[tx_id]
                if not len(angle_sum) == len(pair_count) == len(seq) // 3:
                    raise ValueError(f"geomap['{tx_id}'] needs one angle and one pair value per codon ({len(seq) // 3}).")
            else:
                angle_sum, pair_count = compute_geometry_features(seq, cache_dir=self.cache_dir, version=self.version)
            s = self.stasep
            cods, runs = simulate_runs(seq, angle_sum, pair_count, self.angle_edges, s["tau"], s["alpha"], s["beta"],
                                       s["gamma"], n_stasep_runs, init_p,
                                       seed=None if seed is None else zlib.crc32(f"{seed}:{seq}".encode()))
            L = len(cods)
            item = {
                "length": L,
                "runs": runs,
                "cod_ids": torch.tensor([CODON2IDX.get(c, 64) for c in cods], dtype=torch.long),
                "angle_bin": torch.tensor(angle_bin_index(angle_sum, self.angle_edges), dtype=torch.long),
                "pair_bin": torch.tensor(np.clip(pair_count, 0, 3), dtype=torch.long),
                "bucket_idx": torch.tensor(bucket_index(L), dtype=torch.long),
            }
            if use_utr:
                u5, u3 = self._tokenize_nucleotides(utr5_values[i]), self._tokenize_nucleotides(utr3_values[i])
                item.update(utr5_ids=torch.tensor(u5), utr3_ids=torch.tensor(u3), utr5_len=len(u5), utr3_len=len(u3))
            items.append(item)

        if not use_polisher:
            return [it["runs"].mean(axis=0).astype(np.float32) for it in items]
        if not items:
            return []

        # 2. one model row per (sequence, run): each run is polished on its own, as in training
        rows = []
        for j, it in enumerate(items):
            for r in range(it["runs"].shape[0]):
                row = {k: v for k, v in it.items() if k != "runs"}
                row["tx"] = f"seq_{j}_run_{r}"
                row["sim_feat"] = torch.tensor(np.log1p(it["runs"][r]), dtype=torch.float32)
                rows.append((j, row))

        model = self._load_model(task, use_utr=use_utr)
        te_transform = self._load_te_transform(use_utr=use_utr) if task == "te" and not return_scaled_te else None

        sums = [None] * len(items)
        reference = mamba_reference_path() if self.device.type == "cpu" else contextlib.nullcontext()
        with torch.no_grad(), reference:
            for start in range(0, len(rows), batch_size):
                chunk = rows[start:start + batch_size]
                batch_rows = [r for _, r in chunk]
                batch = self._pad_collate_te_utr(batch_rows) if use_utr else pad_collate(batch_rows)
                if batch["tx"] != [r["tx"] for r in batch_rows]:
                    raise RuntimeError("Internal error: collate reordered the batch.")
                cod, sim, msk = (batch[k].to(self.device) for k in ("cod_ids", "sim_feat", "mask"))
                kw = {k: batch[k].to(self.device) for k in ("angle_bin", "pair_bin", "bucket_idx")}
                if task == "riboseq":
                    preds = model(cod, sim, msk, **kw)
                    if self.version == "paper":   # published polisher outputs log(1 + count)
                        preds = torch.expm1(preds)
                elif use_utr:
                    _, preds = model(cod, sim, msk, batch["utr5_ids"].to(self.device), batch["utr3_ids"].to(self.device),
                                     batch["utr5_mask"].to(self.device), batch["utr3_mask"].to(self.device), **kw)
                else:
                    _, preds = model(cod, sim, msk, **kw)
                preds = preds.cpu().numpy().astype(np.float64)
                for (j, row), p in zip(chunk, preds):
                    if task == "riboseq":
                        val = p[:row["length"]]
                    elif te_transform is not None:
                        val = self._inverse_minmax_te(np.array([p]), te_transform)[0]
                    else:
                        val = float(p)
                    sums[j] = val if sums[j] is None else sums[j] + val

        results = []
        for j, it in enumerate(items):
            mean = sums[j] / it["runs"].shape[0]
            results.append(mean.astype(np.float32) if task == "riboseq" else float(mean))
        return results
