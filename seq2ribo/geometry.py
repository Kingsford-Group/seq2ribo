"""Per-codon RNA structure features (ViennaRNA MFE structure and its 2D layout).

version="corrected" (default): the turning angle of nucleotide k is assigned to nucleotide k.
version="paper": features as used for the published checkpoints (the turning angle of nucleotide k
is assigned to nucleotide k+1, and a failed fold or layout gives zero features).
"""

import hashlib
import math
import os
import pickle
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

# cache-file tag per feature version (bump when a feature definition changes)
CACHE_TAG = {"corrected": "g2", "paper": "g1"}

try:
    import RNA
    HAS_VIENNARNA = True
except ImportError:
    HAS_VIENNARNA = False


def _turning_angles(struct: str, n: int) -> Optional[list]:
    """Turning angle (radians) at nucleotides 1..n-2 between the bond vectors (k-1 -> k) and (k -> k+1)."""
    try:
        RNA.cvar.rna_plot_type = RNA.PLOT_TYPE_TURTLE
        coords = RNA.get_xy_coordinates(struct)
        if coords is None:
            return None
        xs = [coords.get(i).X for i in range(n)]
        ys = [coords.get(i).Y for i in range(n)]
        out = []
        for i in range(1, n - 1):
            v1x, v1y = xs[i] - xs[i - 1], ys[i] - ys[i - 1]
            v2x, v2y = xs[i + 1] - xs[i], ys[i + 1] - ys[i]
            n1, n2 = math.hypot(v1x, v1y), math.hypot(v2x, v2y)
            if n1 == 0 or n2 == 0:
                out.append(0.0)
            else:
                out.append(math.acos(max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))))
        return out
    except Exception:
        return None


def fold_features(seq: str, version: str = "corrected") -> Tuple[np.ndarray, np.ndarray]:
    """Per-codon structure features of an RNA CDS (length multiple of 3).

    Returns (angle_sum, pair_count): angle_sum[j] is the sum of the turning angles of the three
    nucleotides of codon j along the 2D layout of the MFE structure; pair_count[j] in {0..3} is the
    number of paired nucleotides in codon j.
    """
    if not HAS_VIENNARNA:
        raise ImportError("ViennaRNA (python module RNA) is required for structure features")
    seq = seq.upper().replace("T", "U")
    L = len(seq)
    if L == 0 or L % 3:
        raise ValueError(f"sequence length {L} is not a positive multiple of 3")
    struct, _ = RNA.fold_compound(seq).mfe()
    if not struct or len(struct) != L:
        raise RuntimeError("MFE folding failed")
    pt = RNA.ptable(struct)
    paired = np.array([1 if pt[i + 1] > 0 else 0 for i in range(L)], dtype=np.int64)
    ang = _turning_angles(struct, L)
    if ang is None:
        raise RuntimeError("2D layout failed")
    if version == "corrected":
        per_nt = [0.0] + ang + [0.0]
    elif version == "paper":
        per_nt = ([0.0, 0.0] + ang + [0.0] * L)[:L]
    else:
        raise ValueError(f"unknown version {version!r}")
    idx = np.arange(0, L, 3)
    return np.add.reduceat(np.asarray(per_nt, dtype=np.float64), idx), np.add.reduceat(paired, idx)


def compute_geometry_features(
    seq: str,
    cache_dir: Optional[Union[str, Path]] = None,
    version: str = "corrected",
) -> Tuple[np.ndarray, np.ndarray]:
    """fold_features() with an optional on-disk cache keyed by sequence and feature version."""
    if version not in CACHE_TAG:
        raise ValueError(f"unknown version {version!r}")
    seq = seq.upper().replace("T", "U")
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{hashlib.md5(seq.encode('utf-8')).hexdigest()}.{CACHE_TAG[version]}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data["angle_sum"], data["pair_count"]
    try:
        angle_sum, pair_count = fold_features(seq, version)
    except RuntimeError:
        if version != "paper":
            raise
        L = len(seq) // 3  # published behaviour: zero features when folding or the layout fails
        angle_sum, pair_count = np.zeros(L, dtype=np.float64), np.zeros(L, dtype=np.int64)
    if cache_path is not None:
        tmp = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"angle_sum": angle_sum, "pair_count": pair_count}, f)
        os.replace(tmp, cache_path)  # atomic: readers never see a partial file
    return angle_sum, pair_count
