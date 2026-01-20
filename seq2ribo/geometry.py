"""Geometry feature computation using ViennaRNA for RNA secondary structure."""

import math
from pathlib import Path
from typing import Optional, Tuple, Union
import hashlib
import pickle

import numpy as np

try:
    import RNA
    HAS_VIENNARNA = True
except ImportError:
    HAS_VIENNARNA = False


def _compute_vertex_angles(seq: str, struct: str) -> Optional[list]:
    """Compute vertex angles from RNA secondary structure layout."""
    try:
        RNA.cvar.rna_plot_type = RNA.PLOT_TYPE_TURTLE
        coords = RNA.get_xy_coordinates(struct)
        if coords is None:
            return None
            
        xs, ys = [], []
        for i in range(len(seq)):
            p = coords.get(i)
            xs.append(p.X)
            ys.append(p.Y)

        vertex_angles = []
        for i in range(1, len(xs) - 1):
            v1x, v1y = xs[i] - xs[i - 1], ys[i] - ys[i - 1]
            v2x, v2y = xs[i + 1] - xs[i], ys[i + 1] - ys[i]
            n1 = math.hypot(v1x, v1y)
            n2 = math.hypot(v2x, v2y)
            if n1 == 0 or n2 == 0:
                ang = 0.0
            else:
                dot = v1x * v2x + v1y * v2y
                cosang = max(-1.0, min(1.0, dot / (n1 * n2)))
                ang = math.acos(cosang)
            vertex_angles.append(ang)
        
        vertex_angles = [0.0, 0.0] + vertex_angles
        if len(vertex_angles) < len(seq):
            vertex_angles = vertex_angles + [0.0] * (len(seq) - len(vertex_angles))
        elif len(vertex_angles) > len(seq):
            vertex_angles = vertex_angles[:len(seq)]
        return vertex_angles
    except Exception:
        return None


def compute_geometry_features(
    seq: str, 
    cache_dir: Optional[Union[str, Path]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute geometry features from RNA sequence using ViennaRNA.
    
    Args:
        seq: RNA sequence (codons, length must be multiple of 3)
        cache_dir: Directory for caching results
        
    Returns:
        Tuple of (angle_dev_sum, pair_count) arrays per codon
    """
    L = len(seq)
    L_cod = L // 3
    zeros_ang = np.zeros(L_cod, dtype=np.float64)
    zeros_pair = np.zeros(L_cod, dtype=np.int32)

    if L % 3 != 0:
        return zeros_ang, zeros_pair
    
    if not HAS_VIENNARNA:
        print("[WARN] ViennaRNA not installed. Returning zero geometry features.")
        return zeros_ang, zeros_pair

    # Normalize sequence
    seq = seq.upper().replace("T", "U")

    # Check cache
    if cache_dir is None:
        cache_dir = Path("cache/geometry")
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    seq_hash = hashlib.md5(seq.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{seq_hash}.pkl"
    
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data["angle_dev_sum"], data["pair_count"]
        except Exception as e:
            print(f"[WARN] Failed to load cache {cache_path}: {e}")

    # Compute folding
    try:
        fc = RNA.fold_compound(seq)
        if fc is None:
            return zeros_ang, zeros_pair
        struct, _ = fc.mfe()
        
        if not struct:
            return zeros_ang, zeros_pair

        # Get pairing info
        pt = RNA.ptable(struct)
        is_paired = [1 if pt[i + 1] > 0 else 0 for i in range(L)]
        
        # Get vertex angles
        va = _compute_vertex_angles(seq, struct)
        if va is None:
            return zeros_ang, zeros_pair

        # Aggregate to codons
        angle_arr = np.array(va, dtype=np.float64)
        pair_arr = np.array(is_paired, dtype=np.int32)
        
        angle_dev_sum = np.add.reduceat(angle_arr, np.arange(0, L, 3))
        pair_count = np.add.reduceat(pair_arr, np.arange(0, L, 3))

        # Save to cache
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"angle_dev_sum": angle_dev_sum, "pair_count": pair_count}, f)
        except Exception as e:
            print(f"[WARN] Failed to save cache {cache_path}: {e}")

        return angle_dev_sum, pair_count

    except Exception as e:
        print(f"[WARN] Geometry computation failed: {e}")
        return zeros_ang, zeros_pair
