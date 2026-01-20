"""Utility constants and functions for seq2ribo."""

import numpy as np
import torch

CODONS = [
    "AAA","AAC","AAG","AAU","ACA","ACC","ACG","ACU","AGA","AGC","AGG","AGU",
    "AUA","AUC","AUG","AUU","CAA","CAC","CAG","CAU","CCA","CCC","CCG","CCU",
    "CGA","CGC","CGG","CGU","CUA","CUC","CUG","CUU","GAA","GAC","GAG","GAU",
    "GCA","GCC","GCG","GCU","GGA","GGC","GGG","GGU","GUA","GUC","GUG","GUU",
    "UAA","UAC","UAG","UAU","UCA","UCC","UCG","UCU","UGA","UGC","UGG","UGU",
    "UUA","UUC","UUG","UUU"]

CODON2IDX = {c:i for i,c in enumerate(CODONS)}
STOP_CODONS = {"UAA", "UAG", "UGA"}
STOP_IDX = [CODON2IDX[c] for c in STOP_CODONS]
NONSTOP_IDX = [i for i in range(len(CODONS)) if i not in STOP_IDX]

# Angle bins for discretization
DEFAULT_ANGLE_BINS = np.array(
    [0.0, 1.0471990, 2.09439815, 3.14159722, 6.51421177],
    dtype=np.float64
)
K_ANGLE_BINS = len(DEFAULT_ANGLE_BINS) - 1

def build_full_rate(rate_fit):
    """Build full 64-codon rate array from fitted non-stop codon rates."""
    rate_full = np.ones(len(CODONS), dtype=np.float64)
    rate_full[NONSTOP_IDX] = rate_fit
    for i in STOP_IDX:
        rate_full[i] = 1.0 
    return rate_full

def load_state_dict_safely(ckpt_path, device):
    """
    Load a checkpoint safely, handling weights_only=True/False discrepancies.
    """
    try:
        sd = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(str(ckpt_path), map_location=device)
    return sd
