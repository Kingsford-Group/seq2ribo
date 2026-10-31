import numpy as np


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
K_ANGLE_BINS = 4





def build_full_rate(rate_fit):
    rate_full = np.ones(len(CODONS), dtype=np.float64)
    rate_full[NONSTOP_IDX] = rate_fit
    for i in STOP_IDX:
        rate_full[i] = 1.0 
    return rate_full





