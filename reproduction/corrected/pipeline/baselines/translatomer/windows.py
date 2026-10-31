"""Genomic windows, targets and prediction-to-codon mapping for the sequence-only Translatomer.

Strand convention (used identically for training, validation, test inference, TE and expression):
every model input is in TRANSCRIPT orientation. A window is the genomic interval [ws, ws + SEQLEN)
(0-based, half-open); for '-' strand transcripts its sequence is reverse-complemented, so window
index i corresponds to genomic coordinate
    '+': g = ws + i             '-': g = ws + SEQLEN - 1 - i
and output bins run 5'->3' along the transcript. Transcript-oriented per-nt A-site counts are placed
into the window through the CDS blocks of data/cds.pkl, so CDS nucleotide k (transcript order) lands
at window index window_index(...)[k]; by construction window_seq[window_index[k]] == cds_seq[k]
(checked by check_strand.py).

Windows (as in the published seq-only setup and supplement): SEQLEN = 65,536 bp, 1,024 output bins
of 64 bp. CDS genomic span <= SEQLEN: one window centred on the CDS midpoint; otherwise windows start
at the CDS start every STRIDE = 32,768 bp plus one final window ending at the CDS end. The same
tiling is used for training and inference.

Targets: per-window, per-nt counts (zero outside this transcript's CDS) summed into 64-bp bins.

Bins -> codons: a bin prediction is the model's estimate of the summed counts of the CDS nucleotides
inside that bin, so it is divided over those nucleotides ("cds" mapping, the inverse of the target
construction; mass-conserving). The alternative "uniform" mapping spreads it over all 64 nt (mass on
intronic/UTR positions is dropped). Neither assigns a whole bin to a codon. Per-nt values are
averaged over the windows covering a nucleotide and summed per codon.
"""
import os

import numpy as np
import pyfaidx

SEQLEN = 65_536
NBINS = 1024
BIN = SEQLEN // NBINS  # 64
STRIDE = SEQLEN // 2   # 32,768

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
# one-hot channel order of Translatomer: A, T, C, G, other
_LUT = np.full(256, 4, dtype=np.int64)
for _ch, _i in zip(b"ATCGatcg", [0, 1, 2, 3, 0, 1, 2, 3]):
    _LUT[_ch] = _i
_EYE5 = np.eye(5, dtype=np.float32)


def revcomp(seq):
    return seq.translate(_COMP)[::-1]


def onehot(seq):
    """DNA string -> [L, 5] float32 one-hot (A, T, C, G, other)."""
    return _EYE5[_LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]]


class Genome:
    """hg38 window extraction; positions outside the chromosome are filled with N.

    The pyfaidx handle is opened lazily per process so that DataLoader workers do not share a file
    offset with the parent.
    """

    def __init__(self, fasta):
        self.fasta = str(fasta)
        self._fa, self._pid = None, None

    def _handle(self):
        if self._pid != os.getpid():
            self._fa = pyfaidx.Fasta(self.fasta, as_raw=True, sequence_always_upper=True)
            self._pid = os.getpid()
        return self._fa

    def fetch(self, chrom, start, end):
        fa = self._handle()
        n = len(fa[chrom])
        s, e = max(start, 0), min(end, n)
        seq = fa[chrom][s:e] if s < e else ""
        return "N" * (s - start) + seq + "N" * (end - e)

    def window(self, chrom, strand, ws):
        """Window sequence in transcript orientation (reverse complement for '-')."""
        seq = self.fetch(chrom, ws, ws + SEQLEN)
        return revcomp(seq) if strand == "-" else seq


def cds_genomic_positions(entry):
    """Genomic coordinate of every CDS nucleotide, in transcript 5'->3' order."""
    g = np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in zip(entry["starts"], entry["ends"])])
    return g[::-1] if entry["strand"] == "-" else g


def window_starts(entry):
    """Genomic start coordinates of the windows tiling the CDS (strand independent)."""
    cs, ce = int(entry["starts"][0]), int(entry["ends"][-1])
    if ce - cs <= SEQLEN:
        return [(cs + ce) // 2 - SEQLEN // 2]
    return list(range(cs, ce - SEQLEN, STRIDE)) + [ce - SEQLEN]


def window_index(gpos, strand, ws):
    """Window index (transcript-oriented window) of each CDS nucleotide; outside -> -1."""
    w = gpos - ws if strand == "+" else (ws + SEQLEN - 1) - gpos
    return np.where((w >= 0) & (w < SEQLEN), w, -1)


def target_bins(widx, counts_nt):
    """Per-nt counts at window indices widx (-1 = not in window) summed into NBINS bins."""
    ok = widx >= 0
    return np.bincount(widx[ok] // BIN, weights=np.asarray(counts_nt, np.float64)[ok],
                       minlength=NBINS).astype(np.float32)


def bins_to_nt(pred_bins, widx, mode="cds"):
    """Per-nt values of the CDS nucleotides covered by one window (others 0) and a coverage mask.

    mode="cds": bin value divided by the number of CDS nucleotides in that bin (mass-conserving);
    mode="uniform": bin value / 64.
    """
    ok = widx >= 0
    b = widx[ok] // BIN
    if mode == "cds":
        denom = np.bincount(b, minlength=NBINS)[b].astype(np.float64)
    elif mode == "uniform":
        denom = float(BIN)
    else:
        raise ValueError(mode)
    v = np.zeros(len(widx), np.float64)
    v[ok] = np.asarray(pred_bins, np.float64)[b] / denom
    return v, ok


def aggregate_windows(pred_bins_list, widx_list, n_nt, mode="cds"):
    """Average per-nt predictions over all windows covering each CDS nucleotide."""
    tot = np.zeros(n_nt, np.float64)
    cov = np.zeros(n_nt, np.int64)
    for p, w in zip(pred_bins_list, widx_list):
        v, ok = bins_to_nt(p, w, mode)
        tot += v
        cov += ok
    if (cov == 0).any():
        raise RuntimeError(f"{int((cov == 0).sum())} CDS nucleotides not covered by any window")
    return tot / cov


def codon_sum(nt_values):
    L = len(nt_values) // 3
    return np.asarray(nt_values[:3 * L], np.float64).reshape(L, 3).sum(axis=1)


def transcript_windows(tx, entry):
    """[(tx, chrom, strand, ws, widx)] for every window of a transcript."""
    g = cds_genomic_positions(entry)
    return [(tx, entry["chrom"], entry["strand"], ws, window_index(g, entry["strand"], ws))
            for ws in window_starts(entry)]


def centered_placement(n_nt):
    """Expression constructs: CDS centred in an all-N window (as in the published expression script).

    Returns (left padding, window indices of the CDS nucleotides).
    """
    if n_nt > SEQLEN:
        raise ValueError(f"sequence of {n_nt} nt does not fit a {SEQLEN}-bp window")
    pad = (SEQLEN - n_nt) // 2
    return pad, pad + np.arange(n_nt, dtype=np.int64)
