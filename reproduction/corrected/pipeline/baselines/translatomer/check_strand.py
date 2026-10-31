#!/usr/bin/env python
"""Consistency checks of the window / strand convention (CPU only, no model).

For transcripts of both strands (records of data/<cell>/<split>.pkl):
  1. sequence: in every window, window_seq[window_index[k]] == CDS nucleotide k (data/cds.pkl,
     U -> T) for all CDS nucleotides inside the window; the training dataset's one-hot input
     decodes to the same string;
  2. coverage: every CDS nucleotide is inside at least one window;
  3. targets: per window, the binned target sums to the counts of the CDS nucleotides inside it,
     and the training dataset returns exactly these bins;
  4. orientation: for '-' strand transcripts, the first CDS nucleotide (start codon) is at a
     smaller window index than the last one (bins run 5'->3' along the transcript);
  5. round trip: feeding the target bins back as "predictions" through the "cds" bins->nt mapping
     reproduces each window's CDS count total exactly (no inflation), and with one window the
     codon-level total equals the observed total.
Exits non-zero on the first failure.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4]))   # repo root (seq2ribo package)
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import config  # noqa: E402
from common import load_records  # noqa: E402
from window_datasets import TrainWindows, load_cds  # noqa: E402
from windows import (BIN, Genome, aggregate_windows, bins_to_nt, codon_sum, target_bins,  # noqa: E402
                     transcript_windows)

DECODE = np.array(list("ATCGN"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell-line", default="ipsc", choices=config.CELL_LINES)
    p.add_argument("--split", default="test")
    p.add_argument("--subset", default="_01")
    p.add_argument("--n-per-strand", type=int, default=25)
    p.add_argument("--data-dir", type=Path, default=config.DATA)
    a = p.parse_args()
    cds = load_cds(config.DATA / "cds.pkl")
    genome = Genome(config.GENOME_FASTA)
    recs = load_records(a.cell_line, a.split, a.data_dir, a.subset)
    # both strands, and include multi-window (long genomic span) transcripts
    pick = []
    for strand in "+-":
        rs = [r for r in recs if cds[r[0]]["strand"] == strand]
        multi = [r for r in rs if len(transcript_windows(r[0], cds[r[0]])) > 1][:5]
        pick += multi + [r for r in rs if r not in multi][:a.n_per_strand - len(multi)]
    n_win = 0
    ds = TrainWindows(pick, cds, genome)
    k_ds = 0
    for tx, seq, counts, *_ in pick:
        e = cds[tx]
        dna = seq.replace("U", "T")
        wins = transcript_windows(tx, e)
        covered = np.zeros(len(seq), bool)
        pred_list, widx_list = [], []
        for _, chrom, strand, ws, widx in wins:
            wseq = genome.window(chrom, strand, ws)
            ok = widx >= 0
            got = "".join(wseq[i] for i in widx[ok])
            want = "".join(dna[k] for k in np.flatnonzero(ok))
            assert got == want, f"{tx} ({strand}) ws={ws}: window sequence != CDS sequence"
            x, y = ds[k_ds]; k_ds += 1
            assert "".join(DECODE[x.numpy().argmax(1)]) == re.sub("[^ACGT]", "N", wseq), f"{tx}: one-hot mismatch"
            tb = target_bins(widx, counts)
            assert np.array_equal(tb, y.numpy()), f"{tx}: dataset target differs"
            assert np.isclose(tb.sum(), np.asarray(counts, float)[ok].sum()), f"{tx}: target total"
            if strand == "-" and ok[0] and ok[-1]:
                assert widx[0] < widx[-1], f"{tx}: '-' strand window not in transcript orientation"
            v, ok2 = bins_to_nt(tb, widx, "cds")
            assert np.isclose(v.sum(), tb.sum()), f"{tx}: cds mapping not mass-conserving"
            vu, _ = bins_to_nt(tb, widx, "uniform")
            assert vu.sum() <= tb.sum() + 1e-6, f"{tx}: uniform mapping inflates"
            covered |= ok
            pred_list.append(tb); widx_list.append(widx)
            n_win += 1
        assert covered.all(), f"{tx}: CDS not fully covered"
        cod = codon_sum(aggregate_windows(pred_list, widx_list, len(seq), "cds"))
        if len(wins) == 1:
            assert np.isclose(cod.sum(), np.sum(counts)), f"{tx}: codon total != observed total"
        print(f"ok {tx} strand {e['strand']} cds {len(seq)} nt span {e['ends'][-1] - e['starts'][0]} "
              f"windows {len(wins)} obs {int(np.sum(counts))} roundtrip {cod.sum():.1f}")
    n_minus = sum(cds[r[0]]["strand"] == "-" for r in pick)
    print(f"ALL CHECKS PASSED: {len(pick)} transcripts ({n_minus} minus strand), {n_win} windows, bin {BIN} bp")


if __name__ == "__main__":
    main()
