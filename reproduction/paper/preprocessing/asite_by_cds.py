#!/usr/bin/env python
"""Map nucleotide-level A-site coverage onto coding sequences.

Inputs
  --bed        CDS BED12 built from GTF "CDS" features (one record per transcript; GENCODE
               CDS features exclude the stop codon)
  --fasta      CDS sequences in transcript orientation (start codon through stop codon),
               headers = BED12 name
  --intersect  output of `bedtools intersect -a <asite.bedGraph> -b <cds.bed12> -wa -wb`
Output
  FASTA-like records: ">tx|length=L", the CDS sequence, and L space-separated counts,
  where count i belongs to nucleotide i of the CDS in transcript orientation. When the
  sequence is three nucleotides longer than the BED12 span (stop codon not covered by the
  BED12 blocks), the stop codon positions are left at 0.

--published-mapping reproduces the counts of the paper: count i is the i-th CDS nucleotide in
ascending genomic order (reversed relative to the transcript on the minus strand), each bedGraph
record contributes only to its first nucleotide, and every transcript with at least one counted record
is written (inconsistent records are removed later by the dataset filters).
"""
import argparse
import collections
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bed", required=True, type=Path)
    p.add_argument("--fasta", required=True, type=Path)
    p.add_argument("--intersect", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--published-mapping", action="store_true", help="counts as in the paper (see above)")
    return p.parse_args()


def read_fasta(path):
    seqs, name, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name, buf = line[1:].split()[0].split("|")[0].split("::")[0].split("(")[0], []
            elif line:
                buf.append(line)
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def cds_offsets(bed_path):
    """Return {tx: (block_starts, block_ends, cumulative_lengths, strand, total_length)}."""
    out = {}
    with open(bed_path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.rstrip("\n").split("\t")
            start, name, strand = int(f[1]), f[3], f[5]
            sizes = [int(x) for x in f[10].rstrip(",").split(",") if x]
            rel = [int(x) for x in f[11].rstrip(",").split(",") if x]
            starts = np.array([start + r for r in rel], dtype=np.int64)
            ends = starts + np.array(sizes, dtype=np.int64)
            cum = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
            out[name] = (starts, ends, cum, strand, int(sum(sizes)))
    return out


def transcript_index(g, blocks, published=False):
    """0-based transcript-orientation index of genomic base g, or None if outside the CDS blocks.

    published=True: index in ascending genomic order on both strands (paper)."""
    starts, ends, cum, strand, L = blocks
    k = np.searchsorted(starts, g, side="right") - 1
    if k < 0 or g >= ends[k]:
        return None
    i = int(cum[k] + (g - starts[k]))  # index in ascending genomic order
    # on the minus strand the start codon is at the highest genomic coordinate
    return L - 1 - i if strand == "-" and not published else i


def main():
    a = parse_args()
    seqs = read_fasta(a.fasta)
    blocks = cds_offsets(a.bed)
    values = collections.defaultdict(lambda: collections.defaultdict(int))
    with open(a.intersect) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            chrom, g0, g1, v, tx = f[0], int(f[1]), int(f[2]), int(float(f[3])), f[7]
            b = blocks.get(tx)
            if b is None:
                continue
            # a bedGraph record covers every base in [g0, g1) with the same value
            # (paper: only its first base)
            for g in (range(g0, g0 + 1) if a.published_mapping else range(g0, g1)):
                i = transcript_index(g, b, a.published_mapping)
                if i is not None:
                    values[tx][i] += v
    n = 0
    with open(a.out, "w") as out:
        for tx, vec in values.items():
            seq = seqs.get(tx)
            if seq is None:
                continue
            L = len(seq)
            if not a.published_mapping and L not in (blocks[tx][4], blocks[tx][4] + 3):
                continue
            out.write(f">{tx}|length={L}\n{seq}\n{' '.join(str(vec.get(i, 0)) for i in range(L))}\n")
            n += 1
    print(f"wrote {n} transcripts to {a.out}")


if __name__ == "__main__":
    main()
