#!/usr/bin/env python
"""Coding sequences from a GENCODE GTF and the genome FASTA.

For every transcript with CDS features, the CDS blocks and the stop_codon feature are merged
(GENCODE CDS features exclude the stop codon), giving the interval from the first base of the
start codon to the last base of the stop codon. Output (pickle):
  {tx_id: {"gene_id", "chrom", "strand", "starts", "ends", "seq"}}
with 0-based half-open blocks in ascending genomic order and `seq` the RNA sequence in
transcript orientation.
"""
import argparse
import gzip
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


class Fasta:
    """Minimal indexed FASTA reader (.fai required)."""

    def __init__(self, path):
        self.fh = open(path, "rb")
        self.idx = {}
        with open(str(path) + ".fai") as f:
            for line in f:
                name, length, offset, linebases, linewidth = line.split("\t")[:5]
                self.idx[name] = (int(length), int(offset), int(linebases), int(linewidth))

    def fetch(self, chrom, start, end):
        length, offset, lb, lw = self.idx[chrom]
        if not (0 <= start <= end <= length):
            raise ValueError(f"{chrom}:{start}-{end} outside 0-{length}")
        first = offset + (start // lb) * lw + start % lb
        last = offset + (end // lb) * lw + end % lb
        self.fh.seek(first)
        return self.fh.read(last - first).replace(b"\n", b"").replace(b"\r", b"").decode().upper()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gtf", type=Path, default=config.RAW / f"gencode.v{config.GENCODE_RELEASE}.annotation.gtf.gz")
    p.add_argument("--genome", type=Path, default=config.GENOME_FASTA)
    p.add_argument("--out", type=Path, default=config.SHARED / "cds.pkl")
    return p.parse_args()


def main():
    a = parse_args()
    blocks = defaultdict(list)
    meta = {}
    attr_re = re.compile(r'(\S+) "([^"]*)"')
    opener = gzip.open if str(a.gtf).endswith(".gz") else open
    with opener(a.gtf, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] not in ("CDS", "stop_codon"):
                continue
            attrs = dict(attr_re.findall(f[8]))
            tx = attrs["transcript_id"]
            blocks[tx].append((int(f[3]) - 1, int(f[4])))  # GTF 1-based inclusive -> 0-based half-open
            meta[tx] = (attrs["gene_id"], f[0], f[6])
    genome = Fasta(a.genome)
    out, skipped = {}, 0
    for tx, bl in blocks.items():
        gene, chrom, strand = meta[tx]
        if chrom not in genome.idx:
            skipped += 1
            continue
        merged = []
        for s, e in sorted(bl):
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        seq = "".join(genome.fetch(chrom, s, e) for s, e in merged)
        if strand == "-":
            seq = seq.translate(COMP)[::-1]
        out[tx] = {"gene_id": gene, "chrom": chrom, "strand": strand,
                   "starts": np.array([s for s, _ in merged], dtype=np.int64),
                   "ends": np.array([e for _, e in merged], dtype=np.int64),
                   "seq": seq.replace("T", "U")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(out, f)
    print(f"{len(out)} transcripts with CDS written to {a.out} ({skipped} on contigs missing from the genome)")


if __name__ == "__main__":
    main()
