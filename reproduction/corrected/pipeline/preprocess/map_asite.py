#!/usr/bin/env python
"""Per-nucleotide A-site counts along every CDS, in transcript orientation.

Reads a GWIPS-viz bigWig through `bigWigToBedGraph` (UCSC), and for each CDS from
build_cds.py fills the coverage of every CDS base. A bedGraph record covers every base of
its interval with the same value. For minus-strand transcripts the vector is reversed so that
position i is nucleotide i of the CDS 5'->3'. Output (pickle): {tx_id: int32 array (len = CDS nt)}.
"""
import argparse
import pickle
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell-line", required=True, choices=config.CELL_LINES)
    p.add_argument("--bigwig", type=Path, default=None)
    p.add_argument("--cds", type=Path, default=config.SHARED / "cds.pkl")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--bigwig-to-bedgraph", default=shutil.which("bigWigToBedGraph") or "bigWigToBedGraph")
    return p.parse_args()


def load_bedgraph(path):
    """{chrom: (starts, ends, values)} sorted by start."""
    per = defaultdict(lambda: ([], [], []))
    with open(path) as fh:
        for line in fh:
            c, s, e, v = line.split()
            t = per[c]
            t[0].append(int(s)); t[1].append(int(e)); t[2].append(float(v))
    out = {}
    for c, (s, e, v) in per.items():
        s, e, v = np.array(s, np.int64), np.array(e, np.int64), np.array(v, np.float64)
        o = np.argsort(s, kind="stable")
        out[c] = (s[o], e[o], v[o])
    return out


def block_coverage(track, start, end):
    """Coverage of bases [start, end) from one chromosome's bedGraph arrays."""
    cov = np.zeros(end - start, dtype=np.float64)
    if track is None:
        return cov
    s, e, v = track
    lo = max(np.searchsorted(e, start, side="right"), 0)   # first record ending after start
    hi = np.searchsorted(s, end, side="left")               # records starting before end
    for k in range(lo, hi):
        a, b = max(s[k], start), min(e[k], end)
        if a < b:
            cov[a - start:b - start] += v[k]
    return cov


def main():
    a = parse_args()
    bw = a.bigwig or config.RAW / config.GWIPS_TRACKS[a.cell_line]
    out = a.out or config.SHARED / a.cell_line / "asite_counts.pkl"
    with open(a.cds, "rb") as f:
        cds = pickle.load(f)
    with tempfile.TemporaryDirectory(dir=config.SHARED) as tmp:
        bg = Path(tmp) / "track.bedGraph"
        subprocess.run([a.bigwig_to_bedgraph, str(bw), str(bg)], check=True)
        tracks = load_bedgraph(bg)
    counts = {}
    for tx, r in cds.items():
        t = tracks.get(r["chrom"])
        cov = np.concatenate([block_coverage(t, s, e) for s, e in zip(r["starts"], r["ends"])])
        if r["strand"] == "-":
            cov = cov[::-1]
        if len(cov) != len(r["seq"]):
            raise RuntimeError(f"{tx}: coverage length {len(cov)} != sequence length {len(r['seq'])}")
        counts[tx] = np.rint(cov).astype(np.int32)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(counts, f)
    nz = sum(1 for v in counts.values() if v.sum() > 0)
    print(f"[{a.cell_line}] {len(counts)} CDSs, {nz} with non-zero A-site total -> {out}")


if __name__ == "__main__":
    main()
