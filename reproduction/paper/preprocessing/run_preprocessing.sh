#!/usr/bin/env bash
# Build one cell line's dataset from a GWIPS-viz A-site bigWig.
#
# usage: run_preprocessing.sh <asite.bw> <gencode.annotation.gtf> <genome.fa> <cell_line> <out_dir> [work_dir]
#   e.g. run_preprocessing.sh Wagner20_All.RiboProElong.bw gencode.v48.annotation.gtf hg38.fa hek293 \
#        reproduction/paper/data/hek293
#
# requires on PATH: bigWigToBedGraph, gtfToGenePred, genePredToBed (UCSC), gffread, bedtools
set -euo pipefail
BW=$1; GTF=$2; GENOME=$3; CELL=$4; OUT=$5; WORK=${6:-$OUT/preprocessing}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$WORK"

# CDS annotation: BED12 from GTF CDS features, CDS sequences (start through stop codon)
if [ ! -s "$WORK/cds.bed12" ]; then
  awk -F'\t' '$3=="CDS"' "$GTF" > "$WORK/cds_only.gtf"
  gtfToGenePred "$WORK/cds_only.gtf" "$WORK/cds_only.gp"
  genePredToBed "$WORK/cds_only.gp" "$WORK/cds.bed12"
fi
[ -s "$WORK/cds.fa" ] || gffread -x "$WORK/cds.fa" -g "$GENOME" "$GTF"

# A-site coverage on CDS exons
bigWigToBedGraph "$BW" "$WORK/asite.bedGraph"
bedtools intersect -a "$WORK/asite.bedGraph" -b "$WORK/cds.bed12" -wa -wb -split > "$WORK/asite_x_cds.tsv"
python "$HERE/asite_by_cds.py" --bed "$WORK/cds.bed12" --fasta "$WORK/cds.fa" \
  --intersect "$WORK/asite_x_cds.tsv" --out "$WORK/asite_by_cds.fa" --published-mapping

# published splits, folding/geometry
python "$HERE/build_dataset.py" --counts "$WORK/asite_by_cds.fa" --cell-line "$CELL" --out-dir "$OUT" \
  --workers "${WORKERS:-8}"
