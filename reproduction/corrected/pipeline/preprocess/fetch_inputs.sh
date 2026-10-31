#!/usr/bin/env bash
# Inputs for preprocessing, under data/raw:
#   GENCODE v48 comprehensive annotation GTF (downloaded here);
#   GRCh38/hg38 genome FASTA + .fai (GENOME_DIR, e.g. from
#     https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz, indexed with samtools faidx);
#   GWIPS-viz global aggregate elongating-ribosome (A-site) bigWigs (GWIPS_DIR, https://gwips.ucc.ie/):
#     Chen20_All.RiboProElong.bw, Wagner20_All.RiboProElong.bw, Cenik2015.bw, Haas20_All.RiboProElong.bw
#   usage: GENOME_DIR=... GWIPS_DIR=... bash pipeline/preprocess/fetch_inputs.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RAW=$ROOT/data/raw
mkdir -p "$RAW"
GTF=gencode.v48.annotation.gtf.gz
[ -s "$RAW/$GTF" ] || curl -fL -o "$RAW/$GTF" \
  "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_48/$GTF"
: "${GENOME_DIR:?set GENOME_DIR to the directory with hg38.fa and hg38.fa.fai}"
: "${GWIPS_DIR:?set GWIPS_DIR to the directory with the GWIPS-viz bigWigs}"
link() {  # link <file> into data/raw (absolute target; nothing to do if it is already there)
  local src; src=$(readlink -f "$1/$2")
  [ "$src" = "$(readlink -f "$RAW/$2")" ] || ln -sf "$src" "$RAW/$2"
}
link "$GENOME_DIR" hg38.fa
link "$GENOME_DIR" hg38.fa.fai
for bw in Chen20_All.RiboProElong.bw Wagner20_All.RiboProElong.bw Cenik2015.bw Haas20_All.RiboProElong.bw; do
  link "$GWIPS_DIR" "$bw"
done
ls -lL "$RAW"
