#!/usr/bin/env bash
# CPU. Full preprocessing: inputs, CDS models, A-site mapping (4 cell lines), datasets on the published
# splits (with folding), TE and expression datasets.
#   usage: GENOME_DIR=... GWIPS_DIR=... nohup bash jobs/cpu_00_preprocess_all.sh > logs/00_preprocess_all.log 2>&1 &
source "$(dirname "$0")/env.sh"
WORKERS=${WORKERS:-64}
bash pipeline/preprocess/fetch_inputs.sh > /dev/null
log "build_cds"; python -u pipeline/preprocess/build_cds.py
for c in ipsc hek293 lcl rpe; do log "map_asite $c"; python -u pipeline/preprocess/map_asite.py --cell-line $c; done
log "build_dataset"; python -u pipeline/preprocess/build_dataset.py --workers "$WORKERS"
log "TE workbook"; [ -s data/raw/zheng2025_te.tsv ] || python -u pipeline/preprocess/convert_te_workbook.py
log "TE datasets"; python -u pipeline/te/prepare.py
log "expression dataset"; python -u pipeline/expression/prepare.py --workers "$WORKERS"
log "done"
