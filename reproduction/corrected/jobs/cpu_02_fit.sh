#!/usr/bin/env bash
# CPU. sTASEP then TASEP fits for the given cell lines (default all four), sequential.
#   usage: nohup bash jobs/cpu_02_fit.sh [cell ...] > logs/02_fit.log 2>&1 &
source "$(dirname "$0")/env.sh"
WORKERS=${WORKERS:-80}
CELLS=${*:-"ipsc hek293 lcl rpe"}
for c in $CELLS; do
  for m in stasep tasep; do
    log "fit $m $c start"
    python -u pipeline/fit/fit.py --model $m --cell-line $c --workers "$WORKERS" > logs/fit_${m}_${c}.log 2>&1
    log "fit $m $c done"
  done
done
