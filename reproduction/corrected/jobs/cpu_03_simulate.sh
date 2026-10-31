#!/usr/bin/env bash
# CPU. Simulation runs with the fitted parameters of each cell line (default all four):
#   sTASEP 32 runs/transcript (polisher train/val/test), TASEP 32 runs (test, TASEP column),
#   sTASEP 100 runs per TE transcript, sTASEP 1000 runs per mRFP sequence.
#   usage: nohup bash jobs/cpu_03_simulate.sh [cell ...] > logs/03_simulate.log 2>&1 &
source "$(dirname "$0")/env.sh"
WORKERS=${WORKERS:-64}
CELLS=${*:-"ipsc hek293 lcl rpe"}
R=results
for c in $CELLS; do
  ST=$R/fit/stasep/$c/params.json; TA=$R/fit/tasep/$c/params.json
  log "$c: polisher shards"
  python -u pipeline/simulate/simulate.py --params $ST --runs 32 --cell-line $c --out-dir $R/sims/stasep/$c --workers $WORKERS
  log "$c: TASEP test shards"
  python -u pipeline/simulate/simulate.py --params $TA --runs 32 --cell-line $c --splits test --out-dir $R/sims/tasep/$c --workers $WORKERS
  log "$c: TE shards"
  for s in train val test; do
    python -u pipeline/simulate/simulate.py --params $ST --runs 100 --records data/te/$c/$s.pkl --name $s --out-dir $R/sims/te/$c --workers $WORKERS
  done
  log "$c: expression shards"
  for s in train val test; do
    python -u pipeline/simulate/simulate.py --params $ST --runs 1000 --records data/expression/$s.pkl --name $s --out-dir $R/sims/expression/$c --workers $WORKERS
  done
  log "$c: done"
done
