#!/usr/bin/env bash
# GPU. Train + evaluate the polisher for the given cell lines.
#   usage: DEVICE=cuda:0 nohup bash jobs/gpu_04_polisher.sh ipsc hek293 > logs/04_polisher_a.log 2>&1 &
#   BATCH sets the batch size; ACCUM=n splits each batch into n micro-batches (same update, less GPU memory) on a shared GPU.
source "$(dirname "$0")/env.sh"
DEVICE=${DEVICE:-cuda:0}
ACCUM=${ACCUM:-1}
BATCH=${BATCH:-192}   # paper: 192
[ $# -gt 0 ] || { echo "usage: bash jobs/gpu_04_polisher.sh <cell lines> (ipsc hek293 lcl rpe)" >&2; exit 1; }
for c in "$@"; do
  log "$c: polisher train"
  python -u pipeline/polisher/train.py --cell-line $c --shards results/sims/stasep/$c --batch-size $BATCH --accum-steps $ACCUM --device $DEVICE > logs/polisher_$c.log 2>&1
  log "$c: evaluate seq2ribo + sTASEP"
  python -u pipeline/evaluate/evaluate_profiles.py --cell-line $c --shards results/sims/stasep/$c \
    --polisher results/polisher/seq2ribo/$c/polisher.pt --bootstrap 1000 --run-variance --device $DEVICE \
    --out results/eval/profiles/$c/seq2ribo_stasep.json
  log "$c: evaluate TASEP"
  python -u pipeline/evaluate/evaluate_profiles.py --cell-line $c --shards results/sims/tasep/$c \
    --bootstrap 1000 --out results/eval/profiles/$c/tasep.json
done
