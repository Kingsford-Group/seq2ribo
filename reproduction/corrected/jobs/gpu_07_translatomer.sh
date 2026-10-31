#!/usr/bin/env bash
# GPU. Sequence-only Translatomer baseline per cell line: strand check, training (train split, best
# val-loss checkpoint on a 10% hold-out of the training transcripts), then test-set profiles, TE and
# mRFP expression evaluation.
#   usage: DEVICE=cuda:0 nohup bash jobs/gpu_07_translatomer.sh ipsc hek293 > logs/07_translatomer_a.log 2>&1 &
#          DEVICE=cuda:1 nohup bash jobs/gpu_07_translatomer.sh lcl rpe   > logs/07_translatomer_b.log 2>&1 &
# Optional: TRAIN_ARGS="--tf32" (faster on A100), WORKERS=8. Re-running resumes training from
# model_last.pt and skips cell lines whose training already finished.
# Outputs: results/baselines/translatomer/<cell>/{model_best.pt, history.json, train_config.json,
#          profiles.json, te_metrics.json, expression_metrics.json, *_predictions.*}
set -eo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${TRANSLATOMER_ENV:-translatomer}"
set -u
# checkout of https://github.com/xiongxslab/Translatomer at commit ad1540c (required)
export TRANSLATOMER_ROOT=${TRANSLATOMER_ROOT:?set TRANSLATOMER_ROOT to a Translatomer checkout}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
mkdir -p logs
log() { echo "[$(date '+%F %T')] $(hostname) $*"; }

DEVICE=${DEVICE:-cuda:0}
WORKERS=${WORKERS:-8}
TRAIN_ARGS=${TRAIN_ARGS:-}
D=pipeline/baselines/translatomer

log "strand / window check"
python -u $D/check_strand.py --cell-line ipsc --split test --subset _01 > logs/translatomer_check_strand.log 2>&1

[ $# -gt 0 ] || { echo "usage: bash jobs/gpu_07_translatomer.sh <cell lines> (ipsc hek293 lcl rpe)" >&2; exit 1; }
for c in "$@"; do
  OUT=results/baselines/translatomer/$c
  mkdir -p "$OUT"
  if [[ -f $OUT/model_best.pt && -f $OUT/history.json ]] && \
     python -c "import json,sys; h=json.load(open('$OUT/history.json')); c=json.load(open('$OUT/train_config.json')); sys.exit(len(h) < c['epochs'])"; then
    log "$c: training already complete"
  else
    log "$c: train on $DEVICE"
    python -u $D/train.py --cell-line $c --device $DEVICE --num-workers $WORKERS --resume $TRAIN_ARGS \
      --out-dir $OUT > logs/translatomer_train_$c.log 2>&1
  fi
  log "$c: evaluate (profiles, TE, expression)"
  python -u $D/evaluate.py --cell-line $c --checkpoint $OUT/model_best.pt --device $DEVICE --num-workers $WORKERS \
    --out-dir $OUT > logs/translatomer_eval_$c.log 2>&1
  log "$c: done"
done
