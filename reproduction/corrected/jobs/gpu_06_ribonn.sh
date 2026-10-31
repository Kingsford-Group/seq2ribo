#!/usr/bin/env bash
# GPU. Clean RiboNN TE baseline (pipeline/baselines/ribonn): export RiboNN-format inputs, train a seed
# ensemble per task (4 individual + 1 multitask) in each setting (utr_cds, cds_only), then ensemble-predict
# and score the shared test split -> results/baselines/ribonn/<setting>/<mode>/<cell>/test_{predictions,metrics}.
#   usage: DEVICE=cuda:0 nohup bash jobs/gpu_06_ribonn.sh > logs/06_ribonn.log 2>&1 &
#   two GPUs: DEVICE=cuda:0 SETTINGS=utr_cds  nohup bash jobs/gpu_06_ribonn.sh > logs/06_ribonn_a.log 2>&1 &
#             DEVICE=cuda:1 SETTINGS=cds_only nohup bash jobs/gpu_06_ribonn.sh > logs/06_ribonn_b.log 2>&1 &
# Environment (defaults): DEVICE cuda:0, N_SEEDS 5, SETTINGS "utr_cds cds_only", MODES "individual multitask",
#   CELLS "ipsc hek293 lcl rpe", RESULTS_DIR results/baselines/ribonn, RIBONN_ROOT (upstream checkout).
#   Smoke tests only: MAX_EPOCHS, LIMIT_TRAIN_BATCHES (fraction or number of training batches per epoch).
# Finished members (seed<k>/member.json with the same configuration) are skipped, so a rerun resumes.
source "$(dirname "$0")/env.sh"
set +u; conda activate "${RIBONN_ENV:-RiboNN}"; set -u
# use the env's CUDA libraries: a system libcublasLt on LD_LIBRARY_PATH breaks cuBLAS in this torch build
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# multiprocessing temp files on NFS cannot be removed at exit (EBUSY); keep them in RAM when possible
if [ -d /dev/shm ] && [ -w /dev/shm ]; then export TMPDIR=/dev/shm/${USER:-$(id -un)}_ribonn; mkdir -p "$TMPDIR"; fi
# checkout of https://github.com/Sanofi-Public/RiboNN at commit 318e758 (required)
export RIBONN_ROOT=${RIBONN_ROOT:?set RIBONN_ROOT to a RiboNN checkout}

DEVICE=${DEVICE:-cuda:0}
N_SEEDS=${N_SEEDS:-5}
SETTINGS=${SETTINGS:-"utr_cds cds_only"}
MODES=${MODES:-"individual multitask"}
CELLS=${CELLS:-"ipsc hek293 lcl rpe"}
RESULTS_DIR=${RESULTS_DIR:-results/baselines/ribonn}
EXTRA=()
[ -n "${MAX_EPOCHS:-}" ] && EXTRA+=(--max-epochs "$MAX_EPOCHS")
[ -n "${LIMIT_TRAIN_BATCHES:-}" ] && EXTRA+=(--limit-train-batches "$LIMIT_TRAIN_BATCHES")
B=pipeline/baselines/ribonn
LOGS=logs/ribonn; [ "$RESULTS_DIR" != results/baselines/ribonn ] && LOGS=$RESULTS_DIR/logs
mkdir -p "$RESULTS_DIR/inputs" "$LOGS"

# inputs (once; the lock makes concurrent instances wait for the first export)
(
  flock 9
  if [ ! -f "$RESULTS_DIR/inputs/summary.json" ]; then
    log "export"
    python -u $B/export.py --results-dir "$RESULTS_DIR" > "$LOGS/export.log" 2>&1
  fi
) 9> "$RESULTS_DIR/inputs/.export.lock"

for s in $SETTINGS; do
  for m in $MODES; do
    if [ "$m" = individual ]; then tasks=$CELLS; else tasks=multitask; fi
    for t in $tasks; do
      log "train $s $m $t ($N_SEEDS seeds, $DEVICE)"
      targ=(); [ "$m" = individual ] && targ=(--task "$t")
      python -u $B/train.py --setting "$s" --mode "$m" "${targ[@]}" --n-seeds "$N_SEEDS" --device "$DEVICE" \
        --results-dir "$RESULTS_DIR" "${EXTRA[@]}" > "$LOGS/train_${s}_${m}_${t}.log" 2>&1
      grep -E "WARNING|best epoch" "$LOGS/train_${s}_${m}_${t}.log" || true
    done
    log "predict $s $m"
    # the multitask model is always trained and scored on all four cell lines
    if [ "$m" = individual ]; then targ=(--tasks $CELLS); else targ=(); fi
    python -u $B/predict.py --setting "$s" --mode "$m" "${targ[@]}" --n-seeds "$N_SEEDS" --device "$DEVICE" \
      --results-dir "$RESULTS_DIR" > "$LOGS/predict_${s}_${m}.log" 2>&1
    grep "^\[TEST" "$LOGS/predict_${s}_${m}.log"
  done
done
log "done"
