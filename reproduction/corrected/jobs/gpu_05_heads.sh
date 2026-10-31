#!/usr/bin/env bash
# GPU. TE (CDS-only, CDS+UTR) and protein expression heads with the released recipes
# (pipeline/config.py: TE_CDS_RECIPE, TE_UTR_RECIPE, EXPRESSION_RECIPE), one run per seed.
#   usage: DEVICE=cuda:0 nohup bash jobs/gpu_05_heads.sh ipsc hek293 > logs/05_heads_a.log 2>&1 &
#          TASKS="te_utr" SEEDS="42" ...   (subsets)
# Outputs: results/heads/<task>/s<seed>/<cell>/{model.pt, transform.json, history.json, test_*}
source "$(dirname "$0")/env.sh"
DEVICE=${DEVICE:-cuda:0}
TASKS=${TASKS:-"te_cds te_utr expression"}
recipe() {  # recipe <task> <cell> -> finetune.py options (without --seed)
  python - "$1" "$2" <<'EOF'
import sys
sys.path.insert(0, "pipeline")
import config
task, cell = sys.argv[1:]
if task == "expression":
    r = dict(config.EXPRESSION_RECIPE[cell])
    print(f"--lr {r['lr']} --batch-size {r['batch_size']} --no-log1p")
else:
    r = dict(config.TE_CDS_RECIPE if task == "te_cds" else config.TE_UTR_RECIPE)
    r.pop("seeds")
    print(" ".join(f"--{k.replace('_', '-')} {v}" for k, v in r.items()))
EOF
}
seeds() {
  python -c "import sys; sys.path.insert(0, 'pipeline'); import config; \
print(*(config.EXPRESSION_SEEDS if '$1' == 'expression' else config.TE_CDS_RECIPE['seeds']))"
}
[ $# -gt 0 ] || { echo "usage: bash jobs/gpu_05_heads.sh <cell lines> (ipsc hek293 lcl rpe)" >&2; exit 1; }
for c in "$@"; do
  for t in $TASKS; do
    shards=results/sims/te/$c; [ "$t" = expression ] && shards=results/sims/expression/$c
    for s in ${SEEDS:-$(seeds $t)}; do
      out=results/heads/$t/s$s/$c
      if [ -f $out/test_metrics.json ]; then log "$c $t seed $s: done already"; continue; fi
      log "$c $t seed $s"
      python -u pipeline/heads/finetune.py --task $t --cell-line $c --polisher results/polisher/seq2ribo/$c/polisher.pt \
        --shards $shards $(recipe $t $c) --seed $s --device $DEVICE --out-dir $out > logs/heads_${t}_s${s}_$c.log 2>&1
      grep "^\[TEST" logs/heads_${t}_s${s}_$c.log || true
    done
  done
done
log "done"
