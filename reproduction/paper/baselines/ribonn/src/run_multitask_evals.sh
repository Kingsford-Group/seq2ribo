#!/usr/bin/env bash
#
# Evaluate multitask model predictions against per-cell-line ground truth.
#
# Usage:
#   bash src/run_multitask_evals.sh                         # full-sequence
#   bash src/run_multitask_evals.sh multitask_CDS_CV        # CDS-only
#
# The prediction files are expected under:
#   results/<experiment_name>/<cell_line>/prediction_output_multitask.txt
#
# Each prediction file contains columns: tx_id, predicted_TE_HEK293,
# predicted_TE_IPSC, predicted_TE_LCL, predicted_TE_RPE-1
#
# For IPSC, the ground truth column is "mean_te" but the prediction column
# is "predicted_TE_IPSC". We create a temp GT file with a renamed column
# so evaluate_predictions.py can match them.

set -euo pipefail

EXPERIMENT_NAME="${1:-multitask_CV}"
RESULTS_DIR="results/${EXPERIMENT_NAME}"

echo "============================================"
echo "Evaluating multitask predictions: ${EXPERIMENT_NAME}"
echo "============================================"

# HEK293
echo ""
echo "--- HEK293 ---"
python3 -m src.evaluate_predictions \
    --pred_txt "${RESULTS_DIR}/HEK293/prediction_output_multitask.txt" \
    --gt_tsv HEK293/test_mean_te.tsv \
    --gt_col TE_HEK293 \
    --out_dir "${RESULTS_DIR}/HEK293/plots"

# LCL
echo ""
echo "--- LCL ---"
python3 -m src.evaluate_predictions \
    --pred_txt "${RESULTS_DIR}/LCL/prediction_output_multitask.txt" \
    --gt_tsv LCL/test_mean_te.tsv \
    --gt_col TE_LCL \
    --out_dir "${RESULTS_DIR}/LCL/plots"

# RPE
echo ""
echo "--- RPE ---"
python3 -m src.evaluate_predictions \
    --pred_txt "${RESULTS_DIR}/RPE/prediction_output_multitask.txt" \
    --gt_tsv RPE/test_mean_te.tsv \
    --gt_col "TE_RPE-1" \
    --out_dir "${RESULTS_DIR}/RPE/plots"

# IPSC: GT has "mean_te" column, but prediction uses "predicted_TE_IPSC".
# Create temp GT file with "TE_IPSC" column so the evaluator can match.
echo ""
echo "--- IPSC ---"
IPSC_TMP_GT=$(mktemp)
head -1 IPSC/test_mean_te.tsv | sed 's/\bmean_te\b/TE_IPSC/' > "${IPSC_TMP_GT}"
tail -n +2 IPSC/test_mean_te.tsv >> "${IPSC_TMP_GT}"

python3 -m src.evaluate_predictions \
    --pred_txt "${RESULTS_DIR}/IPSC/prediction_output_multitask.txt" \
    --gt_tsv "${IPSC_TMP_GT}" \
    --gt_col TE_IPSC \
    --out_dir "${RESULTS_DIR}/IPSC/plots"

rm -f "${IPSC_TMP_GT}"

echo ""
echo "All evaluations complete. Results in ${RESULTS_DIR}/*/plots/"
