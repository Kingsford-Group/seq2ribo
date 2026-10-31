#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate translatomer

# run from the Translatomer work directory (Translatomer/ checkout, data/<cell>/)
# GENOME_FA=hg38.fa GENCODE_GFF=gencode.v49.annotation.gff3 bash <this script>
: "${GENOME_FA:?set GENOME_FA}" "${GENCODE_GFF:?set GENCODE_GFF}"
HERE=$(cd "$(dirname "$0")" && pwd)

# This script is for evaluating the SeqOnly models AFTER training completes.
# It runs sequentially to avoid resource contention.

run_eval() {
    CELL=$1
    GPU_ID=$2
    BATCH_SIZE=32 # Should be safe for inference if training used 32
    
    echo "=================================================="
    echo "Evaluating $CELL (SeqOnly, GPU $GPU_ID)"
    
    # Look in checkpoints_seqonly
    CKPT=$(ls checkpoints_seqonly/${CELL}/*.ckpt 2>/dev/null | tail -n 1)
    
    if [ -z "$CKPT" ]; then
        echo "No checkpoint found for $CELL in checkpoints_seqonly. Skipping."
        return
    fi
    
    DATA_DIR="data/${CELL}"
    TEST_DATA="${DATA_DIR}/test_data_filtered.pkl"
    TRUE_COUNTS="${DATA_DIR}/eval_ready_true_counts.pkl"
    GEO_MAP="${DATA_DIR}/eval_ready_geo_map.pkl"
    PRED_OUT="${DATA_DIR}/translatomer_predictions_seqonly.pkl"
    LOG_FILE="${DATA_DIR}/evaluation_results_seqonly.txt"
    INF_LOG="${DATA_DIR}/inference_run_seqonly.log"
    
    # Run Inference
    echo "Running Inference..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python "$HERE/inference_seqonly.py" \
        --checkpoint "$CKPT" \
        --test_data_pkl "$TEST_DATA" \
        --output_pkl "$PRED_OUT" \
        --batch_size $BATCH_SIZE \
        --fasta "$GENOME_FA" \
        --gff "$GENCODE_GFF" \
        > "$INF_LOG" 2>&1
        
    if [ ! -f "$PRED_OUT" ]; then
         echo "Inference failed for $CELL. Check $INF_LOG" | tee -a "$LOG_FILE"
         return
    fi
        
    # Run Analysis (Same script supports SeqOnly preds as long as format matches)
    echo "Evaluation Results for $CELL (Checkpoint: $CKPT)" > "$LOG_FILE"
    echo "======================================" >> "$LOG_FILE"
    
    echo "--- Standard Metrics ---" >> "$LOG_FILE"
    CUDA_VISIBLE_DEVICES=$GPU_ID python "$HERE/run_analysis.py" \
        --pred_counts "$PRED_OUT" \
        --true_counts "$TRUE_COUNTS" \
        --test_data "$TEST_DATA" \
        --geo_map "$GEO_MAP" \
        >> "$LOG_FILE" 2>&1
        
    echo "" >> "$LOG_FILE"
    echo "--- Scaled Metrics (Match Scale) ---" >> "$LOG_FILE"
    CUDA_VISIBLE_DEVICES=$GPU_ID python "$HERE/run_analysis.py" \
        --pred_counts "$PRED_OUT" \
        --true_counts "$TRUE_COUNTS" \
        --test_data "$TEST_DATA" \
        --geo_map "$GEO_MAP" \
        --match_scale \
        >> "$LOG_FILE" 2>&1
        
    echo "Finished $CELL. Results in $LOG_FILE"
}

# Run sequentially - Prioritize others first since HEK hangs
run_eval ipsc 1
run_eval lcl 2
run_eval rpe 3

# Run HEK last with safer batch size
CELL="hek293"
GPU_ID=0
BATCH_SIZE=1 # Safe mode
echo "=================================================="
echo "Evaluating $CELL (SeqOnly, GPU $GPU_ID, BS=$BATCH_SIZE - SAFE MODE)"
CKPT=$(ls checkpoints_seqonly/${CELL}/*.ckpt 2>/dev/null | tail -n 1)
if [ -z "$CKPT" ]; then
    echo "No checkpoint found for $CELL. Skipping."
else
    DATA_DIR="data/${CELL}"
    TEST_DATA="${DATA_DIR}/test_data_filtered.pkl"
    # ... variables matching run_eval ...
    # Manually calling inference with BS=1
    PRED_OUT="${DATA_DIR}/translatomer_predictions_seqonly.pkl"
    LOG_FILE="${DATA_DIR}/evaluation_results_seqonly.txt"
    INF_LOG="${DATA_DIR}/inference_run_seqonly.log"
    TRUE_COUNTS="${DATA_DIR}/eval_ready_true_counts.pkl"
    GEO_MAP="${DATA_DIR}/eval_ready_geo_map.pkl"
    
    echo "Running Inference (Safe Mode)..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python "$HERE/inference_seqonly.py" \
        --checkpoint "$CKPT" \
        --test_data_pkl "$TEST_DATA" \
        --output_pkl "$PRED_OUT" \
        --batch_size $BATCH_SIZE \
        --fasta "$GENOME_FA" \
        --gff "$GENCODE_GFF" \
        > "$INF_LOG" 2>&1
        
    if [ ! -f "$PRED_OUT" ]; then
         echo "Inference failed for $CELL. Check $INF_LOG" | tee -a "$LOG_FILE"
    else
        echo "Evaluation Results for $CELL (Checkpoint: $CKPT)" > "$LOG_FILE"
        echo "======================================" >> "$LOG_FILE"
        echo "--- Standard Metrics ---" >> "$LOG_FILE"
        CUDA_VISIBLE_DEVICES=$GPU_ID python "$HERE/run_analysis.py" \
            --pred_counts "$PRED_OUT" \
            --true_counts "$TRUE_COUNTS" \
            --test_data "$TEST_DATA" \
            --geo_map "$GEO_MAP" \
            >> "$LOG_FILE" 2>&1
        echo "--- Scaled Metrics (Match Scale) ---" >> "$LOG_FILE"
        CUDA_VISIBLE_DEVICES=$GPU_ID python "$HERE/run_analysis.py" \
            --pred_counts "$PRED_OUT" \
            --true_counts "$TRUE_COUNTS" \
            --test_data "$TEST_DATA" \
            --geo_map "$GEO_MAP" \
            --match_scale \
            >> "$LOG_FILE" 2>&1
        echo "Finished $CELL. Results in $LOG_FILE"
    fi
fi
