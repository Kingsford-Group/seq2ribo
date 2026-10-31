#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate translatomer

# run from the Translatomer work directory (Translatomer/ checkout, data/<cell>/)
# GENOME_FA=hg38.fa GENCODE_GFF=gencode.v49.annotation.gff3 bash <this script>
: "${GENOME_FA:?set GENOME_FA}" "${GENCODE_GFF:?set GENCODE_GFF}"
HERE=$(cd "$(dirname "$0")" && pwd)

# Ensure all GPUs are visible to the shell before individual scripts select them
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Launching Parallel Sequence-Only Training on 4 GPUs..."
echo "Config: Batch Size 32, Num Workers 1"

# Launch HEK293 (GPU 0)
python -u "$HERE/train_seqonly.py" \
  --fasta "$GENOME_FA" --gff "$GENCODE_GFF" \
  --cell_line hek293 \
  --epochs 10 \
  --batch_size 32 \
  --num_workers 1 \
  --gpu_id 0 \
  --checkpoint_out checkpoints_seqonly \
  > data/hek293/train_seqonly.log 2>&1 &

# Launch IPSC (GPU 1)
python -u "$HERE/train_seqonly.py" \
  --fasta "$GENOME_FA" --gff "$GENCODE_GFF" \
  --cell_line ipsc \
  --epochs 10 \
  --batch_size 32 \
  --num_workers 1 \
  --gpu_id 1 \
  --checkpoint_out checkpoints_seqonly \
  > data/ipsc/train_seqonly.log 2>&1 &

# Launch LCL (GPU 2)
python -u "$HERE/train_seqonly.py" \
  --fasta "$GENOME_FA" --gff "$GENCODE_GFF" \
  --cell_line lcl \
  --epochs 10 \
  --batch_size 32 \
  --num_workers 1 \
  --gpu_id 2 \
  --checkpoint_out checkpoints_seqonly \
  > data/lcl/train_seqonly.log 2>&1 &

# Launch RPE (GPU 3)
python -u "$HERE/train_seqonly.py" \
  --fasta "$GENOME_FA" --gff "$GENCODE_GFF" \
  --cell_line rpe \
  --epochs 10 \
  --batch_size 32 \
  --num_workers 1 \
  --gpu_id 3 \
  --checkpoint_out checkpoints_seqonly \
  > data/rpe/train_seqonly.log 2>&1 &

echo "All training jobs launched in background."
echo "Monitor logs at: data/*/train_seqonly.log"
