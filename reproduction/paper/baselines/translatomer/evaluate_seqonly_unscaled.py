
import os
import sys
import argparse
import pickle
import hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add local directory and Translatomer submodule to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'Translatomer'))

# Import model class
from train_seqonly import PatchedTrainModuleSeqOnly, FastaStringExtractor, onehot_dna_fast, SEQLEN, NBINS, BIN, GenomicTrainDataset

# Import inference utilities
from inference_seqonly import MockDataSources, build_gff_index, predict_transcript_tiled_batched, TxNotFound, NoCDSFound, MockArgs

# Import evaluation utilities
from run_analysis import InMemoryDataset, pad_collate, evaluate_test, CODONS, STOP_IDX, K_ANGLE_BINS

# -----------------------------------------------------------------------------
# Data Split Filtering Logic (Ported from test_unscaled.py)
# -----------------------------------------------------------------------------

def _extract_cds_seq(cds_val):
    if isinstance(cds_val, str):
        return cds_val
    if isinstance(cds_val, (list, tuple)) and len(cds_val) >= 1:
        if isinstance(cds_val[0], str):
            return cds_val[0]
    return str(cds_val)

def _normalize_seq(seq: str) -> str:
    return seq.upper().replace("U", "T").strip()

def get_hash(seq_str):
    # Standardize hash generation
    # load_data in test_unscaled uses a specific extraction
    # Here we are working with the raw pickles which might have different structure
    # train_data.pkl items are tuples: (tx_id, sequence, gt_counts, ...) for some, 
    # or (tx_id, sequence, asite, zeros, geo) for others.
    # We'll assume item[1] is the sequence string for hashing.
    if not isinstance(seq_str, str):
        seq_str = str(seq_str)
    s = _normalize_seq(seq_str)
    # Note: test_unscaled used the CDS value to extract the sequence.
    # But here we likely have the full sequence in item[1].
    # Let's match the hashing from prepare_filtered_test_data.py which used:
    # "CODS|" + s
    # Wait, test_unscaled used only md5(seq). prepare_filtered_test_data used "CODS|"+seq.
    # The user said "incorporate the new test, val and train split checks".
    # I should try to match the logic that was successful.
    # test_unscaled.py used: hashlib.md5(seq.encode("utf-8")).hexdigest()
    # Let's look at what the data actually looks like.
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def extract_seq_from_item(item):
    # item structure varies.
    # Typically item[1] is sequence.
    if len(item) > 1:
        return item[1]
    return ""

def filter_dataset(ref_data, target_data, ref_name="Ref", target_name="Target"):
    """
    Filter target_data to exclude items whose sequence hash matches any in ref_data.
    """
    print(f"Building hash set for {ref_name}...")
    ref_hashes = set()
    for item in ref_data:
        seq = extract_seq_from_item(item)
        if seq:
            ref_hashes.add(get_hash(seq))
            
    print(f"Filtering {target_name} against {ref_name}...")
    kept = []
    removed_count = 0
    
    for item in target_data:
        seq = extract_seq_from_item(item)
        if seq:
            h = get_hash(seq)
            if h in ref_hashes:
                removed_count += 1
                continue
        kept.append(item)
        
    print(f"[{target_name} vs {ref_name}] Kept: {len(kept)}, Removed: {removed_count}")
    return kept

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_line', required=True, choices=['hek293', 'ipsc', 'lcl', 'rpe'])
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--fasta', required=True, help='GRCh38 genome FASTA')
    parser.add_argument('--gff', required=True, help='GENCODE v49 annotation GFF3')
    parser.add_argument('--extra_val_file', default=None, help='Path to an additional validation pickle to filter against')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    base_dir = f"data/{args.cell_line}"
    print(f"Loading data from {base_dir}...")
    
    with open(os.path.join(base_dir, "train_data.pkl"), 'rb') as f:
        full_train_data = pickle.load(f)
        
    # Replicate train_seqonly.py splitting logic
    # strict seed=42, 90/10 split
    np.random.seed(42)
    indices = np.random.permutation(len(full_train_data))
    split_idx = int(len(full_train_data) * 0.9)
    train_data = [full_train_data[i] for i in indices[:split_idx]]
    val_data   = [full_train_data[i] for i in indices[split_idx:]]
    
    with open(os.path.join(base_dir, "test_data.pkl"), 'rb') as f:
        test_data = pickle.load(f)
        
    print(f"Original Sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # 2. Split Checks (Disjoint filtering)
    # Val vs Train
    val_data_f = filter_dataset(train_data, val_data, ref_name="Train", target_name="Val")
    
    # Test vs Train
    test_data_f1 = filter_dataset(train_data, test_data, ref_name="Train", target_name="Test")
    
    # Test vs Val (Original filtered Val or just Val? Strict disjointness usually implies disjoint from VALIDATION SET used.)
    # Using the filtered validation set is correct because that's what we would use for validation.
    test_data_f2 = filter_dataset(val_data_f, test_data_f1, ref_name="Val (Split)", target_name="Test")

    # Extra Val Check (for consistency with Seq2Ribo which has separate val_data.pkl)
    if args.extra_val_file and os.path.exists(args.extra_val_file):
        print(f"Loading extra validation file: {args.extra_val_file}")
        with open(args.extra_val_file, 'rb') as f:
            extra_val_data = pickle.load(f)
        test_data_f3 = filter_dataset(extra_val_data, test_data_f2, ref_name="Val (Extra)", target_name="Test")
    else:
        test_data_f3 = test_data_f2
        
    if len(test_data_f3) == 0:
        print("Error: Filtered Test set is empty!")
        return

    print(f"Final Test Set Size: {len(test_data_f3)}")
    
    # 3. Load Model
    ckpt_dir = f"checkpoints_seqonly/{args.cell_line}"
    if not os.path.exists(ckpt_dir):
        print(f"Checkpoint directory not found: {ckpt_dir}")
        return
        
    # Find latest checkpoint
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
    if not ckpts:
        print(f"No .ckpt found in {ckpt_dir}")
        return
    # Simple sort by modification time or name? Usually name contains epoch/loss.
    # Let's pick the one with lowest val_loss if possible, or just last modified.
    # Filenames: epoch=X-val_loss=Y.ckpt
    ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)))
    best_ckpt = os.path.join(ckpt_dir, ckpts[-1])
    print(f"Loading checkpoint: {best_ckpt}")
    
    try:
        model = PatchedTrainModuleSeqOnly.load_from_checkpoint(best_ckpt, args=MockArgs(), map_location=device)
    except TypeError:
        model = PatchedTrainModuleSeqOnly.load_from_checkpoint(best_ckpt, map_location=device)
    model.to(device).eval()

    # 4. Prepare Inference Resources
    print("Preparing inference resources (FASTA/GFF)...")
    fasta_extractor = FastaStringExtractor(args.fasta)
    ds_mock = MockDataSources(fasta_extractor)
    
    gff_cache_path = args.gff + ".index.pkl"
    if os.path.exists(gff_cache_path):
        with open(gff_cache_path, 'rb') as f:
            gff_index = pickle.load(f)
    else:
        print("GFF index not found, building (this might take time)...")
        import pandas as pd # Import locally if needed
        gff_df = pd.read_csv(args.gff, sep='\t', comment='#', header=None)
        gff_index = build_gff_index(gff_df)
    
    # 5. Run Inference
    print("Running inference...")
    pred_counts = {}
    true_counts = {}
    
    for item in tqdm(test_data_f3):
        tx_id = item[0]
        # Item structure: (tx_id, seq, gt_counts, ...) likely
        # Need to verify where GT is. 
        # In train_seqonly.py: gt_counts = item[2]
        if len(item) > 2:
            gt = item[2]
            true_counts[tx_id] = np.array(gt, dtype=np.float32)
        else:
            continue
            
        try:
             pred_codons = predict_transcript_tiled_batched(
                tx_id, gff_index, ds_mock, model, device, batch_size=args.batch_size
             )
             pred_counts[tx_id] = pred_codons.astype(np.float32)
        except (TxNotFound, NoCDSFound, Exception) as e:
            # print(f"Skipping {tx_id}: {e}")
            pass

    print(f"Inference complete. {len(pred_counts)} predictions generated.")
    
    # Check for all-zeros
    num_zeros = 0
    total_preds = len(pred_counts)
    for tx, p in pred_counts.items():
        if np.all(p == 0) or np.sum(p) == 0:
            num_zeros += 1
            
    print(f"Zero Predictions: {num_zeros} / {total_preds} ({num_zeros/total_preds*100:.2f}%)")
    
    # Save Predictions
    preds_out = os.path.join(base_dir, "predictions_seqonly_unscaled.pkl")
    with open(preds_out, 'wb') as f:
        pickle.dump(pred_counts, f)
    print(f"Saved predictions to {preds_out}")
    
    # 6. Evaluation
    print("Evaluating metrics (unscaled + scaled)...")
    
    # Need angle bins? 
    ANGLE_BINS = np.linspace(0, 2*np.pi, K_ANGLE_BINS + 1)
    
    # We need geomap? run_analysis needs it for angle/pairing.
    # geomap is usually in item[4] if available.
    # Let's try to extract it from test_data_f3
    geomap = {}
    for item in test_data_f3:
        tx = item[0]
        if len(item) >= 5:
            geomap[tx] = item[4]

    # Create dataset
    temp_test_pkl = f"temp_test_filtered_{args.cell_line}.pkl"
    with open(temp_test_pkl, 'wb') as f:
        pickle.dump(test_data_f3, f)
        
    ds = InMemoryDataset(pred_counts, true_counts, temp_test_pkl, geomap=geomap, angle_bins=ANGLE_BINS)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)
    
    # evaluate_test now returns both unscaled and scaled metrics in one call
    metrics = evaluate_test(dl)
    
    # Add zero stats to metrics
    metrics["num_zero_predictions"] = num_zeros
    metrics["pct_zero_predictions"] = num_zeros / total_preds if total_preds > 0 else 0.0
    metrics["total_predictions"] = total_preds
    
    # Convert metrics to native python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, torch.Tensor):
            return convert_to_serializable(obj.item())
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    metrics_serializable = convert_to_serializable(metrics)

    print("\n[Evaluation Results]")
    import json
    print(json.dumps(metrics_serializable, indent=2))
    
    # Save Metrics
    metrics_out = os.path.join(base_dir, "evaluation_seqonly_unscaled.json")
    with open(metrics_out, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"Saved metrics to {metrics_out}")

    # Cleanup
    if os.path.exists(temp_test_pkl):
        os.remove(temp_test_pkl)

if __name__ == "__main__":
    main()
