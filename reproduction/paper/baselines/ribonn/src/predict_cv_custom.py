
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from src.model import RiboNN
from src.data import RiboNNDataModule
import os
import glob
from tqdm import tqdm

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_input_path", required=True)
    parser.add_argument("--predict_output_path", required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--target_pattern", type=str, default=None) # Used for column naming
    parser.add_argument("--cds_only", action="store_true", help="Use CDS-only config (pad_5_prime=False)")
    parser.add_argument("--max_seq_len", type=int, default=None, help="Force max sequence length (to match training)")
    
    args = parser.parse_args()

    # Base config
    config = load_config("config/conf.yml")
    config["tx_info_path"] = args.predict_input_path
    config["target_column_pattern"] = None # No targets in predict mode usually, or ignore them
    
    # Common overrides
    config["with_NAs"] = True
    config["max_shift"] = 0
    config["num_conv_layers"] = 10
    config["symmetric_shift"] = True
    config["num_workers"] = 0 # Avoid multiprocessing overhead for small inference
    
    if args.cds_only:
        # CDS-only: no UTR padding
        config["pad_5_prime"] = False
        config["remove_extreme_txs"] = True
        config["max_utr5_len"] = 0
        config["max_cds_utr3_len"] = 20_000
    else:
        # Full-sequence
        config["pad_5_prime"] = True
        config["remove_extreme_txs"] = False
        config["max_utr5_len"] = 1_381
        config["max_cds_utr3_len"] = 11_937 
    
    if args.max_seq_len:
        config["max_seq_len"] = args.max_seq_len
    
    # We need to setup the datamodule to get the loader
    dm = RiboNNDataModule(config)
    if args.max_seq_len:
        dm.max_tx_len = args.max_seq_len
    loader = dm.predict_dataloader()
    
    # Find models
    model_dir = f"models/{args.experiment_name}"
    checkpoints = sorted(glob.glob(f"{model_dir}/*/best_model.ckpt"))
    
    if not checkpoints:
        print(f"No checkpoints found in {model_dir}")
        return

    print(f"Found {len(checkpoints)} models for {args.experiment_name}. Ensembling...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Store predictions from each model
    # shape: (num_samples, num_models) or similar
    # We'll just sum them up and divide by N at the end to save memory if needed, 
    # but storing all is safer for debugging. Let's sum.
    
    ensemble_preds = None
    
    for i, ckpt in enumerate(checkpoints):
        print(f"Loading model {i+1}/{len(checkpoints)}: {ckpt}")
        model = RiboNN.load_from_checkpoint(ckpt)
        model.to(device)
        model.eval()
        
        preds_list = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Model {i+1}"):
                batch = batch.to(device)
                out = model(batch)
                preds_list.append(out.cpu().numpy())
        
        current_preds = np.concatenate(preds_list, axis=0)
        
        # Flatten if necessary (num_samples, 1) -> (num_samples,)
        if current_preds.ndim > 1 and current_preds.shape[1] == 1:
            current_preds = current_preds.flatten()
            
        if ensemble_preds is None:
            ensemble_preds = np.zeros_like(current_preds, dtype=np.float64)
            
        ensemble_preds += current_preds
        
    # Average
    avg_preds = ensemble_preds / len(checkpoints)
    
    # Save
    output_df = dm.df.copy()
    
    # Column name
    col_name = "mean_predicted_TE" # Default standard name
    if args.target_pattern:
         clean_pat = args.target_pattern.replace('^', '').replace('$', '')
         if clean_pat == "mean_te":
             col_name = "mean_predicted_TE"
         else:
             col_name = f"predicted_{clean_pat}"

    output_df[col_name] = avg_preds
    
    os.makedirs(os.path.dirname(args.predict_output_path), exist_ok=True)
    output_df.to_csv(args.predict_output_path, sep="\t", index=False)
    print(f"Ensemble predictions saved to {args.predict_output_path}")

if __name__ == "__main__":
    main()
