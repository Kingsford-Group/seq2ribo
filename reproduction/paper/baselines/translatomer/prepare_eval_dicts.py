import pickle
import numpy as np
import sys
import os
import argparse
from tqdm import tqdm

def map_predictions_to_codons(pred_pkl, test_data_pkl, output_prefix, use_exp=False):
    print(f"Loading predictions from {pred_pkl}...")
    with open(pred_pkl, 'rb') as f:
        preds = pickle.load(f)
        
    print(f"Loading test data from {test_data_pkl}...")
    with open(test_data_pkl, 'rb') as f:
        test_data = pickle.load(f)
        
    pred_counts_dict = {}
    true_counts_dict = {}
    geo_map_dict = {}
    
    # Constants for mapping
    SEQ_LEN = 65536
    BIN_SIZE = 64
    NUM_BINS = 1024
    
    print("Processing samples...")
    for item in tqdm(test_data):
        tx_id = item[0]
        seq = item[1]
        
        # Item 4 structure from deep inspection: (obs_counts_array, codon_ids_array)
        # obs_counts_array is float64, codon_ids_array is int32
        # Lengths are L_cod = len(seq)/3 usually
        
        if len(item) >= 5:
            # Item 2: asite_counts (Ground Truth, NT resolution)
            asite_counts = item[2]
            
            # Item 4: geo_map tuple (angles, pair_counts) (Codon resolution)
            if isinstance(item[4], tuple):
                geo_map_data = item[4]
                # Usually array lengths in geo_map are codons
                num_codons_geo = len(geo_map_data[0])
            else:
                # Fallback or error
                print(f"Warning: Item 4 is not tuple for {tx_id}")
                continue
                
            geo_map_dict[tx_id] = geo_map_data
            
            # Sum NT counts to Codons to match analysis resolution
            # asite_counts is length L_nt.
            # L_nt should be roughly 3 * num_codons.
            
            if len(asite_counts) % 3 == 0:
                obs_codons = asite_counts.reshape(-1, 3).sum(axis=1)
            else:
                valid_len = (len(asite_counts) // 3) * 3
                obs_codons = asite_counts[:valid_len].reshape(-1, 3).sum(axis=1)
                
            # Verify length match with geo_map if possible
            if len(obs_codons) != num_codons_geo:
                # Sometimes lengths differ slightly due to STOP codon handling or trimming.
                # Use the shorter one or trust obs_codons?
                # Let's trust asite_counts sum for now.
                pass
                
            true_counts_dict[tx_id] = obs_codons
            num_codons = len(obs_codons) # Use this for prediction mapping loop
            
        else:
             print(f"Warning: Item structure mismatch for {tx_id}, len={len(item)}")
             continue
        
        if tx_id in preds:
            pred_bins = preds[tx_id] # (1024,)
            
            # Map bins to codons
            # Center alignment logic
            L = len(seq)
            pad_total = SEQ_LEN - L
            start_offset = pad_total // 2
            
            # Number of codons (from our calculated obs_codons)
            # num_codons variable set above

            
            # Predict codon counts
            pred_codons = np.zeros(num_codons, dtype=np.float32)
            
            for i in range(num_codons):
                # Codon i covers seq indices [3*i, 3*i+3)
                # Middle of codon in seq coordinates
                codon_center_seq = 3*i + 1.5
                
                # Middle of codon in centered 65536 window
                codon_center_global = start_offset + codon_center_seq
                
                # Bin index
                bin_idx = int(codon_center_global // BIN_SIZE)
                
                if 0 <= bin_idx < NUM_BINS:
                    # Value is mean signal in bin.
                    # Should we scale by 3 (bp) or leave as density?
                    # Start with density (raw value) since model output is log(mean+1) or similar?
                    # Wait, tutorial says target = log(mean + 1).
                    # analysis.py: sim_feat = np.log1p(sim) if self.log_input else sim
                    # And 'sim' comes from 'sim_vec_scaled'.
                    # If model predicts log(mean+1), we should interpret it as such.
                    # But pred_counts_map usually expects 'counts'.
                    # Let's assume model output is "signal".
                    # We assign the bin value to the codon.
            # Assign bin value to codon
                    pred_codons[i] = pred_bins[bin_idx]
                else:
                    pred_codons[i] = 0.0
            
            # Apply transformation
            if use_exp:
                pred_codons = np.expm1(pred_codons)
                pred_codons = np.maximum(pred_codons, 0)
            else:
                 # Linear/Raw scale (keep as is, just clip negative)
                 pred_codons = np.maximum(pred_codons, 0)
            
            pred_counts_dict[tx_id] = pred_codons
            
    # Save
    with open(output_prefix + "_pred_counts.pkl", 'wb') as f:
        pickle.dump(pred_counts_dict, f)
    with open(output_prefix + "_true_counts.pkl", 'wb') as f:
        pickle.dump(true_counts_dict, f)
    with open(output_prefix + "_geo_map.pkl", 'wb') as f:
        pickle.dump(geo_map_dict, f)
        
    print(f"Saved dictionaries to {output_prefix}_*.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_pkl', required=True)
    parser.add_argument('--test_data_pkl', required=True)
    parser.add_argument('--output_prefix', required=True)
    parser.add_argument('--use_exp', action='store_true', help='Apply expm1 to predictions')
    args = parser.parse_args()
    
    map_predictions_to_codons(args.pred_pkl, args.test_data_pkl, args.output_prefix, args.use_exp)
