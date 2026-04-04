import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from seq2ribo import Seq2Ribo
import torch

def test_cell_line(cell_line, task="riboseq"):
    print(f"\nTesting Cell Line: {cell_line.upper()} (Task: {task})")
    try:
        # Initialize predictor
        predictor = Seq2Ribo(cell_line=cell_line, weights_dir="weights")
        
        # Sample sequence (random short sequence)
        seq = "AUG" + "GCC" * 10 + "UAG" 
        
        # Predict
        if task == "te":
            # Default TE output is inverse-transformed TE.
            res = predictor.predict(seq, task=task)
            res_scaled = predictor.predict(seq, task=task, return_scaled_te=True)
            # UTR-aware TE mode requires separate 5'UTR/CDS/3'UTR inputs.
            utr5 = "AUGGCUA"
            cds = seq
            utr3 = "UAAUCG"
            res_utr = predictor.predict(
                task=task,
                use_utr=True,
                utr5_list=[utr5],
                cds_list=[cds],
                utr3_list=[utr3],
            )
            res_utr_scaled = predictor.predict(
                task=task,
                use_utr=True,
                utr5_list=[utr5],
                cds_list=[cds],
                utr3_list=[utr3],
                return_scaled_te=True,
            )
        elif task == "protein":
            # Protein path uses fixed 32-pass MC mean inference.
            res = predictor.predict(seq, task=task)
            res_repeat = predictor.predict(seq, task=task)
        else:
            res = predictor.predict(seq, task=task)
        
        # Basic Validation
        assert isinstance(res, list)
        assert len(res) == 1
        output = res[0]
        if task == "te":
            assert isinstance(res_scaled, list)
            assert len(res_scaled) == 1
            output_scaled = res_scaled[0]
            assert isinstance(output_scaled, (float, int))
            if not (0.0 <= float(output_scaled) <= 1.0):
                print(f"FAILURE: Scaled TE out of range [0,1]: {output_scaled}")
            assert isinstance(res_utr, list)
            assert len(res_utr) == 1
            assert isinstance(res_utr_scaled, list)
            assert len(res_utr_scaled) == 1
            output_utr_scaled = res_utr_scaled[0]
            assert isinstance(output_utr_scaled, (float, int))
            if not (0.0 <= float(output_utr_scaled) <= 1.0):
                print(f"FAILURE: Scaled TE UTR out of range [0,1]: {output_utr_scaled}")
        elif task == "protein":
            assert isinstance(res_repeat, list)
            assert len(res_repeat) == 1
            assert isinstance(res_repeat[0], (float, int))
        
        if isinstance(output, (float, int)):
             print(f"Success! Output value: {output:.4f}")
        else:
             print(f"Success! Output shape/length: {output.shape if hasattr(output, 'shape') else len(output)}")
             # Check for NaNs or Infinities
             import numpy as np
             if isinstance(output, torch.Tensor):
                  if torch.isnan(output).any():
                       print("FAILURE: Output contains NaNs")
                  else:
                       # Only print mean for tensors with multiple elements
                       if output.numel() > 1:
                            print(f"Mean output: {output.mean().item():.4f}")
                            
             elif isinstance(output, np.ndarray):
                  if np.isnan(output).any():
                       print("FAILURE: Output contains NaNs")
                  else:
                       if output.size > 1:
                            print(f"Mean output: {output.mean():.4f}")
                            
        # Note: Geometry internal state is not exposed for direct verification here.
             
    except Exception as e:
        print(f"FAILURE: {e}")

def main():
    # Test all combinations
    cell_lines = ["hek293", "lcl", "rpe", "ipsc"]
    tasks = ["riboseq", "te", "protein"]
    
    for cl in cell_lines:
        for t in tasks:
            print(f"--- Testing {cl.upper()} : {t} ---")
            test_cell_line(cl, t)
            print("")


if __name__ == "__main__":
    main()
