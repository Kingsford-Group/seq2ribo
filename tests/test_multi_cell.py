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
        res = predictor.predict(seq, task=task)
        
        # Basic Validation
        assert isinstance(res, list)
        assert len(res) == 1
        output = res[0]
        
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
