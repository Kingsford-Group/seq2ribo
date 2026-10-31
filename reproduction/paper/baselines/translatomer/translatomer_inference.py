import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import argparse
from tqdm import tqdm
from torchmetrics import Metric
from typing import Optional
import pytorch_lightning as pl
import torch.nn.functional as F

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import model definition (assuming model/ directory exists and models.py is there)
try:
    import model.models as models
except ImportError:
    # Fallback if model package is not found or we are not in the right dir
    # But usually it's in the same dir.
    print("Could not import model.models. Ensure 'model' directory exists.")
    sys.exit(1)

SEQ_LEN = 65536

# --- Embedded Classes from train_all_11fold.py ---

class MeanPearsonCorrCoefPerChannel(Metric):
    is_differentiable: Optional[bool] = False
    full_state_update:bool = False
    higher_is_better: Optional[bool] = True
    def __init__(self, n_channels:int, dist_sync_on_step=False):
        """Calculates the mean pearson correlation across channels aggregated over regions"""
        # Removed full_state_update as it causes error in newer torchmetrics
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.reduce_dims=(0, 1)
        self.add_state("product", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("true", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("true_squared", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("pred", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("pred_squared", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("count", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        self.product += torch.sum(preds * target, dim=self.reduce_dims)
        self.true += torch.sum(target, dim=self.reduce_dims)
        self.true_squared += torch.sum(torch.square(target), dim=self.reduce_dims)
        self.pred += torch.sum(preds, dim=self.reduce_dims)
        self.pred_squared += torch.sum(torch.square(preds), dim=self.reduce_dims)
        self.count += torch.sum(torch.ones_like(target), dim=self.reduce_dims)

    def compute(self):
        true_mean = self.true / self.count
        pred_mean = self.pred / self.count

        covariance = (self.product
                    - true_mean * self.pred
                    - pred_mean * self.true
                    + self.count * true_mean * pred_mean)

        true_var = self.true_squared - self.count * torch.square(true_mean)
        pred_var = self.pred_squared - self.count * torch.square(pred_mean)
        tp_var = torch.sqrt(true_var) * torch.sqrt(pred_var)
        correlation = covariance / tp_var
        return correlation

class TrainModule(pl.LightningModule):
    def __init__(self, args=None):
        super().__init__()
        # Use simple dict for hparams or convert args to dict
        if args:
            if hasattr(args, '__dict__'):
                self.save_hyperparameters(vars(args))
            else:
                self.save_hyperparameters(args)
            self.args = args 
        
        self.criterion = nn.MSELoss()
        self.pcc = MeanPearsonCorrCoefPerChannel(1)
        
        self.model = self.get_model(args)

    def get_model(self, args):
        model_name = 'TransModel'
        # Handle dict or object
        if args:
            if isinstance(args, dict) and 'model_type' in args:
                model_name = args['model_type']
            elif hasattr(args, 'model_type'):
                model_name = args.model_type
                
        if hasattr(self, 'hparams') and 'model_type' in self.hparams:
            model_name = self.hparams.model_type

        num_genomic_features = 1
        if hasattr(models, model_name):
            ModelClass = getattr(models, model_name)
            model = ModelClass(num_genomic_features, mid_hidden = 512)
            return model
        else:
             raise ValueError(f"Model {model_name} not found in models module")

    def forward(self, x):
        pred = self.model(x)
        return pred 
    
    # Dummy methods to satisfy abstract classes if any (not needed for inference)

# --- End Embedded Classes ---

class MockArgs:
    def __init__(self, model_type='TransModel'):
        self.model_type = model_type

def one_hot_encode(sequence):
    sequence = sequence.replace('U', 'T')
    en_dict = {'A' : 0, 'T' : 1, 'C' : 2, 'G' : 3, 'N' : 4}
    en_seq = [en_dict.get(ch, 4) for ch in sequence]
    np_seq = np.array(en_seq, dtype=int)
    seq_emb = np.zeros((len(np_seq), 5))
    seq_emb[np.arange(len(np_seq)), np_seq] = 1
    return seq_emb.astype(np.float32)

def preprocess_sample(sequence, rna_signal, target_len=SEQ_LEN):
    seq_oh = one_hot_encode(sequence)
    rna_signal = np.nan_to_num(rna_signal, 0)
    rna_signal_log = np.log(rna_signal + 1).astype(np.float32)
    rna_signal_log = rna_signal_log.reshape(-1, 1)
    
    L = seq_oh.shape[0]
    if rna_signal_log.shape[0] != L:
        min_len = min(L, rna_signal_log.shape[0])
        seq_oh = seq_oh[:min_len]
        rna_signal_log = rna_signal_log[:min_len]
        L = min_len

    combined = np.concatenate([seq_oh, rna_signal_log], axis=1) # (L, 6)
    
    if L < target_len:
        pad_total = target_len - L
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_seq_left = np.tile(np.array([0,0,0,0,1], dtype=np.float32), (pad_left, 1))
        pad_seq_right = np.tile(np.array([0,0,0,0,1], dtype=np.float32), (pad_right, 1))
        pad_rna_left = np.zeros((pad_left, 1), dtype=np.float32)
        pad_rna_right = np.zeros((pad_right, 1), dtype=np.float32)
        pad_left_arr = np.concatenate([pad_seq_left, pad_rna_left], axis=1)
        pad_right_arr = np.concatenate([pad_seq_right, pad_rna_right], axis=1)
        final_input = np.concatenate([pad_left_arr, combined, pad_right_arr], axis=0)
    else:
        start = (L - target_len) // 2
        final_input = combined[start:start+target_len]
        
    return final_input

def run_inference(checkpoint_path, data_dirs):
    print(f"Loading model from {checkpoint_path}...", flush=True)
    try:
        # Pass mock args to ensure 'model_type' is available during init if hparams not loaded yet
        model = TrainModule.load_from_checkpoint(checkpoint_path, args=MockArgs())
    except Exception as e:
        print(f"Error loading model: {e}", flush=True)
        # Try without args if that fails, maybe hparams in checkpoint are enough
        try:
             model = TrainModule.load_from_checkpoint(checkpoint_path)
        except Exception as e2:
             print(f"Retry failed: {e2}", flush=True)
             return

    model.eval()
    model.cuda()
    print("Model loaded and moved to CUDA.", flush=True)
    
    for cell_line_dir in data_dirs:
        pkl_path = os.path.join(cell_line_dir, 'test_data.pkl')
        if not os.path.exists(pkl_path):
            print(f"Skipping {cell_line_dir}, test_data.pkl not found.", flush=True)
            continue
            
        print(f"Processing {pkl_path}...", flush=True)
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        predictions = {}
        
        # Batching might be faster, but let's stick to single item for simplicity and robust padding
        for idx, item in tqdm(enumerate(data), total=len(data)):
            transcript_id = item[0]
            sequence = item[1]
            rna_signal = item[2]
            
            input_tensor = preprocess_sample(sequence, rna_signal)
            input_batch = torch.tensor(input_tensor).unsqueeze(0).float().cuda()
            
            with torch.no_grad():
                output = model(input_batch)
                
            predictions[transcript_id] = output.cpu().numpy().flatten()
            
        output_path = os.path.join(cell_line_dir, 'translatomer_predictions.pkl')
        print(f"Saving predictions to {output_path}...", flush=True)
        with open(output_path, 'wb') as f:
            pickle.dump(predictions, f)
        print("Done.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_root', required=True)
    args = parser.parse_args()
    
    cell_lines = ['hek293', 'ipsc', 'lcl', 'rpe']
    data_dirs = [os.path.join(args.data_root, cl) for cl in cell_lines]
    
    run_inference(args.checkpoint, data_dirs)
