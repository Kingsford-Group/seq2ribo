"""Main inference API for seq2ribo predictions."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from . import constants as CONST
from .data import pad_collate
from .geometry import compute_geometry_features
from .models import MambaExprFull, MambaTEFull, RiboPolisherMamba
from .simulation import simulate_transcript
from .utils import CODON2IDX, DEFAULT_ANGLE_BINS, K_ANGLE_BINS, build_full_rate



class Seq2Ribo:
    def __init__(self, 
                 cell_line: str = "hek293", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 weights_dir: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        self.cell_line = cell_line.lower()
        self.device = torch.device(device)
        self.models = {} 
        self.weights_dir = Path(weights_dir) if weights_dir else Path(".")
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/geometry")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load sTASEP parameters based on cell line
        if self.cell_line == "hek293":
            self.rates = build_full_rate(CONST.HEK293_RATES)
            self.alpha_vec = CONST.HEK293_ALPHA
            self.beta_vec = CONST.HEK293_BETA
            self.bucket_vec = CONST.HEK293_BUCKETS
        elif self.cell_line == "lcl":
            self.rates = build_full_rate(CONST.LCL_RATES)
            self.alpha_vec = CONST.LCL_ALPHA
            self.beta_vec = CONST.LCL_BETA
            self.bucket_vec = CONST.LCL_BUCKETS
        elif self.cell_line == "rpe":
            self.rates = build_full_rate(CONST.RPE_RATES)
            self.alpha_vec = CONST.RPE_ALPHA
            self.beta_vec = CONST.RPE_BETA
            self.bucket_vec = CONST.RPE_BUCKETS
        elif self.cell_line == "ipsc":
            self.rates = build_full_rate(CONST.IPSC_RATES)
            self.alpha_vec = CONST.IPSC_ALPHA
            self.beta_vec = CONST.IPSC_BETA
            self.bucket_vec = CONST.IPSC_BUCKETS
        else:
            raise ValueError(f"Unknown cell line: {cell_line}. Supported: hek293, lcl, rpe, ipsc")

    def _load_model(self, task: str):
        if task in self.models:
            return

        print(f"Loading model for task '{task}' (Cell Line: {self.cell_line})...")
        
        # Defaults
        use_mamba2 = False
        activation = "relu"
        ckpt_filename = "ipsc_mamba_final.pt" 
        
        if task == "riboseq":
            # Configure per cell line
            if self.cell_line == "hek293":
                ckpt_filename = "hek293_mamba_final.pt"
                # Mamba1, ReLU
            elif self.cell_line == "lcl":
                ckpt_filename = "lcl_mamba_final.pt"
                activation = "softplus"
                # Mamba1
            elif self.cell_line == "rpe":
                ckpt_filename = "rpe_mamba_final.pt"
                activation = "softplus"
                # Mamba1
            elif self.cell_line == "ipsc":
                ckpt_filename = "ipsc_mamba_final.pt"
                use_mamba2 = True
                # ReLU

            model = RiboPolisherMamba(
                d_model=192, 
                n_layers=4, 
                dropout=0.1,
                use_mamba2=use_mamba2,
                activation=activation
            )
            
        elif task == "te":
            # Identify config based on cell line (same as riboseq)
            # Default to IPSC settings if not specified
            activation = "relu"
            use_mamba2 = False
            ckpt_filename = "ipsc_mamba_te_full_final.pt"

            if self.cell_line == "hek293":
                ckpt_filename = "hek293_mamba_te_full_final.pt"
            elif self.cell_line == "lcl":
                ckpt_filename = "lcl_mamba_te_full_final.pt"
                activation = "softplus"
            elif self.cell_line == "rpe":
                ckpt_filename = "rpe_mamba_te_full_final.pt"
                activation = "softplus"
            elif self.cell_line == "ipsc":
                ckpt_filename = "ipsc_mamba_te_full_final.pt"
                use_mamba2 = True
            
            # TE hidden dimension: HEK293 uses 256, others (LCL, RPE, iPSC) use 128 based on checkpoints.
            te_hidden = 256 if self.cell_line == "hek293" else 128
            
            base = RiboPolisherMamba(
                d_model=192, 
                n_layers=4, 
                dropout=0.1, 
                use_mamba2=use_mamba2, 
                activation=activation
            )
            model = MambaTEFull(base, hidden=te_hidden)

        elif task == "protein":
            if self.cell_line != "ipsc":
                print("Warning: Protein expression task is primarily supported for iPSC. Using available checkpoint anyway.")
            ckpt_filename = "mamba_expr_full_final.pt"
            
            # Protein uses Mamba2 (as per iPSC)
            base = RiboPolisherMamba(
                d_model=192, 
                n_layers=4, 
                dropout=0.1, 
                use_mamba2=True,
                activation="relu"
            )
            model = MambaExprFull(base, hidden=128)

            
        else:
            raise ValueError(f"Unknown task: {task}")

        ckpt_path = self.weights_dir / ckpt_filename
        
        # Check if file exists, if not try just filename in case user is in weights dir
        if not ckpt_path.exists():
            # Try searching in current dir
            if Path(ckpt_filename).exists():
                ckpt_path = Path(ckpt_filename)
            else:
                raise FileNotFoundError(f"Checkpoint for task '{task}' ({ckpt_filename}) not found at {self.weights_dir}. Please provide correct weights_dir.")

        print(f"Loading weights from {ckpt_path}...")
        # Load weights
        state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.models[task] = model
        return model

    def predict(self, 
                sequences: Union[str, List[str]], 
                task: str = "riboseq", 
                geomap: Optional[Dict] = None,
                use_polisher: bool = True,
                n_stasep_runs: int = 1,
                init_p: float = 0.01):
        """
        Run prediction pipeline:
        1. sTASEP simulation (if needed to generate features)
        2. Model forward pass (if use_polisher=True)
        """
        if isinstance(sequences, str):
            sequences = [sequences]
            
        geomap = geomap or {}
        
        results = []
        
        # 1. Run sTASEP simulation for each sequence
        # We process them to create the batch for the model
        batch_data = []
        
        print(f"Running sTASEP simulation for {len(sequences)} sequences with {n_stasep_runs} runs per sequence (init_p={init_p})...")
        
        for i, seq_raw in enumerate(sequences):
            # Normalize sequence: RNA uses U, so replace T with U if present
            seq = seq_raw.replace("T", "U").replace("t", "u")
            tx_id = f"seq_{i}"
            
            # Geometry check
            if tx_id in geomap:
                angle_dev_sum, pair_count = geomap[tx_id]
            else:
                # Fallback: compute on-the-fly using ViennaRNA if available
                # If valid RNA length
                if len(seq) % 3 == 0:
                     angle_dev_sum, pair_count = compute_geometry_features(seq, cache_dir=self.cache_dir)
                else:
                     # Invalid length for codon mapping, default to zeros
                     L_codons = len(seq) // 3
                     angle_dev_sum = np.zeros(L_codons, dtype=np.float64)
                     pair_count = np.zeros(L_codons, dtype=np.int32)
            
            # Run sTASEP simulation
            
            args = (
                tx_id, 
                seq, 
                None, # a_cnts
                None, # p_cnts
                angle_dev_sum, 
                pair_count,
                self.rates,
                self.alpha_vec,
                self.beta_vec,
                self.bucket_vec,
                DEFAULT_ANGLE_BINS,
                n_stasep_runs,
                init_p
            )
            
            # simulate_transcript returns:
            # tx, cods, obs_counts, sim_vec_raw, sim_vec_scaled, scale, completed_total
            res = simulate_transcript(args)
            
            # We construct a dict similar to what PolishPKLDataset returns
            tx, cods, obs, sim_raw, sim_scaled, scale, comp = res
            
            # Normalize to average counts to reduce variance
            if n_stasep_runs > 1:
                sim_scaled = sim_scaled / n_stasep_runs
            
            if not use_polisher:
                results.append(sim_scaled)
                continue
            
            # Prepare data for pad_collate
            # Need: cod_ids, sim_feat, angle_bin, pair_bin, bucket_idx
            
            L = len(cods)
            cod_ids = torch.tensor([CODON2IDX.get(c, 64) for c in cods], dtype=torch.long)
            sim_feat = torch.tensor(np.log1p(sim_scaled), dtype=torch.float32) # Log transform as in Dataset
            
            # Geometry bins
            # Recalculate bins for model (simulate_transcript used them for sim, but model needs indices)
            kk = np.digitize(angle_dev_sum, DEFAULT_ANGLE_BINS) - 1
            kk = np.clip(kk, 0, K_ANGLE_BINS - 1).astype(np.int64)
            pb = np.clip(pair_count, 0, 3).astype(np.int64)
            
            angle_bin = torch.tensor(kk, dtype=torch.long)
            pair_bin = torch.tensor(pb, dtype=torch.long)
            
            bb = np.zeros(L, dtype=np.int64)
            l1 = L // 3
            l2 = (2 * L) // 3
            bb[l1:l2] = 1
            bb[l2:] = 2
            bucket_idx = torch.tensor(bb, dtype=torch.long)
            
            item = {
                "tx": tx_id,
                "length": L,
                "cod_ids": cod_ids,
                "sim_feat": sim_feat,
                "angle_bin": angle_bin,
                "pair_bin": pair_bin,
                "bucket_idx": bucket_idx
            }
            batch_data.append(item)

        # 2. Batch and Predict
        if not batch_data:
            return results

        # We can implement simple batching (all in one for now as it's inference)
        batch = pad_collate(batch_data)
        
        model = self._load_model(task)
        
        with torch.no_grad():
            # Move to device
            cod = batch["cod_ids"].to(self.device)
            sim = batch["sim_feat"].to(self.device)
            msk = batch["mask"].to(self.device)
            ang = batch["angle_bin"].to(self.device)
            pai = batch["pair_bin"].to(self.device)
            buc = batch["bucket_idx"].to(self.device)
            
            # Forward
            if task == "riboseq":
                logits = model(cod, sim, msk, angle_bin=ang, pair_bin=pai, bucket_idx=buc)
                preds = torch.expm1(logits)  # Convert log(counts+1) to counts
                
            elif task == "te":
                cnts, te_preds = model(cod, sim, msk, angle_bin=ang, pair_bin=pai, bucket_idx=buc)
                preds = te_preds # [0,1] scaled TE
                # Output is predicted TE value
                # For this MVP, we return scaled and warn, or if we had params we'd scale back.
                
            elif task == "protein":
                cnts, expr_preds = model(cod, sim, msk, angle_bin=ang, pair_bin=pai, bucket_idx=buc)
                preds = expr_preds
        
        # Collect results
        preds_np = preds.cpu().numpy()
        
        for i, p in enumerate(preds_np):
            # For riboseq, p is (Lmax,), we need to slice to length and handle mask
            L = batch_data[i]["length"]
            if task == "riboseq":
                val = p[:L] # Array of counts per codon
            else:
                val = float(p) # Scalar
            results.append(val)
            
        return results

