"""Main inference API for seq2ribo predictions."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from . import constants as CONST
from .data import pad_collate
from .geometry import compute_geometry_features
from .models import MambaExprFull, MambaTEFull, MambaTEFullUTR, RiboPolisherMamba
from .simulation import simulate_transcript
from .utils import CODON2IDX, DEFAULT_ANGLE_BINS, K_ANGLE_BINS, build_full_rate, load_state_dict_safely

NUC2IDX = {"A": 0, "U": 1, "G": 2, "C": 3}
NUC_PAD_IDX = 4



class Seq2Ribo:
    def __init__(self, 
                 cell_line: str = "hek293", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 weights_dir: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        self.cell_line = cell_line.lower()
        self.device = torch.device(device)
        self.models = {} 
        self.te_transforms = {}
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

    @staticmethod
    def _normalize_rna(seq: str) -> str:
        return seq.replace("T", "U").replace("t", "u").upper()

    @staticmethod
    def _tokenize_nucleotides(seq: str) -> np.ndarray:
        seq_norm = Seq2Ribo._normalize_rna(seq)
        return np.array([NUC2IDX.get(ch, NUC2IDX["A"]) for ch in seq_norm], dtype=np.int64)

    @staticmethod
    def _ensure_sequence_list(seq_input: Union[str, List[str], tuple], name: str) -> List[str]:
        if isinstance(seq_input, str):
            return [seq_input]
        if isinstance(seq_input, (list, tuple)):
            if not all(isinstance(x, str) for x in seq_input):
                raise ValueError(f"All entries in '{name}' must be strings.")
            return list(seq_input)
        raise ValueError(f"'{name}' must be a string or list/tuple of strings.")

    @staticmethod
    def _pad_collate_te_utr(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        B = len(batch)
        Lmax = max(x["length"] for x in batch)
        N5max = max(1, max(x["utr5_len"] for x in batch))
        N3max = max(1, max(x["utr3_len"] for x in batch))

        cod = torch.full((B, Lmax), 64, dtype=torch.long)
        sim = torch.zeros((B, Lmax), dtype=torch.float32)
        msk = torch.zeros((B, Lmax), dtype=torch.bool)
        angle_bin = torch.zeros((B, Lmax), dtype=torch.long)
        pair_bin = torch.zeros((B, Lmax), dtype=torch.long)
        bucket_idx = torch.zeros((B, Lmax), dtype=torch.long)

        utr5 = torch.full((B, N5max), NUC_PAD_IDX, dtype=torch.long)
        utr3 = torch.full((B, N3max), NUC_PAD_IDX, dtype=torch.long)
        utr5_m = torch.zeros((B, N5max), dtype=torch.bool)
        utr3_m = torch.zeros((B, N3max), dtype=torch.bool)
        txs = []

        for i, ex in enumerate(batch):
            L = ex["length"]
            txs.append(ex["tx"])
            cod[i, :L] = ex["cod_ids"]
            sim[i, :L] = ex["sim_feat"]
            msk[i, :L] = True
            angle_bin[i, :L] = ex["angle_bin"]
            pair_bin[i, :L] = ex["pair_bin"]
            bucket_idx[i, :L] = ex["bucket_idx"]

            n5 = ex["utr5_len"]
            if n5 > 0:
                utr5[i, :n5] = ex["utr5_ids"]
                utr5_m[i, :n5] = True
            n3 = ex["utr3_len"]
            if n3 > 0:
                utr3[i, :n3] = ex["utr3_ids"]
                utr3_m[i, :n3] = True

        return {
            "tx": txs,
            "cod_ids": cod,
            "sim_feat": sim,
            "mask": msk,
            "angle_bin": angle_bin,
            "pair_bin": pair_bin,
            "bucket_idx": bucket_idx,
            "utr5_ids": utr5,
            "utr3_ids": utr3,
            "utr5_mask": utr5_m,
            "utr3_mask": utr3_m,
        }

    def _load_te_transform(self, use_utr: bool = False):
        mode = "utr" if use_utr else "cds"
        cache_key = (self.cell_line, mode)
        if cache_key in self.te_transforms:
            return self.te_transforms[cache_key]

        transform_filename = f"{self.cell_line}_te_transform_{mode}.json"
        transform_path = self.weights_dir / transform_filename

        if not transform_path.exists():
            if Path(transform_filename).exists():
                transform_path = Path(transform_filename)
            else:
                raise FileNotFoundError(
                    f"TE transform file ({transform_filename}) not found at {self.weights_dir}. "
                    "Please provide correct weights_dir."
                )

        with open(transform_path, "r", encoding="utf-8") as f:
            te_transform = json.load(f)

        required = {"name", "lo", "hi", "eps"}
        missing = required - set(te_transform.keys())
        if missing:
            raise ValueError(f"Invalid TE transform file {transform_path}: missing keys {sorted(missing)}")
        if te_transform["name"] != "minmax":
            raise ValueError(f"Unsupported TE transform '{te_transform['name']}' in {transform_path}; expected 'minmax'")
        if float(te_transform["hi"]) <= float(te_transform["lo"]):
            raise ValueError(f"Invalid TE transform bounds in {transform_path}: hi must be greater than lo")

        self.te_transforms[cache_key] = te_transform
        return te_transform

    @staticmethod
    def _inverse_minmax_te(te_scaled_np: np.ndarray, te_transform: Dict[str, float]) -> np.ndarray:
        lo = float(te_transform["lo"])
        hi = float(te_transform["hi"])
        eps = float(te_transform["eps"])
        te_scaled_np = np.clip(te_scaled_np, eps, 1.0 - eps)
        return lo + (hi - lo) * te_scaled_np

    def _load_model(self, task: str, use_utr: Optional[bool] = None):
        te_use_utr = bool(use_utr) if task == "te" else False
        model_key = (task, "utr") if task == "te" and te_use_utr else (task, "default")
        if model_key in self.models:
            return self.models[model_key]

        if task == "te":
            print(f"Loading model for task '{task}' (Cell Line: {self.cell_line}, use_utr={te_use_utr})...")
        else:
            print(f"Loading model for task '{task}' (Cell Line: {self.cell_line})...")
        
        ckpt_filename = "ipsc_mamba_final.pt"
        
        if task == "riboseq":
            ckpt_map = {
                "hek293": "hek293_mamba_final_unscaled.pt",
                "lcl": "lcl_mamba_final_unscaled.pt",
                "rpe": "rpe_mamba_final_unscaled.pt",
                "ipsc": "ipsc_mamba_final_unscaled.pt",
            }
            ckpt_filename = ckpt_map[self.cell_line]

            model = RiboPolisherMamba(
                d_model=192, 
                n_layers=4, 
                d_state=16,
                d_conv=4,
                expand=2,
                dropout=0.1,
                use_mamba2=False,
                activation="softplus"
            )
            
        elif task == "te":
            if te_use_utr:
                ckpt_map = {
                    "hek293": "hek293_mamba_te_utr_final.pt",
                    "lcl": "lcl_mamba_te_utr_final.pt",
                    "rpe": "rpe_mamba_te_utr_final.pt",
                    "ipsc": "ipsc_mamba_te_utr_final.pt",
                }
            else:
                ckpt_map = {
                    "hek293": "hek293_mamba_te_full_final_cds.pt",
                    "lcl": "lcl_mamba_te_full_final_cds.pt",
                    "rpe": "rpe_mamba_te_full_final_cds.pt",
                    "ipsc": "ipsc_mamba_te_full_final_cds.pt",
                }
            ckpt_filename = ckpt_map[self.cell_line]

            base = RiboPolisherMamba(
                d_model=192, 
                n_layers=4, 
                d_state=16,
                d_conv=4,
                expand=2,
                dropout=0.1, 
                use_mamba2=False, 
                activation="softplus"
            )
            if te_use_utr:
                model = MambaTEFullUTR(
                    base,
                    d_te=128,
                    n_te_layers=2,
                    te_d_state=16,
                    te_d_conv=4,
                    te_expand=2,
                    dropout=0.1,
                    use_log1p=False,
                )
            else:
                model = MambaTEFull(base, hidden=256)
            self._load_te_transform(use_utr=te_use_utr)

        elif task == "protein":
            ckpt_map = {
                "hek293": "hek293_mamba_expr_full_final.pt",
                "lcl": "lcl_mamba_expr_full_final.pt",
                "rpe": "rpe_mamba_expr_full_final.pt",
                "ipsc": "ipsc_mamba_expr_full_final.pt",
            }
            ckpt_filename = ckpt_map[self.cell_line]
            expr_hidden_map = {
                "hek293": 128,
                "lcl": 128,
                "rpe": 128,
                "ipsc": 64,
            }
            expr_hidden = expr_hidden_map[self.cell_line]

            base = RiboPolisherMamba(
                d_model=192, 
                n_layers=4, 
                d_state=16,
                d_conv=4,
                expand=2,
                dropout=0.1, 
                use_mamba2=False,
                activation="softplus"
            )
            model = MambaExprFull(base, hidden=expr_hidden, use_log1p=False)

            
        else:
            raise ValueError(f"Unknown task: {task}")

        ckpt_path = self.weights_dir / ckpt_filename
        
        if not ckpt_path.exists():
            if Path(ckpt_filename).exists():
                ckpt_path = Path(ckpt_filename)
            else:
                raise FileNotFoundError(f"Checkpoint for task '{task}' ({ckpt_filename}) not found at {self.weights_dir}. Please provide correct weights_dir.")

        print(f"Loading weights from {ckpt_path}...")
        # Load weights
        state_dict = load_state_dict_safely(ckpt_path, self.device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]
        elif isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        self.models[model_key] = model
        return model

    def predict(
        self,
        sequences: Optional[Union[str, List[str]]] = None,
        task: str = "riboseq",
        geomap: Optional[Dict] = None,
        use_polisher: bool = True,
        n_stasep_runs: int = 1,
        init_p: float = 0.01,
        return_scaled_te: bool = False,
        use_utr: bool = False,
        utr5_list: Optional[Union[str, List[str]]] = None,
        cds_list: Optional[Union[str, List[str]]] = None,
        utr3_list: Optional[Union[str, List[str]]] = None,
    ):
        """
        Run prediction pipeline:
        1. sTASEP simulation (if needed to generate features)
        2. Model forward pass (if use_polisher=True)
        """
        if use_utr and task != "te":
            raise ValueError("use_utr=True is only supported for task='te'.")

        if use_utr and task == "te":
            if sequences is not None:
                raise ValueError(
                    "For TE+UTR, do not pass merged transcript sequences. "
                    "Provide split inputs via utr5_list, cds_list, and utr3_list."
                )
            if utr5_list is None or cds_list is None or utr3_list is None:
                raise ValueError("TE+UTR requires utr5_list, cds_list, and utr3_list.")
            utr5_values = self._ensure_sequence_list(utr5_list, "utr5_list")
            cds_values = self._ensure_sequence_list(cds_list, "cds_list")
            utr3_values = self._ensure_sequence_list(utr3_list, "utr3_list")
            if not (len(utr5_values) == len(cds_values) == len(utr3_values)):
                raise ValueError("utr5_list, cds_list, and utr3_list must have equal lengths.")
            sequence_values = cds_values
        else:
            if sequences is None:
                raise ValueError("Please provide 'sequences' for this prediction mode.")
            sequence_values = self._ensure_sequence_list(sequences, "sequences")
            utr5_values = None
            utr3_values = None
            
        geomap = geomap or {}
        
        results = []
        
        # 1. Run sTASEP simulation for each sequence
        # We process them to create the batch for the model
        batch_data = []
        
        if n_stasep_runs < 1:
            raise ValueError("n_stasep_runs must be >= 1.")

        print(f"Running sTASEP simulation for {len(sequence_values)} sequences with {n_stasep_runs} runs per sequence (init_p={init_p})...")
        
        for i, seq_raw in enumerate(sequence_values):
            seq = self._normalize_rna(seq_raw)
            if len(seq) % 3 != 0:
                raise ValueError(
                    f"Sequence {i} has length {len(seq)} which is not divisible by 3. "
                    "All sequences must consist of complete codons (length divisible by 3)."
                )
            tx_id = f"seq_{i}"
            
            # Geometry check
            if tx_id in geomap:
                angle_dev_sum, pair_count = geomap[tx_id]
            else:
                if len(seq) % 3 == 0:
                     angle_dev_sum, pair_count = compute_geometry_features(seq, cache_dir=self.cache_dir)
                else:
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
            
            _, cods, _, sim_raw, _, _, _ = res
            sim_counts = sim_raw.astype(np.float32, copy=False)
            if n_stasep_runs > 1:
                sim_counts = sim_counts / float(n_stasep_runs)
            
            if not use_polisher:
                results.append(sim_counts)
                continue
            
            # Prepare data for pad_collate
            # Need: cod_ids, sim_feat, angle_bin, pair_bin, bucket_idx
            
            L = len(cods)
            cod_ids = torch.tensor([CODON2IDX.get(c, 64) for c in cods], dtype=torch.long)
            # Use simulation counts (averaged over runs) for all tasks.
            # No ribo-load scaling is applied anywhere at inference time.
            sim_feat = torch.tensor(np.log1p(sim_counts), dtype=torch.float32)
            
            # Geometry bins
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
            if use_utr and task == "te":
                utr5_tokens = self._tokenize_nucleotides(utr5_values[i]) if utr5_values is not None else np.array([], dtype=np.int64)
                utr3_tokens = self._tokenize_nucleotides(utr3_values[i]) if utr3_values is not None else np.array([], dtype=np.int64)
                item["utr5_ids"] = torch.tensor(utr5_tokens, dtype=torch.long)
                item["utr3_ids"] = torch.tensor(utr3_tokens, dtype=torch.long)
                item["utr5_len"] = int(len(utr5_tokens))
                item["utr3_len"] = int(len(utr3_tokens))
            batch_data.append(item)

        # 2. Batch and Predict
        if not batch_data:
            return results

        if use_utr and task == "te":
            batch = self._pad_collate_te_utr(batch_data)
        else:
            batch = pad_collate(batch_data)
        
        model = self._load_model(task, use_utr=use_utr if task == "te" else None)
        
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
                if use_utr:
                    u5 = batch["utr5_ids"].to(self.device)
                    u3 = batch["utr3_ids"].to(self.device)
                    u5m = batch["utr5_mask"].to(self.device)
                    u3m = batch["utr3_mask"].to(self.device)
                    cnts, te_preds = model(
                        cod, sim, msk,
                        u5, u3, u5m, u3m,
                        angle_bin=ang, pair_bin=pai, bucket_idx=buc
                    )
                else:
                    cnts, te_preds = model(cod, sim, msk, angle_bin=ang, pair_bin=pai, bucket_idx=buc)
                preds = te_preds # [0,1] scaled TE
                
            elif task == "protein":
                # MC inference
                mc_preds = []
                model.train()
                for _ in range(32):
                    _, expr_preds = model(cod, sim, msk, angle_bin=ang, pair_bin=pai, bucket_idx=buc)
                    mc_preds.append(expr_preds)
                model.eval()
                preds = torch.stack(mc_preds, dim=0).mean(dim=0)
        
        # Collect results
        preds_np = preds.cpu().numpy()
        
        for i, p in enumerate(preds_np):
            # For riboseq, p is (Lmax,), we need to slice to length and handle mask
            L = batch_data[i]["length"]
            if task == "riboseq":
                val = p[:L] # Array of counts per codon
            elif task == "te":
                if return_scaled_te:
                    val = float(p)
                else:
                    te_transform = self._load_te_transform(use_utr=use_utr)
                    val = float(self._inverse_minmax_te(np.array([p], dtype=np.float64), te_transform)[0])
            else:
                val = float(p) # Scalar
            results.append(val)
            
        return results

