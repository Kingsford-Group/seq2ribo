
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from src.train import train_model_nested_cv
from src.model import RiboNN
from src.data import RiboNNDataModule
import os

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--target_pattern", required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--inner_cv_folds", type=int, default=9)
    # Added to control if we want to run full 10 folds or fewer for testing/speed
    parser.add_argument("--outer_cv_folds", type=int, default=10) 
    parser.add_argument("--stop_after_n_inner_cv_folds", type=int, default=None)
    
    args = parser.parse_args()

    # Training
    config = load_config("config/conf.yml")
    
    # Overrides
    config["tx_info_path"] = args.train_path
    config["target_column_pattern"] = args.target_pattern
    
    config["pad_5_prime"] = True
    config["optimizer"] = "AdamW"
    config["lr"] = 0.0001
    config["l2_scale"] = 0.001
    config["with_NAs"] = True
    config["max_shift"] = 0
    config["num_conv_layers"] = 10
    config["symmetric_shift"] = True
    config["remove_extreme_txs"] = True
    config["activation"] = "relu"
    config["max_utr5_len"] = 1_381
    config["max_cds_utr3_len"] = 11_937
    
    print(f"Starting Cross-Validation training for {args.experiment_name}...")
    # Update config max_epochs as train_model_nested_cv uses it from config or argument?
    # Checked train.py: it takes max_epochs as arg
    
    train_model_nested_cv(
        config,
        experiment_name=args.experiment_name,
        max_epochs=args.epochs,
        outer_cv_folds=args.outer_cv_folds,
        inner_cv_folds=args.inner_cv_folds,
        stop_after_n_inner_cv_folds=args.stop_after_n_inner_cv_folds
    )
    
if __name__ == "__main__":
    main()
