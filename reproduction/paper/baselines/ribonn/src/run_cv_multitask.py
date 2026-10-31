"""
Multitask nested cross-validation training for RiboNN (full sequence: UTR5 + CDS + UTR3).

Trains a single model that predicts TE for all 4 cell lines simultaneously:
  TE_HEK293, TE_LCL, TE_RPE-1, TE_IPSC

Uses masked_mse_loss (with_NAs=True) so transcripts missing a TE value
for some cell lines are handled gracefully.

Usage:
  python -m src.run_cv_multitask
  python -m src.run_cv_multitask --epochs 100 --outer_cv_folds 5
"""

import argparse
import yaml
from src.train import train_model_nested_cv


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/training_multitask.txt")
    parser.add_argument("--target_pattern", default="^TE_")
    parser.add_argument("--experiment_name", default="multitask_CV")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--inner_cv_folds", type=int, default=9)
    parser.add_argument("--outer_cv_folds", type=int, default=10)
    parser.add_argument("--stop_after_n_inner_cv_folds", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2048)

    args = parser.parse_args()

    config = load_config("config/conf.yml")

    config["tx_info_path"] = args.train_path
    config["target_column_pattern"] = args.target_pattern
    config["train_batch_size"] = args.batch_size
    config["val_batch_size"] = args.batch_size
    config["test_batch_size"] = args.batch_size

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

    print(f"Starting multitask Cross-Validation training for {args.experiment_name}...")
    print(f"  Training data: {args.train_path}")
    print(f"  Target pattern: {args.target_pattern}")

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
