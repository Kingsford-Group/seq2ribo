"""
Predict TE using ensemble of multitask CV models, then split predictions
into per-cell-line output files for evaluation.

The multitask model outputs 4 values per transcript (one per cell line).
The column order matches the alphabetical order of TE_* columns in the
training data, which is determined by how RiboNNDataModule's
  df.filter(regex=r"^TE_")
selects and orders them.

Usage (full sequence):
  python -m src.predict_cv_multitask \
      --experiment_name multitask_CV

Usage (CDS only):
  python -m src.predict_cv_multitask \
      --experiment_name multitask_CDS_CV \
      --cds_only
"""

import argparse
import os
import glob
import yaml
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.model import RiboNN
from src.data import RiboNNDataModule


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CELL_LINES = {
    "HEK293": {
        "pred_input": "HEK293/prediction_input.txt",
        "pred_input_cds": "HEK293/prediction_input_cds_only.txt",
    },
    "LCL": {
        "pred_input": "LCL/prediction_input.txt",
        "pred_input_cds": "LCL/prediction_input_cds_only.txt",
    },
    "RPE": {
        "pred_input": "RPE/prediction_input.txt",
        "pred_input_cds": "RPE/prediction_input_cds_only.txt",
    },
    "IPSC": {
        "pred_input": "IPSC/prediction_input.txt",
        "pred_input_cds": "IPSC/prediction_input_cds_only.txt",
    },
}


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_target_columns(training_data_path):
    """Read the header of the multitask training file to determine the
    order of TE_* columns, which matches the model output order."""
    df_head = pd.read_csv(training_data_path, sep="\t", nrows=0)
    te_cols = sorted([c for c in df_head.columns if c.startswith("TE_")])
    return te_cols


def predict_ensemble(loader, checkpoints, device):
    """Run prediction with all models and return averaged predictions."""
    ensemble_preds = None

    for i, ckpt in enumerate(checkpoints):
        print(f"  Loading model {i+1}/{len(checkpoints)}: {os.path.basename(os.path.dirname(ckpt))}")
        model = RiboNN.load_from_checkpoint(ckpt)
        model.to(device)
        model.eval()

        preds_list = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"  Model {i+1}", leave=False):
                batch = batch.to(device)
                out = model(batch)
                preds_list.append(out.cpu().numpy())

        current_preds = np.concatenate(preds_list, axis=0)

        if ensemble_preds is None:
            ensemble_preds = np.zeros_like(current_preds, dtype=np.float64)

        ensemble_preds += current_preds

    return ensemble_preds / len(checkpoints)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", required=True,
                        help="e.g. multitask_CV or multitask_CDS_CV")
    parser.add_argument("--cds_only", action="store_true",
                        help="Use CDS-only config (pad_5_prime=False)")
    parser.add_argument("--train_data_path", default=None,
                        help="Path to multitask training file (to read column order). "
                             "Defaults based on --cds_only flag.")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: results/<experiment_name>)")
    args = parser.parse_args()

    if args.train_data_path is None:
        if args.cds_only:
            args.train_data_path = "data/training_multitask_cds_only.txt"
        else:
            args.train_data_path = "data/training_multitask.txt"

    if args.out_dir is None:
        args.out_dir = f"results/{args.experiment_name}"

    te_cols = get_target_columns(args.train_data_path)
    print(f"Target columns (model output order): {te_cols}")

    model_dir = f"models/{args.experiment_name}"
    checkpoints = sorted(glob.glob(f"{model_dir}/*/best_model.ckpt"))
    if not checkpoints:
        print(f"No checkpoints found in {model_dir}")
        return

    print(f"Found {len(checkpoints)} models for ensemble.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_config = load_config("config/conf.yml")
    base_config["target_column_pattern"] = None
    base_config["with_NAs"] = True
    base_config["max_shift"] = 0
    base_config["num_conv_layers"] = 10
    base_config["symmetric_shift"] = True
    base_config["num_workers"] = 0

    if args.cds_only:
        base_config["pad_5_prime"] = False
        base_config["remove_extreme_txs"] = True
        base_config["max_utr5_len"] = 0
        base_config["max_cds_utr3_len"] = 20_000
    else:
        base_config["pad_5_prime"] = True
        base_config["remove_extreme_txs"] = False
        base_config["max_utr5_len"] = 1_381
        base_config["max_cds_utr3_len"] = 11_937

    os.makedirs(args.out_dir, exist_ok=True)

    # When pad_5_prime=False, tensor length depends on max_tx_len which varies
    # per dataset. We must match the training data's max_tx_len so that
    # len_after_conv matches the saved model's linear layer dimensions.
    train_max_tx_len = None
    if not base_config["pad_5_prime"]:
        train_config = base_config.copy()
        train_config["tx_info_path"] = args.train_data_path
        train_config["target_column_pattern"] = "^TE_"
        train_dm = RiboNNDataModule(train_config)
        train_max_tx_len = train_dm.max_tx_len
        print(f"Training data max_tx_len: {train_max_tx_len}")
        del train_dm

    for cell_line, paths in CELL_LINES.items():
        print(f"\n{'='*60}")
        print(f"Predicting for {cell_line}...")
        print(f"{'='*60}")

        input_key = "pred_input_cds" if args.cds_only else "pred_input"
        pred_input_path = paths[input_key]

        if not os.path.exists(pred_input_path):
            print(f"  WARNING: {pred_input_path} not found, skipping.")
            continue

        config = base_config.copy()
        config["tx_info_path"] = pred_input_path

        dm = RiboNNDataModule(config)

        if train_max_tx_len is not None and dm.max_tx_len > train_max_tx_len:
            n_before = len(dm.df)
            dm.df = dm.df.query("tx_size <= @train_max_tx_len").reset_index(drop=True)
            dm.max_tx_len = train_max_tx_len
            print(f"  Capped max_tx_len: {dm.max_tx_len} (dropped {n_before - len(dm.df)} transcripts)")

        loader = dm.predict_dataloader()

        avg_preds = predict_ensemble(loader, checkpoints, device)
        # avg_preds shape: (num_samples, num_targets)

        output_df = dm.df[["tx_id"]].copy()

        if avg_preds.ndim == 1:
            avg_preds = avg_preds.reshape(-1, 1)

        for idx, col in enumerate(te_cols):
            output_df[f"predicted_{col}"] = avg_preds[:, idx]

        cell_out_dir = os.path.join(args.out_dir, cell_line)
        os.makedirs(cell_out_dir, exist_ok=True)
        out_path = os.path.join(cell_out_dir, "prediction_output_multitask.txt")
        output_df.to_csv(out_path, sep="\t", index=False)
        print(f"  Saved to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
