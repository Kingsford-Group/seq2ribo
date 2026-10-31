"""Command-line inference: `seq2ribo ...` (after `pip install -e .`) or `python scripts/run_inference.py ...`."""
import argparse
import csv
import json
import sys

import numpy as np


def read_fasta(path):
    ids, sequences, header, seq = [], [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    ids.append(header)
                    sequences.append("".join(seq))
                header, seq = line[1:], []
            elif line:
                seq.append(line)
    if header is not None:
        ids.append(header)
        sequences.append("".join(seq))
    return ids, sequences


def main(argv=None):
    parser = argparse.ArgumentParser(description="seq2ribo Inference CLI")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--seq", type=str, help="Single RNA sequence string")
    source.add_argument("--fasta", type=str, help="Path to FASTA file")
    parser.add_argument("--task", type=str, choices=["riboseq", "te", "protein"], default="riboseq",
                        help="Task to perform: ribosome profiling, TE prediction, or protein expression.")
    parser.add_argument("--use_utr", action="store_true",
                        help="For task=te, use CDS+UTR TE model and require split --utr5/--cds/--utr3 inputs.")
    parser.add_argument("--cell-line", type=str, default="ipsc", help="Cell line (e.g., ipsc, hek293, lcl, rpe)")
    parser.add_argument("--weights-dir", type=str, default=None,
                        help="Directory containing model weights (default: the repository's weights/)")
    parser.add_argument("--version", type=str, choices=["corrected", "paper"], default="corrected",
                        help="Model version: corrected (default) or paper (published checkpoints in weights/paper/)")
    parser.add_argument("--cache-dir", type=str, default="cache/geometry", help="Directory for geometry cache")
    parser.add_argument("--n-stasep-runs", type=int, default=100,
                        help="Number of sTASEP simulations per sequence; each run is polished and the predictions are averaged")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the sTASEP simulations (per sequence)")
    parser.add_argument("--batch-size", type=int, default=64, help="Model batch size (rows = sequence x run)")
    parser.add_argument("--init-p", type=float, default=None,
                        help="Initiation probability for sTASEP (default: the value used for training, 0.01)")
    parser.add_argument("--return-scaled-te", action="store_true",
                        help="For task=te, return scaled TE in [0,1] instead of inverse-transformed TE.")
    parser.add_argument("--utr5", type=str, help="5' UTR RNA sequence (required with --use_utr for task=te).")
    parser.add_argument("--cds", type=str, help="CDS RNA sequence (required with --use_utr for task=te).")
    parser.add_argument("--utr3", type=str, help="3' UTR RNA sequence (required with --use_utr for task=te).")
    parser.add_argument("--no-polisher", action="store_true",
                        help="task=riboseq only: return the simulated sTASEP profile, skip the polisher model")
    parser.add_argument("--output", type=str,
                        help="Output file: .csv writes CSV (profiles as space-separated values), anything else JSON. "
                             "If not set, prints JSON to stdout.")
    args = parser.parse_args(argv)

    if args.no_polisher and args.task != "riboseq":
        parser.error("--no-polisher applies to --task riboseq only.")
    sequences, ids, utr5_list, cds_list, utr3_list = [], [], None, None, None
    if args.task == "te" and args.use_utr:
        if args.seq or args.fasta:
            parser.error("Do not pass --seq/--fasta with --use_utr for task=te. Use --utr5/--cds/--utr3.")
        if args.utr5 is None or args.cds is None or args.utr3 is None:
            parser.error("--use_utr for task=te requires --utr5, --cds, and --utr3.")
        utr5_list, cds_list, utr3_list = [args.utr5], [args.cds], [args.utr3]
        ids.append("input_seq")
    else:
        if args.utr5 or args.cds or args.utr3:
            parser.error("--utr5/--cds/--utr3 are only valid with --task te --use_utr.")
        if args.seq:
            sequences.append(args.seq)
            ids.append("input_seq")
        elif args.fasta:
            try:
                ids, sequences = read_fasta(args.fasta)
            except OSError as e:
                parser.error(f"cannot read FASTA: {e}")
        else:
            parser.error("Must provide --seq or --fasta")

    print(f"Loaded {len(ids)} sequences.")
    from .inference import Seq2Ribo   # needs torch and mamba-ssm
    predictor = Seq2Ribo(cell_line=args.cell_line, weights_dir=args.weights_dir, cache_dir=args.cache_dir,
                         version=args.version)
    try:
        results = predictor.predict(
            None if (args.task == "te" and args.use_utr) else sequences,
            task=args.task,
            use_polisher=not args.no_polisher,
            n_stasep_runs=args.n_stasep_runs,
            init_p=args.init_p,
            return_scaled_te=args.return_scaled_te,
            use_utr=args.use_utr,
            utr5_list=utr5_list,
            cds_list=cds_list,
            utr3_list=utr3_list,
            seed=args.seed,
            batch_size=args.batch_size,
        )
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    output_data = [{"id": i, "prediction": r.tolist() if isinstance(r, (np.ndarray, np.generic)) else r}
                   for i, r in zip(ids, results)]
    if args.output and args.output.lower().endswith(".csv"):
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "prediction"])
            for d in output_data:
                p = d["prediction"]
                w.writerow([d["id"], " ".join(f"{v:.6g}" for v in p) if isinstance(p, list) else p])
        print(f"Results written to {args.output}")
    elif args.output:
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results written to {args.output}")
    else:
        print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    main()
