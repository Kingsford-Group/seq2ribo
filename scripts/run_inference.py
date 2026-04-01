#!/usr/bin/env python3
import argparse
import sys
import json
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from seq2ribo import Seq2Ribo

def main():
    parser = argparse.ArgumentParser(description="seq2ribo Inference CLI")
    parser.add_argument("--seq", type=str, help="Single RNA sequence string")
    parser.add_argument("--fasta", type=str, help="Path to FASTA file")
    parser.add_argument("--task", type=str, choices=["riboseq", "te", "protein"], default="riboseq",
                        help="Task to perform: ribosome profiling, TE prediction, or protein expression.")
    parser.add_argument("--use_utr", action="store_true",
                        help="For task=te, use CDS+UTR TE model and require split --utr5/--cds/--utr3 inputs.")
    parser.add_argument("--cell-line", type=str, default="ipsc", help="Cell line (e.g., ipsc, hek293, lcl, rpe)")
    parser.add_argument("--weights-dir", type=str, default=str(Path(__file__).parent.parent / "weights"), help="Directory containing model weights")
    parser.add_argument("--cache-dir", type=str, default="cache/geometry", help="Directory for geometry cache")
    parser.add_argument("--n-stasep-runs", type=int, default=100, help="Number of sTASEP simulations per sequence")
    parser.add_argument("--init-p", type=float, default=0.01, help="Initiation probability for sTASEP (default: 0.01)")
    parser.add_argument("--return-scaled-te", action="store_true",
                        help="For task=te, return scaled TE in [0,1] instead of inverse-transformed TE.")
    parser.add_argument("--utr5", type=str, help="5' UTR RNA sequence (required with --use_utr for task=te).")
    parser.add_argument("--cds", type=str, help="CDS RNA sequence (required with --use_utr for task=te).")
    parser.add_argument("--utr3", type=str, help="3' UTR RNA sequence (required with --use_utr for task=te).")
    parser.add_argument("--no-polisher", action="store_true", help="Run only sTASEP simulation, skip polisher model")
    parser.add_argument("--output", type=str, help="Output file (JSON/CSV). If not set, prints to stdout.")
    
    args = parser.parse_args()
    
    # Input validation
    sequences = []
    ids = []
    utr5_list = None
    cds_list = None
    utr3_list = None

    if args.task == "te" and args.use_utr:
        if args.seq or args.fasta:
            print("Error: Do not pass --seq/--fasta with --use_utr for task=te. Use --utr5/--cds/--utr3.")
            parser.print_help()
            sys.exit(1)
        if not (args.utr5 is not None and args.cds is not None and args.utr3 is not None):
            print("Error: --use_utr for task=te requires --utr5, --cds, and --utr3.")
            parser.print_help()
            sys.exit(1)
        utr5_list = [args.utr5]
        cds_list = [args.cds]
        utr3_list = [args.utr3]
        ids.append("input_seq")
    else:
        if args.utr5 or args.cds or args.utr3:
            print("Error: --utr5/--cds/--utr3 are only valid with --task te --use_utr.")
            parser.print_help()
            sys.exit(1)

        if args.seq:
            sequences.append(args.seq)
            ids.append("input_seq")
        elif args.fasta:
            # Simple FASTA parser
            try:
                with open(args.fasta, 'r') as f:
                    header = None
                    seq = []
                    for line in f:
                        line = line.strip()
                        if line.startswith(">"):
                            if header:
                                ids.append(header)
                                sequences.append("".join(seq))
                            header = line[1:]
                            seq = []
                        else:
                            seq.append(line)
                    if header:
                        ids.append(header)
                        sequences.append("".join(seq))
            except Exception as e:
                print(f"Error reading FASTA: {e}")
                sys.exit(1)
        else:
            print("Error: Must provide --seq or --fasta")
            parser.print_help()
            sys.exit(1)

    loaded_n = len(ids) if args.task == "te" and args.use_utr else len(sequences)
    print(f"Loaded {loaded_n} sequences.")
    
    # Initialize Predictor
    predictor = Seq2Ribo(cell_line=args.cell_line, weights_dir=args.weights_dir, cache_dir=args.cache_dir)
    
    # Run Prediction
    try:
        input_sequences = None if (args.task == "te" and args.use_utr) else sequences
        results = predictor.predict(
            input_sequences,
            task=args.task, 
            use_polisher=not args.no_polisher,
            n_stasep_runs=args.n_stasep_runs,
            init_p=args.init_p,
            return_scaled_te=args.return_scaled_te,
            use_utr=args.use_utr,
            utr5_list=utr5_list,
            cds_list=cds_list,
            utr3_list=utr3_list,
        )
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # Output logic
    output_data = []
    for i, res in enumerate(results):
        # Convert numpy types to native python for JSON serialization
        if isinstance(res, (np.ndarray, np.generic)):
            val = res.tolist()
        else:
            val = res
            
        output_data.append({
            "id": ids[i],
            "prediction": val
        })
        
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results written to {args.output}")
    else:
        print(json.dumps(output_data, indent=2))

if __name__ == "__main__":
    main()
