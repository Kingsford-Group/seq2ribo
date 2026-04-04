# seq2ribo

Structure-aware integration of machine learning and simulation to predict ribosome location profiles from RNA sequences.

[[Read the Paper on bioRxiv]](https://www.biorxiv.org/content/10.64898/2026.02.08.700508v2)

## Installation

### Prerequisites

- **Linux** (required for mamba-ssm)
- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit** 11.8+ (check with `nvcc --version`)
- **Conda** package manager

### Quick Install

```bash
# Clone the repository
git clone https://github.com/Kingsford-Group/seq2ribo.git
cd seq2ribo

# Create conda environment
conda env create -f environment.yml

# Activate
conda activate seq2ribo

# Install mamba-ssm (compiles from source, ~5-10 min)
python -m pip install --no-build-isolation mamba-ssm causal-conv1d

# Install seq2ribo package
pip install -e .
```

### Verify Installation

```bash
python -c "import RNA; import mamba_ssm; import torch; print('All imports OK!')"
```

> **Note**: If your CUDA version differs from 11.8, edit `pytorch-cuda=11.8` in `environment.yml` to match your system.  
> See [INSTALL.md](INSTALL.md) for detailed troubleshooting.

## Usage

### Python API

```python
from seq2ribo import Seq2Ribo

# Initialize predictor
predictor = Seq2Ribo(cell_line="hek293", weights_dir="weights")

# Predict ribosome density
sequence = "AUGGCCAAGCUGAAG..."
results = predictor.predict(sequence, task="riboseq")
```

### Command Line

```bash
# Show all CLI options
python scripts/run_inference.py --help

# Riboseq from one sequence
python scripts/run_inference.py --task riboseq --cell-line hek293 --seq "AUGGCC..."

# Riboseq from FASTA (multiple sequences)
python scripts/run_inference.py --task riboseq --cell-line ipsc --fasta input.fa --output riboseq_results.json

# TE (CDS-only), output (default)
python scripts/run_inference.py --task te --cell-line rpe --seq "AUGGCC..."

# TE (CDS-only), scaled output in [0,1]
python scripts/run_inference.py --task te --cell-line rpe --seq "AUGGCC..." --return-scaled-te

# TE (CDS+UTR)
python scripts/run_inference.py --task te --use_utr --cell-line hek293 \
  --utr5 "AUGGCUA..." --cds "AUGGCC..." --utr3 "UAAUCG..."

# Protein expression
python scripts/run_inference.py --task protein --cell-line lcl --seq "AUGGCC..."

# sTASEP-only (no polisher), with custom simulation controls
python scripts/run_inference.py --task riboseq --cell-line hek293 --seq "AUGGCC..." \
  --no-polisher --n-stasep-runs 50 --init-p 0.02
```

#### CLI Parameters

`scripts/run_inference.py` supports:

- `--seq`: Single RNA sequence string (mutually exclusive with `--fasta`).
- `--fasta`: FASTA file with one or more RNA sequences.
- `--task`: One of `riboseq`, `te`, `protein` (default: `riboseq`).
- `--use_utr`: Enable UTR-aware TE mode (valid only with `--task te`).
- `--cell-line`: One of `hek293`, `lcl`, `rpe`, `ipsc` (default: `ipsc`).
- `--weights-dir`: Model checkpoint directory (default: `weights` in repo).
- `--cache-dir`: Geometry cache directory (default: `cache/geometry`).
- `--n-stasep-runs`: Number of sTASEP runs per sequence (default: `100`).
- `--init-p`: sTASEP initiation probability (default: `0.01`).
- `--return-scaled-te`: For `te`, return scaled TE in `[0,1]` instead of inverse-transformed TE.
- `--utr5`: 5' UTR sequence (required with `--task te --use_utr`).
- `--cds`: CDS sequence (required with `--task te --use_utr`).
- `--utr3`: 3' UTR sequence (required with `--task te --use_utr`).
- `--no-polisher`: Return simulation-only output (skip neural model).
- `--output`: Output path for JSON results (stdout if omitted).

### Supported Tasks

| Task | Description | Output |
|------|-------------|--------|
| `riboseq` | Ribosome profiling | Per-codon counts |
| `te` | Translation efficiency (CDS-only or CDS+UTR) | Scalar (inverse by default; scaled with `--return-scaled-te`) |
| `protein` | Protein expression | Scalar (fixed 32-pass MC mean) |

### Supported Cell Lines

- `hek293` - HEK293
- `lcl` - Lymphoblastoid Cell Line
- `rpe` - RPE-1
- `ipsc` - iPSC

### Riboseq Checkpoints (Current)

Riboseq inference expects unscaled checkpoints in the `weights/` directory:

- `hek293_mamba_final_unscaled.pt`
- `lcl_mamba_final_unscaled.pt`
- `rpe_mamba_final_unscaled.pt`
- `ipsc_mamba_final_unscaled.pt`

### TE Checkpoints (CDS-only)

TE inference expects CDS-only checkpoints and per-cell transform files in `weights/`:

- `hek293_mamba_te_full_final_cds.pt` + `hek293_te_transform_cds.json`
- `lcl_mamba_te_full_final_cds.pt` + `lcl_te_transform_cds.json`
- `rpe_mamba_te_full_final_cds.pt` + `rpe_te_transform_cds.json`
- `ipsc_mamba_te_full_final_cds.pt` + `ipsc_te_transform_cds.json`

For `task="te"`, output is inverse-transformed TE by default. Use `--return-scaled-te` in CLI to return scaled TE in `[0,1]`.

### TE Checkpoints (CDS+UTR)

TE UTR-aware inference (enabled with `--use_utr`) expects:

- `hek293_mamba_te_utr_final.pt` + `hek293_te_transform_utr.json`
- `lcl_mamba_te_utr_final.pt` + `lcl_te_transform_utr.json`
- `rpe_mamba_te_utr_final.pt` + `rpe_te_transform_utr.json`
- `ipsc_mamba_te_utr_final.pt` + `ipsc_te_transform_utr.json`

For TE+UTR mode, users must provide split sequence parts (`utr5`, `cds`, `utr3`) instead of a merged transcript.

```bash
python scripts/run_inference.py --task te --use_utr --cell-line hek293 \
  --utr5 "AUGGCUA..." --cds "AUGGCC..." --utr3 "UAAUCG..."
```

### Protein Expression Checkpoints

Protein expression inference expects per-cell checkpoints in `weights/`:

- `hek293_mamba_expr_full_final.pt`
- `lcl_mamba_expr_full_final.pt`
- `rpe_mamba_expr_full_final.pt`
- `ipsc_mamba_expr_full_final.pt`

Protein expression inference uses fixed 32-pass MC forward averaging (mean only) to mirror finetune-time test behavior.

## Project Structure

```
seq2ribo/
├── seq2ribo/          # Core package
│   ├── inference.py   # Main API
│   ├── models.py      # Neural network models
│   ├── simulation.py  # sTASEP simulation
│   └── geometry.py    # RNA structure features
├── scripts/           # CLI scripts
├── weights/           # Model checkpoints
├── tests/             # Test suite
└── environment.yml    # Conda environment
```

## License

This software is licensed for **Academic or Non-Profit Organization Noncommercial Research Use Only**.

See the [LICENSE](LICENSE) file for the full terms.

**For commercial use or any use not permitted by the academic license, please contact the options below to discuss licensing:**
- Carl Kingsford (carlk@cs.cmu.edu)

## Citation

If you use seq2ribo in your research, please cite:


```bibtex
@article{kaynar2026seq2ribo,
	title = {seq2ribo: Structure-aware integration of machine learning and
	         simulation to predict ribosome locations profiles from {RNA}
	         sequences},
	author = {G{\"u}n Kaynar and Carl Kingsford},
	year = {2026},
	journal = {bioRxiv},
	url = {https://www.biorxiv.org/content/10.64898/2026.02.08.700508v2},
}
```

