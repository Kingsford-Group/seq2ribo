# seq2ribo

Structure-aware integration of machine learning and simulation to predict ribosome location profiles from RNA sequences.

[[Read the Paper on bioRxiv]](https://www.biorxiv.org/content/10.64898/2026.02.08.700508v1)

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
# Predict from sequence
python scripts/run_inference.py --seq "AUGGCC..." --cell-line hek293 --task riboseq

# Predict from FASTA
python scripts/run_inference.py --fasta input.fa --cell-line ipsc --output results.json
```

### Supported Tasks

| Task | Description | Output |
|------|-------------|--------|
| `riboseq` | Ribosome profiling | Per-codon counts |
| `te` | Translation efficiency | Scalar |
| `protein` | Protein expression | Scalar |

### Supported Cell Lines

- `hek293` - HEK293
- `lcl` - Lymphoblastoid Cell Line
- `rpe` - RPE-1
- `ipsc` - iPSC

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
@article{Kaynar2026Feb,
	author = {Kaynar, Gun and Kingsford, Carl},
	title = {{seq2ribo: Structure-aware integration of machine learning and simulation to predict ribosome location profiles from RNA sequences}},
	journal = {bioRxiv},
	pages = {2026.02.08.700508},
	year = {2026},
	month = feb,
	issn = {2692-8205},
	publisher = {Cold Spring Harbor Laboratory},
	eprint = {2026.02.08.700508},
	url = {https://doi.org/10.64898/2026.02.08.700508}
}
```

