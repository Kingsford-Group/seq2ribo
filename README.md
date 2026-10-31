# seq2ribo

Structure-aware integration of machine learning and simulation to predict ribosome location profiles from RNA sequences.

[[Read the Paper in Bioinformatics]](https://academic.oup.com/bioinformatics/article/42/Supplement_1/btag296/8726312)

> **Update.** We found and corrected an error in the data preprocessing of the published version (minus-strand
> A-site profiles were reversed) and retrained all models and baselines. The corrected models are the default
> and perform better than reported in the paper; see [UPDATE.md](UPDATE.md). The published models remain
> available with `version="paper"`.

## Installation

### Prerequisites

- **Linux** (required for mamba-ssm)
- **NVIDIA GPU** with CUDA support (prediction also runs on CPU with `device="cpu"`, much slower; installing
  mamba-ssm still needs the CUDA toolkit)
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

The conda environment includes **Jupyter** (`notebook`, `ipykernel`) for the tutorials in `notebooks/`. If your env is older, run `conda env update -f environment.yml --prune` after `git pull`.

### Verify Installation

```bash
python -c "import RNA; import mamba_ssm; import torch; print('All imports OK!')"
```

> **Note**: If your CUDA version differs from 11.8, edit `pytorch-cuda=11.8` in `environment.yml` to match your system.  
> See [INSTALL.md](INSTALL.md) for detailed troubleshooting.

## Notebooks

Interactive notebooks are provided in the `notebooks/` directory for exploring seq2ribo without writing code from scratch.

### Local Jupyter Notebook

From the repo root with `seq2ribo` activated:

```bash
cd notebooks
jupyter notebook demo.ipynb
```

`demo.ipynb` walks through every prediction task. Ribo-seq demos use real test-set transcripts. In the **cross-cell-line panel**, protein is run on the same demo CDS for direct side-by-side comparison; dedicated protein demos use held-out **mRFP reporter** test sequences in `notebooks/sample_data/mrfp_expr_test_samples.csv` (the training domain of the expression heads).

1. **A-site profile prediction** — sTASEP-only vs sTASEP + Polisher, with ground-truth comparison and Pearson correlation
2. **Cross-cell-line comparison** — same ribo-seq transcript for profiles, TE, and protein bars across all four lines
3. **Translation efficiency (CDS-only and CDS+UTR)**
4. **Protein expression prediction** — mRFP test-set variants (not ribo-seq)
5. **Synonymous codon mutation analysis** — synonymous codon swaps and profile / TE comparison
6. **sTASEP parameter sensitivity** — effects of `init_p` and `n_stasep_runs` on simulation output
7. **Custom sequence input** — paste your own RNA CDS and get a full prediction panel
8. **Batch prediction from FASTA** — example FASTA holds mRFP test CDS (expression column on-distribution)

**`rna_fold_features_stasep_anim.ipynb`** (methods / supplementary) walks through ViennaRNA MFE layout (`PLOT_TYPE_SIMPLE`), per-codon geometry channels (angle / pair / bucket), and illustrative **sTASEP** animations (linear tracks, wait-residual arrows, and 2D MFE layout) using a demo CDS from `demo_transcripts.pkl`. For full **ribo-seq prediction** panels (ground truth vs sTASEP vs polisher), use **`demo.ipynb`**. Paths and transcript ID are set in-notebook; run Jupyter from the repo root or from `notebooks/`.

### Google Colab Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kingsford-Group/seq2ribo/blob/main/notebooks/colab_seq2ribo.ipynb)

`colab_seq2ribo.ipynb` includes the same demos as above plus automated environment setup on a Colab GPU runtime (installs ViennaRNA, mamba-ssm, and seq2ribo).

## Usage

### Python API

```python
from seq2ribo import Seq2Ribo

# Initialize predictor
predictor = Seq2Ribo(cell_line="hek293")                    # corrected models (weights/hek293/)
# predictor = Seq2Ribo(cell_line="hek293", version="paper")  # published models (weights/paper/)

# Predict ribosome density (one array of per-codon counts per sequence)
sequence = "AUGGCCAAGCUGAAG..."
profile = predictor.predict(sequence, task="riboseq", seed=0)[0]

# Translation efficiency (CDS only) and protein expression (one value per sequence)
te = predictor.predict(sequence, task="te")[0]
expr = predictor.predict(sequence, task="protein")[0]

# Translation efficiency with UTRs: pass the three parts separately
te_utr = predictor.predict(task="te", use_utr=True, utr5_list=["GCAGAGCC..."], cds_list=[sequence],
                           utr3_list=["GGUUGCCG..."])[0]
```

Each prediction averages `n_stasep_runs` polished sTASEP runs (default 100, as used to evaluate the models);
`seed` makes it reproducible.

### Command Line

`pip install -e .` also installs the same interface as the `seq2ribo` command.

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
- `--version`: `corrected` (default) or `paper` (published checkpoints, `weights/paper/`).
- `--cache-dir`: Geometry cache directory (default: `cache/geometry`).
- `--n-stasep-runs`: Number of sTASEP runs per sequence (default: `100`). Each run is polished separately and the predictions are averaged; with `--no-polisher` the simulated counts are averaged.
- `--init-p`: sTASEP initiation probability (default: the value used for training, `0.01`).
- `--seed`: Seed for the sTASEP simulations; each sequence is seeded from the seed and its own sequence, so results do not depend on batch composition.
- `--batch-size`: Model batch size in rows (sequence × run, default: `64`).
- `--return-scaled-te`: For `te`, return scaled TE in `[0,1]` instead of inverse-transformed TE.
- `--utr5`: 5' UTR sequence (required with `--task te --use_utr`).
- `--cds`: CDS sequence (required with `--task te --use_utr`).
- `--utr3`: 3' UTR sequence (required with `--task te --use_utr`).
- `--no-polisher`: With `--task riboseq`, return the simulated sTASEP profile (skip the polisher).
- `--output`: Output path; `.csv` writes CSV (profiles as space-separated values), any other name JSON
  (stdout if omitted).

### Supported Tasks

| Task | Description | Output |
|------|-------------|--------|
| `riboseq` | Ribosome profiling | Per-codon counts |
| `te` | Translation efficiency (CDS-only or CDS+UTR) | Scalar (inverse by default; scaled with `--return-scaled-te`) |
| `protein` | Protein expression | Scalar |

### Supported Cell Lines

- `hek293` - HEK293
- `lcl` - Lymphoblastoid Cell Line
- `rpe` - RPE-1
- `ipsc` - iPSC

### Checkpoints

`weights/<cell_line>/` holds the corrected models of each cell line:

- `polisher.pt` (riboseq), `te_cds.pt` + `te_cds_transform.json` (TE, CDS only),
  `te_utr.pt` + `te_utr_transform.json` (TE, CDS+UTR), `expression.pt` (protein expression)
- `config.json`: fitted sTASEP parameters, angle-bin edges and head settings

The published checkpoints are a release asset: `tar -xzf seq2ribo-paper-weights.tar.gz -C weights` creates
`weights/paper/`, used with `version="paper"`.
For `task="te"`, output is inverse-transformed TE by default (`--return-scaled-te` returns the scaled value in
`[0,1]`). For TE+UTR, pass the parts separately (`utr5_list`, `cds_list`, `utr3_list` in Python; `--utr5`, `--cds`,
`--utr3` on the command line). Predictions are averaged over the
polished sTASEP runs (`n_stasep_runs`).

The notebooks ship a small held-out test slice of the mRFP expression benchmark as
`notebooks/sample_data/mrfp_expr_test_samples.csv`.

## Project Structure

```
seq2ribo/
├── seq2ribo/          # Core package
│   ├── inference.py   # Main API (Seq2Ribo)
│   ├── cli.py         # Command line
│   ├── models.py      # Neural network models
│   ├── simulation.py  # sTASEP simulation
│   ├── geometry.py    # RNA structure features
│   ├── metrics.py     # Profile metrics
│   ├── data.py        # Batching
│   ├── utils.py       # Codon tables, checkpoint loading
│   └── constants.py   # sTASEP parameters of the published models
├── scripts/           # CLI script (run_inference.py)
├── reproduction/      # paper/ (published pipeline), corrected/ (corrected pipeline), splits/
├── notebooks/         # Jupyter & Colab demo notebooks
├── weights/           # Corrected model checkpoints (weights/<cell_line>/)
├── tests/             # Test suite
├── UPDATE.md          # Corrected pipeline and results
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
  title={{seq2ribo}: structure-aware integration of machine learning and simulation to predict ribosome location profiles from {RNA} sequences},
  author={Kaynar, G{\"u}n and Kingsford, Carl},
  journal={Bioinformatics},
  volume={42},
  number={Supplement\_1},
  pages={btag296},
  year={2026},
  publisher={Oxford University Press},
  doi={10.1093/bioinformatics/btag296}
}
```

