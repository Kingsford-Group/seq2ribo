# Installation Guide

## Prerequisites

- **Linux** (required for mamba-ssm)
- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit** 11.8+ installed (check with `nvcc --version`)
- **Conda** or **Mamba** package manager

## Quick Install

### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/seq2ribo.git
cd seq2ribo

# Create conda environment
conda env create -f environment.yml

# Activate
conda activate seq2ribo

# Install mamba-ssm (requires compilation, may take 5-10 minutes)
python -m pip install --no-build-isolation mamba-ssm causal-conv1d

# Install seq2ribo package
pip install -e .
```

### Option 2: Manual Installation

If you have an existing environment with PyTorch + CUDA:

```bash
# Install ViennaRNA (requires bioconda channel)
conda install -c bioconda viennarna

# Install mamba-ssm (use --no-build-isolation to use existing PyTorch)
python -m pip install --no-build-isolation mamba-ssm causal-conv1d

# Install other dependencies
pip install einops transformers numpy pandas scipy matplotlib

# Install seq2ribo
pip install -e /path/to/seq2ribo
```

## CUDA Version Matching

> **Important**: Your PyTorch CUDA version must match your system CUDA version.

Check your system CUDA:
```bash
nvcc --version
```

Check PyTorch CUDA:
```bash
python -c "import torch; print(torch.version.cuda)"
```

If they don't match, the mamba-ssm build will fail. Adjust `pytorch-cuda` in `environment.yml` accordingly:
- CUDA 11.8: `pytorch-cuda=11.8`
- CUDA 12.1: `pytorch-cuda=12.1`
- CUDA 12.4: `pytorch-cuda=12.4`

## Verify Installation

```bash
python -c "import RNA; import mamba_ssm; import torch; print('All imports OK!')"
```

## Troubleshooting

### "CUDA mismatch" error during mamba-ssm install
- Ensure `pytorch-cuda` version in `environment.yml` matches your `nvcc --version`
- Use `--no-build-isolation` flag with pip

### ViennaRNA import fails
- Make sure bioconda channel is added: `conda config --add channels bioconda`

### pip uses wrong environment
- Always use `python -m pip install` instead of `pip install` to ensure correct environment
