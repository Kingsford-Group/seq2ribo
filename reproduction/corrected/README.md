# Corrected pipeline

Reproduces the corrected models and every table of [UPDATE.md](../../UPDATE.md): data, sTASEP/TASEP fits,
simulations, polishers, TE and expression heads, RiboNN and Translatomer. Transcripts and
train/validation/test splits are those of the paper ([`../splits`](../splits)). All settings are in
[`pipeline/config.py`](pipeline/config.py). Run everything from this directory; outputs go to `data/`,
`results/` and `logs/`.

## Setup

- `seq2ribo` environment and package (top-level [README](../../README.md)), plus
  `conda install -c conda-forge pandas scipy openpyxl "numpy<2"` and `conda install -c bioconda ucsc-bigwigtobedgraph`
  (numpy 1.x, so that RiboNN's environment can read the datasets; the released runs used ViennaRNA 2.7.2).
- RiboNN: a checkout of https://github.com/Sanofi-Public/RiboNN at commit `318e758` and its environment
  (`RIBONN_ROOT`, `RIBONN_ENV`, default `RiboNN`). With recent pandas versions, `src/data.py` needs
  `engine="python"` in the `query` that filters on `max_cds_utr3_len`.
- Translatomer: a checkout of https://github.com/xiongxslab/Translatomer at commit `ad1540c` and its
  environment (`TRANSLATOMER_ROOT`, `TRANSLATOMER_ENV`, default `translatomer`), plus `pyfaidx`.

## Inputs (`data/raw/`)

- GRCh38/hg38 FASTA with index and the GWIPS-viz elongating-ribosome bigWigs
  (`Chen20_All.RiboProElong.bw`, `Wagner20_All.RiboProElong.bw`, `Cenik2015.bw`,
  `Haas20_All.RiboProElong.bw`): `GENOME_DIR=... GWIPS_DIR=... bash pipeline/preprocess/fetch_inputs.sh`
  links them and downloads the GENCODE v48 annotation.
- TE labels: Zheng et al. 2025, supplementary workbook (`config.TE_WORKBOOK_URL`) as
  `data/raw/41587_2025_2712_MOESM3_ESM.xlsx`.
- mRFP expression: `data/raw/mRFP_Expression.csv` (Nieuwkoop et al. 2023, split of CodonBERT):
  `curl -L -o data/raw/mRFP_Expression.csv https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/master/benchmarks/CodonBERT/data/fine-tune/mRFP_Expression.csv` (sha256 `ed32c1e547ca4db464ef873768fcc6b3f4738704981a88bd9b842bd4e630ed99`).

## Steps

| Step | Script | Hardware |
|---|---|---|
| 1. CDS models, A-site counts, datasets (profiles, TE, expression) | `jobs/cpu_00_preprocess_all.sh` | CPU |
| 2. sTASEP and TASEP fits | `jobs/cpu_02_fit.sh` | CPU |
| 3. Simulations (polisher, TASEP test, TE, expression) | `jobs/cpu_03_simulate.sh` | CPU |
| 4. Polisher training and profile evaluation | `jobs/gpu_04_polisher.sh <cells>` | GPU |
| 5. TE and expression heads (all seeds) | `jobs/gpu_05_heads.sh <cells>` | GPU |
| 6. RiboNN | `jobs/gpu_06_ribonn.sh` | GPU |
| 7. Translatomer | `jobs/gpu_07_translatomer.sh <cells>` | GPU |
| 8. pertranscriptMAE and codonMAE as in the paper's evaluation code | `python pipeline/evaluate/published_mae.py --device cuda:0` | GPU |
| 9. Tables | `python pipeline/evaluate/results_tables.py` | CPU |

Each job script documents its options at the top (e.g. `DEVICE=cuda:0`, `WORKERS=64`). Steps 4-7 need the
outputs of the previous steps of the same cell line only, so cell lines can run on different GPUs.

## Outputs

- `results/fit/{stasep,tasep}/<cell>/params.json`: fitted wait times (`weights/<cell>/config.json` of the
  released models).
- `results/polisher/seq2ribo/<cell>/polisher.pt`, `results/eval/profiles/<cell>/*.json`: polisher and
  profile metrics (bootstrap 95% intervals included); `results/eval/profiles/published_mae.json`:
  pertranscriptMAE and codonMAE of step 8.
- `results/heads/{te_cds,te_utr,expression}/s<seed>/<cell>/`: heads, test predictions and metrics. The
  released single model of each head is the seed with the lowest validation MSE.
- `results/baselines/{ribonn,translatomer}/`: baseline models and metrics.
- `results/eval/results_tables.md`: the tables of UPDATE.md.

The release asset `seq2ribo-corrected-results.tar.gz` holds these outputs for the released runs
(`tar -xzf seq2ribo-corrected-results.tar.gz -C reproduction/corrected`); step 9 then rebuilds the
tables without retraining.
