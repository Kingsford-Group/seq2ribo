# Paper reproduction

Scripts of the published pipeline: datasets, TASEP/sTASEP fits, polishers, profile evaluation, TE and
protein expression heads for the four cell lines (`ipsc`, `hek293`, `rpe`, `lcl`), and the baselines
as run for the paper (`baselines/`). The A-site profiles are built as in the paper (minus-strand
profiles in genomic order, see [UPDATE.md](../../UPDATE.md)). The corrected pipeline is in
[`../corrected`](../corrected).

Run every command from the repository root. Scripts write to `reproduction/paper/results/` unless
told otherwise. The published checkpoints (release asset) go to `weights/paper/`:
`python -c "from seq2ribo import download_paper_weights; download_paper_weights()"`.

## Requirements

On top of the `seq2ribo` environment (see the top-level README):

```bash
conda install -c conda-forge openpyxl
# preprocessing only
conda install -c bioconda bedtools gffread ucsc-bigwigtobedgraph ucsc-gtftogenepred ucsc-genepredtobed
```

## Layout

```
reproduction/paper/
├── preprocessing/   GWIPS-viz A-site bigWig + GENCODE -> split pickles and geometry maps
├── TASEP/           codon-only TASEP fit
├── sTASEP/          structure-aware TASEP fit
├── polisher/        polisher training data (sTASEP simulations) and polisher training
├── evaluation/      test-set metrics for sTASEP and polished profiles
├── TE/              TE simulations and CDS / CDS+UTR head finetuning
├── expression/      mRFP simulations and expression head finetuning
└── data/<cell>/     {train,val,test}_data[_01].pkl, polisher/geo_map_{train,val,test}.pkl
```

Each record in `{split}_data.pkl` is
`(tx_id, cds_rna, asite_counts_per_nt, zeros, (angle_dev_sum, pair_count))`.
The `_01` files are 10% subsets for quick runs; `--subset 001` / `0001` subsample them further.

## 1. Preprocessing

Inputs per cell line: the global aggregate elongating-ribosome (A-site) track for human hg38
from [GWIPS-viz](https://gwips.ucc.ie/) (bigWig), the
[GENCODE](https://www.gencodegenes.org/) release 48 comprehensive annotation GTF
(`gencode.v48.annotation.gtf`, uncompressed) and the GRCh38/hg38 genome FASTA.

| cell line | study | GWIPS-viz track |
|---|---|---|
| ipsc | Chen et al. 2020 | `Chen20_All.RiboProElong.bw` |
| hek293 | Wagner et al. 2020 | `Wagner20_All.RiboProElong.bw` |
| rpe | Haas et al. 2020 | `Haas20_All.RiboProElong.bw` |
| lcl | Cenik et al. 2015 | `Cenik2015.bw` |

```bash
WORKERS=32 bash reproduction/paper/preprocessing/run_preprocessing.sh \
  Wagner20_All.RiboProElong.bw gencode.v48.annotation.gtf hg38.fa hek293 reproduction/paper/data/hek293
```

`run_preprocessing.sh` runs:

1. CDS annotation: GTF `CDS` features -> `gtfToGenePred` -> `genePredToBed` (CDS BED12), and
   `gffread -x` for CDS sequences (start through stop codon).
2. `bigWigToBedGraph`, then `bedtools intersect -wa -wb -split` against the CDS BED12.
3. `asite_by_cds.py --published-mapping`: per-nucleotide A-site counts as in the paper (positions in
   genomic order, each bedGraph record at its first nucleotide).
4. `build_dataset.py`: the published train/val/test transcripts (`reproduction/splits/<cell>.tsv.gz`;
   the original split cannot be regenerated), ViennaRNA MFE folding and per-codon angle/pair features
   (`seq2ribo.geometry`, `version="paper"`).

## 2. TASEP and sTASEP fits

```bash
python reproduction/paper/TASEP/main_corrected_loss.py --cell-line ipsc
python reproduction/paper/sTASEP/main_corrected_loss_bias.py --cell-line ipsc
# quick check
python reproduction/paper/sTASEP/main_corrected_loss_bias.py --cell-line hek293 --subset 0001 --iters 2
```

100 outer iterations, all wait times initialized at 0.5, step size 0.01 (λ = 0.5 for the
structural bias terms). Simulated counts are rescaled to each transcript's observed total before
the updates. Initiation probability is 0.01 for every transcript.

Outputs: `results/{TASEP,sTASEP}/<cell>/*_fit_<subset>_<timestamp>.{json,npz}`. The JSON
`params` block has the same layout as `seq2ribo/constants.py`
(`rates`: 61 sense codons in alphabetical order without stops; `alpha`: pair counts;
`beta`: angle bins; `buckets`: CDS thirds). The simulator rescales `rates` to mean 1.

## 3. Polisher training data

```bash
python reproduction/paper/polisher/data_generator.py --cell-line ipsc --times 32
# with freshly fitted parameters instead of seq2ribo/constants.py
python reproduction/paper/polisher/data_generator.py --cell-line ipsc --times 32 \
  --params reproduction/paper/results/sTASEP/ipsc/stasep_fit_full_<timestamp>.json
```

Writes 32 independent sTASEP runs per transcript to
`reproduction/paper/data/<cell>/polisher/data_new/{train,val,test}/<tx>_sample_<k>.pkl`. Use `--overwrite` to
replace existing shards and `--seed` to change the simulation seed.

## 4. Polisher training

```bash
python reproduction/paper/polisher/main_mamba_geo_unscaled.py --cell-line ipsc
```

Poisson NLL with the Softplus output of the polisher as the log-rate (predicted counts are
`expm1` of that output), Adam (lr 1e-4, weight decay 2e-5), gradient clipping 1.0, LR halved after three
epochs without validation improvement, best checkpoint by validation NLL. HEK293 upweights
non-zero positions by 5 (`--non-zero-weight`). Output:
`results/polisher/<cell>/mamba_final_unscaled.pt` (the file shipped as
`weights/paper/<cell>_mamba_final_unscaled.pt`).

## 5. Profile evaluation

```bash
python reproduction/paper/evaluation/evaluate_profiles.py --cell-line ipsc --bootstrap 1000
python reproduction/paper/evaluation/evaluate_profiles.py --cell-line ipsc \
  --checkpoint reproduction/paper/results/polisher/ipsc/mamba_final_unscaled.pt
```

Test transcripts whose CDS also occurs in the training or validation split are removed. Every
simulation shard is polished separately and the predictions are averaged per transcript. Reports
Tx-level r, Shape r, Elemwise r and the six MAEs, unscaled and per-transcript scaled (`sc_`),
for both sTASEP and seq2ribo, plus bootstrap 95% CIs. Metric definitions are in
`evaluation/metrics.py`.

To evaluate the TASEP baseline, generate shards with the TASEP parameters into a separate output
root (`data_generator.py --params <tasep_fit>.json --output-root <dir>`) and pass
`<dir>/<cell>/polisher/data_new` (the folder that contains `test/`) via `--shards`; its `sTASEP` entry
then holds the TASEP metrics.

## 6. Translation efficiency

TE labels come from Zheng et al. 2025, Supplementary Table (`41587_2025_2712_MOESM3_ESM.xlsx`).
Download it into `reproduction/paper/TE/`:

```bash
curl -L -o reproduction/paper/TE/41587_2025_2712_MOESM3_ESM.xlsx \
  "https://static-content.springer.com/esm/art%3A10.1038%2Fs41587-025-02712-x/MediaObjects/41587_2025_2712_MOESM3_ESM.xlsx"
```

The `ipsc` run predicts mean TE; `hek293`, `rpe` and `lcl` predict the cell-line TE columns.

```bash
python reproduction/paper/TE/run_unscaled.py --cell-line ipsc --times 10
python reproduction/paper/TE/finetune_unscaled.py --cell-line ipsc        # CDS-only head
python reproduction/paper/TE/finetune_unscaled_utr.py --cell-line ipsc    # CDS + UTR head
```

`run_unscaled.py` matches labeled transcripts to the Ribo-seq splits, removes test/validation
CDSs shared with earlier splits, and writes `--times` sTASEP runs per transcript plus
`matched_mean_te_with_completed_total_train_val_test.csv` under `results/TE/<cell>/run_unscaled/`.

The finetuning scripts start from `results/polisher/<cell>/mamba_final_unscaled.pt` if present,
otherwise from `weights/paper/<cell>_mamba_final_unscaled.pt` (override with `--weights`). Batch size,
epochs (64 and 10 for all cell lines), the TE min-max range and, for the CDS head, whether the head
reads log1p counts (all but HEK293) are in `CELL_CONFIG` at the top of each script.
The UTR head reads 5'/3' UTRs from the same workbook (transcripts within RiboNN's length limits,
exact versioned ids; `--utr_unversioned` matches ids without version). `test_metrics.json`
reports Pearson r over all simulation rows and, under `per_transcript`, after averaging the
predictions of each transcript.

Outputs map to the shipped files as `finetune_cds/mamba_te_full_final.pt` ->
`weights/paper/<cell>_mamba_te_full_final_cds.pt`, `finetune_cds_utr/mamba_te_utr_final.pt` ->
`weights/paper/<cell>_mamba_te_utr_final.pt`, and `te_transform.json` ->
`weights/paper/<cell>_te_transform_{cds,utr}.json`.

## 7. Protein expression (mRFP)

```bash
python reproduction/paper/expression/run.py --cell-line ipsc --reps 1000
python reproduction/paper/expression/finetune_unscaled.py --cell-line ipsc
```

`run.py` folds the 1,459 mRFP CDSs of `expression/mRFP_Expression.csv` (Nieuwkoop et al. 2023, split of CodonBERT;
`curl -L -o reproduction/paper/expression/mRFP_Expression.csv https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/master/benchmarks/CodonBERT/data/fine-tune/mRFP_Expression.csv`;
predefined 1,021/219/219 split) and writes `--reps` sTASEP runs per sequence to
`results/expression/<cell>/run/`. The finetuning script trains the expression head and the
polisher end to end (3 epochs, batch 512, lr 1e-4, weight decay 2e-4) and writes
`results/expression/<cell>/finetune/mamba_expr_full_final.pt`
(shipped as `weights/paper/<cell>_mamba_expr_full_final.pt`).

## 8. Baselines (as run for the paper)

`baselines/ribonn/` and `baselines/translatomer/` are the scripts that produced the published RiboNN and
Translatomer numbers. `upstream_*.patch` are the local changes to the upstream repositories (RiboNN
commit 318e758, Translatomer commit ad1540c). Inputs default to the files of this directory (`data/<cell>/`,
the TE workbook in `TE/`, `expression/mRFP_Expression.csv`, `results/TE/<cell>/run_unscaled/`).

- RiboNN: run from a patched RiboNN checkout with `src/` copied into it. `<CELL>/prepare*.py --out_dir <CELL>`
  build the inputs (`src/prepare_cds_only.py` before `src/prepare_multitask.py`), `src/run_cv*.py` train
  (individual, multitask, CDS-only), `src/predict_cv_*.py` predict (to `results/<CELL>/prediction_output_cv.txt`
  for `src/run_evals.sh`), and `src/run_*evals.sh` and `<CELL>/corr_*.py --pred_txt <predictions>` score.
- Translatomer: run from a work directory holding the patched checkout (`Translatomer/`) and the split
  pickles of each cell line (`data/<cell>/`, copied from `reproduction/paper/data/<cell>/`), with
  `GENOME_FA` (GRCh38) and `GENCODE_GFF` (GENCODE v49 GFF3) set; `B=<repo>/reproduction/paper/baselines/translatomer`.
  1. `bash $B/run_training_seqonly.sh`: sequence-only models in `checkpoints_seqonly/<cell>/` (the first run
     caches a GFF index next to `GENCODE_GFF`, so its directory must be writable).
  2. `python $B/prepare_filtered_test_data.py --train_pkl data/<cell>/train_data.pkl --test_pkl data/<cell>/test_data.pkl --output_pkl data/<cell>/test_data_filtered.pkl`
  3. `PYTHONPATH=Translatomer python $B/translatomer_inference.py --checkpoint <pretrained Translatomer checkpoint> --data_root data`
     (sequence + RNA-seq model, with each transcript's A-site counts as the RNA-seq input; writes
     `data/<cell>/translatomer_predictions.pkl`), then
     `python $B/prepare_eval_dicts.py --pred_pkl data/<cell>/translatomer_predictions.pkl --test_data_pkl data/<cell>/test_data.pkl --output_prefix data/<cell>/eval_ready`.
  4. `bash $B/run_evaluation_seqonly.sh` scores the sequence-only profiles, `python $B/te_analysis_all.py`
     TE (step 3 predictions) and `python $B/expr_analysis_all.py` expression (the checkpoint with the lowest
     validation loss of each cell line).

The baselines of the corrected pipeline are in [`../corrected`](../corrected); [UPDATE.md](../../UPDATE.md)
describes how they differ.
