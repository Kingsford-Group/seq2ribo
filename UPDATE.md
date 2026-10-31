# Update: corrected preprocessing and retrained models

After publication we found a bug in the data preprocessing: for transcripts on the minus strand, the A-site
profiles were stored in genomic order, i.e. reversed relative to the transcript. We fixed it and retrained
all models on the same transcripts and train/validation/test splits as the paper, together with a few
improvements to training. With the fix, seq2ribo predicts ribosome profiles considerably better, its
ensembles exceed the published numbers on every task, and it outperforms RiboNN on TE with and without UTRs on all cell lines.

The repository now ships the corrected models by default (`weights/`). The published models remain
available (`Seq2Ribo(..., version="paper")`, weights in the release assets), and both pipelines can be
reproduced (`reproduction/paper`, `reproduction/corrected`).

## The fix

A-site counts of minus-strand transcripts are now in transcript (5'→3') order. We revised the rest of the
preprocessing at the same time: bedGraph records spanning several nucleotides now contribute to every
nucleotide they cover (previously only the first), reads on the stop codon are kept, and each nucleotide's
turning angle is assigned to that nucleotide (previously the next one).

## Improvements

- Polisher: the Poisson loss uses the polisher output (Softplus) directly as the rate.
- TE labels are min-max scaled with the range of the training labels in all cell lines.
- TE heads: AdamW (weight decay 0.05), learning rate 5e-5 for the head and 5e-6 for the polisher, 10 (CDS+UTR: 5) of
  the 100 simulation runs per transcript per epoch, early stopping on validation MSE. For CDS+UTR, the polisher
  is also frozen for the first 3 epochs and the UTR branches use a 10x learning rate.
- Expression heads: learning rate and batch size chosen on the validation set.

**Baselines.** RiboNN and Translatomer were retrained on the corrected data, each following its manuscript:
RiboNN as 5-seed ensembles with early stopping on the validation split, with multitask training restricted
to transcripts outside the validation and test sets of every cell line; Translatomer as the sequence-only
model for both profiles and TE.

## Results

Results on test sets of the paper, in the layout of the paper's main-text tables. Ribosome profiles: the three Pearson
correlations are computed on unscaled predictions (higher is better) and the six MAEs after per-transcript
scaling (lower is better). TE and
expression: Pearson r per transcript, with the prediction averaged over the simulation runs of a transcript.
Single model: the seed with the lowest validation MSE; ensemble: mean prediction of the seeds. Cells read
paper → corrected; the paper reports no ensembles. Bold marks the best corrected value per cell line and
metric (TE: within each input setting).

**Ribosome profiles**

| Cell line | Metric | TASEP | sTASEP | Translatomer (seq.) | seq2ribo |
|---|---|---|---|---|---|
| iPSC | Tx-level r | 0.200 → 0.157 | 0.210 → 0.157 | 0.051 → 0.053 | 0.920 → **0.923** |
|  | Shape r | -0.015 → 0.055 | 0.039 → 0.058 | 0.001 → -0.002 | 0.162 → **0.349** |
|  | Elemwise r | 0.056 → 0.024 | 0.078 → 0.026 | -0.002 → 0.047 | 0.598 → **0.871** |
|  | pertranscriptMAE | 1.339 → 1.354 | 1.308 → 1.312 | 1.532 → 1.524 | 1.386 → **1.145** |
|  | elemwiseMAE | 4.895 → 4.498 | 4.561 → 4.479 | 6.638 → 7.102 | 4.135 → **2.904** |
|  | codonMAE | 40,894 → 2,795 | 3,725 → 2,820 | 5,916 → 29,513 | 12,028 → **2,770** |
|  | pairMAE | 40,181 → 73,827 | 15,120 → 25,759 | 94,648 → 251,241 | 44,153 → **8,646** |
|  | angleMAE | 16,182 → 45,373 | 12,608 → **9,189** | 94,648 → 251,241 | 85,032 → 19,398 |
|  | bucketMAE | 43,280 → 199,978 | 27,123 → 73,678 | 311,640 → 731,771 | 287,231 → **32,622** |
| HEK293 | Tx-level r | 0.006 → 0.008 | 0.013 → 0.007 | 0.023 → -0.019 | 0.657 → **0.715** |
|  | Shape r | 0.002 → 0.002 | 0.007 → 0.006 | 0.000 → 0.018 | 0.054 → **0.381** |
|  | Elemwise r | -0.004 → -0.001 | -0.001 → -0.002 | 0.000 → 0.001 | 0.396 → **0.642** |
|  | pertranscriptMAE | 1.811 → 1.809 | 1.807 → 1.804 | 1.001 → **1.111** | 1.909 → 1.537 |
|  | elemwiseMAE | 0.225 → 0.224 | 0.222 → 0.224 | 0.150 → **0.169** | 0.247 → 0.174 |
|  | codonMAE | 244 → 289 | 183 → 330 | 4,859 → 4,450 | 3,627 → **188** |
|  | pairMAE | 300 → **413** | 318 → 547 | 77,744 → 71,337 | 15,164 → 509 |
|  | angleMAE | 431 → **217** | 259 → 234 | 77,744 → 71,337 | 23,899 → 482 |
|  | bucketMAE | 7,403 → 13,814 | 6,685 → 12,445 | 103,658 → 95,116 | 66,934 → **2,977** |
| LCL | Tx-level r | 0.079 → 0.104 | 0.101 → 0.104 | 0.055 → 0.025 | 0.901 → **0.918** |
|  | Shape r | -0.012 → 0.084 | 0.046 → 0.089 | 0.000 → 0.006 | 0.186 → **0.477** |
|  | Elemwise r | -0.006 → 0.029 | 0.014 → 0.031 | 0.000 → 0.017 | 0.612 → **0.878** |
|  | pertranscriptMAE | 1.642 → 1.535 | 1.518 → 1.502 | 1.599 → 1.625 | 1.471 → **1.134** |
|  | elemwiseMAE | 32.927 → 28.037 | 28.956 → 27.978 | 32.164 → 34.337 | 22.418 → **10.785** |
|  | codonMAE | 800,628 → 45,520 | 35,579 → 45,119 | 33,724 → 224,914 | 38,902 → **20,031** |
|  | pairMAE | 298,006 → 268,260 | 248,121 → 246,934 | 539,576 → 465,455 | 187,622 → **11,474** |
|  | angleMAE | 119,977 → 189,134 | 155,088 → 73,859 | 562,607 → 451,200 | 84,351 → **22,724** |
|  | bucketMAE | 464,613 → 242,979 | 299,512 → 401,200 | 789,398 → 3,166,027 | 920,437 → **224,212** |
| RPE-1 | Tx-level r | 0.137 → 0.138 | 0.130 → 0.139 | 0.035 → 0.015 | 0.771 → **0.788** |
|  | Shape r | -0.005 → 0.079 | 0.041 → 0.092 | 0.003 → -0.004 | 0.139 → **0.434** |
|  | Elemwise r | 0.003 → 0.033 | 0.017 → 0.042 | 0.003 → 0.018 | 0.610 → **0.718** |
|  | pertranscriptMAE | 1.411 → 1.385 | 1.369 → 1.340 | 1.590 → 1.539 | 1.503 → **1.091** |
|  | elemwiseMAE | 3.750 → 3.337 | 3.506 → 3.321 | 5.100 → 5.374 | 3.362 → **1.785** |
|  | codonMAE | 23,034 → 3,664 | 2,383 → 3,360 | 4,740 → 23,612 | 12,404 → **1,897** |
|  | pairMAE | 27,098 → 19,804 | 6,858 → **7,618** | 75,832 → 218,624 | 24,184 → 8,284 |
|  | angleMAE | 17,323 → 24,745 | 4,286 → 7,294 | 76,336 → 218,624 | 15,336 → **4,122** |
|  | bucketMAE | 15,733 → 206,042 | 22,854 → 103,330 | 497,259 → 511,393 | 356,474 → **9,829** |

**Translation efficiency**, single models (Pearson r)

| | CDS only | | | | CDS + UTR | | |
|---|---|---|---|---|---|---|---|
| **Cell line** | **Translatomer (seq.)** | **RiboNN (ind.)** | **RiboNN (mul.)** | **seq2ribo** | **RiboNN (ind.)** | **RiboNN (mul.)** | **seq2ribo** |
| Mean | 0.372 → -0.095 | 0.523 → 0.651 | 0.529 → 0.652 | 0.688 → **0.682** | 0.698 → 0.776 | 0.682 → 0.736 | 0.708 → **0.784** |
| HEK293 | 0.209 → -0.100 | 0.419 → 0.601 | 0.442 → 0.583 | 0.570 → **0.636** | 0.602 → 0.719 | 0.587 → 0.704 | 0.652 → **0.739** |
| LCL | 0.301 → -0.162 | 0.604 → 0.714 | 0.590 → 0.707 | 0.697 → **0.731** | 0.711 → 0.773 | 0.707 → 0.762 | 0.732 → **0.783** |
| RPE-1 | 0.339 → -0.103 | 0.584 → 0.628 | 0.523 → 0.619 | 0.616 → **0.703** | 0.667 → 0.703 | 0.652 → 0.692 | 0.624 → **0.788** |

**Translation efficiency**, ensembles (Pearson r; RiboNN 5 seeds, seq2ribo 3 seeds)

| | CDS only | | | CDS + UTR | | |
|---|---|---|---|---|---|---|
| **Cell line** | **RiboNN (ind.)** | **RiboNN (mul.)** | **seq2ribo** | **RiboNN (ind.)** | **RiboNN (mul.)** | **seq2ribo** |
| Mean | 0.661 | 0.653 | **0.693** | 0.777 | 0.744 | **0.800** |
| HEK293 | 0.605 | 0.589 | **0.661** | 0.723 | 0.699 | **0.739** |
| LCL | 0.720 | 0.712 | **0.753** | 0.783 | 0.764 | **0.784** |
| RPE-1 | 0.634 | 0.611 | **0.711** | 0.709 | 0.692 | **0.792** |

**Protein expression** (mRFP, Pearson r; ensemble of 4 seeds)

| Cell line | Translatomer (seq.) | seq2ribo | seq2ribo, ensemble |
|---|---|---|---|
| iPSC | 0.052 → -0.007 | 0.903 → 0.888 | **0.906** |
| HEK293 | 0.000 → all zeros | 0.883 → 0.895 | **0.919** |
| LCL | 0.261 → 0.062 | 0.873 → 0.860 | **0.878** |
| RPE-1 | 0.027 → 0.114 | 0.830 → **0.900** | 0.889 |

## Files

- `weights/<cell>/`: corrected single models (polisher, TE CDS-only, TE CDS+UTR, expression), TE label
  transforms and `config.json` (fitted sTASEP parameters, angle bins, head settings).
- Release assets: `seq2ribo-paper-weights.tar.gz` (published weights; `version="paper"` downloads them into
  `weights/paper/`) and `seq2ribo-corrected-results.tar.gz` (every checkpoint behind the tables above,
  including the ensemble members, the fitted TASEP/sTASEP parameters and the baseline checkpoints; extract
  into `reproduction/corrected/`).
- `reproduction/paper/`: the published pipeline (minus-strand profiles as published).
- `reproduction/corrected/`: the corrected pipeline, including RiboNN and Translatomer.
