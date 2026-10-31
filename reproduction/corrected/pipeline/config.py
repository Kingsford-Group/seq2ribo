"""Paths and parameters of the corrected seq2ribo pipeline (one place for every setting)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
RESULTS = ROOT / "results"
SHARED = DATA / "inputs"   # CDS annotation (cds.pkl), corrected A-site counts (<cell>/asite_counts.pkl), fold cache

CELL_LINES = ["ipsc", "hek293", "lcl", "rpe"]

# GWIPS-viz global aggregate elongating-ribosome (A-site) tracks, hg38
GWIPS_TRACKS = {
    "ipsc": "Chen20_All.RiboProElong.bw",     # Chen et al. 2020
    "hek293": "Wagner20_All.RiboProElong.bw",  # Wagner et al. 2020
    "lcl": "Cenik2015.bw",                     # Cenik et al. 2015
    "rpe": "Haas20_All.RiboProElong.bw",       # Haas et al. 2020
}
GENCODE_RELEASE = 48
GENCODE_GTF_URL = (f"https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{GENCODE_RELEASE}/"
                   f"gencode.v{GENCODE_RELEASE}.annotation.gtf.gz")
GENOME_FASTA = RAW / "hg38.fa"  # GRCh38/hg38, with .fai index

# CDS filters
START_CODONS = ("AUG", "CUG", "GUG")
STOP_CODONS = ("UAA", "UAG", "UGA")
MAX_CDS_NT = 10_000

# published train/val/test transcript sets (profiles) and TE sets of the paper; cds_trim_5p is the number
# of leading nucleotides of the annotated CDS removed in the published sequence (incomplete 5' CDS ends)
SPLITS_DIR = ROOT.parent / "splits"
PROFILE_SPLITS = {c: SPLITS_DIR / f"{c}.tsv.gz" for c in CELL_LINES}
TE_SPLITS = {c: SPLITS_DIR / f"te_{c}.tsv.gz" for c in CELL_LINES}

# structure features
ANGLE_BINS_K = 4
ANGLE_BINS_QUANTILES = (0.0, 0.995)  # linear bins between these training quantiles

# sTASEP / TASEP
INIT_P = 0.01                # initiation probability per step, shared by all transcripts
EXCLUSION = 3                # codons
BURN_IN_TERMINATIONS = 10
SNAPSHOT_P = 0.1
SNAPSHOTS_PER_RUN = 200
FIT_ITERS = 100
FIT_INIT = 0.5               # initial value of every wait time
FIT_LR = 0.01                # eta for tau, alpha, beta, gamma
FIT_BIAS_LAMBDA = 0.5
WAIT_EPS = 1e-12

# polisher
POLISHER = dict(d_model=192, n_layers=4, d_state=16, d_conv=4, expand=2, dropout=0.1,
                lr=1e-4, weight_decay=2e-5, grad_clip=1.0, batch_size=192, epochs=100,
                plateau_factor=0.5, plateau_patience=3, early_stop_patience=6, sims_per_tx=32,
                nonzero_weight={"hek293": 5.0})

# downstream heads
TE_COLUMNS = {"ipsc": "mean_te", "hek293": "TE_HEK293", "lcl": "TE_LCL", "rpe": "TE_RPE-1"}
TE = dict(sims_per_tx=100, lr=1e-4, weight_decay=1e-4, grad_clip=1.0, batch_size=64, epochs=10)
TE_WORKBOOK_URL = ("https://static-content.springer.com/esm/art%3A10.1038%2Fs41587-025-02712-x/"
                   "MediaObjects/41587_2025_2712_MOESM3_ESM.xlsx")
EXPRESSION = dict(sims_per_seq=1000, lr=1e-4, weight_decay=2e-4, grad_clip=1.0, batch_size=512, epochs=3)

# released head recipes (see UPDATE.md); the finetune.py options of each head
TE_CDS_RECIPE = dict(optimizer="adamw", lr=5e-5, weight_decay=0.05, batch_size=64, backbone_lr_mult=0.1,
                     train_runs_per_epoch=10, val_runs=20, epochs=40, patience=6, seeds=(42, 1, 2))
TE_UTR_RECIPE = dict(optimizer="adamw", lr=5e-5, weight_decay=0.05, batch_size=64, backbone_lr_mult=0.1,
                     utr_lr_mult=10, freeze_backbone_epochs=3, train_runs_per_epoch=5, val_runs=20, epochs=40,
                     patience=6, seeds=(42, 1, 2))
# expression: Adam, weight decay 2e-4, 3 epochs, all runs, head on the counts (no log1p); learning rate and
# batch size per cell line chosen on the validation set
EXPRESSION_RECIPE = {"ipsc": dict(lr=1.5e-4, batch_size=1024), "hek293": dict(lr=2e-4, batch_size=512),
                     "lcl": dict(lr=1e-4, batch_size=512), "rpe": dict(lr=1.5e-4, batch_size=1024)}
EXPRESSION_SEEDS = (42, 1, 2, 3)
