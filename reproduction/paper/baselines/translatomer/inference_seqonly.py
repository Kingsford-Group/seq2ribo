import os, re, pickle, warnings, sys, argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add module path
sys.path.append('Translatomer') 

# Import from training script to get Model Config
# This requires train_seqonly.py to be in the same directory
from train_seqonly import PatchedTrainModuleSeqOnly, FastaStringExtractor, onehot_dna_fast, SEQLEN, NBINS, BIN

LOG_FILE = "script_progress.log"
def log_to_file(msg):
    with open(LOG_FILE, "a") as f:
        f.write(str(msg) + "\n")

# kipoiseq might be needed for Interval, or we mock it
try:
    from kipoiseq import Interval
except ImportError:
    class Interval:
        def __init__(self, chrom, start, end):
            self.chrom = chrom
            self.start = start
            self.end = end
        def __repr__(self): return f"{self.chrom}:{self.start}-{self.end}"

# ------------------ HELPERS ------------------
def _attrs(s):
    d = {}
    for kv in s.split(';'):
        if '=' in kv:
            k,v = kv.split('=',1); d[k]=v
    return d
def _enst_nov(enst): 
    return enst.split('.')[0]

STRICT_REQUIRE_CDS = True

class MockDataSources:
    def __init__(self, fasta_extractor):
        self.fasta = fasta_extractor
        
    def close(self):
        pass

# ------------------ GFF Indexing ------------------
def _key_variants(id_like: str):
    base = id_like.split(':', 1)[-1] if ':' in id_like else id_like
    en0  = _enst_nov(base)
    return {
        id_like, base, en0,
        f"transcript:{base}", f"transcript:{en0}"
    }

def build_gff_index(gff_df):
    idx = {"tx": {}, "cds": {}, "exon": {}}
    # transcripts
    tx_rows = gff_df[gff_df[2].isin(['transcript','mRNA'])]
    for _, r in tx_rows.iterrows():
        a = _attrs(r[8]); tid = a.get('ID','')
        chrom, strand = r[0], r[6]
        tstart0, tend0 = int(r[3]) - 1, int(r[4])
        for k in _key_variants(tid):
            idx["tx"][k] = (chrom, strand, tstart0, tend0)

    # CDS
    cds_rows = gff_df[gff_df[2]=='CDS']
    cds_map = {}
    for _, r in cds_rows.iterrows():
        a = _attrs(r[8]); parent = a.get('Parent','')
        for k in _key_variants(parent):
            cds_map.setdefault(k, []).append((int(r[3]) - 1, int(r[4])))
    for k, ex in cds_map.items():
        ex.sort(key=lambda x: x[0])
        merged=[list(ex[0])]
        for s,e in ex[1:]:
            if s>merged[-1][1]: merged.append([s,e])
            else: merged[-1][1]=max(merged[-1][1], e)
        idx["cds"][k] = [(s,e) for s,e in merged]

    return idx

class TxNotFound(Exception): pass
class NoCDSFound(Exception): pass

def lookup_tx_and_intervals(gff_index, enst, strict_require_cds=True):
    keys = list(_key_variants(enst))
    tx_info = None
    for k in keys:
        if k in gff_index["tx"]:
            tx_info = gff_index["tx"][k]; break
    if tx_info is None:
        raise TxNotFound(enst)

    for k in keys:
        if k in gff_index["cds"]:
            chrom, strand, tx_s, tx_e = tx_info
            return chrom, strand, tx_s, tx_e, gff_index["cds"][k], 'CDS'
    raise NoCDSFound(enst)

# ------------------ Inference ------------------
class MockArgs:
    def __init__(self):
        self.model_type = 'TransModel'
        self.hparams = {}
        self.learning_rate = 1e-5

@torch.inference_mode()
def predict_windows_batched(model, ds: MockDataSources, chrom, starts, device, batch_size=4):
    preds = []
    
    for i in range(0, len(starts), batch_size):
        batch = starts[i:i+batch_size]
        # Intervals
        ivals = [Interval(chrom, s, s+SEQLEN) for s in batch]

        # Extract DNA
        xs_dna = [onehot_dna_fast(ds.fasta.extract(iv)) for iv in ivals] # [L,5]
        
        # Stack -> [B, L, 5]
        x = torch.stack(xs_dna).float().to(device) 

        y = model(x) # [B, NBINS]
        preds.append(y.detach().cpu().numpy())

    return np.concatenate(preds, axis=0) if preds else np.zeros((0, NBINS), dtype=np.float32)

def predict_transcript_tiled_batched(enst, gff_index, ds, model, device, overlap=16384, batch_size=4):
    chrom, strand, tx_s, tx_e, intervals, kind = lookup_tx_and_intervals(
        gff_index, enst, strict_require_cds=STRICT_REQUIRE_CDS
    )
    cds_s = min(s for s,_ in intervals)
    cds_e = max(e for _,e in intervals)
    region_len = cds_e - cds_s

    # Tiling logic
    if (tx_e - tx_s) <= SEQLEN:
        mid = (tx_s + tx_e)//2
        starts = [mid - SEQLEN//2]
    else:
        stride = SEQLEN - overlap
        starts = list(range(cds_s, cds_e, stride))
        if starts[-1] + SEQLEN < cds_e:
            starts.append(cds_e - SEQLEN)

    win_preds = predict_windows_batched(
        model, ds, chrom, starts, device, batch_size=batch_size
    )

    # Aggregation
    region_bins = (region_len + BIN - 1) // BIN
    agg   = np.zeros(region_bins, dtype=np.float32)
    count = np.zeros(region_bins, dtype=np.int32)

    for wstart, pred1024 in zip(starts, win_preds):
        wend = wstart + SEQLEN
        ov_s = max(wstart, cds_s)
        ov_e = min(wend,   cds_e)
        if ov_s >= ov_e: continue
        
        win_lo_bin = (ov_s - wstart) // BIN
        win_hi_bin = (ov_e - wstart + BIN - 1) // BIN
        reg_lo_bin = (ov_s - cds_s) // BIN
        reg_hi_bin = reg_lo_bin + (win_hi_bin - win_lo_bin)

        win_lo_bin = max(0, win_lo_bin); win_hi_bin = min(NBINS, win_hi_bin)
        reg_lo_bin = max(0, reg_lo_bin); reg_hi_bin = min(region_bins, reg_hi_bin)

        w = pred1024[win_lo_bin:win_hi_bin]
        if w.size == 0: continue
        agg[reg_lo_bin:reg_hi_bin]   += w[:reg_hi_bin-reg_lo_bin]
        count[reg_lo_bin:reg_hi_bin] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        region_bins_pred = np.where(count>0, agg/np.maximum(count,1), 0.0).astype(np.float32)

    # Map to BP then Codon
    region_bp = np.repeat(region_bins_pred, BIN)[:region_len]
    pieces=[]
    for s,e in intervals:
        rs = s - cds_s; re = e - cds_s
        pieces.append(region_bp[rs:re])
    pred_tx_bp = np.concatenate(pieces) if pieces else np.array([], dtype=np.float32)

    if strand == '-':
        pred_tx_bp = pred_tx_bp[::-1]

    trim = (pred_tx_bp.size // 3) * 3
    pred_tx_codon = pred_tx_bp[:trim].reshape(-1,3).mean(1) if trim>0 else np.array([], dtype=np.float32)

    return pred_tx_codon

# ------------------ Main ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--fasta', required=True, help='GRCh38 genome FASTA')
    parser.add_argument('--gff', required=True, help='GENCODE v49 annotation GFF3')
    parser.add_argument('--test_data_pkl', required=True, help='Pickle containing list of transcript IDs to predict')
    parser.add_argument('--output_pkl', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    global LOG_FILE
    LOG_FILE = args.output_pkl + ".log" 

    log_to_file("Starting main (SeqOnly)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_to_file(f"Device: {device}")
    
    # Load Model
    log_to_file(f"Loading model from {args.checkpoint}...")
    try:
        model = PatchedTrainModuleSeqOnly.load_from_checkpoint(args.checkpoint, args=MockArgs(), map_location=device)
    except TypeError:
         model = PatchedTrainModuleSeqOnly.load_from_checkpoint(args.checkpoint, map_location=device)
    model.to(device).eval()
    
    # Load Data Sources
    log_to_file("Loading Genome and GFF...")
    fasta_extractor = FastaStringExtractor(args.fasta)
    ds = MockDataSources(fasta_extractor) # Only FASTA
    
    # GFF Index
    gff_cache_path = args.gff + ".index.pkl"
    if os.path.exists(gff_cache_path):
        log_to_file(f"Loading GFF index from cache: {gff_cache_path}...")
        with open(gff_cache_path, 'rb') as f:
            gff_index = pickle.load(f)
    else:
        log_to_file("Parsing GFF (this may take a while)...")
        gff_df = pd.read_csv(args.gff, sep='\t', comment='#', header=None)
        gff_index = build_gff_index(gff_df)
        with open(gff_cache_path, 'wb') as f:
            pickle.dump(gff_index, f)
    
    # Load keys
    with open(args.test_data_pkl, 'rb') as f:
        data = pickle.load(f)
    
    tx_ids = []
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) > 0:
            tx_ids.append(item[0])
            
    log_to_file(f"Found {len(tx_ids)} transcripts to predict.")
    
    pred_counts_dict = {}
    
    log_to_file("Starting loop...")
    for tx in tqdm(tx_ids):
        try:
            pred_codons = predict_transcript_tiled_batched(
                tx, gff_index, ds, model, device, batch_size=args.batch_size
            )
            pred_counts_dict[tx] = pred_codons.astype(np.float32)
        except (TxNotFound, NoCDSFound, Exception) as e:
            pass
            
    log_to_file(f"Prediction complete. Valid predictions: {len(pred_counts_dict)} / {len(tx_ids)}")
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(pred_counts_dict, f)
    log_to_file(f"Saved to {args.output_pkl}")

if __name__ == "__main__":
    main()
