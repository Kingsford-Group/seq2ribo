import os, re, pickle, warnings, sys, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# Append path to import Translatomer modules
sys.path.append('Translatomer')
from train_all_11fold import TrainModule, FastaStringExtractor
import model.blocks as blocks

# ------------------ CONSTANTS ------------------
SEQLEN   = 65536
NBINS    = 1024
BIN      = SEQLEN // NBINS  # 64
TRAIN_STRIDE = SEQLEN // 2 

# ------------------ HELPERS ------------------
def _attrs(s):
    d = {}
    for kv in s.split(';'):
        if '=' in kv:
            k,v = kv.split('=',1); d[k]=v
    return d
def _enst_nov(enst): 
    return enst.split('.')[0]

# 5-channel one-hot (A,T,C,G,other)
_LUT = np.full(256, 4, dtype=np.uint8)
for _ch, idx in zip(b'ATCGatcg', [0,1,2,3,0,1,2,3]): _LUT[_ch] = idx
_EYE5 = np.eye(5, dtype=np.float32)

def onehot_dna_fast(seq: str) -> torch.Tensor:
    idx = _LUT[np.frombuffer(seq.encode('ascii','ignore').upper(), dtype=np.uint8)]
    return torch.from_numpy(_EYE5[idx])  # [L,5]

class Interval:
    def __init__(self, chrom, start, end):
        self.chrom = chrom
        self.start = start
        self.end = end
    def __repr__(self): return f"{self.chrom}:{self.start}-{self.end}"

# ------------------ GFF Indexing ------------------
def _key_variants(id_like: str):
    base = id_like.split(':', 1)[-1] if ':' in id_like else id_like
    en0  = _enst_nov(base)
    return {
        id_like, base, en0,
        f"transcript:{base}", f"transcript:{en0}"
    }

def build_gff_index(gff_df):
    idx = {"tx": {}, "cds": {}}
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

def lookup_tx_and_intervals(gff_index, enst):
    keys = list(_key_variants(enst))
    tx_info = None
    for k in keys:
        if k in gff_index["tx"]:
            tx_info = gff_index["tx"][k]; break
    if tx_info is None:
        return None 
    for k in keys:
        if k in gff_index["cds"]:
            chrom, strand, tx_s, tx_e = tx_info
            return chrom, strand, tx_s, tx_e, gff_index["cds"][k]
    return None

# ------------------ Dataset ------------------
class GenomicTrainDataset(Dataset):
    def __init__(self, data_list, gff_index, fasta_extractor):
        self.data_list = data_list # List of tuples from pkl
        self.gff_index = gff_index
        self.fasta = fasta_extractor
        
        # Prepare samples (Windows)
        self.samples = []
        self._prepare_samples()
        
    def _prepare_samples(self):
        print("Preparing training samples (tiling)...")
        count = 0
        for item in self.data_list:
            if not (isinstance(item, (list, tuple)) and len(item) >= 3): continue
            
            tx_id = item[0]
            gt_counts = item[2] # Item 2 is GT
            
            # Lookup GFF
            info = lookup_tx_and_intervals(self.gff_index, tx_id)
            if info is None: continue
            chrom, strand, tx_s, tx_e, intervals = info
            
            # CDS limits (Genomic)
            cds_s = min(s for s,_ in intervals)
            cds_e = max(e for _,e in intervals)
            
            # Tiling Strategy for Training:
            if (cds_e - cds_s) <= SEQLEN:
                mid = (cds_s + cds_e)//2
                starts = [mid - SEQLEN//2]
            else:
                starts = list(range(cds_s, cds_e, TRAIN_STRIDE))
                if starts[-1] + SEQLEN < cds_e:
                    starts.append(cds_e - SEQLEN)
            
            for s in starts:
                self.samples.append({
                    "tx_id": tx_id,
                    "chrom": chrom,
                    "strand": strand,
                    "win_start": s,
                    "win_end": s + SEQLEN,
                    "gt_counts": gt_counts, 
                    "intervals": intervals,
                    "cds_s": cds_s
                })
                count += 1
        print(f"Prepared {len(self.samples)} samples from {len(self.data_list)} transcripts.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # 1. Sequence
        ival = Interval(s["chrom"], s["win_start"], s["win_end"])
        seq_onehot = onehot_dna_fast(self.fasta.extract(ival)) # [L, 5]
        
        # 2. RNA (REMOVED for SeqOnly)
        
        # 3. Target Construction
        target_bp = np.zeros(SEQLEN, dtype=np.float32)
        
        current_spliced_idx = 0
        win_start = s["win_start"]
        win_end = s["win_end"]
        
        for (is_s, is_e) in s["intervals"]:
            seg_len = is_e - is_s
            ov_s = max(is_s, win_start)
            ov_e = min(is_e, win_end)
            
            if ov_s < ov_e:
                seg_offset_start = ov_s - is_s
                seg_offset_end = ov_e - is_s
                
                spliced_idx_start = current_spliced_idx + seg_offset_start
                spliced_idx_end   = current_spliced_idx + seg_offset_end
                
                if spliced_idx_start < len(s["gt_counts"]):
                    limit = min(spliced_idx_end, len(s["gt_counts"]))
                    gt_chunk = s["gt_counts"][spliced_idx_start:limit]
                    
                    win_offset_start = ov_s - win_start
                    win_offset_end = win_offset_start + len(gt_chunk)
                    
                    target_bp[win_offset_start:win_offset_end] = gt_chunk
            
            current_spliced_idx += seg_len
            
        # Handle Reverse Strand
        if s["strand"] == '-':
            # Flip Sequence (Flip L dim, Swap Channels A<->T, C<->G)
            seq_onehot = torch.flip(seq_onehot, [0]) 
            # Manual complement swap for channels
            s0 = seq_onehot[:, 0].clone()
            s1 = seq_onehot[:, 1].clone()
            s2 = seq_onehot[:, 2].clone()
            s3 = seq_onehot[:, 3].clone()
            seq_onehot[:, 0] = s1
            seq_onehot[:, 1] = s0
            seq_onehot[:, 2] = s3
            seq_onehot[:, 3] = s2
            
            # Flip Target BP
            target_bp = np.flip(target_bp)
            
        # 4. Bin Target
        target_binned = target_bp.reshape(NBINS, BIN).sum(axis=1)
        
        return {
            'sequence': seq_onehot.float(),
            'output_features': torch.tensor(target_binned, dtype=torch.float32) # [1024]
        }

class GenomicDataModule(pl.LightningDataModule):
    def __init__(self, train_pkl, gff_index, fasta_path, batch_size=32, num_workers=1):
        super().__init__()
        self.train_pkl = train_pkl
        self.gff_index = gff_index
        self.fasta_path = fasta_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        with open(self.train_pkl, 'rb') as f:
            full_data = pickle.load(f)
        np.random.seed(42)
        indices = np.random.permutation(len(full_data))
        split = int(len(full_data) * 0.9)
        train_data = [full_data[i] for i in indices[:split]]
        val_data   = [full_data[i] for i in indices[split:]]
        
        self.fasta_extractor = FastaStringExtractor(self.fasta_path)
        self.train_ds = GenomicTrainDataset(train_data, self.gff_index, self.fasta_extractor)
        self.val_ds   = GenomicTrainDataset(val_data,   self.gff_index, self.fasta_extractor)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

# ------------------ Model Patch ------------------
# Define Sequence-Only Model (5 Channels)
class TransModelSeqOnly(nn.Module):
    def __init__(self, num_genomic_features, mid_hidden = 512, record_attn = False):
        super(TransModelSeqOnly, self).__init__()
        print('Initializing TransModelSeqOnly (5 Channels)')
        self.conv1 = nn.Sequential(
            nn.Conv1d(5, 512, 129, 128, 64), # Changed from 6 to 5
            #nn.Conv1d(1, 512, 129, 128, 64),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.attn = blocks.AttnModule(hidden = mid_hidden, record_attn = record_attn, inpu_dim =512)
        self.conv2 = nn.Conv1d(512, 1, 3, 1, 1)
        self.Linear4 = nn.Linear(in_features =512, out_features = 1024)
        self.record_attn = record_attn
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x):
        '''
        Input feature:
        batch_size, length = 65536, feature_dim = 5
        '''
        x = x.transpose(1, 2).contiguous().float()
        x = self.conv1(x)
        x = x.transpose(1, 2).contiguous().float()
        if self.record_attn:
            x, attn_weights = self.attn(x)
        else:
            x = self.attn(x)
        x = self.dropout(x)
        x = x.transpose(1, 2).contiguous().float()
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.Linear4(x).squeeze(1)
        x = F.relu(x)
        if self.record_attn:
            return x, attn_weights
        else:
            return x         

# PL Patching
if hasattr(TrainModule, 'training_epoch_end'):
    del TrainModule.training_epoch_end
if hasattr(TrainModule, 'validation_epoch_end'):
    del TrainModule.validation_epoch_end
if hasattr(TrainModule, 'test_epoch_end'):
    del TrainModule.test_epoch_end

class PatchedTrainModuleSeqOnly(TrainModule):
    def __init__(self, args):
        super().__init__(args)
        # Re-initialize model to be SeqOnly
        self.model = self.get_model(args)

    def get_model(self, args):
        # Override to return custom model
        return TransModelSeqOnly(args.hparams)

    def proc_batch(self, batch):
        # Only have sequence
        inputs = batch['sequence'] # [B, L, 5]
        targets = batch['output_features']
        targets = targets.float() 
        return inputs, targets

    # Override steps to ensure we use our proc_batch and logging
    def training_step(self, batch, batch_idx):
        inputs, targets = self.proc_batch(batch)
        pred = self(inputs)   
        loss = self.criterion(pred, targets).mean()
        pcc = self.pcc(pred, targets).mean()
        
        ranks1 = pred.argsort().argsort().type(torch.float32)
        ranks2 = targets.argsort().argsort().type(torch.float32)
        scc = self.pcc(ranks1, ranks2).mean()
        
        metrics = {
            'loss/train_step' : loss, 
            'pearson/train_step': pcc,
            'spearman/train_step': scc,
            'train_loss': loss,
            'train_pcc': pcc
        }
        self.log_dict(metrics, batch_size=inputs.shape[0], prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        return {'loss': loss, 'pcc': pcc, 'scc': scc}

    def validation_step(self, batch, batch_idx):
        inputs, targets = self.proc_batch(batch)
        pred = self(inputs)   
        loss = self.criterion(pred, targets).mean()
        pcc = self.pcc(pred, targets).mean()
        
        ranks1 = pred.argsort().argsort().type(torch.float32)
        ranks2 = targets.argsort().argsort().type(torch.float32)
        scc = self.pcc(ranks1, ranks2).mean()
        
        metrics = {
            'val_loss': loss,
            'val_pcc': pcc,
            'val_scc': scc
        }
        self.log_dict(metrics, batch_size=inputs.shape[0], prog_bar=True, sync_dist=True, on_epoch=True)
        return metrics

# ------------------ Main ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_line', required=True, choices=['hek293', 'ipsc', 'lcl', 'rpe'])
    parser.add_argument('--checkpoint_out', default='checkpoints_seqonly')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--fasta', required=True, help='GRCh38 genome FASTA')
    parser.add_argument('--gff', required=True, help='GENCODE v49 annotation GFF3')
    parser.add_argument('--resume_ckpt', default=None)
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID to use (e.g. 0 for cuda:0)")
    parser.add_argument('--num_workers', type=int, default=1)
    
    args = parser.parse_args()
    
    pl.seed_everything(42)
    
    base_dir = f"data/{args.cell_line}"
    train_pkl = f"{base_dir}/train_data.pkl"
    
    gff_cache_path = args.gff + ".index.pkl"
    if os.path.exists(gff_cache_path):
        print(f"Loading GFF index from {gff_cache_path}...")
        with open(gff_cache_path, 'rb') as f:
            gff_index = pickle.load(f)
    else:
        from inference_seqonly import build_gff_index   # the index the inference script builds and caches
        print(f"Building GFF index {gff_cache_path}...")
        gff_index = build_gff_index(pd.read_csv(args.gff, sep='\t', comment='#', header=None))
        with open(gff_cache_path, 'wb') as f:
            pickle.dump(gff_index, f)
        
    print(f"Loading {train_pkl}...")
    dm = GenomicDataModule(train_pkl, gff_index, args.fasta, batch_size=args.batch_size, num_workers=args.num_workers)
    
    class ModelArgs:
        def __init__(self):
            self.model_type = 'TransModel'
            self.learning_rate = 1e-5 
            self.hparams = {} 
            
    model_args = ModelArgs()
    
    if args.resume_ckpt:
        print(f"Loading checkpoint {args.resume_ckpt}...")
        model = PatchedTrainModuleSeqOnly.load_from_checkpoint(args.resume_ckpt, args=model_args)
    else:
        print("Initializing new SeqOnly model...")
        model = PatchedTrainModuleSeqOnly(model_args)

    ckpt_dir = f"{args.checkpoint_out}/{args.cell_line}"
    print(f"Outputting checkpoints to {ckpt_dir}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=[args.gpu_id],
        callbacks=[checkpoint_callback],
        log_every_n_steps=50,
        enable_progress_bar=True
    )
    
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()
