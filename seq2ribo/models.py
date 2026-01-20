"""Neural network models for ribosome profiling prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba, Mamba2

# -----------------------------
# Core Polisher Model
# -----------------------------

class RiboPolisherMamba(nn.Module):
    def __init__(
        self,
        d_model=192,
        n_layers=4,
        d_state=16,   # Mamba-specific: state space dimension
        d_conv=4,     # Mamba-specific: local convolution width
        expand=2,     # Mamba-specific: expansion factor
        dropout=0.1,
        k_angle_bins=4,
        use_mamba2=False,
        activation="relu" # "relu" or "softplus"
    ):
        super().__init__()
        self.activation = activation
        # token & sim
        self.cod_emb  = nn.Embedding(65, d_model)  # 64 codons + padding id 64
        self.sim_proj = nn.Linear(1, d_model)

        # geo feature embeddings (active immediately; no gate)
        self.angle_emb  = nn.Embedding(k_angle_bins, d_model)
        self.pair_emb   = nn.Embedding(4, d_model)
        self.bucket_emb = nn.Embedding(3, d_model)

        MambaLayer = Mamba2 if use_mamba2 else Mamba

        # backbone
        self.encoder = nn.ModuleList([
            MambaLayer(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

        # inits
        nn.init.xavier_uniform_(self.sim_proj.weight)
        nn.init.zeros_(self.sim_proj.bias)
        nn.init.xavier_uniform_(self.angle_emb.weight)
        nn.init.xavier_uniform_(self.pair_emb.weight)
        nn.init.xavier_uniform_(self.bucket_emb.weight)

        # OPTIONAL: bias last layer to be slightly positive
        last = self.head[-1]
        if isinstance(last, nn.Linear) and last.bias is not None:
            nn.init.constant_(last.bias, 1.0)

    def forward(self, cod_ids, sim_feat, pad_mask,
                angle_bin=None, pair_bin=None, bucket_idx=None):
        """
        cod_ids   : (B, L) long in [0..64], where 64 is padding id
        sim_feat  : (B, L) float
        pad_mask  : (B, L) bool
        angle_bin : (B, L) long in [0..k_angle_bins-1] (optional)
        pair_bin  : (B, L) long in [0..3]             (optional)
        bucket_idx: (B, L) long in [0..2]             (optional)
        """
        # base tokenization
        tok = self.cod_emb(cod_ids)                       # (B, L, d)
        sim = self.sim_proj(sim_feat.unsqueeze(-1))       # (B, L, d)
        x = tok + sim

        # fuse geo features by addition (no gating)
        if (angle_bin is not None) or (pair_bin is not None) or (bucket_idx is not None):
            add = 0
            if angle_bin is not None:
                add = add + self.angle_emb(angle_bin.clamp(0, self.angle_emb.num_embeddings - 1))
            if pair_bin is not None:
                add = add + self.pair_emb(pair_bin.clamp(0, self.pair_emb.num_embeddings - 1))
            if bucket_idx is not None:
                add = add + self.bucket_emb(bucket_idx.clamp(0, self.bucket_emb.num_embeddings - 1))
            x = x + add

        # mamba stack
        for i, mamba_block in enumerate(self.encoder):
            residual = x
            x = self.norms[i](x)
            x = mamba_block(x)
            x = x + residual

        logits = self.head(x).squeeze(-1)      # (B, L)
        
        if self.activation == "softplus":
            logits = F.softplus(logits)
        else:
            logits = torch.relu(logits)            # enforce non-negativity
            
        logits = logits * pad_mask.float()     # keep masked positions zero
        return logits


# -----------------------------
# Translation Efficiency (TE) Head
# -----------------------------

class TEHeadFullCounts(nn.Module):
    """
    TE head that reads FULL per-position counts:
      - element-wise small MLP over counts (log1p optional)
      - masked mean pooling -> scalar
      - final linear + sigmoid to produce a value in [0,1] (scaled TE)
    """
    def __init__(self, hidden=64, use_log1p=True, positive_output=True):
        super().__init__()
        self.use_log1p = use_log1p
        self.positive_output = positive_output
        self.ff = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1)
        )
        self.post = nn.Linear(1, 1)
        self.final_act = nn.Sigmoid() if positive_output else nn.Identity()

    def forward(self, counts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.log1p(counts) if self.use_log1p else counts
        x = self.ff(x.unsqueeze(-1)).squeeze(-1)   # (B,L)
        m = mask.float()
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (x * m).sum(dim=1, keepdim=True) / denom
        out = self.post(pooled).squeeze(1)         # (B,)
        out = self.final_act(out)
        return out                                  # scaled TE in [0,1]


class MambaTEFull(nn.Module):
    def __init__(self, base: RiboPolisherMamba, hidden=64, use_log1p=True):
        super().__init__()
        self.base = base
        self.te_head = TEHeadFullCounts(hidden=hidden, use_log1p=use_log1p, positive_output=True)

    def forward(self, cod_ids, sim_feat, mask, angle_bin=None, pair_bin=None, bucket_idx=None):
        logits = self.base(cod_ids, sim_feat, mask,
                           angle_bin=angle_bin, pair_bin=pair_bin, bucket_idx=bucket_idx)
        # Base polisher output is usually ReLU'd (counts-like). 
        # Clamping and expm1 implies treating it as log-counts in TE logic.
        counts = torch.expm1(torch.clamp(logits, max=20.0)) * mask.float()
        
        te_pred_scaled = self.te_head(counts, mask)   # in [0,1]
        return counts, te_pred_scaled


# -----------------------------
# Protein Expression Head
# -----------------------------

class ExprHeadFullCounts(nn.Module):
    """Element-wise MLP on counts, masked-mean pooling -> scalar expression."""
    def __init__(self, hidden=64, use_log1p=True):
        super().__init__()
        self.use_log1p = use_log1p
        self.ff = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden //2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1)
        )
        self.post = nn.Linear(1, 1)

    def forward(self, counts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.log1p(counts) if self.use_log1p else counts
        x = self.ff(x.unsqueeze(-1)).squeeze(-1)   # (B,L)
        m = mask.float()
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (x * m).sum(dim=1, keepdim=True) / denom
        return self.post(pooled).squeeze(1)        # (B,)


class MambaExprFull(nn.Module):
    def __init__(self, base: RiboPolisherMamba, hidden=64, use_log1p=True):
        super().__init__()
        self.base = base
        self.expr_head = ExprHeadFullCounts(hidden=hidden, use_log1p=use_log1p)

    def forward(self, cod_ids, sim_feat, mask, angle_bin=None, pair_bin=None, bucket_idx=None):
        logits = self.base(cod_ids, sim_feat, mask, angle_bin=angle_bin, pair_bin=pair_bin, bucket_idx=bucket_idx)
        counts = torch.expm1(torch.clamp(logits, max=20.0)) * mask.float()
        expr_pred = self.expr_head(counts, mask)
        return counts, expr_pred
