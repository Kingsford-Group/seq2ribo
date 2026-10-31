"""Sequence-only Translatomer (He et al. 2024) built from the released Translatomer code.

The transformer trunk is Translatomer's own `model.blocks.AttnModule`, imported unchanged from a
clone of https://github.com/xiongxslab/Translatomer, commit ad1540c (path: env TRANSLATOMER_ROOT).
The sequence-only variant replaces the 6-channel input convolution of TransModel (DNA one-hot +
RNA-seq) by a 5-channel one (one-hot A, T, C, G, other); everything else is as released:
Input [B, 65536, 5] -> conv (k=129, stride 128) -> 512 tokens x 512 -> AttnModule (12 layers) ->
conv to 1 channel -> Linear(512 -> 1024) -> ReLU: one non-negative value per 64-bp bin.
"""
import os
import subprocess
import sys

import torch.nn as nn
import torch.nn.functional as F

TRANSLATOMER_ROOT = os.environ.get("TRANSLATOMER_ROOT")
if not TRANSLATOMER_ROOT:
    raise RuntimeError("set TRANSLATOMER_ROOT to a checkout of https://github.com/xiongxslab/Translatomer")
sys.path.insert(0, TRANSLATOMER_ROOT)
import model.blocks as blocks  # noqa: E402  (released Translatomer)


def translatomer_commit():
    try:
        return subprocess.run(["git", "-C", TRANSLATOMER_ROOT, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


class TransModelSeqOnly(nn.Module):
    def __init__(self, mid_hidden=512, record_attn=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(5, 512, 129, 128, 64),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.attn = blocks.AttnModule(hidden=mid_hidden, record_attn=record_attn, inpu_dim=512)
        self.conv2 = nn.Conv1d(512, 1, 3, 1, 1)
        self.Linear4 = nn.Linear(in_features=512, out_features=1024)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous().float()
        x = self.conv1(x)
        x = x.transpose(1, 2).contiguous().float()
        x = self.attn(x)
        x = self.dropout(x)
        x = x.transpose(1, 2).contiguous().float()
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.Linear4(x).squeeze(1)
        return F.relu(x)
