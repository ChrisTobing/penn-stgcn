"""
6-block lightweight ST-GCN.

Channels: 2 → 32 → 32 → 64 → 64 → 128 → 128
Temporal stride-2 at blocks 3 and 5 (T: 100 → 50 → 25)
Head: GlobalAvgPool → BN → Dropout → FC(num_class)
"""

import torch
import torch.nn as nn

from src.models.adjacency import get_penn_action_adjacency
from src.models.stgcn_block import STGCNBlock


class STGCN_Light(nn.Module):

    def __init__(self, num_class=15, in_channels=2, A=None, dropout=0.0,
                 adaptive=False):
        super().__init__()
        if A is None:
            A = get_penn_action_adjacency()

        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, 32, A, stride=1, residual=False,
                       dropout=dropout, adaptive=adaptive),
            STGCNBlock(32, 32, A, stride=1,
                       dropout=dropout, adaptive=adaptive),
            STGCNBlock(32, 64, A, stride=2,
                       dropout=dropout, adaptive=adaptive),
            STGCNBlock(64, 64, A, stride=1,
                       dropout=dropout, adaptive=adaptive),
            STGCNBlock(64, 128, A, stride=2,
                       dropout=dropout, adaptive=adaptive),
            STGCNBlock(128, 128, A, stride=1,
                       dropout=dropout, adaptive=adaptive),
        ])

        self.bn_out = nn.BatchNorm1d(128)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_class)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=[2, 3])              # GlobalAvgPool → (N*M, 128)
        x = x.view(N, M, -1).mean(dim=1)    # average persons → (N, 128)
        x = self.bn_out(x)
        x = self.fc_dropout(x)
        return self.fc(x)
