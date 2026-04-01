"""
Single ST-GCN block: spatial graph conv → temporal conv → residual add.

Supports both fixed adjacency (baseline) and adaptive A+B+C (Shi et al. CVPR 2019).
"""

import torch
import torch.nn as nn

from src.models.adjacency import normalize_adjacency


class STGCNBlock(nn.Module):

    def __init__(self, in_ch, out_ch, A, stride=1, residual=True, dropout=0.0,
                 adaptive=False):
        """
        Args:
            in_ch    : input channels
            out_ch   : output channels
            A        : raw (unnormalised) adjacency matrix, shape (V, V)
            stride   : temporal stride (1 = keep, 2 = halve T)
            residual : whether to use a residual connection
            dropout  : dropout probability after temporal conv
            adaptive : if True, use learnable A + B + C adjacency
        """
        super().__init__()
        self.adaptive = adaptive
        V = A.shape[0]

        # Fixed component: normalise once and freeze
        A_norm = normalize_adjacency(A)
        self.register_buffer('A', torch.tensor(A_norm, dtype=torch.float32))

        # Learnable components (adaptive only)
        if self.adaptive:
            self.B = nn.Parameter(torch.zeros(V, V))
            C_reduced = max(in_ch // 4, 1)
            self.theta = nn.Conv2d(in_ch, C_reduced, kernel_size=1)
            self.phi = nn.Conv2d(in_ch, C_reduced, kernel_size=1)

        # Spatial graph convolution
        self.gcn_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn_gcn = nn.BatchNorm2d(out_ch)

        # Temporal convolution
        self.tcn_conv = nn.Conv2d(
            out_ch, out_ch,
            kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1),
        )
        self.bn_tcn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif in_ch == out_ch and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        # x: (N, C, T, V)
        res = self.residual(x)

        if self.adaptive:
            th = self.theta(x).mean(dim=2)  # (N, C//4, V)
            ph = self.phi(x).mean(dim=2)

            C = torch.softmax(
                torch.einsum('nci,ncj->nij', th, ph),
                dim=-1,
            )

            A_hat = self.A + self.B + C  # (N, V, V)

            # Row-normalise to stabilise aggregation scale.
            A_hat = A_hat / (A_hat.sum(dim=-1, keepdim=True) + 1e-6)

            x = torch.einsum('nctv,nvw->nctw', x, A_hat)
        else:
            x = torch.einsum('nctv,vw->nctw', x, self.A)

        x = self.relu(self.bn_gcn(self.gcn_conv(x)))
        x = self.bn_tcn(self.tcn_conv(x))
        x = self.dropout(x)
        return self.relu(x + res)
