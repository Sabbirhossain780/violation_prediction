"""
baselines/deepcrime.py
======================
Tier 1 Baseline: DeepCrime (AAAI 2018) adapted for violation forecasting.

Key adaptations:
- Input: 58-channel violation counts instead of crime categories
- Zero-inflation gate: Explicitly models P(y=0) vs P(y>0)
- Temporal: GRU-based burstiness capture instead of LSTM
- Output: Count-only (no severity/admin heads)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict 
from .base_baseline import BaseBaseline


class DeepCrime(BaseBaseline):
    def __init__(
        self,
        n_nodes: int,
        n_channels: int,
        adj_matrix: torch.Tensor,
        hidden_dim: int = 128,
        n_gcn_layers: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(n_nodes, n_channels, adj_matrix, **kwargs)
        self.hidden_dim = hidden_dim
        
        # Spatial GCN layers
        self.gcn_layers = nn.ModuleList()
        in_dim = n_channels
        for _ in range(n_gcn_layers):
            self.gcn_layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
        
        # Temporal GRU for burstiness
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Zero-inflation gate: predicts P(y=0) per channel
        self.zero_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_channels),
            nn.Sigmoid()
        )
        
        # Count prediction head (for non-zero values)
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_channels),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor, tim: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, H, N, C = x.shape
        
        # Spatial encoding: apply GCN independently per time step
        spatial_outs = []
        for t in range(H):
            h = x[:, t, :, :]  # [B, N, C]
            for gcn in self.gcn_layers:
                # Simple spectral GCN: A @ H @ W
                h = torch.bmm(self.adj_matrix.unsqueeze(0).expand(B, -1, -1), h)
                h = F.relu(gcn(h))
            spatial_outs.append(h)
        
        # Stack temporal sequence: [B, N, H, hidden_dim] -> [B*N, H, hidden_dim]
        z = torch.stack(spatial_outs, dim=2)  # [B, N, H, D]
        z = z.reshape(B * N, H, -1)
        
        # Temporal GRU
        gru_out, _ = self.gru(z)  # [B*N, H, D]
        last_hidden = gru_out[:, -1, :]  # [B*N, D]
        last_hidden = last_hidden.reshape(B, N, -1)  # [B, N, D]
        
        # Zero-inflated count prediction
        p_zero = self.zero_gate(last_hidden)  # [B, N, C]
        count_nonzero = self.count_head(last_hidden)  # [B, N, C]
        
        # Expected count = (1 - p_zero) * count_nonzero
        expected_count = (1 - p_zero) * count_nonzero
        
        # Reshape to [B, F=1, N, C]
        mu = expected_count.unsqueeze(1)
        
        return {"count_mu": mu}