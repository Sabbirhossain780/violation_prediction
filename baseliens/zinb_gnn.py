"""
baselines/zinb_gnn.py
=====================
Tier 1 Baseline: ZINB-GNN (Spatial Statistics 2022).

GNN with native Zero-Inflated Negative Binomial output layer.
Uses its OWN ZINB loss (not our hybrid loss) for fair comparison.
Purpose: Validate that our Event-Skip Attention + TopK GCN adds value
beyond just using ZINB parameterization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .base_baseline import BaseBaseline


class ZINBGNN(BaseBaseline):
    def __init__(
        self,
        n_nodes: int,
        n_channels: int,
        adj_matrix: torch.Tensor,
        hidden_dim: int = 128,
        n_gcn_layers: int = 2,
        zero_inflation_gate: bool = True,
        **kwargs
    ):
        super().__init__(n_nodes, n_channels, adj_matrix, **kwargs)
        self.hidden_dim = hidden_dim
        self.use_zi_gate = zero_inflation_gate
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        in_dim = n_channels
        for _ in range(n_gcn_layers):
            self.gcn_layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
        
        # NB parameters
        self.mu_head = nn.Sequential(nn.Linear(hidden_dim, n_channels), nn.Softplus())
        self.alpha_head = nn.Sequential(nn.Linear(hidden_dim, n_channels), nn.Softplus())
        
        # Zero-inflation gate
        if self.use_zi_gate:
            self.pi_head = nn.Sequential(nn.Linear(hidden_dim, n_channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor, tim: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, H, N, C = x.shape
        
        # Aggregate temporal dim (ZINB-GNN doesn't model temporal dynamics explicitly)
        z = x.mean(dim=1)  # [B, N, C]
        
        # GCN message passing
        for gcn in self.gcn_layers:
            z = torch.bmm(self.adj_matrix.unsqueeze(0).expand(B, -1, -1), z)
            z = F.relu(gcn(z))
        
        mu = self.mu_head(z)       # [B, N, C]
        alpha = self.alpha_head(z) # [B, N, C]
        
        result = {
            "count_mu": mu.unsqueeze(1),      # [B, 1, N, C]
            "count_alpha": alpha.unsqueeze(1) # [B, 1, N, C]
        }
        
        if self.use_zi_gate:
            pi = self.pi_head(z)
            result["count_pi"] = pi.unsqueeze(1)
            # Adjusted mu accounting for zero inflation
            result["count_mu"] = (mu * (1 - pi)).unsqueeze(1)
        
        return result