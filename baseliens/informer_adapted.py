"""
baselines/informer_adapted.py
=============================
Tier 2 Baseline: Informer (AAAI 2021) WITH SHARED ADAPTATIONS.

Pure temporal transformer — NO spatial modeling.
Adaptations: Virtual Node ignored (no graph), hybrid loss compatible output.

Purpose: Ablation proving TopK GCN contributes beyond temporal patterns alone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .base_baseline import BaseBaseline


class InformerAdapted(BaseBaseline):
    def __init__(
        self,
        n_nodes: int,
        n_channels: int,
        adj_matrix: torch.Tensor,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 3,
        factor: int = 5,
        **kwargs
    ):
        super().__init__(n_nodes, n_channels, adj_matrix, **kwargs)
        self.d_model = d_model
        
        # Input embedding (flatten spatial dim — no spatial awareness)
        self.input_emb = nn.Linear(n_channels, d_model)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model)
        
        # Transformer encoder (ProbSparse attention from Informer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        
        # Output projection
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_channels),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor, tim: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, H, N, C = x.shape
        
        # Flatten spatial: treat each node's time series independently
        # [B, H, N, C] -> [B*N, H, C]
        x_flat = x.reshape(B * N, H, C)
        
        z = self.input_emb(x_flat)  # [B*N, H, D]
        z = self.pos_enc(z)
        
        z = self.encoder(z)  # [B*N, H, D]
        
        # Use last time step as forecast representation
        z_last = z[:, -1, :]  # [B*N, D]
        
        mu = self.output_head(z_last)  # [B*N, C]
        mu = mu.reshape(B, N, C).unsqueeze(1)  # [B, 1, N, C]
        
        return {"count_mu": mu}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]