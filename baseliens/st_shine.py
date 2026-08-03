"""
baselines/st_shine.py
=====================
Tier 1 Baseline: ST-SHINE (KDD 2020) adapted for violation forecasting.

Hierarchical attention network capturing:
- Level 1: Intra-category spatial patterns (per violation type)
- Level 2: Inter-category correlations across all 58 channels
Output: Count-only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict  
from .base_baseline import BaseBaseline


class STSHINE(BaseBaseline):
    def __init__(
        self,
        n_nodes: int,
        n_channels: int,
        adj_matrix: torch.Tensor,
        hidden_dim: int = 128,
        n_attention_heads: int = 4,
        hierarchy_levels: int = 2,
        **kwargs
    ):
        super().__init__(n_nodes, n_channels, adj_matrix, **kwargs)
        self.hidden_dim = hidden_dim
        self.n_heads = n_attention_heads
        
        # Channel embedding
        self.channel_emb = nn.Linear(n_channels, hidden_dim)
        
        # Hierarchical attention layers
        self.intra_attn = nn.MultiheadAttention(hidden_dim, n_attention_heads, batch_first=True)
        self.inter_attn = nn.MultiheadAttention(hidden_dim, n_attention_heads, batch_first=True)
        
        # Temporal MLP
        self.temporal_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_channels),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor, tim: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, H, N, C = x.shape
        
        # Embed channels: [B, H, N, C] -> [B, H, N, D]
        z = self.channel_emb(x)
        
        # Aggregate temporal dimension via mean pooling (ST-SHINE uses hierarchical temporal aggregation)
        z = z.mean(dim=1)  # [B, N, D]
        
        # Level 1: Intra-spatial attention (node-to-node within same representation)
        attn_out, _ = self.intra_attn(z, z, z)
        z = z + attn_out
        
        # Level 2: Inter-category attention (treat channels as sequence)
        # Reshape: [B, N, D] -> [B*N, 1, D] for single-token attention across nodes
        # Simplified: use global attention across nodes
        attn_out2, _ = self.inter_attn(z, z, z)
        z = z + attn_out2
        
        # Temporal refinement
        z = z + self.temporal_mlp(z)
        
        # Output
        mu = self.output_head(z).unsqueeze(1)  # [B, 1, N, C]
        
        return {"count_mu": mu}