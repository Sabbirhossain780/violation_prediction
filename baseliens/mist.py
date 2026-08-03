"""
baselines/mist.py
=================
Tier 1 Baseline: MiST (ICDE 2021) adapted for violation forecasting.

Neural Hawkes Process modeling event self-excitation:
- Intensity function λ(t) depends on history of ALL 58 channels
- Cross-channel excitation: Equipment violation → triggers DUI intensity
- Continuous-time formulation discretized to 6-hour bins
Output: Count-only (integrated intensity over forecast window)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .base_baseline import BaseBaseline


class MiST(BaseBaseline):
    def __init__(
        self,
        n_nodes: int,
        n_channels: int,
        adj_matrix: torch.Tensor,
        hidden_dim: int = 128,
        n_event_types: int = 58,
        intensity_layers: int = 2,
        **kwargs
    ):
        super().__init__(n_nodes, n_channels, adj_matrix, **kwargs)
        self.hidden_dim = hidden_dim
        
        # Event type embedding
        self.event_emb = nn.Embedding(n_event_types, hidden_dim)
        
        # History encoder: LSTM over event sequence
        self.history_encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Intensity network: maps hidden state to per-channel intensity
        intensity_net = []
        in_dim = hidden_dim
        for _ in range(intensity_layers):
            intensity_net.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim
        intensity_net.append(nn.Linear(hidden_dim, n_event_types))
        intensity_net.append(nn.Softplus())
        self.intensity_net = nn.Sequential(*intensity_net)

    def forward(self, x: torch.Tensor, tim: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, H, N, C = x.shape
        
        # Flatten spatial dims for batch processing: [B*N, H, C]
        x_flat = x.reshape(B * N, H, C)
        
        # Weighted sum of event embeddings by count (handles multiple events per bin)
        # Create index tensor for embedding lookup
        indices = torch.arange(C, device=x.device).unsqueeze(0).unsqueeze(0).expand(B * N, H, -1)
        emb = self.event_emb(indices)  # [B*N, H, C, D]
        
        # Weight by normalized counts
        x_norm = x_flat / (x_flat.sum(dim=-1, keepdim=True).clamp(min=1.0))
        weighted_emb = (emb * x_norm.unsqueeze(-1)).sum(dim=2)  # [B*N, H, D]
        
        # Encode history
        lstm_out, _ = self.history_encoder(weighted_emb)  # [B*N, H, D]
        last_state = lstm_out[:, -1, :]  # [B*N, D]
        
        # Compute intensity (expected rate per channel)
        intensity = self.intensity_net(last_state)  # [B*N, C]
        
        # Integrated intensity over forecast window (6 hours) ≈ intensity * window_duration
        # Since our bins are already 6-hour units, intensity IS the expected count
        mu = intensity.reshape(B, N, C).unsqueeze(1)  # [B, 1, N, C]
        
        return {"count_mu": mu}