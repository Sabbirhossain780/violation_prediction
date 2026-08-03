"""
src/models/embeddings.py
========================
Input Embedding Layer for Multivariate Traffic Violation Forecasting.

Adapts TopKNet's embedding blueprint for sparse, multivariate event data:
- Sparse Categorical Embedding: Learns dense representations for 58 violation
  channels with built-in sparsity handling via learnable channel embeddings.
- Time-Aware Spatial Identity Embedding (E_tas): Modulates grid identity by
  6-hour quarter instead of 5-minute intervals.
- Temporal Embedding: Day-of-week + quarter-of-day codebooks.
"""

import torch
import torch.nn as nn


class ViolationEmbedding(nn.Module):
    """
    Complete input embedding layer producing E ∈ R[B, H, N, d_h].

    Args:
        n_channels: Number of violation categories (58)
        n_nodes: Total nodes including Virtual Sinkhole
        d_f: Data embedding dimension
        d_t: Temporal embedding dimension (d_diw + d_tid)
        d_s: Spatial identity embedding dimension
        n_days: Number of day-of-week categories (7)
        n_quarters: Number of time-of-day quarters (4)
        dropout: Dropout rate for time information projection
    """

    def __init__(
        self,
        n_channels: int = 58,
        n_nodes: int = 601,
        d_f: int = 64,
        d_t: int = 32,
        d_s: int = 32,
        n_days: int = 7,
        n_quarters: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.d_h = d_f + d_t + d_s

        # ── Data Embedding: Sparse Categorical → Dense ────────────────────────
        # Learnable per-channel embeddings capture semantic relationships
        # between violation types (e.g., Equipment ↔ Suspended License)
        self.channel_emb = nn.Embedding(n_channels, d_f)
        # Linear projection to mix channel embeddings with count magnitudes
        self.data_proj = nn.Linear(d_f, d_f)

        # ── Temporal Embedding: Codebooks ─────────────────────────────────────
        d_diw = d_t // 2
        d_tid = d_t - d_diw
        self.day_emb = nn.Embedding(n_days, d_diw)
        self.quarter_emb = nn.Embedding(n_quarters, d_tid)

        # ── Time-Aware Spatial Identity Embedding ─────────────────────────────
        # TE: Time information projection from temporal metadata
        self.time_fc = nn.Sequential(
            nn.Linear(2, d_s),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # E_s: Learnable spatial identity per node
        self.spatial_emb = nn.Parameter(torch.randn(n_nodes, d_s))

    def forward(
        self,
        x: torch.Tensor,
        tim: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:   [B, H, N, C] violation counts
            tim: [B, H, 2] temporal metadata (day_of_week, quarter_of_day)

        Returns:
            E: [B, H, N, d_h] fused embedding
        """
        B, H, N, C = x.shape

        # ── Data Embedding ────────────────────────────────────────────────────
        # Weight channel embeddings by normalized count magnitude
        # This preserves both semantic identity AND event intensity
        x_norm = x / (x.sum(dim=-1, keepdim=True).clamp(min=1.0))  # [B,H,N,C]
        # Weighted sum of channel embeddings: captures active violation mix
        ch_emb = self.channel_emb.weight  # [C, d_f]
        E_d = torch.einsum('bhnc,cd->bhnd', x_norm, ch_emb)  # [B,H,N,d_f]
        E_d = self.data_proj(E_d)

        # ── Temporal Embedding ────────────────────────────────────────────────
        dow = tim[..., 0].long()      # [B, H]
        qtr = tim[..., 1].long()      # [B, H]
        E_diw = self.day_emb(dow)     # [B, H, d_diw]
        E_tid = self.quarter_emb(qtr) # [B, H, d_tid]
        E_t = torch.cat([E_diw, E_tid], dim=-1)  # [B, H, d_t]
        # Broadcast across nodes: [B, H, N, d_t]
        E_t = E_t.unsqueeze(2).expand(-1, -1, N, -1)

        # ── Time-Aware Spatial Identity Embedding ─────────────────────────────
        # Project temporal metadata to spatial dimension
        TE = self.time_fc(tim.float())  # [B, H, d_s]
        # Broadcast TE across nodes and modulate with learnable spatial identity
        TE = TE.unsqueeze(2).expand(-1, -1, N, -1)       # [B, H, N, d_s]
        E_s = self.spatial_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, N, d_s]
        E_tas = TE * E_s  # Hadamard product: [B, H, N, d_s]

        # ── Concatenate All Embeddings ────────────────────────────────────────
        E = torch.cat([E_d, E_t, E_tas], dim=-1)  # [B, H, N, d_h]
        return E