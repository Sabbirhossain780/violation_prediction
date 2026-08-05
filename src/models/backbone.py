"""
src/models/backbone.py
======================
Spatio-Temporal Backbone for Multivariate Traffic Violation Forecasting.

Adapts TopKNet's encoder blueprint for sparse event data:
- Event-Skip Temporal Attention: Cross-attention with event-presence masking
  that skips empty 6-hour bins and attends only to historically active periods.
- TopK GCN: Graph convolution on distance-based + Virtual Sinkhole adjacency,
  pruning to top-K_s most relevant spatial neighbors per node.
- Gated MLP: Dual-branch feature interaction with gating mechanism.
- All modules use LayerNorm + residual connections for training stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EventSkipTemporalAttention(nn.Module):
    """
    Time-Aware TopK Attention adapted for sparse violation data.
    
    Key adaptation: Queries/Keys derive from temporal metadata AND an
    event-presence mask derived from input counts. This allows the model
    to "skip" empty 6-hour bins and directly attend to historical bins
    where violations actually occurred, regardless of temporal distance.
    """

    def __init__(self, d_h: int, n_heads: int = 4, top_k_temporal: int = 3, dropout: float = 0.1):
        super().__init__()
        assert d_h % n_heads == 0, f"d_h ({d_h}) must be divisible by n_heads ({n_heads})"
        self.d_h = d_h
        self.n_heads = n_heads
        self.d_k = d_h // n_heads
        self.top_k = top_k_temporal

        self.W_Q = nn.Linear(2, d_h)       # Temporal metadata → Query
        self.W_K = nn.Linear(2, d_h)       # Temporal metadata → Key
        self.W_V = nn.Linear(d_h, d_h)     # Hidden states → Value
        self.W_O = nn.Linear(d_h, d_h)     # Output projection
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_h)

    def forward(
        self,
        z: torch.Tensor,
        tim: torch.Tensor,
        x_counts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z:          [B, H, N, d_h] hidden representation
            tim:        [B, H, 2] temporal metadata (day_of_week, quarter)
            x_counts:   [B, H, N, C] raw violation counts (for event mask)

        Returns:
            z_out:      [B, H, N, d_h] attended representation
        """
        B, H, N, _ = z.shape

        # Event presence mask: True if ANY violation occurred in this bin
        # Shape: [B, H] — shared across nodes and channels
        event_mask = (x_counts.sum(dim=(-1, -2)) > 0).float()  # [B, H]

        # Project temporal metadata to Q, K
        Q = self.W_Q(tim.float())  # [B, H, d_h]
        K = self.W_K(tim.float())  # [B, H, d_h]
        V = self.W_V(z)            # [B, H, N, d_h]

        # Reshape for multi-head attention
        Q = Q.view(B, H, self.n_heads, self.d_k).permute(0, 2, 1, 3)  # [B, nh, H, dk]
        K = K.view(B, H, self.n_heads, self.d_k).permute(0, 2, 1, 3)  # [B, nh, H, dk]
        V = V.view(B, H, N, self.n_heads, self.d_k).permute(0, 2, 3, 1, 4)  # [B, N, nh, H, dk]

        # Attention scores: [B, nh, H, H]
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # EVENT-SKIP MASKING: Zero out attention to empty time bins
        # Expand event_mask: [B, 1, 1, H] → broadcast over heads and query positions
        skip_mask = event_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, H]
        attn = attn.masked_fill(skip_mask == 0, float('-inf'))

        # TOP-K TEMPORAL FILTERING: Keep only top-k most relevant time steps
        # For each query position, retain only top-k keys (after event masking)
        top_k_vals, top_k_idx = torch.topk(attn, k=min(self.top_k, H), dim=-1)
        top_k_mask = torch.full_like(attn, float('-inf'))
        top_k_mask.scatter_(-1, top_k_idx, top_k_vals)
        attn = top_k_mask

        attn = F.softmax(attn, dim=-1)
        # Guard against all-empty-bin batches: softmax(-inf,...) → NaN → replace with 0
        # so the residual connection z+out carries signal unchanged (correct for zero-event bins)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        # Aggregate values: [B, N, nh, H, dk] × [B, nh, H, H] → [B, N, nh, H, dk]
        # Note: matmul broadcasts over N dimension
        out = torch.matmul(attn.unsqueeze(1), V.permute(0, 1, 3, 2, 4))  # [B, N, nh, H, dk]
        out = out.permute(0, 3, 1, 2, 4).contiguous().view(B, H, N, self.d_h)
        out = self.W_O(out)

        return self.norm(z + out)


class TopKGCN(nn.Module):
    """
    TopK Graph Convolution adapted for violation forecasting.
    
    Operates on precomputed distance-based + Virtual Sinkhole adjacency.
    Dynamically prunes to top-K_s strongest spatial neighbors per node
    at each forward pass, filtering out irrelevant distant grids.
    """

    def __init__(self, d_h: int, adj_matrix: torch.Tensor, top_k_spatial_ratio: float = 0.2):
        super().__init__()
        self.d_h = d_h
        self.top_k_spatial_ratio = top_k_spatial_ratio
        self.linear = nn.Linear(d_h, d_h)
        self.norm = nn.LayerNorm(d_h)

        # Register adjacency as buffer (not a learnable parameter)
        self.register_buffer('adj', adj_matrix.float())  # [N, N]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, H, N, d_h] hidden representation

        Returns:
            z_out: [B, H, N, d_h] spatially aggregated representation
        """
        B, H, N, _ = z.shape
        K_s = max(int(N * self.top_k_spatial_ratio), 1)

        # Dynamic TopK pruning of adjacency matrix per forward pass
        # Keep only top-K_s strongest connections for each node
        _, top_k_idx = torch.topk(self.adj, k=K_s, dim=-1)  # [N, K_s]
        sparse_adj = torch.zeros_like(self.adj)
        row_idx = torch.arange(N, device=z.device).unsqueeze(-1).expand(-1, K_s)
        sparse_adj[row_idx, top_k_idx] = self.adj[row_idx, top_k_idx]

        # Normalize adjacency (symmetric normalization: D^{-1/2} A D^{-1/2})
        deg = sparse_adj.sum(dim=-1).clamp(min=1e-6)
        deg_inv_sqrt = deg.pow(-0.5)
        norm_adj = deg_inv_sqrt.unsqueeze(-1) * sparse_adj * deg_inv_sqrt.unsqueeze(0)

        # Graph convolution: aggregate neighbor messages
        # Reshape z for batched matmul: [B*H, N, d_h]
        z_flat = z.reshape(B * H, N, self.d_h)
        z_agg = torch.bmm(norm_adj.unsqueeze(0).expand(B * H, -1, -1), z_flat)  # [B*H, N, d_h]
        z_agg = z_agg.reshape(B, H, N, self.d_h)

        z_out = self.linear(z_agg)
        return self.norm(z + z_out)


class GatedMLP(nn.Module):
    """
    Gated MLP for feature interaction after spatio-temporal modeling.
    Dual branch: linear path × gated nonlinear path → fused output.
    """

    def __init__(self, d_h: int, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.linear_branch = nn.Linear(d_h, d_model)
        self.gate_branch = nn.Sequential(
            nn.Linear(d_h, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fuse = nn.Linear(d_model, d_h)
        self.norm = nn.LayerNorm(d_h)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, H, N, d_h] → [B, H, N, d_h]"""
        g1 = self.linear_branch(z)
        g2 = self.gate_branch(z)
        fused = self.fuse(g1 * g2)
        return self.norm(z + fused)


class SpatioTemporalBackbone(nn.Module):
    """
    Complete spatio-temporal encoder stack.
    
    Stacks L layers of: EventSkipTemporalAttention → TopKGCN → GatedMLP
    Each sub-module has its own residual connection + LayerNorm.
    """

    def __init__(
        self,
        d_h: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        top_k_temporal: int = 3,
        top_k_spatial_ratio: float = 0.2,
        d_model: int = 256,
        adj_matrix: Optional[torch.Tensor] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_layers = n_layers

        if adj_matrix is None:
            raise ValueError("adj_matrix must be provided. Load from data/processed/adj_matrix.pt")

        self.temporal_attns = nn.ModuleList([
            EventSkipTemporalAttention(d_h, n_heads, top_k_temporal, dropout)
            for _ in range(n_layers)
        ])
        self.spatial_gcns = nn.ModuleList([
            TopKGCN(d_h, adj_matrix, top_k_spatial_ratio)
            for _ in range(n_layers)
        ])
        self.gated_mlps = nn.ModuleList([
            GatedMLP(d_h, d_model, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        z: torch.Tensor,
        tim: torch.Tensor,
        x_counts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z:         [B, H, N, d_h] embedded input
            tim:       [B, H, 2] temporal metadata
            x_counts:  [B, H, N, C] raw violation counts (for event-skip mask)

        Returns:
            z:         [B, H, N, d_h] encoded spatio-temporal representation
        """
        for i in range(self.n_layers):
            z = self.temporal_attns[i](z, tim, x_counts)
            z = self.spatial_gcns[i](z)
            z = self.gated_mlps[i](z)
        return z