"""
baselines/base_baseline.py
==========================
Abstract base class enforcing fair adaptation protocols for all baselines.

Ensures:
- Standardized forward(x, tim) -> dict interface
- Virtual Sinkhole Node injection for Tier 2 models
- Output format compatible with HybridHeterogeneousLoss
- Consistent parameter initialization

FIXED: Added missing typing imports (Dict, Optional) to prevent NameError.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional  # ← FIXED: Was missing, causing NameError


class BaseBaseline(ABC, nn.Module):
    """
    All baselines must implement forward() returning a dict with at least 'count_mu'.
    This guarantees compatibility with our unified training loop and loss function.
    """

    def __init__(
        self,
        n_nodes: int,
        n_channels: int,
        adj_matrix: torch.Tensor,
        virtual_sinkhole_node: bool = True,
        **kwargs
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_channels = n_channels
        self.virtual_sinkhole_node = virtual_sinkhole_node
        
        # Register adjacency matrix as buffer (not a learnable parameter)
        self.register_buffer('adj_matrix', adj_matrix.float())

    @abstractmethod
    def forward(self, x: torch.Tensor, tim: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass interface.
        
        Args:
            x:   [B, H, N, C] violation counts
            tim: [B, H, 2] temporal metadata
            
        Returns:
            Dict with at least 'count_mu': [B, F, N, C]
            Optionally 'threat_probs' and 'admin_logits' if model supports them.
        """
        pass

    def _get_output_shape(self, x: torch.Tensor) -> tuple:
        """Helper to extract B, F, N, C from input tensor."""
        B, H, N, C = x.shape
        return B, 1, N, C  # F=1 for single-step forecast