"""
src/models/decoder.py
=====================
Hybrid Multi-Task Decoder for Traffic Violation Forecasting.

Replaces TopKNet's single Linear Decoder with three specialized heads:
1. Count Head: Outputs Negative Binomial (mu, alpha) parameters for
   overdispersed violation counts. Uses Softplus for positivity.
2. Admin Head: Categorical logits for 4 administrative dispositions.
3. Threat Head: Independent sigmoid probabilities for S severity flags.

All heads share the same backbone representation but have independent
output projections, enabling multi-task learning with task-specific
activation functions and loss-compatible parameterizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CountHead(nn.Module):
    """
    Negative Binomial parameter estimation head for violation counts.

    Outputs mu (mean) and alpha (inverse dispersion) per channel.
    Both are constrained to be positive via Softplus activation.
    The NB distribution handles overdispersion inherent in sparse
    event count data far better than Poisson or Gaussian assumptions.
    """

    def __init__(self, d_h: int, n_channels: int = 58):
        super().__init__()
        self.mu_proj = nn.Linear(d_h, n_channels)
        self.alpha_proj = nn.Linear(d_h, n_channels)
        self.n_channels = n_channels

    def forward(self, z: torch.Tensor) -> dict:
        """
        Args:
            z: [B, F, N, d_h] encoded spatio-temporal representation

        Returns:
            dict with 'mu' and 'alpha', both [B, F, N, C], strictly positive
        """
        # Softplus ensures positivity; +1e-6 prevents numerical issues
        mu = F.softplus(self.mu_proj(z)) + 1e-6       # [B, F, N, C]
        alpha = F.softplus(self.alpha_proj(z)) + 1e-6  # [B, F, N, C]
        return {"mu": mu, "alpha": alpha}


class AdminHead(nn.Module):
    """
    Categorical classification head for administrative dispositions.
    Outputs raw logits for 4 mutually exclusive categories:
    Warning, Citation, ESERO, SERO.
    Softmax is applied in the loss function, not here, for numerical stability.
    """

    def __init__(self, d_h: int, n_admin: int = 4):
        super().__init__()
        self.proj = nn.Linear(d_h, n_admin)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, F, N, d_h]

        Returns:
            logits: [B, F, N, n_admin] raw logits (no softmax)
        """
        return self.proj(z)


class ThreatHead(nn.Module):
    """
    Multi-label binary classification head for severity/threat flags.
    Outputs independent sigmoid probabilities for each flag.
    Flags are NOT mutually exclusive (a stop can involve both
    Alcohol AND Arrest simultaneously).
    """

    def __init__(self, d_h: int, n_threat: int = 9):
        super().__init__()
        self.proj = nn.Linear(d_h, n_threat)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, F, N, d_h]

        Returns:
            probs: [B, F, N, n_threat] probabilities in [0, 1]
        """
        return torch.sigmoid(self.proj(z))


class HybridDecoder(nn.Module):
    """
    Complete hybrid multi-task decoder combining all three heads.

    Takes the final backbone output and produces predictions for
    all three tasks simultaneously from the shared representation.
    This enables multi-task learning where common violation patterns
    (e.g., Speeding) provide gradient signal that helps rare tasks
    (e.g., Fatal crashes) learn meaningful representations.
    """

    def __init__(
        self,
        d_h: int = 128,
        n_channels: int = 58,
        n_admin: int = 4,
        n_threat: int = 9,
    ):
        super().__init__()
        self.count_head = CountHead(d_h, n_channels)
        self.admin_head = AdminHead(d_h, n_admin)
        self.threat_head = ThreatHead(d_h, n_threat)

    def forward(self, z: torch.Tensor) -> dict:
        """
        Args:
            z: [B, F, N, d_h] final encoded representation from backbone

        Returns:
            dict with keys:
                'count_mu':    [B, F, N, C] NB mean parameters
                'count_alpha': [B, F, N, C] NB inverse dispersion parameters
                'admin_logits':[B, F, N, 4] raw categorical logits
                'threat_probs':[B, F, N, S] sigmoid probabilities
        """
        count_params = self.count_head(z)
        admin_logits = self.admin_head(z)
        threat_probs = self.threat_head(z)

        return {
            "count_mu": count_params["mu"],
            "count_alpha": count_params["alpha"],
            "admin_logits": admin_logits,
            "threat_probs": threat_probs,
        }