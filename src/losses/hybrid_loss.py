"""
src/losses/hybrid_loss.py
=========================
Hybrid Heterogeneity-Aware Multi-Task Loss for Traffic Violation Forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ZeroInflatedNBRelativeLoss(nn.Module):
    """
    Zero-Inflated Negative Binomial loss with Relative Error penalty.
    """

    def __init__(self, beta: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(
        self,
        mu: torch.Tensor,
        alpha: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # NUMERICAL STABILITY FIX: Clamp to prevent log(0) and division by zero
        mu = mu.clamp(min=1e-6)
        alpha = alpha.clamp(min=1e-6)

        log_likelihood = (
            torch.lgamma(target + alpha)
            - torch.lgamma(alpha)
            - torch.lgamma(target + 1)
            + alpha * torch.log(alpha / (mu + alpha + self.eps))
            + target * torch.log(mu / (mu + alpha + self.eps))
        )
        nll = -log_likelihood

        rel_error = torch.abs(target - mu) / (target + self.eps)
        element_loss = nll + self.beta * rel_error

        return element_loss.mean()


class FocalBCELoss(nn.Module):
    """
    Focal Binary Cross-Entropy for rare severity/threat flags.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        pred_probs: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        p = pred_probs.clamp(min=1e-7, max=1 - 1e-7)
        bce = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
        
        p_t = target * p + (1 - target) * (1 - p)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
            focal_weight = alpha_t * focal_weight

        return (focal_weight * bce).mean()


class HybridHeterogeneousLoss(nn.Module):
    """
    Complete multi-task loss combining all three heads with
    heterogeneity-aware spatial weighting.
    """

    def __init__(
        self,
        lambda_count: float = 1.0,
        lambda_admin: float = 0.5,
        lambda_threat: float = 0.8,
        beta_nb: float = 1.0,
        gamma_focal: float = 2.0,
        hetero_eps: float = 1e-6,
    ):
        super().__init__()
        self.lambda_count = lambda_count
        self.lambda_admin = lambda_admin
        self.lambda_threat = lambda_threat
        self.hetero_eps = hetero_eps

        self.count_loss_fn = ZeroInflatedNBRelativeLoss(beta=beta_nb)
        self.threat_loss_fn = FocalBCELoss(gamma=gamma_focal)

        self._hetero_weights: Optional[torch.Tensor] = None

    def compute_heterogeneity_weights(
        self,
        full_count_tensor: torch.Tensor,
        chunk_size: int = 1000,
        target_device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        T, N, C = full_count_tensor.shape
        M = T * C
        
        # Always compute accumulation on CPU to save CUDA VRAM
        sum_x = torch.zeros(N, dtype=torch.float64, device="cpu")
        sum_x2 = torch.zeros(N, dtype=torch.float64, device="cpu")
        
        for i in range(0, T, chunk_size):
            chunk = full_count_tensor[i:i + chunk_size].to(device="cpu", dtype=torch.float64)
            sum_x.add_(chunk.sum(dim=(0, 2)))
            sum_x2.add_((chunk ** 2).sum(dim=(0, 2)))
            
        grid_means = (sum_x / M).float()
        grid_vars = torch.clamp((sum_x2 / M) - (grid_means.double() ** 2), min=0.0).float()

        raw_weights = grid_vars / (grid_means + self.hetero_eps)
        normalized = raw_weights / (raw_weights.mean() + self.hetero_eps)

        device = target_device or full_count_tensor.device
        self._hetero_weights = normalized.to(device)
        print(f"[Loss] Heterogeneity weights computed: "
              f"min={normalized.min():.3f}, max={normalized.max():.3f}, "
              f"median={normalized.median():.3f}")
        return self._hetero_weights

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # ── Count Loss (with heterogeneity weighting) ────────────────────────
        count_loss_raw = self.count_loss_fn(
            mu=predictions["count_mu"],
            alpha=predictions["count_alpha"],
            target=targets["y_count"],
        )

        if self._hetero_weights is not None:
            w = self._hetero_weights.view(1, 1, -1, 1)
            
            # Apply clamp here too for the weighted recomputation
            mu = predictions["count_mu"].clamp(min=1e-6)
            alpha = predictions["count_alpha"].clamp(min=1e-6)
            
            nb_ll = -(
                torch.lgamma(targets["y_count"] + alpha)
                - torch.lgamma(alpha)
                - torch.lgamma(targets["y_count"] + 1)
                + alpha * torch.log(alpha / (mu + alpha + self.hetero_eps))
                + targets["y_count"] * torch.log(mu / (mu + alpha + self.hetero_eps))
            )
            rel_err = torch.abs(targets["y_count"] - mu) / (targets["y_count"] + self.hetero_eps)
            element_loss = nb_ll + self.count_loss_fn.beta * rel_err
            count_loss = (element_loss * w).mean()
        else:
            count_loss = count_loss_raw

        # ── Admin Loss ───────────────────────────────────────────────────────
        B, forecast_len, N, S_admin = predictions["admin_logits"].shape
        admin_logits_flat = predictions["admin_logits"].reshape(-1, S_admin)
        admin_target_flat = targets["y_admin"].reshape(-1, S_admin).argmax(dim=-1)
        admin_loss = F.cross_entropy(admin_logits_flat, admin_target_flat)

        # ── Threat Loss ──────────────────────────────────────────────────────
        threat_loss = self.threat_loss_fn(
            pred_probs=predictions["threat_probs"],
            target=targets["y_threat"],
        )

        # ── Weighted Combination ─────────────────────────────────────────────
        total_loss = (
            self.lambda_count * count_loss
            + self.lambda_admin * admin_loss
            + self.lambda_threat * threat_loss
        )

        return {
            "total_loss": total_loss,
            "count_loss": count_loss.detach(),
            "admin_loss": admin_loss.detach(),
            "threat_loss": threat_loss.detach(),
        }