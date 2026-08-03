"""
src/utils/metrics.py
====================
Custom Evaluation Metrics for Sparse Multivariate Violation Forecasting.

UPDATED: Added graceful handling for partial baseline outputs.
Tier 1 baselines (MiST, ZINB-GNN) may only produce count predictions
without admin/threat heads. All functions now accept None gracefully.
"""

import torch
import numpy as np
from typing import Dict, Optional


def nonzero_relative_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    Mean Relative Error computed ONLY on non-zero target elements.
    Works identically for our model and all baselines since all produce counts.
    Returns NaN if no non-zero targets exist in the batch.
    """
    mask = target > 0
    if mask.sum() == 0:
        return float('nan')
    rel_err = torch.abs(pred[mask] - target[mask]) / (target[mask] + eps)
    return rel_err.mean().item()


def heterogeneity_weighted_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    """
    MAE weighted by per-node heterogeneity scores (Fano factor).
    Works identically for our model and all baselines.
    """
    mae = torch.abs(pred - target)
    w = weights.view(1, 1, -1, 1)
    return (mae * w).mean().item()


def threat_f1_score(
    pred_probs: Optional[torch.Tensor],  # UPDATED: Now accepts None
    target: Optional[torch.Tensor],     # UPDATED: Now accepts None
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    UPDATED: Returns zero F1 when pred/target are None (baseline doesn't
    produce threat head). Prevents crashes during baseline evaluation.
    """
    # UPDATED: Graceful handling for baselines without threat head
    if pred_probs is None or target is None:
        return {"macro_f1": 0.0, "note": "threat_head_not_available"}

    pred_binary = (pred_probs >= threshold).float()
    pb = pred_binary.reshape(-1, pred_binary.shape[-1])
    tb = target.reshape(-1, target.shape[-1])

    tp = (pb * tb).sum(dim=0)
    fp = (pb * (1 - tb)).sum(dim=0)
    fn = ((1 - pb) * tb).sum(dim=0)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    result = {f"threat_f1_flag_{i}": f1[i].item() for i in range(f1.shape[0])}
    result["macro_f1"] = f1.mean().item()
    return result


def admin_accuracy(
    pred_logits: Optional[torch.Tensor],  # UPDATED: Now accepts None
    target_probs: Optional[torch.Tensor], # UPDATED: Now accepts None
) -> float:
    """
    UPDATED: Returns 0.0 when pred/target are None (baseline doesn't
    produce admin head). Prevents crashes during baseline evaluation.
    """
    # UPDATED: Graceful handling for baselines without admin head
    if pred_logits is None or target_probs is None:
        return 0.0

    pred_class = pred_logits.argmax(dim=-1)
    target_class = target_probs.argmax(dim=-1)
    return (pred_class == target_class).float().mean().item()


def zero_inflation_match(
    pred: torch.Tensor,
    target: torch.Tensor,
    tolerance: float = 0.05,
) -> float:
    """Unchanged — works for all models that produce count outputs."""
    pred_zero_rate = (pred < 0.5).float().mean()
    true_zero_rate = (target == 0).float().mean()
    match = torch.abs(pred_zero_rate - true_zero_rate) <= tolerance
    return match.float().item()


def compute_composite_score(
    pred_counts: torch.Tensor,
    target_counts: torch.Tensor,
    pred_threat: Optional[torch.Tensor] = None,   # UPDATED: Optional
    target_threat: Optional[torch.Tensor] = None, # UPDATED: Optional
    hetero_weights: Optional[torch.Tensor] = None,
    baseline_name: Optional[str] = None,          # UPDATED: Tag for logging
) -> Dict[str, float]:
    """
    UPDATED: Handles partial baseline outputs gracefully.

    When pred_threat/target_threat are None (Tier 1 baselines like MiST,
    ZINB-GNN), the threat F1 component is set to 0.0 rather than causing
    a crash. The composite score is still computable using count metrics.

    Added baseline_name parameter for clear logging of which metrics
    are applicable vs. skipped for each model.
    """
    nz_rel = nonzero_relative_error(pred_counts, target_counts)

    # UPDATED: Safe threat metric computation
    threat_metrics = threat_f1_score(pred_threat, target_threat)

    if hetero_weights is not None:
        hw_mae = heterogeneity_weighted_mae(pred_counts, target_counts, hetero_weights)
    else:
        hw_mae = torch.abs(pred_counts - target_counts).mean().item()

    # Composite scoring with graceful degradation
    nz_rel_safe = nz_rel if not np.isnan(nz_rel) else 1.0

    # UPDATED: Adjust weighting when threat head is unavailable
    if pred_threat is None:
        # Baseline without threat head: redistribute weight to count metrics
        composite = (
            0.5 * (1.0 / (1.0 + nz_rel_safe))
            + 0.5 * (1.0 / (1.0 + hw_mae))
        )
        threat_note = "excluded_no_threat_head"
    else:
        # Full model or adapted baseline with threat head
        composite = (
            0.4 * threat_metrics["macro_f1"]
            + 0.3 * (1.0 / (1.0 + nz_rel_safe))
            + 0.3 * (1.0 / (1.0 + hw_mae))
        )
        threat_note = "included"

    result = {
        "nonzero_rel_error": nz_rel,
        "hetero_weighted_mae": hw_mae,
        "threat_macro_f1": threat_metrics["macro_f1"],
        "composite_score": composite,
        "threat_status": threat_note,
    }

    # UPDATED: Tag results with baseline name for clear experiment tracking
    if baseline_name is not None:
        result["model"] = baseline_name

    return result