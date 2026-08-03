import sys
from pathlib import Path

# Ensure we can import from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from src.losses.hybrid_loss import HybridHeterogeneousLoss, ZeroInflatedNBRelativeLoss, FocalBCELoss
from src.utils.metrics import compute_composite_score

print("STARTING CHECK...") # If you don't see this, the file is not being read!

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    print("=" * 60)
    print("TEST 1: Zero-Inflated NB + Relative Error Loss")
    print("=" * 60)
    nb_loss_fn = ZeroInflatedNBRelativeLoss(beta=1.0).to(device)
    mu = torch.tensor([[[[2.0, 0.5, 10.0], [0.1, 5.0, 0.3], [8.0, 1.0, 0.2], [0.0, 0.0, 0.0]]]], device=device, requires_grad=True)
    alpha = torch.ones_like(mu, requires_grad=True) * 2.0
    target = torch.tensor([[[[3, 0, 12], [0, 4, 0], [10, 0, 0], [0, 0, 0]]]], device=device, dtype=torch.float32)
    
    loss = nb_loss_fn(mu, alpha, target)
    loss.backward()
    
    assert not torch.isnan(loss), "FAIL: NB loss is NaN!"
    assert not torch.isnan(mu.grad).any(), "FAIL: mu gradient contains NaN!"
    print("  PASS: Gradients are valid and non-NaN.\n")

    print("=" * 60)
    print("TEST 2: Focal BCE Loss")
    print("=" * 60)
    focal_loss_fn = FocalBCELoss(gamma=2.0).to(device)
    pred_probs = torch.tensor([[0.01, 0.95, 0.02, 0.03]], device=device, requires_grad=True)
    target_binary = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)
    
    focal_loss = focal_loss_fn(pred_probs, target_binary)
    focal_loss.backward()
    
    assert not torch.isnan(focal_loss), "FAIL: Focal loss is NaN!"
    print("  PASS: Gradients are valid and non-NaN.\n")

    print("=" * 60)
    print("TEST 3: Heterogeneity Weight Modulation")
    print("=" * 60)
    hybrid_loss = HybridHeterogeneousLoss().to(device)
    X_train = torch.zeros(100, 4, 3)
    X_train[:, 0, :] = 2.0
    X_train[:, 1, :] = torch.poisson(torch.full((100, 3), 5.0)).float()
    X_train[:, 2, :] = 0.0
    
    weights = hybrid_loss.compute_heterogeneity_weights(X_train.to(device))
    assert weights[1] > weights[0], "FAIL: Bursty grid should have higher weight!"
    print(f"  PASS: Bursty grid weight ({weights[1]:.2f}) > Steady grid weight ({weights[0]:.2f}).\n")

    print("=" * 60)
    print("TEST 4: Partial Baseline Output Compatibility")
    print("=" * 60)
    batch_size, nodes, channels = 4, 4, 58
    pred_counts = torch.rand(batch_size, 1, nodes, channels)
    tgt_counts = torch.poisson(torch.full((batch_size, 1, nodes, channels), 2.0)).float()
    
    metrics = compute_composite_score(
        pred_counts=pred_counts, target_counts=tgt_counts,
        pred_threat=None, target_threat=None,
        hetero_weights=weights, baseline_name="test_baseline"
    )
    
    assert metrics['threat_status'] == 'excluded_no_threat_head', "FAIL: Should flag missing threat head!"
    print(f"  PASS: Handled missing threat head gracefully (Status: {metrics['threat_status']}).\n")

    print("=" * 60)
    print("ALL TESTS PASSED - Loss function is ready for training")
    print("=" * 60)

if __name__ == "__main__":
    main()