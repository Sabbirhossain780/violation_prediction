"""
scripts/evaluate.py
===================
UPDATED: Baseline-Aware Test-Set Evaluation for Violation Forecasting.

Changes from original:
- Added --model argument to evaluate our model or any registered baseline
- Uses BASELINE_REGISTRY from train.py for consistent model routing
- Enforces IDENTICAL temporal test split as training (no leakage)
- Routes through updated metrics.py with baseline_name tagging
- Saves results to outputs/baselines/{model_name}/test_results.json
- Reuses heterogeneity weights from training data only
"""

import sys
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import ViolationForecastDataset
from src.models.embeddings import ViolationEmbedding
from src.models.backbone import SpatioTemporalBackbone
from src.models.decoder import HybridDecoder
from src.losses.hybrid_loss import HybridHeterogeneousLoss
from src.utils.metrics import (
    nonzero_relative_error,
    heterogeneity_weighted_mae,
    threat_f1_score,
    admin_accuracy,
    zero_inflation_match,
    compute_composite_score,
)

# Import shared baseline registry from train.py for consistency
from scripts.train import BASELINE_REGISTRY, build_baseline_model, forward_pass


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def temporal_split(n_samples: int, val_ratio: float, test_ratio: float):
    """
    IDENTICAL temporal split function as in train.py.
    This MUST remain synchronized to guarantee fair comparison.
    """
    n_test = int(n_samples * test_ratio)
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_val - n_test
    test_idx = list(range(n_train + n_val, n_samples))
    return test_idx


def build_our_model_for_eval(cfg, metadata, adj_matrix, device):
    """Rebuild our model architecture for checkpoint loading."""
    embedding = ViolationEmbedding(
        n_channels=metadata["n_channels"], n_nodes=metadata["n_nodes"],
        d_f=cfg["model"]["d_f"], d_t=cfg["model"]["d_t"], d_s=cfg["model"]["d_s"],
        n_days=cfg["model"]["n_days"], n_quarters=cfg["model"]["n_quarters"],
        dropout=cfg["model"]["embedding_dropout"],
    ).to(device)
    backbone = SpatioTemporalBackbone(
        d_h=embedding.d_h, n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"], top_k_temporal=cfg["model"]["top_k_temporal"],
        top_k_spatial_ratio=cfg["model"]["top_k_spatial_ratio"],
        d_model=cfg["model"]["d_model"], adj_matrix=adj_matrix,
        dropout=cfg["model"]["backbone_dropout"],
    ).to(device)
    decoder = HybridDecoder(
        d_h=embedding.d_h, n_channels=metadata["n_channels"],
        n_admin=metadata["n_admin"], n_threat=metadata["n_threat"],
    ).to(device)
    return {"embedding": embedding, "backbone": backbone, "decoder": decoder}


@torch.no_grad()
def evaluate(model, loader, criterion, device, hetero_weights=None, model_name="ours"):
    """Full test-set evaluation with graceful handling of partial baseline outputs."""
    # Set eval mode
    if model_name == "ours":
        for m in model.values():
            m.eval()
    else:
        model.eval()

    total_losses = {"total_loss": 0, "count_loss": 0, "admin_loss": 0, "threat_loss": 0}
    all_preds_count, all_targets_count = [], []
    all_preds_threat, all_targets_threat = [], []
    all_preds_admin, all_targets_admin = [], []
    n_batches = 0

    for batch in tqdm(loader, desc=f"Evaluating [{model_name}]", leave=False):
        targets = {
            "y_count": batch["y_count"].to(device),
            "y_admin": batch["y_admin"].to(device),
            "y_threat": batch["y_threat"].to(device),
        }

        preds = forward_pass(model, batch, device, model_name)
        loss_dict = criterion(preds, targets)
        for k in total_losses:
            total_losses[k] += loss_dict[k].item()
        n_batches += 1

        # Collect predictions — count always available
        all_preds_count.append(preds["count_mu"].cpu())
        all_targets_count.append(targets["y_count"].cpu())

        # UPDATED: Safely collect optional heads for baselines
        if preds.get("threat_probs") is not None:
            all_preds_threat.append(preds["threat_probs"].cpu())
            all_targets_threat.append(targets["y_threat"].cpu())

        if preds.get("admin_logits") is not None:
            all_preds_admin.append(preds["admin_logits"].cpu())
            all_targets_admin.append(targets["y_admin"].cpu())

    avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}

    # Concatenate batches
    pred_counts = torch.cat(all_preds_count, dim=0)
    tgt_counts = torch.cat(all_targets_count, dim=0)

    pred_threat = torch.cat(all_preds_threat, dim=0) if all_preds_threat else None
    tgt_threat = torch.cat(all_targets_threat, dim=0) if all_targets_threat else None

    pred_admin = torch.cat(all_preds_admin, dim=0) if all_preds_admin else None
    tgt_admin = torch.cat(all_targets_admin, dim=0) if all_targets_admin else None

    # Compute all metrics via updated metrics.py (handles None gracefully)
    nz_rel = nonzero_relative_error(pred_counts, tgt_counts)
    hw_mae = heterogeneity_weighted_mae(pred_counts, tgt_counts, hetero_weights) if hetero_weights is not None else float("nan")
    threat_metrics = threat_f1_score(pred_threat, tgt_threat)
    admin_acc = admin_accuracy(pred_admin, tgt_admin)
    zi_match = zero_inflation_match(pred_counts, tgt_counts)
    composite = compute_composite_score(
        pred_counts, tgt_counts, pred_threat, tgt_threat,
        hetero_weights, baseline_name=model_name,
    )

    results = {
        "model": model_name,
        "losses": avg_losses,
        "nonzero_relative_error": nz_rel,
        "heterogeneity_weighted_mae": hw_mae,
        "admin_accuracy": admin_acc,
        "zero_inflation_match": zi_match,
        "composite_score": composite["composite_score"],
        "threat_macro_f1": threat_metrics["macro_f1"],
        "threat_status": threat_metrics.get("note", "included"),
        "threat_per_flag_f1": {k: v for k, v in threat_metrics.items() if k.startswith("threat_f1_flag_")},
    }
    return results


def main(config_path: str = None, checkpoint_path: str = None, model_name: str = "ours"):
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "base_config.yaml"
    cfg = load_config(str(config_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device} | Model: {model_name}")

    metadata = torch.load(f"{cfg['data']['processed_dir']}/metadata.pt")
    adj_matrix = torch.load(f"{cfg['data']['processed_dir']}/adj_matrix.pt").to(device)

    # Build model
    if model_name == "ours":
        model = build_our_model_for_eval(cfg, metadata, adj_matrix, device)
    else:
        model = build_baseline_model(model_name, cfg, metadata, adj_matrix, device)

    # Load checkpoint
    if checkpoint_path is None:
        if model_name == "ours":
            exp_name = cfg["experiment"]["name"]
            ckpt_dir = PROJECT_ROOT / "outputs" / exp_name
        else:
            ckpt_dir = PROJECT_ROOT / "outputs" / f"baselines/{model_name}"
        checkpoint_path = ckpt_dir / "best_model.pt"

    ckpt = torch.load(checkpoint_path, map_location=device)
    if model_name == "ours":
        model["embedding"].load_state_dict(ckpt["embedding_state"])
        model["backbone"].load_state_dict(ckpt["backbone_state"])
        model["decoder"].load_state_dict(ckpt["decoder_state"])
    else:
        model.load_state_dict(ckpt["model_state"])
    print(f"[INFO] Loaded checkpoint: epoch={ckpt['epoch']} composite={ckpt['composite_score']:.4f}")

    # IDENTICAL temporal test split as training
    full_dataset = ViolationForecastDataset(
        data_dir=cfg["data"]["processed_dir"],
        hist_steps=cfg["data"]["hist_steps"],
        forecast_steps=cfg["data"]["forecast_steps"],
    )
    test_idx = temporal_split(
        len(full_dataset), cfg["training"]["val_ratio"], cfg["training"]["test_ratio"],
    )
    test_ds = Subset(full_dataset, test_idx)
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=2)
    print(f"[INFO] Test set: {len(test_ds)} samples (strict temporal split)")

    # Reconstruct loss with heterogeneity weights from TRAINING data only
    criterion = HybridHeterogeneousLoss(
        lambda_count=cfg["loss"]["lambda_count"],
        lambda_admin=cfg["loss"]["lambda_admin"],
        lambda_threat=cfg["loss"]["lambda_threat"],
        beta_nb=cfg["loss"]["beta_nb"],
        gamma_focal=cfg["loss"]["gamma_focal"],
        hetero_eps=cfg["loss"]["hetero_eps"],
    ).to(device)

    X_full = torch.load(f"{cfg['data']['processed_dir']}/X.pt")
    n_total = len(full_dataset)
    n_test = int(n_total * cfg["training"]["test_ratio"])
    n_val = int(n_total * cfg["training"]["val_ratio"])
    n_train = n_total - n_val - n_test
    train_bins = n_train + cfg["data"]["hist_steps"] + cfg["data"]["forecast_steps"]
    X_train = X_full[:train_bins]
    criterion.compute_heterogeneity_weights(X_train.to(device))

    # Run evaluation
    print("\n" + "=" * 60)
    print(f"TEST SET EVALUATION [{model_name}]")
    print("=" * 60)
    results = evaluate(model, test_loader, criterion, device, criterion._hetero_weights, model_name)

    # Print formatted results
    print(f"\n{'Metric':<35} {'Value':>10}")
    print("-" * 47)
    print(f"{'Composite Score':<35} {results['composite_score']:>10.4f}")
    print(f"{'Non-Zero Relative Error':<35} {results['nonzero_relative_error']:>10.4f}")
    print(f"{'Heterogeneity-Weighted MAE':<35} {results['heterogeneity_weighted_mae']:>10.4f}")
    print(f"{'Threat Macro F1':<35} {results['threat_macro_f1']:>10.4f}")
    print(f"{'Admin Disposition Accuracy':<35} {results['admin_accuracy']:>10.4f}")
    print(f"{'Zero-Inflation Match':<35} {results['zero_inflation_match']:>10.4f}")
    print(f"{'Threat Status':<35} {results['threat_status']:>10}")
    print(f"\n{'Loss Component':<35} {'Value':>10}")
    print("-" * 47)
    for k, v in results["losses"].items():
        print(f"{k:<35} {v:>10.4f}")

    # Save results to model-specific directory
    if model_name == "ours":
        output_dir = Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["name"]
    else:
        output_dir = Path(cfg["experiment"]["output_dir"]) / f"baselines/{model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Results saved to {results_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate violation forecasting model on test set")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--model", type=str, default="ours",
                        choices=["ours"] + list(BASELINE_REGISTRY.keys()),
                        help="Model to evaluate: 'ours' or any registered baseline name")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.model)