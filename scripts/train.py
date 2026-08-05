"""
scripts/train.py
================
UPDATED: Baseline-Aware Training Loop for Violation Forecasting.

Changes from original:
- Added --model argument to select between 'ours' and baseline models
- Added baseline model registry with dynamic import and instantiation
- Applied shared_adaptations (Virtual Node, loss replacement) to Tier 2 baselines
- Routes all models through updated metrics.py with baseline_name tagging
- Saves baseline checkpoints to separate subdirectories
- Identical temporal split enforced for ALL models
"""

import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import importlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import ViolationForecastDataset
from src.models.embeddings import ViolationEmbedding
from src.models.backbone import SpatioTemporalBackbone
from src.models.decoder import HybridDecoder
from src.losses.hybrid_loss import HybridHeterogeneousLoss
from src.utils.metrics import compute_composite_score


# =============================================================================
# BASELINE MODEL REGISTRY
# =============================================================================
# Maps config baseline names to their module paths and class names.
# Each baseline module must expose a class with:
#   __init__(self, n_nodes, n_channels, adj_matrix, **params)
#   forward(self, x, tim) -> dict with at least 'count_mu' key
BASELINE_REGISTRY = {
    # Tier 1: Event/Crime Forecasting
    "deepcrime": {"module": "baselines.deepcrime", "class": "DeepCrime"},
    "st_shine": {"module": "baselines.st_shine", "class": "STSHINE"},
    "mist": {"module": "baselines.mist", "class": "MiST"},
    "zinb_gnn": {"module": "baselines.zinb_gnn", "class": "ZINBGNN"},
    # Tier 2: Adapted Traffic Flow
    "stgcn_adapted": {"module": "baselines.stgcn_adapted", "class": "STGCNAdapted"},
    "informer_adapted": {"module": "baselines.informer_adapted", "class": "InformerAdapted"},
}


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def temporal_split(n_samples: int, val_ratio: float, test_ratio: float):
    """Strict temporal split — identical across ALL models for fair comparison."""
    n_test = int(n_samples * test_ratio)
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_val - n_test
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n_train + n_val))
    test_idx = list(range(n_train + n_val, n_samples))
    print(f"[Split] Train: {n_train} | Val: {n_val} | Test: {n_test}")
    return train_idx, val_idx, test_idx


def build_our_model(cfg, metadata, adj_matrix, device):
    """Build our hybrid multi-task model."""
    embedding = ViolationEmbedding(
        n_channels=metadata['n_channels'], n_nodes=metadata['n_nodes'],
        d_f=cfg['model']['d_f'], d_t=cfg['model']['d_t'], d_s=cfg['model']['d_s'],
        n_days=cfg['model']['n_days'], n_quarters=cfg['model']['n_quarters'],
        dropout=cfg['model']['embedding_dropout'],
    ).to(device)
    backbone = SpatioTemporalBackbone(
        d_h=embedding.d_h, n_layers=cfg['model']['n_layers'],
        n_heads=cfg['model']['n_heads'], top_k_temporal=cfg['model']['top_k_temporal'],
        top_k_spatial_ratio=cfg['model']['top_k_spatial_ratio'],
        d_model=cfg['model']['d_model'], adj_matrix=adj_matrix,
        dropout=cfg['model']['backbone_dropout'],
    ).to(device)
    decoder = HybridDecoder(
        d_h=embedding.d_h, n_channels=metadata['n_channels'],
        n_admin=metadata['n_admin'], n_threat=metadata['n_threat'],
    ).to(device)
    return {'embedding': embedding, 'backbone': backbone, 'decoder': decoder}


def build_baseline_model(model_name, cfg, metadata, adj_matrix, device):
    """
    Dynamically import and instantiate a baseline model.
    Applies shared_adaptations for Tier 2 models automatically.
    """
    if model_name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {model_name}. Available: {list(BASELINE_REGISTRY.keys())}")

    entry = BASELINE_REGISTRY[model_name]
    module = importlib.import_module(entry["module"])
    model_class = getattr(module, entry["class"])

    # Get baseline-specific params from config
    tier1_cfg = cfg.get('baselines', {}).get('tier1_event_models', {})
    tier2_cfg = cfg.get('baselines', {}).get('tier2_adapted_traffic', {})

    if model_name in tier1_cfg:
        params = tier1_cfg[model_name].get('params', {})
    elif model_name in tier2_cfg:
        params = tier2_cfg[model_name].get('params', {})
        # Inject shared adaptation flags for Tier 2
        shared = tier2_cfg.get('shared_adaptations', {})
        params['virtual_sinkhole_node'] = shared.get('virtual_sinkhole_node', True)
        params['loss_adaptation'] = shared.get('loss_replacement', 'hybrid')
    else:
        params = {}

    model = model_class(
        n_nodes=metadata['n_nodes'],
        n_channels=metadata['n_channels'],
        adj_matrix=adj_matrix,
        **params,
    ).to(device)

    print(f"[INFO] Built baseline: {model_name} ({entry['module']}.{entry['class']})")
    return model


def forward_pass(model, batch, device, model_name='ours'):
    """
    Unified forward pass dispatcher.
    Our model uses dict-of-modules; baselines use single nn.Module.
    All outputs normalized to dict with at least 'count_mu' key.
    """
    x = batch['x'].to(device)
    tim_hist = batch['tim_hist'].to(device)

    if model_name == 'ours':
        emb = model['embedding'](x, tim_hist)
        z = model['backbone'](emb, tim_hist, x)
        forecast_steps = batch['y_count'].shape[1]
        z_forecast = z[:, -forecast_steps:]
        preds = model['decoder'](z_forecast)
    else:
        # Baseline models have unified forward(x, tim) interface
        preds = model(x, tim_hist)

    return preds


def get_all_parameters(model, model_name='ours'):
    """Extract parameters from either dict-of-modules or single nn.Module."""
    if model_name == 'ours':
        return list(model['embedding'].parameters()) + \
               list(model['backbone'].parameters()) + \
               list(model['decoder'].parameters())
    else:
        return list(model.parameters())


def clip_gradients(model, model_name, max_norm):
    """Clip gradients for either dict-of-modules or single nn.Module."""
    if model_name == 'ours':
        for key in ['embedding', 'backbone', 'decoder']:
            torch.nn.utils.clip_grad_norm_(model[key].parameters(), max_norm)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip_norm, model_name='ours'):
    model_module = model if model_name == 'ours' else model
    if model_name == 'ours':
        for m in model.values():
            m.train()
    else:
        model.train()

    total_losses = {'total_loss': 0, 'count_loss': 0, 'admin_loss': 0, 'threat_loss': 0}
    n_batches = 0

    pbar = tqdm(loader, desc=f"Training [{model_name}]", leave=False)
    for batch in pbar:
        targets = {
            'y_count': batch['y_count'].to(device),
            'y_admin': batch['y_admin'].to(device),
            'y_threat': batch['y_threat'].to(device),
        }

        preds = forward_pass(model, batch, device, model_name)
        loss_dict = criterion(preds, targets)

        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        clip_gradients(model, model_name, grad_clip_norm)
        optimizer.step()

        for k in total_losses:
            total_losses[k] += loss_dict[k].item()
        n_batches += 1
        pbar.set_postfix({k: f"{v / n_batches:.4f}" for k, v in total_losses.items()})

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, hetero_weights=None, model_name='ours'):
    if model_name == 'ours':
        for m in model.values():
            m.eval()
    else:
        model.eval()

    total_losses = {'total_loss': 0, 'count_loss': 0, 'admin_loss': 0, 'threat_loss': 0}
    all_preds_count, all_targets_count = [], []
    all_preds_threat, all_targets_threat = [], []
    n_batches = 0

    for batch in tqdm(loader, desc=f"Validating [{model_name}]", leave=False):
        targets = {
            'y_count': batch['y_count'].to(device),
            'y_admin': batch['y_admin'].to(device),
            'y_threat': batch['y_threat'].to(device),
        }

        preds = forward_pass(model, batch, device, model_name)
        loss_dict = criterion(preds, targets)
        for k in total_losses:
            total_losses[k] += loss_dict[k].item()
        n_batches += 1

        all_preds_count.append(preds['count_mu'].cpu())
        all_targets_count.append(targets['y_count'].cpu())
        # UPDATED: Safely collect threat predictions (may be None for baselines)
        threat_pred = preds.get('threat_probs', None)
        if threat_pred is not None:
            all_preds_threat.append(threat_pred.cpu())
            all_targets_threat.append(targets['y_threat'].cpu())

    avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}

    pred_counts = torch.cat(all_preds_count, dim=0)
    tgt_counts = torch.cat(all_targets_count, dim=0)

    # UPDATED: Pass None safely for baselines without threat head
    if len(all_preds_threat) > 0:
        pred_threat = torch.cat(all_preds_threat, dim=0)
        tgt_threat = torch.cat(all_targets_threat, dim=0)
    else:
        pred_threat = None
        tgt_threat = None

    # UPDATED: Tag metrics with baseline_name for experiment tracking
    metrics = compute_composite_score(
        pred_counts, tgt_counts, pred_threat, tgt_threat,
        hetero_weights, baseline_name=model_name,
    )
    return avg_losses, metrics


def main(config_path: str = None, model_name: str = 'ours'):
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "base_config.yaml"
    cfg = load_config(str(config_path))

    torch.manual_seed(cfg['experiment']['seed'])
    np.random.seed(cfg['experiment']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device} | Model: {model_name}")

    metadata = torch.load(f"{cfg['data']['processed_dir']}/metadata.pt", weights_only=False)
    if 'n_admin' not in metadata:
        metadata['n_admin'] = len(cfg['data']['admin_columns'])
    if 'n_threat' not in metadata:
        metadata['n_threat'] = len(cfg['data']['threat_columns'])
    adj_matrix = torch.load(f"{cfg['data']['processed_dir']}/adj_matrix.pt", weights_only=False).to(device)

    # Create datasets with IDENTICAL temporal split for all models
    full_dataset = ViolationForecastDataset(
        data_dir=cfg['data']['processed_dir'],
        hist_steps=cfg['data']['hist_steps'],
        forecast_steps=cfg['data']['forecast_steps'],
    )
    train_idx, val_idx, _ = temporal_split(
        len(full_dataset), cfg['training']['val_ratio'], cfg['training']['test_ratio'],
    )
    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)
    num_workers = cfg['training'].get('num_workers', 0)
    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=num_workers)

    # Build model (ours or baseline)
    if model_name == 'ours':
        model = build_our_model(cfg, metadata, adj_matrix, device)
    else:
        model = build_baseline_model(model_name, cfg, metadata, adj_matrix, device)

    # Precompute heterogeneity weights from training data only
    X_full = torch.load(f"{cfg['data']['processed_dir']}/X.pt", weights_only=False)
    train_bins = train_idx[-1] + cfg['data']['hist_steps'] + cfg['data']['forecast_steps']
    X_train = X_full[:train_bins]

    criterion = HybridHeterogeneousLoss(
        lambda_count=cfg['loss']['lambda_count'],
        lambda_admin=cfg['loss']['lambda_admin'],
        lambda_threat=cfg['loss']['lambda_threat'],
        beta_nb=cfg['loss']['beta_nb'],
        gamma_focal=cfg['loss']['gamma_focal'],
        hetero_eps=cfg['loss']['hetero_eps'],
    ).to(device)
    criterion.compute_heterogeneity_weights(X_train, target_device=device)

    # Optimizer & Scheduler
    all_params = get_all_parameters(model, model_name)
    optimizer = torch.optim.AdamW(all_params, lr=cfg['training']['learning_rate'],
                                   weight_decay=cfg['training']['weight_decay'])

    warmup_epochs = cfg['training']['warmup_epochs']
    max_epochs = cfg['training']['max_epochs']
    min_lr = cfg['training']['min_lr']

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
        cosine = 0.5 * (1 + np.cos(np.pi * progress))
        return max(cosine, min_lr / cfg['training']['learning_rate'])

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Output directory: separate subdirs for baselines
    if model_name == 'ours':
        output_dir = Path(cfg['experiment']['output_dir']) / cfg['experiment']['name']
    else:
        output_dir = Path(cfg['experiment']['output_dir']) / f"baselines/{model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_composite = -float('inf')
    patience_counter = 0
    history = []

    print(f"\n{'='*60}")
    print(f"Training [{model_name}]: {max_epochs} epochs | Warmup: {warmup_epochs}")
    print(f"{'='*60}\n")

    for epoch in range(max_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{max_epochs} | LR: {current_lr:.6f}")

        train_losses = train_one_epoch(model, train_loader, criterion, optimizer,
                                        device, cfg['training']['grad_clip_norm'], model_name)
        val_losses, val_metrics = validate(model, val_loader, criterion, device,
                                            criterion._hetero_weights, model_name)
        scheduler.step()

        record = {'epoch': epoch + 1, 'lr': current_lr, 'model': model_name,
                  **{f'train_{k}': v for k, v in train_losses.items()},
                  **{f'val_{k}': v for k, v in val_losses.items()},
                  **{f'val_{k}': v for k, v in val_metrics.items()}}
        history.append(record)

        composite = val_metrics['composite_score']
        print(f"  Composite: {composite:.4f} | NZ-RelErr: {val_metrics['nonzero_rel_error']:.4f} | "
              f"HW-MAE: {val_metrics['hetero_weighted_mae']:.4f} | Threat-F1: {val_metrics['threat_macro_f1']:.4f}")

        if composite > best_composite:
            best_composite = composite
            patience_counter = 0
            ckpt = {'epoch': epoch + 1, 'composite_score': composite,
                    'config': cfg, 'model_name': model_name}
            if model_name == 'ours':
                ckpt.update({'embedding_state': model['embedding'].state_dict(),
                             'backbone_state': model['backbone'].state_dict(),
                             'decoder_state': model['decoder'].state_dict()})
            else:
                ckpt['model_state'] = model.state_dict()
            ckpt['optimizer_state'] = optimizer.state_dict()
            torch.save(ckpt, output_dir / "best_model.pt")
            print(f"  New best (composite={composite:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg['training']['patience']:
                print(f"\n Early stopping at epoch {epoch+1}. Best: {best_composite:.4f}")
                break

    import json
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n[DONE] Results saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train violation forecasting model or baseline")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default="ours",
                        choices=["ours"] + list(BASELINE_REGISTRY.keys()),
                        help="Model to train: 'ours' or any registered baseline name")
    args = parser.parse_args()
    main(args.config, args.model)