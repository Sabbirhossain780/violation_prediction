# Multivariate Spatio-Temporal Traffic Violation Forecaster

A novel spatio-temporal forecasting framework for predicting traffic violation counts, administrative dispositions, and severity/threat flags across 1km spatial grids using Montgomery County, MD police stop data.

Built on conceptual principles from TopKNet (CIKM 2025), but fundamentally redesigned for sparse, zero-inflated, multivariate event data rather than continuous traffic flow.

## Problem Statement

Traditional traffic forecasting models predict continuous variables (speed/flow) on dense road networks. This project solves a fundamentally different problem:

-   Sparse Events: 90%+ of grid-time cells contain zero violations
-   Multivariate Categories: 58 distinct AI-classified violation types as interacting channels
-   Multi-Task Output: Simultaneously predicts counts, admin dispositions (Warning/Citation/ESERO/SERO), and threat flags (Arrest/Alcohol/Fatal/etc.)
-   Bursty Temporal Patterns: Violations cluster in time (weekend nights, rush hours) with long empty gaps

## Architecture Overview

```text
Input: [H=4, N~600, C=58] violation counts + temporal metadata
                    |
        +-----------v------------+
        |   Sparse Categorical   | <-- Weighted channel embeddings
        |      Embedding         |     (captures cross-violation semantics)
        +-----------+------------+
                    |
        +-----------v------------+
        |  Time-Aware Spatial    | <-- Hadamard modulation of grid identity
        |  Identity Embedding    |     by 6-hour quarter
        +-----------+------------+
                    |
    +---------------v----------------+
    |   Event-Skip Temporal Attention| <-- Skips empty bins; attends only to
    |   (TopK + Event Presence Mask) |     historically active time steps
    +---------------+----------------+
                    |
    +---------------v----------------+
    |   TopK GCN                     | <-- Dynamic spatial pruning + Virtual
    |   (Distance + Virtual Node)    |     Sinkhole Node for (0,0) events
    +---------------+----------------+
                    |
    +---------------v----------------+
    |   Gated MLP                    | <-- Dual-branch feature interaction
    +---------------+----------------+
                    | x L layers
        +-----------v------------+
        |   Hybrid 3-Head Decoder|
        +------------------------+
        | Count: NB(mu, alpha)   | --> Expected violation counts
        | Admin: Softmax(4)      | --> Disposition probabilities
        | Threat: Sigmoid(S)     | --> Severity flag probabilities
        +------------------------+
                    |
Output: [F=1, N~600, 58+4+S] multi-task predictions



Project Structure:

violation_forecast/
├── configs/base_config.yaml       # All hyperparameters & baseline configs
├── data/
│   ├── raw/                       # Original datasets (gitignored)
│   └── processed/                 # Generated tensors (.pt files)
├── src/
│   ├── preprocessing.py           # GPS cleaning, gridding, binning, aggregation
│   ├── dataset.py                 # PyTorch Dataset with sliding windows
│   ├── models/
│   │   ├── embeddings.py          # Sparse categorical + time-aware spatial
│   │   ├── backbone.py            # Event-Skip Attention + TopK GCN
│   │   └── decoder.py             # Hybrid 3-head decoder
│   ├── losses/
│   │   └── hybrid_loss.py         # ZINB + RelError + Focal BCE + Hetero weights
│   └── utils/
│       └── metrics.py             # Custom eval metrics (partial output safe)
├── baselines/                     # Domain-appropriate comparison models
│   ├── base_baseline.py           # Shared interface enforcement
│   ├── deepcrime.py               # Tier 1: GNN + zero-inflation
│   ├── st_shine.py                # Tier 1: Hierarchical attention
│   ├── mist.py                    # Tier 1: Neural Hawkes process
│   ├── zinb_gnn.py                # Tier 1: Native ZINB output
│   ├── stgcn_adapted.py           # Tier 2: STGCN w/ shared adaptations
│   └── informer_adapted.py        # Tier 2: Pure temporal transformer
├── scripts/
│   ├── preprocess.py              # CLI: data forging pipeline
│   ├── train.py                   # CLI: baseline-aware training loop
│   └── evaluate.py                # CLI: test-set evaluation
├── notebooks/
│   └── 02_loss_gradient_check.ipynb  # Loss sanity verification
└── outputs/                       # Checkpoints, logs, results (gitignored)



QUICK START:
Prerequisites:
  pip install torch numpy pandas openpyxl tqdm pyyaml


Step-1:Preprocess Data
python scripts/preprocess.py \
    --raw-data data/raw/demo_data.xlsx \
    --vlookup data/raw/violation_categories_gpt-5.4.xlsx \
    --output-dir data/processed


Step 2: Verify Loss Function (Recommended Before Training)
  jupyter notebook notebooks/02_loss_gradient_check.ipynb



Step 3: Train Model
# Train our model
python scripts/train.py --model ours

# Train a baseline
python scripts/train.py --model deepcrime
python scripts/train.py --model stgcn_adapted


Step 4: Evaluate
# Evaluate our model
python scripts/evaluate.py --model ours

# Evaluate a baseline
python scripts/evaluate.py --model mist





Baseline Framework:
| Decision | Choice | Rationale |
| :--- | :--- | :--- |
| Spatial Unit | 1km grid + Virtual Node | Goldilocks zone for GNN; handles missing GPS |
| Temporal Bin | 6-hour quarters | Avoids 97% zero tensor at 1-hr resolution |
| Input Channels | 58 AI-classified types | Multivariate interaction via shared backbone |
| Output | Hybrid 3-head | Real-world dispatch utility; MTL synergy |
| Loss | ZINB + RelError + Focal | Value accuracy > non-zero detection |
| Base Paper Role | Conceptual blueprint only | Novel math for sparse event data |



Evaluation Metrics
- Standard MAE/RMSE are misleading on zero-inflated data. We use:
- Non-Zero Relative Error: Value accuracy on active bins only
- Heterogeneity-Weighted MAE: Prioritizes surge-prone grids
- Threat Macro F1: Detection quality for rare severe events
- Composite Score: Weighted combination for early stopping



@article{violation_forecast_2026,
  title={Multivariate Spatio-Temporal Traffic Violation Forecasting with Event-Skip Attention and Zero-Inflated Multi-Task Learning},
  year={2026},
  note={Code available at [repository URL]}
}