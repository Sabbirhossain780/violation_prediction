# =============================================================================
# VIOLATION FORECAST PIPELINE CONTROLLER
# Usage: make <target> [CONFIG=path/to/config.yaml] [MODEL=model_name]
# =============================================================================

.PHONY: help setup preprocess check train evaluate baseline clean

# Default config and model (override via CLI: make train CONFIG=custom.yaml)
CONFIG ?= configs/base_config.yaml
MODEL  ?= ours

help: ## Show this help message
	@echo "=== Violation Forecast Pipeline ==="
	@echo ""
	@echo "Setup:"
	@echo "  make setup              Install dependencies from requirements.txt"
	@echo ""
	@echo "Data:"
	@echo "  make preprocess         Run data forging pipeline ($(CONFIG))"
	@echo ""
	@echo "Validation:"
	@echo "  make check              Open loss gradient check notebook"
	@echo ""
	@echo "Training:"
	@echo "  make train              Train our model"
	@echo "  make baseline MODEL=x   Train baseline (deepcrime|st_shine|mist|zinb_gnn|stgcn_adapted|informer_adapted)"
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate           Evaluate our model on test set"
	@echo "  make eval-baseline MODEL=x  Evaluate a baseline on test set"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean              Remove all generated outputs and processed data"
	@echo ""
	@echo "Examples:"
	@echo "  make train CONFIG=configs/experiments/ablation_no_event_skip.yaml"
	@echo "  make baseline MODEL=deepcrime"
	@echo "  make eval-baseline MODEL=mist"

setup: ## Install all dependencies
	pip install -r requirements.txt
	@echo "[DONE] Dependencies installed."

preprocess: ## Run data forging pipeline
	@if [ ! -f "$(CONFIG)" ]; then echo "ERROR: Config not found: $(CONFIG)"; exit 1; fi
	python scripts/preprocess.py --config $(CONFIG)
	@echo "[DONE] Preprocessing complete."

check: ## Open loss gradient sanity check notebook
	jupyter notebook notebooks/02_loss_gradient_check.ipynb

train: ## Train our model or a specific model via MODEL variable
	@if [ ! -f "$(CONFIG)" ]; then echo "ERROR: Config not found: $(CONFIG)"; exit 1; fi
	python scripts/train.py --config $(CONFIG) --model $(MODEL)

baseline: ## Train a baseline model (usage: make baseline MODEL=deepcrime)
	@if [ "$(MODEL)" = "ours" ]; then echo "ERROR: Use 'make train' for our model. Use 'make baseline MODEL=<name>' for baselines."; exit 1; fi
	$(MAKE) train MODEL=$(MODEL)

evaluate: ## Evaluate our model on test set
	python scripts/evaluate.py --config $(CONFIG) --model $(MODEL)

eval-baseline: ## Evaluate a baseline (usage: make eval-baseline MODEL=mist)
	@if [ "$(MODEL)" = "ours" ]; then echo "ERROR: Use 'make evaluate' for our model."; exit 1; fi
	$(MAKE) evaluate MODEL=$(MODEL)

clean: ## Remove all generated artifacts
	rm -rf data/processed/*.pt
	rm -rf outputs/*/best_model.pt
	rm -rf outputs/*/training_history.json
	rm -rf outputs/*/test_results.json
	rm -rf outputs/baselines/
	rm -rf outputs/logs/
	@echo "[DONE] All generated outputs cleaned."





    Standard Execution Workflow
Follow this exact sequence for any experiment:
# 1. First-time setup
make setup

# 2. Forge tensors from raw data
make preprocess

# 3. Validate loss function BEFORE training (critical)
make check

# 4. Train our model
make train

# 5. Evaluate on held-out test set
make evaluate

# 6. Train and evaluate baselines for comparison
make baseline MODEL=deepcrime
make eval-baseline MODEL=deepcrime

make baseline MODEL=stgcn_adapted
make eval-baseline MODEL=stgcn_adapted
