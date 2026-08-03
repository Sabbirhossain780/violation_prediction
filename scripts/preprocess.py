"""
scripts/preprocess.py
=====================
CLI Entry Point for Phase 1: Data Engineering & Tensor Forging.

Usage:
    python scripts/preprocess.py --config configs/base_config.yaml
"""

import argparse
import sys
import logging
from pathlib import Path

# Add project root to path so src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import preprocess


def setup_logging():
    """Configure console + file logging."""
    log_dir = PROJECT_ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "preprocess.log", mode="w"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: Convert raw traffic stop data into PyTorch tensors using a config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "base_config.yaml"),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    logging.info("=" * 60)
    logging.info("Phase 1: Data Engineering & Tensor Forging Pipeline")
    logging.info("=" * 60)
    logging.info(f"  Config file: {args.config}")
    logging.info("=" * 60)

    if not Path(args.config).exists():
        logging.error(f"Config file not found: {args.config}")
        sys.exit(1)

    try:
        preprocess(config_path=args.config)
        logging.info(" Preprocessing completed successfully.")
    except Exception as e:
        logging.exception(f" Preprocessing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()