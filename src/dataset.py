"""
src/dataset.py
==============
PyTorch Dataset for Multivariate Spatio-Temporal Traffic Violation Forecasting.

Implements sliding window sampling (H=4 history → F=1 future),
temporal metadata generation for Time-Aware Embeddings, and
proper handling of the Virtual Sinkhole Node.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple


class ViolationForecastDataset(Dataset):
    """
    Sliding-window dataset for violation forecasting.

    Args:
        data_dir: Path to directory containing X.pt, Y_count.pt, Y_admin.pt,
                  Y_threat.pt, and metadata.pt
        split: One of 'train', 'val', 'test' (future use; currently loads full series)
        hist_steps: Number of historical 6-hour bins as input (H). Default: 4
        forecast_steps: Number of future 6-hour bins to predict (F). Default: 1
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        hist_steps: int = 4,
        forecast_steps: int = 1,
    ):
        super().__init__()
        self.hist_steps = hist_steps
        self.forecast_steps = forecast_steps
        self.split = split

        # Load preprocessed tensors
        self.X = torch.load(f"{data_dir}/X.pt", weights_only=False)              # [T, N, C]
        self.Y_count = self.X                                # Share memory reference with X to save 4GB RAM
        self.Y_admin = torch.load(f"{data_dir}/Y_admin.pt", weights_only=False)   # [T, N, S_admin]
        self.Y_threat = torch.load(f"{data_dir}/Y_threat.pt", weights_only=False) # [T, N, S_threat]
        self.metadata = torch.load(f"{data_dir}/metadata.pt", weights_only=False)

        self.n_nodes = self.metadata["n_nodes"]
        self.n_channels = self.metadata["n_channels"]
        self.hours_per_bin = self.metadata["hours_per_bin"]
        self.all_bins = self.metadata["all_bins"]  # List of pd.Timestamp

        # Total valid samples = T - H - F + 1
        self.total_timesteps = self.X.shape[0]
        self.n_samples = self.total_timesteps - hist_steps - forecast_steps + 1

        if self.n_samples <= 0:
            raise ValueError(
                f"Not enough timesteps ({self.total_timesteps}) for "
                f"H={hist_steps}, F={forecast_steps}. Need at least {hist_steps + forecast_steps}."
            )

        # Precompute temporal metadata for all bins
        # Shape: [T, 2] where col0=day_of_week (0-6), col1=quarter_of_day (0-3)
        self.temporal_meta = self._build_temporal_metadata()

        print(f"[Dataset:{split}] {self.n_samples} samples | "
              f"H={hist_steps} F={forecast_steps} | "
              f"Nodes={self.n_nodes} Channels={self.n_channels}")

    def _build_temporal_metadata(self) -> torch.Tensor:
        """
        Build temporal metadata tensor for Time-Aware Embeddings.
        Col 0: day_of_week (0=Mon ... 6=Sun)
        Col 1: quarter_of_day (0=00-06, 1=06-12, 2=12-18, 3=18-24)
        """
        import pandas as pd

        bins = pd.DatetimeIndex(self.all_bins)
        dow = torch.tensor(bins.dayofweek, dtype=torch.long)       # [T]
        hour = torch.tensor(bins.hour, dtype=torch.long)           # [T]
        quarter = hour // self.hours_per_bin                       # [T], values 0-3

        meta = torch.stack([dow, quarter], dim=-1)                 # [T, 2]
        return meta

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single sliding window sample.

        Returns dict with keys:
            'x':          [H, N, C]       Historical violation counts
            'y_count':    [F, N, C]       Future violation counts
            'y_admin':    [F, N, S_admin] Future admin disposition probs
            'y_threat':   [F, N, S_threat]Future threat flag probs
            'tim_hist':   [H, 2]          Temporal metadata for history window
            'tim_future': [F, 2]          Temporal metadata for forecast window
        """
        start = idx
        hist_end = start + self.hist_steps
        future_end = hist_end + self.forecast_steps

        x = self.X[start:hist_end].float()                    # [H, N, C]
        y_count = self.Y_count[hist_end:future_end].float()    # [F, N, C]
        y_admin = self.Y_admin[hist_end:future_end].float()    # [F, N, S_admin]
        y_threat = self.Y_threat[hist_end:future_end].float()  # [F, N, S_threat]

        tim_hist = self.temporal_meta[start:hist_end]      # [H, 2]
        tim_future = self.temporal_meta[hist_end:future_end]  # [F, 2]

        return {
            "x": x,
            "y_count": y_count,
            "y_admin": y_admin,
            "y_threat": y_threat,
            "tim_hist": tim_hist,
            "tim_future": tim_future,
        }