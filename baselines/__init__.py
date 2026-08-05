"""
baselines/__init__.py
=====================
Baseline model registry for fair comparison.
All baselines share a common interface: forward(x, tim) -> dict with 'count_mu'.
"""

from .deepcrime import DeepCrime
from .st_shine import STSHINE
from .mist import MiST
from .zinb_gnn import ZINBGNN
from .stgcn_adapted import STGCNAdapted
from .informer_adapted import InformerAdapted

__all__ = [
    "DeepCrime",
    "STSHINE", 
    "MiST",
    "ZINBGNN",
    "STGCNAdapted",
    "InformerAdapted",
]