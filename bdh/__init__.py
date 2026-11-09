"""
BDH: The Dragon Hatchling (arXiv:2509.26507)
Implementation of the BDH-GPU architecture in PyTorch.
"""

from .config import BDHConfig
from .rope import RoPE
from .model import LinearAttention, BDH_GPU, BDH_GPU_Prime

__all__ = [
    "BDHConfig",
    "RoPE",
    "LinearAttention",
    "BDH_GPU",
    "BDH_GPU_Prime",
]
