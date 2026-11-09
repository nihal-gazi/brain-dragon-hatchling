"""
Configuration and model presets for BDH-GPU architecture.
Reference: "The Dragon Hatchling: The Missing Link Between the Transformer and Models of the Brain"
           (arXiv:2509.26507v1)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BDHConfig:
    # Model dimensions
    d_internal: int = 256        # D: low-rank internal dimension (d in paper)
    n_neurons: int = 32768       # N: concept dimension / number of neurons (n in paper)
    n_heads: int = 4             # H: number of heads (h in paper)
    n_layers: int = 6            # L: number of recurrent layer iterations (shared weights)
    vocab_size: int = 256        # Token alphabet size (default: 256 for raw UTF-8 bytes)
    dropout: float = 0.05        # Dropout probability

    # Positional encoding settings
    rope_base: float = 10000.0   # RoPE base frequency
    max_seq_len: int = 4096      # Maximum sequence length for precomputed RoPE tables
    use_alibi_damping: bool = False  # Enable state damping (ALiBi-style geometric relaxation)
    alibi_decay_rate: float = 0.99   # Damping factor per step (U matrix damping)

    # Architectural variants
    use_prime_gating: bool = False   # BDH-GPU' extension with xLSTM-like gating
    merge_layer_logits: bool = False # BDH-GPU' extension to pool logits across layers

    @property
    def d_head(self) -> int:
        """Neuron dimension per head."""
        assert self.n_neurons % self.n_heads == 0, (
            f"n_neurons ({self.n_neurons}) must be divisible by n_heads ({self.n_heads})"
        )
        return self.n_neurons // self.n_heads

    @property
    def total_param_count(self) -> int:
        """
        Theoretical parameter count:
        E: N * D
        Dx: N * D
        Dy: N * D
        wte: vocab_size * D
        readout: D * vocab_size
        Total: 3 * N * D + 2 * vocab_size * D = (3 + o(1)) * N * D
        """
        return 3 * self.n_neurons * self.d_internal + 2 * self.vocab_size * self.d_internal

    @classmethod
    def appendix_e(cls) -> "BDHConfig":
        """Didactic configuration directly from Appendix E of the paper."""
        return cls(
            d_internal=256,
            n_neurons=32768,
            n_heads=4,
            n_layers=6,
            vocab_size=256,
            dropout=0.05,
        )

    @classmethod
    def model_25m(cls) -> "BDHConfig":
        """25M parameter configuration from Table 5."""
        return cls(
            d_internal=256,
            n_neurons=32768,
            n_heads=4,
            n_layers=8,
            vocab_size=256,
            dropout=0.1,
        )

    @classmethod
    def model_50m(cls) -> "BDHConfig":
        """50M parameter configuration from Table 5."""
        return cls(
            d_internal=256,
            n_neurons=65536,
            n_heads=4,
            n_layers=8,
            vocab_size=256,
            dropout=0.1,
        )

    @classmethod
    def model_100m(cls) -> "BDHConfig":
        """100M parameter configuration from Table 5."""
        return cls(
            d_internal=256,
            n_neurons=131072,
            n_heads=4,
            n_layers=8,
            vocab_size=256,
            dropout=0.1,
        )

    @classmethod
    def tiny(cls) -> "BDHConfig":
        """Lightweight configuration for quick local testing and unit tests."""
        return cls(
            d_internal=64,
            n_neurons=512,
            n_heads=2,
            n_layers=2,
            vocab_size=256,
            dropout=0.0,
            max_seq_len=512,
        )
