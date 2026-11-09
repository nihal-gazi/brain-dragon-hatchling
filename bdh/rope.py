"""
Rotary Position Embedding (RoPE) and Positional Dynamics for BDH-GPU.
Reference: Su et al. (2024), "RoFormer: Enhanced Transformer with Rotary Position Embedding".
Paper references: Eq. (4), (5), (8), Section 3.2 (U operator), and Appendix E.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


class RoPE(nn.Module):
    """
    Rotary Position Embedding operating over head dimension (d_head = N // H).
    Applies SO(2) orthogonal rotation on paired coordinates:
      R_t * x = x * cos(t * theta) + rotate_half(x) * sin(t * theta)
    """

    def __init__(
        self,
        d_head: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
        use_alibi_damping: bool = False,
        alibi_decay_rate: float = 0.99,
    ):
        super().__init__()
        assert d_head % 2 == 0, f"d_head ({d_head}) must be even for RoPE coordinate pairing."
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.base = base
        self.use_alibi_damping = use_alibi_damping
        self.alibi_decay_rate = alibi_decay_rate

        # Inverse frequencies for d_head // 2 pairs
        inv_freq = 1.0 / (self.base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        self.max_seq_len = max(self.max_seq_len, seq_len)
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        # freqs: (seq_len, d_head // 2)
        freqs = torch.outer(t, self.inv_freq)
        # Concatenate for full d_head: (seq_len, d_head)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

        if self.use_alibi_damping:
            # Build causal decay matrix: M[i, j] = decay_rate ** (i - j) for i >= j
            i_indices = torch.arange(self.max_seq_len).unsqueeze(1)
            j_indices = torch.arange(self.max_seq_len).unsqueeze(0)
            diff = (i_indices - j_indices).clamp(min=0).float()
            decay = self.alibi_decay_rate ** diff
            self.register_buffer("alibi_decay_matrix", decay, persistent=False)
        else:
            self.register_buffer("alibi_decay_matrix", None, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Apply RoPE to tensor of shape (B, H, T, d_head).
        offset allows applying RoPE starting from a non-zero sequence index.
        """
        T = x.shape[2]
        if offset + T > self.cos_cached.shape[0]:
            self._build_cache(offset + T + 1024)

        cos = self.cos_cached[offset : offset + T, :].to(dtype=x.dtype)
        sin = self.sin_cached[offset : offset + T, :].to(dtype=x.dtype)

        # Reshape for broadcasting: (1, 1, T, d_head)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        return (x * cos) + (self._rotate_half(x) * sin)

    def forward_step(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """
        Apply RoPE for a single token at position `step`.
        x shape: (B, H, 1, d_head)
        """
        return self.forward(x, offset=step)

    def get_decay_matrix(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        """Returns the causal ALiBi geometric decay matrix if enabled."""
        if not self.use_alibi_damping:
            return None
        if seq_len > self.alibi_decay_matrix.shape[0]:
            self._build_cache(seq_len + 1024)
        return self.alibi_decay_matrix[:seq_len, :seq_len].to(device=device, dtype=dtype)
