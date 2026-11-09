"""
PyTorch implementation of BDH-GPU ("Dragon Hatchling").
Reference: "The Dragon Hatchling: The Missing Link Between the Transformer and Models of the Brain"
           (arXiv:2509.26507v1)
Primary references: Definition 4, Equations (4), (5), (8), and Appendix E code listing.
"""

from typing import List, Optional, Tuple, Union, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BDHConfig
from .rope import RoPE


class LinearAttention(nn.Module):
    """
    Linear Attention module for BDH-GPU.
    Computes causal linear attention:
      a^*_t = sum_{tau < t} (Qr_t @ Kr_tau^T) * V_tau
    Supports:
      - Parallel training mode over full context sequences (Appendix E)
      - O(1) step-by-step recurrent state-space mode for autoregressive generation (Eq. 5, 8)
      - Rotary Position Embedding (RoPE)
      - Optional ALiBi exponential damping (relaxing state over time)
    """

    def __init__(
        self,
        d_head: int,
        max_seq_len: int = 4096,
        rope_base: float = 10000.0,
        use_alibi_damping: bool = False,
        alibi_decay_rate: float = 0.99,
    ):
        super().__init__()
        self.d_head = d_head
        self.use_alibi_damping = use_alibi_damping
        self.alibi_decay_rate = alibi_decay_rate

        self.rope = RoPE(
            d_head=d_head,
            max_seq_len=max_seq_len,
            base=rope_base,
            use_alibi_damping=use_alibi_damping,
            alibi_decay_rate=alibi_decay_rate,
        )

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        offset: int = 0,
    ) -> torch.Tensor:
        """
        Parallel causal attention forward pass (Appendix E).
        Args:
          Q: Queries, shape (B, H, T, d_head)
          K: Keys, shape (B, H, T, d_head)
          V: Values, shape (B, 1, T, D) or (B, H, T, D)
          offset: Positional offset for RoPE
        Returns:
          a_ast: Attention output, shape (B, H, T, D)
        """
        Qr = self.rope(Q, offset=offset)
        Kr = self.rope(K, offset=offset)

        # Pairwise attention scores: (B, H, T, T)
        scores = Qr @ Kr.transpose(-2, -1)

        # Apply ALiBi geometric damping if enabled
        if self.use_alibi_damping:
            T = Q.size(2)
            decay = self.rope.get_decay_matrix(T, device=scores.device, dtype=scores.dtype)
            scores = scores * decay

        # Causal mask: strictly past keys (tau < t), diagonal=-1 zeroes current and future
        causal_scores = scores.tril(diagonal=-1)

        # Weighted values: (B, H, T, T) @ (B, 1, T, D) -> (B, H, T, D)
        return causal_scores @ V

    def step(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        state: torch.Tensor,
        step_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step recurrent linear attention update (Definition 4, Eq. 5 & 8).
        Args:
          Q: Query at current step, shape (B, H, 1, d_head)
          K: Key at current step, shape (B, H, 1, d_head)
          V: Value at current step, shape (B, 1, 1, D) or (B, H, 1, D)
          state: Recurrent state rho_{t-1}, shape (B, H, d_head, D)
          step_idx: Current temporal position (0-based)
        Returns:
          a_ast: Output attention vector, shape (B, H, 1, D)
          new_state: Updated recurrent state rho_t, shape (B, H, d_head, D)
        """
        Qr = self.rope.forward_step(Q, step=step_idx)
        Kr = self.rope.forward_step(K, step=step_idx)

        # Query existing state from tau < step_idx: a^*_t = Qr_t @ rho_{t-1}
        a_ast = Qr @ state # (B, H, 1, d_head) @ (B, H, d_head, D) -> (B, H, 1, D)

        # Decay previous state if ALiBi damping is active
        if self.use_alibi_damping:
            new_state = state * self.alibi_decay_rate
        else:
            new_state = state

        # Incorporate current key & value into state for future steps: rho_t = rho_{t-1} + Kr_t^T @ V_t
        # Kr.transpose(-2, -1): (B, H, d_head, 1)
        # V: (B, 1, 1, D) -> broadcast to (B, H, 1, D)
        new_state = new_state + (Kr.transpose(-2, -1) @ V)

        return a_ast, new_state


class BDH_GPU(nn.Module):
    """
    BDH-GPU Architecture ("Dragon Hatchling") as specified in arXiv:2509.26507v1.
    Directly reflects Appendix E and Definition 4.
    """

    def __init__(self, config: Optional[BDHConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = BDHConfig(**kwargs) if kwargs else BDHConfig.appendix_e()
        self.config = config

        D = config.d_internal
        H = config.n_heads
        N = config.n_neurons
        dropout = config.dropout
        vocab_size = config.vocab_size

        # Non-parametric LayerNorm in internal dimension D (Section 3.1 & Appendix E)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.wte = nn.Embedding(vocab_size, D)
        self.drop = nn.Dropout(dropout)

        # Trainable parameter matrices (3 * N * D scalable weights, shared across layers)
        self.encoder = nn.Parameter(
            torch.zeros((N, D)).normal_(std=0.02)
        )
        self.decoder_x = nn.Parameter(
            torch.zeros((H, D, N // H)).normal_(std=0.02)
        )
        self.decoder_y = nn.Parameter(
            torch.zeros((H, D, N // H)).normal_(std=0.02)
        )
        self.readout = nn.Parameter(
            torch.zeros((D, vocab_size)).normal_(std=0.02)
        )

        self.attn = LinearAttention(
            d_head=config.d_head,
            max_seq_len=config.max_seq_len,
            rope_base=config.rope_base,
            use_alibi_damping=config.use_alibi_damping,
            alibi_decay_rate=config.alibi_decay_rate,
        )

        # Initialize embedding weights
        nn.init.normal_(self.wte.weight, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        return_activations: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Parallel forward pass over sequences (Appendix E).
        Args:
          idx: Token tensor, shape (B, T)
          return_activations: If True, returns (logits, [(x_0, y_0), ...]) for sparsity analysis.
        Returns:
          logits: (B, T, vocab_size)
        """
        B, T = idx.size()
        v_ast = self.ln(self.wte(idx).unsqueeze(1)) # (B, 1, T, D)

        saved_activations = [] if return_activations else None

        for _ in range(self.config.n_layers):
            # 1. Project to neuron space: (B, 1, T, D) @ (H, D, N // H) -> (B, H, T, N // H)
            x = F.relu(v_ast @ self.decoder_x)

            # 2. Linear Attention in neuron space
            a_ast = self.attn(
                Q=x,
                K=x,
                V=v_ast,
            ) # (B, H, T, D)

            # 3. Project normalized attention back to neuron space and modulate by x
            # ln(a_ast) @ decoder_y -> (B, H, T, N // H)
            y = F.relu(self.ln(a_ast) @ self.decoder_y) * x # (B, H, T, N // H)
            y = y.transpose(1, 2).reshape(B, 1, T, self.config.n_neurons) # (B, 1, T, N)

            if return_activations:
                saved_activations.append((x.detach(), y.detach()))

            y = self.drop(y)

            # 4. Lift back to internal dimension D via encoder and add residual
            v_ast = v_ast + self.ln(y @ self.encoder) # (B, 1, T, D)
            v_ast = self.ln(v_ast)

        logits = v_ast.squeeze(1) @ self.readout # (B, T, vocab_size)

        if return_activations:
            return logits, saved_activations
        return logits

    def init_recurrent_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> List[torch.Tensor]:
        """
        Initializes recurrent attention state rho for each layer.
        Each state tensor has shape: (B, H, N // H, D)
        """
        H = self.config.n_heads
        d_head = self.config.d_head
        D = self.config.d_internal
        return [
            torch.zeros((batch_size, H, d_head, D), device=device, dtype=dtype)
            for _ in range(self.config.n_layers)
        ]

    def step(
        self,
        idx: torch.Tensor,
        states: List[torch.Tensor],
        step_idx: int,
        return_activations: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        O(1) recurrent step for next-token generation (State-space Eq. 8).
        Args:
          idx: Single token index tensor, shape (B,) or (B, 1)
          states: List of L state tensors, each (B, H, d_head, D)
          step_idx: Current sequence step (0, 1, 2, ...)
        Returns:
          logits: (B, vocab_size)
          new_states: Updated list of L state tensors
          activations: Optional list of (x, y) for layer monitoring
        """
        if idx.dim() == 1:
            idx = idx.unsqueeze(1) # (B, 1)
        B, T = idx.size()
        assert T == 1, "step() processes exactly one token at a time (T=1)."

        v_ast = self.ln(self.wte(idx).unsqueeze(1)) # (B, 1, 1, D)

        new_states = []
        saved_activations = [] if return_activations else None

        for layer_l in range(self.config.n_layers):
            state_l = states[layer_l]

            # 1. Project to neuron space
            x = F.relu(v_ast @ self.decoder_x) # (B, H, 1, N // H)

            # 2. Linear Attention recurrent step
            a_ast, next_state_l = self.attn.step(
                Q=x,
                K=x,
                V=v_ast,
                state=state_l,
                step_idx=step_idx,
            ) # (B, H, 1, D)
            new_states.append(next_state_l)

            # 3. Project to neuron space and modulate by x
            y = F.relu(self.ln(a_ast) @ self.decoder_y) * x # (B, H, 1, N // H)
            y = y.transpose(1, 2).reshape(B, 1, 1, self.config.n_neurons)

            if return_activations:
                saved_activations.append((x.detach(), y.detach()))

            y = self.drop(y)

            # 4. Lift back to internal dimension and add residual
            v_ast = v_ast + self.ln(y @ self.encoder) # (B, 1, 1, D)
            v_ast = self.ln(v_ast)

        # Logits at step t: (B, vocab_size)
        logits = (v_ast.squeeze(1).squeeze(1)) @ self.readout

        if return_activations:
            return logits, new_states, saved_activations
        return logits, new_states, None

    @torch.no_grad()
    def generate(
        self,
        prompt_idx: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        mode: str = "recurrent",
    ) -> torch.Tensor:
        """
        Autoregressive generation from prompt tokens.
        Modes:
          - 'recurrent': O(1) state updates using recurrent attention state
          - 'parallel': Full causal recomputation at each step (for verification)
        """
        self.eval()
        device = prompt_idx.device
        B, T = prompt_idx.size()

        if mode == "parallel":
            generated = prompt_idx.clone()
            for _ in range(max_new_tokens):
                logits = self.forward(generated) # (B, curr_T, vocab_size)
                next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
            return generated

        elif mode == "recurrent":
            states = self.init_recurrent_state(B, device=device, dtype=self.encoder.dtype)
            # Warm up state with prompt tokens
            for t in range(T - 1):
                token_t = prompt_idx[:, t : t + 1]
                _, states, _ = self.step(token_t, states, step_idx=t)

            # Last prompt token gives first generated token logits
            token_last = prompt_idx[:, -1:]
            logits, states, _ = self.step(token_last, states, step_idx=T - 1)

            generated = [prompt_idx]
            for step_i in range(max_new_tokens):
                curr_logits = logits / max(temperature, 1e-5)
                if top_k is not None:
                    v, _ = torch.topk(curr_logits, min(top_k, curr_logits.size(-1)))
                    curr_logits[curr_logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(curr_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated.append(next_token)

                # Feed next token into recurrent step
                curr_step_idx = T + step_i
                logits, states, _ = self.step(next_token, states, step_idx=curr_step_idx)

            return torch.cat(generated, dim=1)
        else:
            raise ValueError(f"Unknown generation mode: {mode}. Choose 'recurrent' or 'parallel'.")


class BDH_GPU_Prime(BDH_GPU):
    """
    BDH-GPU' Architecture ("Dragon Hatchling Prime").
    Extends canonical BDH-GPU with:
      1. xLSTM-like input/forget gating on state updates (Beck et al., 2024; Section 4.2 & B.2)
      2. Multi-layer logit pooling/merging (Appendix B.2)
    """

    def __init__(self, config: Optional[BDHConfig] = None, **kwargs):
        if config is None:
            config = BDHConfig(**kwargs) if kwargs else BDHConfig.model_25m()
        config.use_prime_gating = True
        config.merge_layer_logits = True
        super().__init__(config=config)

        D = config.d_internal
        L = config.n_layers

        # xLSTM-like input and forget gates conditioned on internal representation
        self.gate_in = nn.Linear(D, config.n_heads, bias=True)
        self.gate_forget = nn.Linear(D, config.n_heads, bias=True)

        # Multi-layer logit mixing weights: learned scalar weighting for each layer's prediction
        self.layer_weights = nn.Parameter(torch.ones(L) / L)

    def forward(
        self,
        idx: torch.Tensor,
        return_activations: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, T = idx.size()
        v_ast = self.ln(self.wte(idx).unsqueeze(1)) # (B, 1, T, D)

        saved_activations = [] if return_activations else None
        layer_logits_list = []

        for layer_idx in range(self.config.n_layers):
            x = F.relu(v_ast @ self.decoder_x) # (B, H, T, N // H)

            # Compute xLSTM-like gating
            # v_ast: (B, 1, T, D) -> squeeze to (B, T, D)
            v_flat = v_ast.squeeze(1)
            f_gate = torch.sigmoid(self.gate_forget(v_flat)).transpose(1, 2).unsqueeze(-1) # (B, H, T, 1)
            i_gate = torch.sigmoid(self.gate_in(v_flat)).transpose(1, 2).unsqueeze(-1)     # (B, H, T, 1)

            # Modulate attention input with gates
            a_ast = self.attn(
                Q=x,
                K=x * i_gate,
                V=v_ast,
            )

            y = F.relu(self.ln(a_ast) @ self.decoder_y) * x
            y = y.transpose(1, 2).reshape(B, 1, T, self.config.n_neurons)

            if return_activations:
                saved_activations.append((x.detach(), y.detach()))

            y = self.drop(y)
            v_ast = v_ast + self.ln(y @ self.encoder)
            v_ast = self.ln(v_ast)

            # Collect intermediate layer prediction
            layer_logits_list.append(v_ast.squeeze(1) @ self.readout)

        # Merge next-token predictions from all layers with softmax weighting
        norm_weights = F.softmax(self.layer_weights, dim=0)
        final_logits = sum(w * l_logits for w, l_logits in zip(norm_weights, layer_logits_list))

        if return_activations:
            return final_logits, saved_activations
        return final_logits
