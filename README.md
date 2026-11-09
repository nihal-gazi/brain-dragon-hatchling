# Dragon Hatchling (BDH): The Missing Link Between the Transformer and Models of the Brain

A clean, modular, and mathematically verified PyTorch implementation of the **BDH-GPU** architecture described in:

> **The Dragon Hatchling: The Missing Link Between the Transformer and Models of the Brain**  
> *Adrian Kosowski, Przemysław Uznański, Jan Chorowski, Zuzanna Stamirowska, Michał Bartoszkiewicz*  
> Pathway, Palo Alto, USA  
> arXiv: [2509.26507v1](https://arxiv.org/abs/2509.26507)

---

## 1. Key Insights & Architecture Overview

BDH introduces a bridge between the **Transformer** (centralized tensor operations) and **Brain Models** (local distributed graph dynamics, Hebbian learning, and spiking integrate-and-fire thresholding).

### Fundamental Properties
- **Scales in a single neuron dimension $n$**: Almost all model parameters reside in three matrices $E \in \mathbb{R}^{n \times d}$, $D_x \in \mathbb{R}^{n \times d}$, $D_y \in \mathbb{R}^{n \times d}$ (shared across all $L$ layers). Total parameters: $(3 + o(1))nd$.
- **Positive, Sparse Activations**: Activations $x, y \in (\mathbb{R}^+)^n$ live strictly in the positive orthant. Passing through ReLU gates induces natural sparsity ($\approx 5\%$ in $y$ during language modeling).
- **Dual Representation**:
  1. **Parallel causal training mode**: Standard token-parallel training via causal linear self-attention:
     $$\text{scores} = (Q_r K_r^T)_{\text{tril}(-1)}, \quad A^* = \text{scores} \cdot V$$
  2. **Recurrent state-space mode**: $O(1)$ memory & $O(nd)$ FLOPs per step inference with state $\rho \in \mathbb{R}^{n \times d}$ per layer:
     $$\rho_{t, l} = (\rho_{t-1, l} + K_{r, t}^T V_t) \cdot U, \quad a^*_{t, l} = Q_{r, t} \cdot \rho_{t-1, l}$$

---

## 2. Core Equations

For token sequence $t = 0, 1, \dots$ across layers $l = 1, \dots, L$:

$$\begin{aligned}
x_{t, l} &= \text{ReLU}\left(v^*_{t, l-1} D_x\right) \\
a^*_{t, l} &= \sum_{\tau < t} v^*_{\tau, l-1} x_{\tau, l}^T U^{t - \tau} x_{t, l} \\
y_{t, l} &= \text{ReLU}\left(\text{LN}(a^*_{t, l}) D_y\right) \odot x_{t, l} \\
v^*_{t, l} &= \text{LN}\left(v^*_{t, l-1} + \text{LN}(y_{t, l} E)\right)
\end{aligned}$$

where:
- $v^*_{t, 0}$ is initialized from input embeddings $\text{LN}(W_{te}[\text{token}_t])$.
- $U$ is the local rotation / damping matrix (RoPE + optional ALiBi damping).
- Output logits are obtained via $v^*_{t, L} W_{\text{readout}}$.

---

## 3. Repository Structure

```
├── bdh/
│   ├── __init__.py      # Module exports: BDH_GPU, BDH_GPU_Prime, BDHConfig, RoPE, LinearAttention
│   ├── config.py        # BDHConfig dataclass & paper presets (Tiny, Appendix E, 25M, 50M, 100M)
│   ├── rope.py          # RoPE (Su et al., 2024) across head dim (N // H) & optional ALiBi damping
│   └── model.py         # BDH_GPU (Appendix E) & BDH_GPU_Prime with parallel and recurrent step modes
├── test_bdh.py          # Comprehensive test suite (9 tests verifying equivalence, sparsity, and gradients)
├── train.py             # Self-contained training CLI with synthetic repetition puzzle & sparsity profiling
└── 2509.26507v1.pdf     # Reference paper
```

---

## 4. Quickstart

### Run the Test Suite
Verifies exact algebraic equivalence between parallel attention and recurrent state updates, gradient flow, and parameter counts:
```bash
python test_bdh.py
```

### Train on Synthetic Repetition Puzzle (Section 6.4)
Train a tiny model on the synthetic memory benchmark (13-byte warmup prefix + 8 repeated 8-byte facts):
```bash
python train.py --preset tiny --steps 100 --batch_size 4 --seq_len 128
```

### Train on Natural Language Bytes
Train on UTF-8 raw byte stream:
```bash
python train.py --preset tiny --dataset text --steps 150 --prompt "To be, or not "
```

### Train with BDH-GPU' (xLSTM Gating + Multi-Layer Logits) and ALiBi Damping
```bash
python train.py --preset tiny --prime --alibi --steps 100
```

---

## 5. Model Presets (Table 5 & Appendix E)

| Preset | $N$ (Neurons) | $D$ (Internal) | $H$ (Heads) | $L$ (Layers) | Scalable Params |
|---|---|---|---|---|---|
| `tiny` | 512 | 64 | 2 | 2 | ~131 K |
| `appendix_e` | 32,768 | 256 | 4 | 6 | ~25.3 M |
| `25m` | 32,768 | 256 | 4 | 8 | ~25.3 M |
| `50m` | 65,536 | 256 | 4 | 8 | ~50.5 M |
| `100m` | 131,072 | 256 | 4 | 8 | ~100.9 M |

---

## 6. Mathematical Equivalence Verification

In `test_bdh.py`, the parallel causal attention matrix computation:
```python
scores = (Qr @ Kr.transpose(-2, -1)).tril(diagonal=-1)
a_ast = scores @ V
```
is tested against the $O(1)$ state-space recurrence:
```python
a_ast_t = Qr_t @ rho
rho = rho + Kr_t.transpose(-2, -1) @ V_t
```
Under floating point precision, both formulations produce identical outputs down to machine precision ($\approx 10^{-14}$ in float64, $< 10^{-5}$ in float32).
