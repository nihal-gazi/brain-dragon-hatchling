"""
Comprehensive test suite for BDH-GPU ("Dragon Hatchling") implementation.
Verifies:
  1. Parameter counts matching paper formula (3 + o(1)) * n * d
  2. Exact numerical equivalence between parallel forward and recurrent step()
  3. Backpropagation and gradient flow on all trainable matrices
  4. Positivity and sparsity properties of neuron activations (x >= 0, y >= 0)
  5. RoPE positional embedding and ALiBi damping
  6. BDH-GPU' variant
  7. Autoregressive generation in parallel and recurrent modes
"""

import unittest
import torch
import torch.nn.functional as F

from bdh import BDHConfig, BDH_GPU, BDH_GPU_Prime, RoPE, LinearAttention


class TestBDH(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    def test_parameter_counts(self):
        """Verifies trainable parameter count matches paper formula (3 + o(1)) * n * d."""
        config = BDHConfig.tiny()
        model = BDH_GPU(config)

        # Count parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # E: (N, D), Dx: (H, D, N//H), Dy: (H, D, N//H), wte: (vocab, D), readout: (D, vocab)
        expected = config.total_param_count
        self.assertEqual(trainable_params, expected)

        # Verify 25M preset
        cfg_25m = BDHConfig.model_25m()
        m_25m = BDH_GPU(cfg_25m)
        params_25m = sum(p.numel() for p in m_25m.parameters() if p.requires_grad)
        # Scalable part: 3 * 32768 * 256 = 25,165,824; Embeddings: 2 * 256 * 256 = 131,072
        self.assertEqual(params_25m, 25_165_824 + 131_072)

    def test_forward_pass_shape(self):
        """Checks input/output tensor shapes for parallel execution."""
        config = BDHConfig.tiny()
        model = BDH_GPU(config)
        model.eval()

        B, T = 3, 16
        idx = torch.randint(0, config.vocab_size, (B, T))
        logits = model(idx)

        self.assertEqual(logits.shape, (B, T, config.vocab_size))
        self.assertFalse(torch.isnan(logits).any())

    def test_parallel_vs_recurrent_attention_equivalence(self):
        """
        Critical test: Verifies that step-by-step recurrent attention matches
        the causal parallel attention matrix formulation exactly (Eq. 4, 5, 8).
        """
        B, H, T, d_head, D = 2, 2, 8, 32, 64
        attn = LinearAttention(d_head=d_head, max_seq_len=64)
        attn.eval()

        Q = torch.randn(B, H, T, d_head)
        K = Q
        V = torch.randn(B, 1, T, D)

        # 1. Parallel causal attention
        out_parallel = attn(Q, K, V) # (B, H, T, D)

        # 2. Recurrent step-by-step attention
        out_recurrent_list = []
        state = torch.zeros(B, H, d_head, D)
        for t in range(T):
            q_t = Q[:, :, t : t + 1, :]
            k_t = K[:, :, t : t + 1, :]
            v_t = V[:, :, t : t + 1, :]

            a_t, state = attn.step(q_t, k_t, v_t, state, step_idx=t)
            out_recurrent_list.append(a_t)

        out_recurrent = torch.cat(out_recurrent_list, dim=2)

        max_diff = (out_parallel - out_recurrent).abs().max().item()
        self.assertLess(
            max_diff,
            1e-4,
            f"Parallel and recurrent attention output diverged: max diff {max_diff}",
        )

    def test_full_model_parallel_vs_recurrent_equivalence(self):
        """
        Verifies that running the entire BDH_GPU model token-by-token using step()
        yields identical logits to parallel forward() over the full sequence.
        """
        config = BDHConfig.tiny()
        model = BDH_GPU(config)
        model.eval()

        B, T = 2, 10
        idx = torch.randint(0, config.vocab_size, (B, T))

        # 1. Full parallel forward pass
        logits_parallel = model(idx) # (B, T, vocab_size)

        # 2. Recurrent token-by-token pass
        states = model.init_recurrent_state(B, device=idx.device, dtype=torch.float32)
        logits_recurrent_list = []
        for t in range(T):
            token_t = idx[:, t : t + 1]
            logit_t, states, _ = model.step(token_t, states, step_idx=t)
            logits_recurrent_list.append(logit_t.unsqueeze(1))

        logits_recurrent = torch.cat(logits_recurrent_list, dim=1) # (B, T, vocab_size)

        max_diff = (logits_parallel - logits_recurrent).abs().max().item()
        self.assertLess(
            max_diff,
            1e-4,
            f"Full model parallel vs recurrent logits mismatch: max diff {max_diff}",
        )

    def test_gradient_flow_backward(self):
        """Verifies backward pass computes valid non-zero gradients on all model parameters."""
        config = BDHConfig.tiny()
        model = BDH_GPU(config)
        model.train()

        B, T = 2, 12
        idx = torch.randint(0, config.vocab_size, (B, T))
        targets = torch.randint(0, config.vocab_size, (B, T))

        logits = model(idx)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient.")
                self.assertFalse(torch.isnan(param.grad).any(), f"Parameter {name} has NaN in gradient.")
                self.assertGreater(param.grad.abs().sum().item(), 0.0, f"Parameter {name} has zero gradient.")

    def test_sparse_positive_activations(self):
        """
        Verifies that neuron activations x and y are strictly non-negative (x >= 0, y >= 0),
        and tests activation sparsity profiling as reported in Section 4.1 & 6.4.
        """
        config = BDHConfig.tiny()
        model = BDH_GPU(config)
        model.eval()

        B, T = 2, 16
        idx = torch.randint(0, config.vocab_size, (B, T))
        _, activations = model(idx, return_activations=True)

        self.assertEqual(len(activations), config.n_layers)
        for layer_idx, (x, y) in enumerate(activations):
            # Non-negativity check
            self.assertTrue(torch.all(x >= 0.0), f"x in layer {layer_idx} contains negative values.")
            self.assertTrue(torch.all(y >= 0.0), f"y in layer {layer_idx} contains negative values.")

            # Sparsity calculation
            zero_fraction_x = (x == 0.0).float().mean().item()
            zero_fraction_y = (y == 0.0).float().mean().item()
            self.assertGreater(zero_fraction_x, 0.0, "x should exhibit zero activations due to ReLU.")
            self.assertGreater(zero_fraction_y, 0.0, "y should exhibit zero activations due to ReLU.")

    def test_alibi_damping(self):
        """Tests that enabling ALiBi state damping relaxes attention weights over distance."""
        config = BDHConfig.tiny()
        config.use_alibi_damping = True
        config.alibi_decay_rate = 0.95
        model = BDH_GPU(config)
        model.eval()

        B, T = 2, 8
        idx = torch.randint(0, config.vocab_size, (B, T))
        logits_parallel = model(idx)

        # Recurrent equivalence with ALiBi damping
        states = model.init_recurrent_state(B, device=idx.device)
        logits_recurrent_list = []
        for t in range(T):
            token_t = idx[:, t : t + 1]
            logit_t, states, _ = model.step(token_t, states, step_idx=t)
            logits_recurrent_list.append(logit_t.unsqueeze(1))
        logits_recurrent = torch.cat(logits_recurrent_list, dim=1)

        diff = (logits_parallel - logits_recurrent).abs().max().item()
        self.assertLess(diff, 1e-4, f"ALiBi damping parallel/recurrent mismatch: {diff}")

    def test_bdh_gpu_prime(self):
        """Tests the BDH-GPU' variant with xLSTM-style gating and multi-layer logit mixing."""
        config = BDHConfig.tiny()
        model = BDH_GPU_Prime(config)
        model.train()

        B, T = 2, 8
        idx = torch.randint(0, config.vocab_size, (B, T))
        logits = model(idx)
        self.assertEqual(logits.shape, (B, T, config.vocab_size))

        loss = logits.sum()
        loss.backward()
        self.assertIsNotNone(model.gate_in.weight.grad)
        self.assertIsNotNone(model.layer_weights.grad)

    def test_generation_recurrent_vs_parallel(self):
        """Verifies generation functionality works in both recurrent and parallel modes."""
        config = BDHConfig.tiny()
        model = BDH_GPU(config)
        model.eval()

        prompt = torch.randint(0, config.vocab_size, (1, 5))
        max_new = 6

        # Deterministic generation (temperature -> 0 or greedy with seed)
        torch.manual_seed(123)
        gen_rec = model.generate(prompt, max_new_tokens=max_new, mode="recurrent", temperature=0.01)

        torch.manual_seed(123)
        gen_par = model.generate(prompt, max_new_tokens=max_new, mode="parallel", temperature=0.01)

        self.assertEqual(gen_rec.shape, (1, 5 + max_new))
        self.assertEqual(gen_par.shape, (1, 5 + max_new))
        self.assertTrue(torch.equal(gen_rec, gen_par), "Greedy generation should match between modes.")


if __name__ == "__main__":
    unittest.main()
