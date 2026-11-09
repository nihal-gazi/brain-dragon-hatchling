"""
Self-contained training, evaluation, and sparsity profiling CLI for BDH-GPU.
Implements the training regime described in Section 4.2, Section 6.4, and Appendix B.2:
  - Raw UTF-8 byte modeling (vocab_size=256)
  - Next token prediction
  - AdamW with warmup and linear learning rate decay
  - Adaptive gradient clipping
  - Sparsity profiling on activation vectors (x, y)
  - Recurrent state-space generation
"""

import argparse
import math
import os
import random
import time
from typing import Generator, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from bdh import BDHConfig, BDH_GPU, BDH_GPU_Prime


# Built-in sample text for instant training/testing without external downloads
SAMPLE_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.

For who would bear the whips and scorns of time,
Th' oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office, and the spurns
That patient merit of th' unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country from whose bourn
No traveller returns, puzzles the will,
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all.
"""


def generate_synthetic_repetition_data(
    num_samples: int = 1000,
    seq_len: int = 256,
) -> torch.Tensor:
    """
    Generates synthetic memorization & repetition sequences as described in Section 6.4 of the paper:
    "The input sequence started with a fixed 13-letter warm-up sequence, followed by 8 repetitions
     of an 8-letter random word ('fact'), with the same pattern repeating every 13 + 8 * 8 = 77 letters."
    """
    tokens = []
    warmup_prefix = b"WARMUP_PREFIX" # 13 bytes
    alphabet = b"abcdefghijklmnopqrstuvwxyz"

    for _ in range(num_samples):
        seq = bytearray()
        while len(seq) < seq_len:
            seq.extend(warmup_prefix)
            # Pick a random 8-letter fact
            fact = bytes(random.choices(alphabet, k=8))
            # 8 repetitions of the 8-letter fact
            for _ in range(8):
                seq.extend(fact)
        tokens.append(list(seq[:seq_len]))

    return torch.tensor(tokens, dtype=torch.long)


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts a batch of sequences and targets."""
    if data.dim() == 2:
        # Pre-chunked dataset
        idx = torch.randint(0, data.size(0), (batch_size,))
        x = data[idx, :seq_len].to(device)
        y = data[idx, 1 : seq_len + 1].to(device) if data.size(1) > seq_len else data[idx, :seq_len].to(device)
        return x, y
    else:
        # Continuous 1D byte tensor
        ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
        x = torch.stack([data[i : i + seq_len] for i in ix]).to(device)
        y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix]).to(device)
        return x, y


def measure_sparsity(activations: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[dict]:
    """
    Measures percentage of non-zero entries in activation vectors x and y.
    In BDH, vectors are positive (x, y >= 0) and y is empirically ~5% sparse (Section 4.1 & 6.2).
    """
    stats = []
    for layer_idx, (x, y) in enumerate(activations):
        # Active fraction: percentage of non-zero entries
        x_active = (x > 0.0).float().mean().item() * 100.0
        y_active = (y > 0.0).float().mean().item() * 100.0
        stats.append({
            "layer": layer_idx,
            "x_active_pct": x_active,
            "y_active_pct": y_active,
        })
    return stats


def main():
    parser = argparse.ArgumentParser(description="BDH-GPU ('Dragon Hatchling') Training & Sparsity Profiling")
    parser.add_argument("--preset", type=str, default="tiny", choices=["tiny", "appendix_e", "25m", "50m", "100m"],
                        help="Model preset configuration")
    parser.add_argument("--prime", action="store_true", help="Use BDH-GPU' (with xLSTM gating & multi-layer logits)")
    parser.add_argument("--alibi", action="store_true", help="Enable ALiBi exponential damping on attention state")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "text", "file"],
                        help="Dataset source: synthetic repetition (Section 6.4), built-in text, or custom file")
    parser.add_argument("--data_path", type=str, default="", help="Path to custom text file when dataset=file")
    parser.add_argument("--steps", type=int, default=150, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=128, help="Context sequence length (T)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate (paper uses 1e-3)")
    parser.add_argument("--warmup_steps", type=int, default=20, help="Linear warmup steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--eval_interval", type=int, default=25, help="Interval for evaluation and logging")
    parser.add_argument("--prompt", type=str, default="To be, or not ", help="Prompt for generation test")
    parser.add_argument("--max_gen_tokens", type=int, default=40, help="Number of bytes to generate")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Build config
    if args.preset == "tiny":
        config = BDHConfig.tiny()
    elif args.preset == "appendix_e":
        config = BDHConfig.appendix_e()
    elif args.preset == "25m":
        config = BDHConfig.model_25m()
    elif args.preset == "50m":
        config = BDHConfig.model_50m()
    elif args.preset == "100m":
        config = BDHConfig.model_100m()
    else:
        config = BDHConfig.tiny()

    if args.alibi:
        config.use_alibi_damping = True

    # Instantiate model
    if args.prime:
        model = BDH_GPU_Prime(config).to(device)
        model_name = "BDH-GPU' (Prime with xLSTM Gating)"
    else:
        model = BDH_GPU(config).to(device)
        model_name = "BDH-GPU (Canonical Definition 4)"

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Dimensions: N={config.n_neurons} neurons, D={config.d_internal}, H={config.n_heads} heads, L={config.n_layers} layers")
    print(f"Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print("=" * 70)

    # Prepare dataset
    if args.dataset == "synthetic":
        print("Using synthetic repetition puzzle dataset (Section 6.4: 13-byte warmup + 8 repeated 8-byte facts)...")
        data_tensor = generate_synthetic_repetition_data(num_samples=200, seq_len=args.seq_len + 1)
        is_chunked = True
    elif args.dataset == "file" and os.path.exists(args.data_path):
        print(f"Loading custom dataset from: {args.data_path}")
        with open(args.data_path, "rb") as f:
            raw_bytes = f.read()
        data_tensor = torch.tensor(list(raw_bytes), dtype=torch.long)
        is_chunked = False
    else:
        print("Using built-in natural language UTF-8 byte stream...")
        raw_bytes = SAMPLE_TEXT.encode("utf-8")
        data_tensor = torch.tensor(list(raw_bytes), dtype=torch.long)
        is_chunked = False

    # Optimizer with decoupled weight decay (AdamW) as in Appendix B.2
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    # Warmup and linear learning rate scheduler
    def get_lr(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        # Linear decay to 1e-4 as reported in Appendix B.2
        target_min_lr = 1e-4
        return target_min_lr + (args.lr - target_min_lr) * max(0.0, 1.0 - progress)

    print("\nStarting training loop...")
    model.train()
    start_time = time.time()

    for step in range(1, args.steps + 1):
        # Update learning rate
        curr_lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = curr_lr

        # Sample batch
        x, y_true = get_batch(data_tensor, args.batch_size, args.seq_len, device)

        optimizer.zero_grad()
        return_acts = (step % args.eval_interval == 0 or step == args.steps)

        if return_acts:
            logits, activations = model(x, return_activations=True)
        else:
            logits = model(x, return_activations=False)

        # Cross-entropy next-token prediction
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), y_true.view(-1))
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # Evaluation & reporting
        if return_acts:
            ppl = math.exp(min(loss.item(), 20.0))
            sparsity_info = measure_sparsity(activations)
            avg_y_active = sum(s["y_active_pct"] for s in sparsity_info) / len(sparsity_info)
            avg_x_active = sum(s["x_active_pct"] for s in sparsity_info) / len(sparsity_info)

            print(f"Step {step:4d}/{args.steps} | Loss: {loss.item():.4f} | PPL: {ppl:6.2f} | "
                  f"LR: {curr_lr:.2e} | Neuron Active %: x={avg_x_active:.1f}%, y={avg_y_active:.1f}%")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f}s ({args.steps / elapsed:.1f} steps/s).")

    # Layer-by-layer sparsity report
    print("\n" + "=" * 70)
    print("LAYER-BY-LAYER ACTIVATION SPARSITY REPORT (Section 4.1 & 6.4):")
    print("In BDH, non-negative activations (x, y >= 0) become sparse via ReLU gates.")
    print("-" * 70)
    for s in sparsity_info:
        print(f"  Layer {s['layer']}: x active: {s['x_active_pct']:5.2f}% | "
              f"y active (fresh signal): {s['y_active_pct']:5.2f}% (sparse: {100.0 - s['y_active_pct']:5.2f}%)")
    print("=" * 70)

    # Generation demo with recurrent state-space mode
    print("\n" + "=" * 70)
    print("AUTOREGRESSIVE GENERATION DEMO (Recurrent O(1) State-Space Mode):")
    prompt_bytes = args.prompt.encode("utf-8")
    prompt_tensor = torch.tensor([list(prompt_bytes)], dtype=torch.long, device=device)

    print(f"Prompt: {repr(args.prompt)}")
    gen_tokens = model.generate(
        prompt_tensor,
        max_new_tokens=args.max_gen_tokens,
        temperature=0.8,
        top_k=40,
        mode="recurrent",
    )

    gen_bytes = bytes(gen_tokens[0].cpu().tolist())
    try:
        decoded_text = gen_bytes.decode("utf-8", errors="replace")
    except Exception:
        decoded_text = str(gen_bytes)

    print(f"Generated text: {repr(decoded_text)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
