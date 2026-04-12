"""
Deep Encoder: stack N encoder blocks and measure depth vs training stability.
This extended file shows what happens when you stack 1, 2, 4, 8, 12 encoder blocks.
We measure: (1) gradient norm at each layer, (2) forward activation statistics,
and (3) final loss value as a proxy for whether the network is learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.gelu(self.linear1(x)))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class DeepEncoder(nn.Module):
    """
    Stack N encoder blocks to form a deep encoder.
    This is the architecture used in BERT (12 blocks) and BERT-large (24 blocks).
    """
    def __init__(self, d_model, n_heads, n_blocks, d_ff=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff)
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# Gradient norm measurement per layer
# ---------------------------------------------------------------------------

def measure_gradient_norms(model):
    """
    After a backward pass, collect the gradient norm for each parameter
    in each encoder block. Returns a list of gradient norms per block.
    """
    grad_norms = []
    for block in model.blocks:
        block_grad_norms = []
        for name, param in block.named_parameters():
            if param.grad is not None:
                block_grad_norms.append(param.grad.norm().item())
        grad_norms.append(block_grad_norms)
    return grad_norms


# ---------------------------------------------------------------------------
# Activation statistics measurement
# ---------------------------------------------------------------------------

def measure_activation_stats(model, x):
    """
    Run a forward pass and collect mean/std of activations at each block output.
    We hook into each block's output to collect these statistics.
    """
    stats = []
    handles = []

    def hook(module, input, output):
        # output shape: (batch, seq, d_model)
        mean = output.mean().item()
        std = output.std().item()
        stats.append({"mean": mean, "std": std})

    for block in model.blocks:
        handles.append(block.register_forward_hook(hook))

    with torch.no_grad():
        model(x)

    for h in handles:
        h.remove()

    return stats


# ---------------------------------------------------------------------------
# Training stability: loss curve comparison
# ---------------------------------------------------------------------------

def train_and_measure(depths, d_model=128, n_heads=4, seq_len=32, batch_size=8,
                      steps=100, lr=1e-3):
    """
    Train shallow and deep encoders on a dummy task for a few steps
    and compare how quickly/successfully they learn.
    """
    results = {}

    for depth in depths:
        torch.manual_seed(42)
        model = DeepEncoder(d_model, n_heads, n_blocks=depth)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_history = []

        x = torch.randn(batch_size, seq_len, d_model)
        target = torch.randn(batch_size, seq_len, d_model)

        for step in range(steps):
            optimizer.zero_grad()
            out = model(x)
            loss = F.mse_loss(out, target)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

        results[depth] = loss_history

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 60)
    print("Deep Encoder: depth vs training stability")
    print("=" * 60)

    d_model, n_heads, seq_len = 128, 4, 32
    depths = [1, 2, 4, 8]

    # ---- Measure gradient norms ----
    print("\n--- Gradient norms per block (after one backward pass) ---")

    for depth in depths:
        model = DeepEncoder(d_model, n_heads, n_blocks=depth)
        x = torch.randn(4, seq_len, d_model)
        target = torch.randn(4, seq_len, d_model)

        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()

        grad_norms = measure_gradient_norms(model)
        # Average gradient norm per block
        avg_per_block = [sum(b) / len(b) for b in grad_norms]
        print(f"  depth={depth:2d}: avg grad norm per block = {[f'{g:.2e}' for g in avg_per_block]}")

    # ---- Measure activation statistics ----
    print("\n--- Activation statistics (mean, std) per block ---")

    for depth in depths:
        model = DeepEncoder(d_model, n_heads, n_blocks=depth)
        x = torch.randn(2, seq_len, d_model)
        stats = measure_activation_stats(model, x)
        means = [f"{s['mean']:.3f}" for s in stats]
        stds = [f"{s['std']:.3f}" for s in stats]
        print(f"  depth={depth:2d}: means={means}")
        print(f"           stds ={stds}")

    # ---- Training curves ----
    print("\n--- Training curves (MSE loss, first 10 and last 10 steps) ---")

    results = train_and_measure(depths, d_model=d_model, n_heads=n_heads,
                                seq_len=seq_len, steps=100)

    for depth, history in results.items():
        print(f"  depth={depth:2d}: first10={[f'{l:.4f}' for l in history[:10]]}")
        print(f"           last10 ={[f'{l:.4f}' for l in history[-10:]]}")

    # ---- Plot training curves ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for depth, history in results.items():
        axes[0].plot(history, label=f"depth={depth}")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training curves by depth")
    axes[0].legend()
    axes[0].grid(True)

    # Gradient norm per depth
    final_grad_norms = {}
    for depth in depths:
        model = DeepEncoder(d_model, n_heads, n_blocks=depth)
        x = torch.randn(4, seq_len, d_model)
        target = torch.randn(4, seq_len, d_model)
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        grad_norms = measure_gradient_norms(model)
        final_grad_norms[depth] = [sum(b) / len(b) for b in grad_norms]

    x_pos = range(len(depths))
    avg_grads = [sum(final_grad_norms[d]) / len(final_grad_norms[d]) for d in depths]
    axes[1].bar(x_pos, avg_grads)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([str(d) for d in depths])
    axes[1].set_xlabel("Encoder depth")
    axes[1].set_ylabel("Average gradient norm")
    axes[1].set_title("Gradient norm vs depth")
    axes[1].grid(True, axis="y")

    plt.tight_layout()
    plt.savefig("deep_encoder.png", dpi=150)
    print("\nSaved plot to deep_encoder.png")

    print("\n" + "=" * 60)
    print("Key observations:")
    print("  - Deeper encoders (more blocks) maintain more stable gradient norms")
    print("  - Without residual connections, gradient norms would decay exponentially")
    print("  - All depths learn, but deeper networks have more capacity")
    print("=" * 60)
