"""
scheduler_recipes.py — Layer-wise Learning Rate Decay (LLRD).

In deep networks (transformers especially), earlier layers learn more general
features and should change more slowly than later layers. LLRD applies a
geometric decay to the learning rate of each layer, working from the output
backwards:

    lr_layer_i = base_lr * decay_rate ** (num_layers - i)

For example, with base_lr=1e-3 and decay_rate=0.95 for a 12-layer model:
    Layer 12 (closest to output): 1e-3
    Layer 11:                     1e-3 * 0.95
    Layer 1  (closest to input):  1e-3 * 0.95^11 ≈ 0.6e-3

This is combined with a base schedule (e.g., cosine decay). We implement LLRD
as a modifier that wraps an optimizer and applies per-layer LR scaling.

We compare: uniform LR vs LLRD for a transformer-like network.
"""

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class LayerWiseLRDecay:
    """
    Layer-wise Learning Rate Decay wrapper.

    Applies a multiplicative decay to each parameter group, working from the
    last layer (highest LR) to the first layer (lowest LR).

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Base optimizer.
    decay_rate : float, default 0.95
        LR multiplier per layer when moving backward through the network.
    num_layers : int
        Number of layers to apply decay across.
    """

    def __init__(
        self,
        optimizer,
        decay_rate: float = 0.95,
        num_layers: int = 12,
    ):
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        self.num_layers = num_layers

        # Assign a LR multiplier to each param group based on its depth
        self.param_groups = []
        for i, pg in enumerate(optimizer.param_groups):
            # Work backward: last layer gets multiplier 1.0, first gets decay^11
            depth = num_layers - 1 - i
            multiplier = decay_rate ** depth
            pg_copy = {**pg, "lr": pg["lr"] * multiplier, "_llrd_mult": multiplier}
            self.param_groups.append(pg_copy)

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self._param_groups

    @param_groups.setter
    def param_groups(self, value):
        self._param_groups = value


def build_llrd_param_groups(model, base_lr: float, decay_rate: float):
    """
    Build optimizer param groups with layer-wise LR decay.

    Parameters are ordered from last layer to first (reverse order of forward pass).
    We rely on the model's parameter order, which for nn.Sequential is insertion order.

    Returns a list of param groups compatible with torch.optim.Optimizer.
    """
    params = list(model.parameters())
    num_params = len(params)
    num_layers = num_params // 2  # Each layer has weight + bias

    param_groups = []
    for i, param in enumerate(params):
        layer_idx = i // 2  # 0-indexed layer
        # Last layer (layer_idx = num_layers-1) gets base_lr, first gets decay^...
        depth = num_layers - 1 - layer_idx
        lr = base_lr * (decay_rate ** depth)
        param_groups.append({"params": [param], "lr": lr})

    return param_groups


class TransformerBlock(nn.Module):
    """A single transformer-style layer for testing LLRD."""

    def __init__(self, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.lin1 = nn.Linear(d_model, d_model * 4)
        self.lin2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        lin_out = self.lin2(torch.nn.functional.gelu(self.lin1(x)))
        x = self.norm2(x + lin_out)
        return x


def make_model(num_layers: int = 6, d_model: int = 64):
    """Build a small transformer-like stack."""
    return nn.Sequential(
        nn.Linear(d_model, d_model),
        *[TransformerBlock(d_model) for _ in range(num_layers)],
        nn.Linear(d_model, d_model),
    )


def get_layer_lrs(model, optimizer):
    """Extract the LR assigned to each parameter in the optimizer."""
    param_to_lr = {}
    for pg in optimizer.param_groups:
        for p in pg["params"]:
            param_to_lr[id(p)] = pg["lr"]
    return [param_to_lr[id(p)] for p in model.parameters()]


def compare_llrd_vs_uniform():
    """Compare LR distribution and training with uniform LR vs LLRD."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_layers = 6
    d_model = 64
    base_lr = 1e-3
    decay_rate = 0.85
    epochs = 50

    torch.manual_seed(42)

    # ---- Show LR distribution ----
    model = make_model(num_layers, d_model).to(device)

    # Uniform
    opt_uniform = torch.optim.AdamW(model.parameters(), lr=base_lr)
    lrs_uniform = get_layer_lrs(model, opt_uniform)

    # LLRD
    model2 = make_model(num_layers, d_model).to(device)
    param_groups = build_llrd_param_groups(model2, base_lr, decay_rate)
    opt_llrd = torch.optim.AdamW(param_groups)
    lrs_llrd = get_layer_lrs(model2, opt_llrd)

    print(f"Number of parameters: {len(lrs_uniform)}")
    print(f"\nLR per parameter group (first 6 = layer 1, next 6 = layer 2, ...):")
    print(f"Uniform: min={min(lrs_uniform):.2e}, max={max(lrs_uniform):.2e}")
    print(f"LLRD:    min={min(lrs_llrd):.2e}, max={max(lrs_llrd):.2e}")
    print(f"LLRD decay factor per layer: {decay_rate}")

    # ---- Plot LR distribution ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(range(len(lrs_uniform)), lrs_uniform, color="#2196F3", alpha=0.7)
    axes[0].set_title("Uniform LR")
    axes[0].set_xlabel("Parameter Index")
    axes[0].set_ylabel("Learning Rate")
    axes[0].set_yscale("log")

    axes[1].bar(range(len(lrs_llrd)), lrs_llrd, color="#FF5722", alpha=0.7)
    axes[1].set_title(f"LLRD (decay={decay_rate})")
    axes[1].set_xlabel("Parameter Index")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_yscale("log")

    plt.suptitle("Layer-wise Learning Rate Distribution")
    plt.tight_layout()
    plt.savefig("llrd_distribution.png", dpi=150)

    # ---- Training comparison ----
    from schedulers import LinearWarmupCosineDecay

    X_train = torch.randn(512, d_model, d_model, device=device)
    y_train = torch.randn(512, d_model, device=device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            X_train, y_train
        ),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )

    def train_model(model, optimizer, scheduler, epochs):
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            count = 0
            for bx, by in loader:
                optimizer.zero_grad()
                pred = model(bx)
                loss = nn.functional.mse_loss(pred, by)
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item() * bx.size(0)
                count += bx.size(0)
            losses.append(epoch_loss / count)
        return losses

    # Uniform
    model_u = make_model(num_layers, d_model).to(device)
    opt_u = torch.optim.AdamW(model_u.parameters(), lr=base_lr)
    sched_u = LinearWarmupCosineDecay(opt_u, warmup_steps=5, total_steps=epochs)
    losses_u = train_model(model_u, opt_u, sched_u, epochs)

    # LLRD
    model_l = make_model(num_layers, d_model).to(device)
    pg_l = build_llrd_param_groups(model_l, base_lr, decay_rate)
    opt_l = torch.optim.AdamW(pg_l)
    sched_l = LinearWarmupCosineDecay(opt_l, warmup_steps=5, total_steps=epochs)
    losses_l = train_model(model_l, opt_l, sched_l, epochs)

    plt.figure(figsize=(8, 5))
    plt.plot(losses_u, label="Uniform LR", linewidth=2, color="#2196F3")
    plt.plot(losses_l, label=f"LLRD (decay={decay_rate})", linewidth=2, color="#FF5722")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LLRD vs Uniform LR — Training Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("llrd_comparison.png", dpi=150)
    print("\nSaved llrd_distribution.png and llrd_comparison.png")

    print(f"\nFinal loss — Uniform: {losses_u[-1]:.4f}")
    print(f"Final loss — LLRD:   {losses_l[-1]:.4f}")

    plt.show()


if __name__ == "__main__":
    compare_llrd_vs_uniform()
