"""
Compare SGD, Adam, and AdamW on a Simple Training Problem

This script trains three identical models with different optimizers
and plots the loss curves for comparison.

The problem: minimize (w - target)^2
Initial: w = 0, target = 5

Expected behavior:
- SGD with momentum: smooth convergence, slowest initially
- Adam: fast initial progress due to adaptive learning rates
- AdamW: similar to Adam but with proper weight decay regularization
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_with_optimizer(optimizer_cls, optimizer_kwargs, steps=100, verbose=False):
    """
    Train a single weight toward a target using the specified optimizer.

    Args:
        optimizer_cls: Optimizer class (e.g., torch.optim.SGD)
        optimizer_kwargs: Dict of kwargs for the optimizer
        steps: Number of training steps
        verbose: If True, print progress

    Returns:
        List of loss values at each step
    """
    target = 5.0
    w = torch.tensor([0.0], requires_grad=True)

    opt = optimizer_cls([w], **optimizer_kwargs)

    losses = []
    for step in range(steps):
        loss = (w - target) ** 2
        losses.append(loss.item())

        if verbose and step % 20 == 0:
            print(f"  Step {step:3d}: loss = {loss.item():.6f}")

        loss.backward()
        opt.step()
        opt.zero_grad()

    return losses


def compare_optimizers():
    """Compare all three optimizers on the same problem."""
    steps = 100

    print("Training with SGD (momentum=0.9, lr=0.1)...")
    sgd_losses = train_with_optimizer(
        torch.optim.SGD,
        {"lr": 0.1, "momentum": 0.9},
        steps=steps,
        verbose=True
    )

    print("Training with Adam (lr=0.1)...")
    adam_losses = train_with_optimizer(
        torch.optim.Adam,
        {"lr": 0.1},
        steps=steps,
        verbose=True
    )

    print("Training with AdamW (lr=0.1, weight_decay=0.01)...")
    adamw_losses = train_with_optimizer(
        torch.optim.AdamW,
        {"lr": 0.1, "weight_decay": 0.01},
        steps=steps,
        verbose=True
    )

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sgd_losses, label="SGD + Momentum", linewidth=2)
    plt.plot(adam_losses, label="Adam", linewidth=2)
    plt.plot(adamw_losses, label="AdamW", linewidth=2)

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Optimizer Comparison: SGD vs Adam vs AdamW")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    plt.tight_layout()
    plt.savefig("optimizer_comparison.png", dpi=150)
    plt.show()

    # Print final losses
    print("\n" + "=" * 50)
    print("Final Losses:")
    print(f"  SGD:     {sgd_losses[-1]:.6f}")
    print(f"  Adam:    {adam_losses[-1]:.6f}")
    print(f"  AdamW:   {adamw_losses[-1]:.6f}")

    return {
        "sgd": sgd_losses,
        "adam": adam_losses,
        "adamw": adamw_losses,
    }


def compare_on_mlp():
    """
    Compare optimizers on a small MLP for more realistic behavior.
    Uses a simple regression task: predict y = x @ W + b.
    """
    torch.manual_seed(42)

    # Simple dataset: y = x @ true_w + true_b
    true_w = torch.tensor([[2.0, -1.0], [0.5, 1.0]])
    true_b = torch.tensor([1.0, -0.5])
    n_samples = 1000

    x = torch.randn(n_samples, 2)
    y = x @ true_w + true_b

    # Simple MLP
    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 2)
    )

    # Optimizer configs to compare
    configs = [
        ("SGD (momentum=0.9)", torch.optim.SGD, {"lr": 1e-3, "momentum": 0.9}),
        ("Adam (lr=1e-3)", torch.optim.Adam, {"lr": 1e-3}),
        ("AdamW (lr=1e-3, wd=0.01)", torch.optim.AdamW, {"lr": 1e-3, "weight_decay": 0.01}),
    ]

    results = {}

    for name, opt_cls, opt_kwargs in configs:
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        optimizer = opt_cls(model.parameters(), **opt_kwargs)
        criterion = nn.MSELoss()

        losses = []
        for epoch in range(100):
            pred = model(x)
            loss = criterion(pred, y)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        results[name] = losses
        print(f"{name}: final loss = {losses[-1]:.6f}")

    # Plot
    plt.figure(figsize=(10, 6))
    for name, losses in results.items():
        plt.plot(losses, label=name, linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Optimizer Comparison on MLP Regression")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    plt.tight_layout()
    plt.savefig("optimizer_comparison_mlp.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Part 1: Simple 1D Problem")
    print("=" * 60)
    compare_optimizers()

    print("\n" + "=" * 60)
    print("Part 2: MLP on Synthetic Regression Data")
    print("=" * 60)
    compare_on_mlp()
