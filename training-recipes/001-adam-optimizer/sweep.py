"""
sweep.py — Learning rate sweep with both our Adam and torch.optim.Adam.

We sample learning rates logarithmically from 1e-5 to 1e-1 and run a short
training run for each. The resulting loss landscape is shown as a heatmap
so you can see which LR ranges work well.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from adam import Adam as AdamFromScratch


class SimpleNet(nn.Module):
    def __init__(self, input_dim: int = 20, hidden: int = 64, output_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def run_training(lr: float, use_scratch: bool, seed: int = 42, epochs: int = 20):
    """Train a model with the given LR and return final loss."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    input_dim, hidden, output_dim = 20, 64, 5
    model = SimpleNet(input_dim, hidden, output_dim)

    if use_scratch:
        optimizer = AdamFromScratch(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X = torch.randn(512, input_dim)
    y = torch.randint(0, output_dim, (512,))

    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(X), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def main():
    # Learning rates to sweep (log-spaced)
    lrs = np.logspace(-5, -1, 12).tolist()
    epochs = 20
    optimizer_names = ["Our Adam", "torch.optim.Adam"]

    results = {name: [] for name in optimizer_names}

    print(f"Sweeping {len(lrs)} learning rates from {lrs[0]:.2e} to {lrs[-1]:.2e}")
    print(f"Running {epochs} epochs per configuration\n")

    for lr in lrs:
        for name, use_scratch in [("Our Adam", True), ("torch.optim.Adam", False)]:
            losses = run_training(lr, use_scratch, seed=42, epochs=epochs)
            final_loss = losses[-1]
            results[name].append(final_loss)
            print(f"  lr={lr:.4e} | {name} | final_loss={final_loss:.4f}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, name in zip(axes, optimizer_names):
        ax.semilogx(lrs, results[name], "o-", linewidth=2, markersize=5)
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Final Loss")
        ax.set_title(f"{name} — LR Sweep")
        ax.grid(True, alpha=0.3)
        ax.axvline(lrs[np.argmin(results[name])], color="red", linestyle="--",
                   label=f"Best LR = {lrs[np.argmin(results[name])]:.2e}")
        ax.legend()

    fig.suptitle("Learning Rate Sweep: Our Adam vs torch.optim.Adam", fontsize=14)
    plt.tight_layout()
    plt.savefig("sweep.png", dpi=150)
    print("\nSaved sweep.png")

    # Summary table
    print("\n--- Summary ---")
    for name in optimizer_names:
        best_idx = np.argmin(results[name])
        print(f"  {name}: best_lr={lrs[best_idx]:.2e}, best_loss={results[name][best_idx]:.4f}")

    plt.show()


if __name__ == "__main__":
    main()
