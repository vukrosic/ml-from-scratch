"""
compare.py — Compare learning rate schedules by training identical models.

We run the same MLP with four different schedulers and plot the loss curves
side-by-side. This shows how the choice of schedule affects convergence speed
and final loss.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from schedulers import ConstantLR, StepDecay, CosineAnnealing, LinearWarmupCosineDecay


class SimpleMLP(nn.Module):
    """Small MLP for a classification task."""

    def __init__(self, input_dim: int = 20, hidden: int = 128, output_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def make_dataset(n: int, input_dim: int = 20, output_dim: int = 5, seed: int = 42):
    """Create a synthetic classification dataset."""
    torch.manual_seed(seed)
    X = torch.randn(n, input_dim)
    y = torch.randint(0, output_dim, (n,))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return loader


def train_one_epoch(model, optimizer, scheduler, loader, device):
    """Train for one epoch, step the scheduler, return average loss."""
    model.train()
    total_loss = 0.0
    count = 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = nn.functional.cross_entropy(logits, batch_y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * batch_x.size(0)
        count += batch_x.size(0)

    return total_loss / count


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_scheduler(name: str, optimizer, total_steps: int):
    """Create a scheduler by name."""
    if name == "constant":
        return ConstantLR(optimizer)
    elif name == "step":
        return StepDecay(optimizer, step_size=15, gamma=0.5)
    elif name == "cosine":
        return CosineAnnealing(optimizer, total_steps=total_steps)
    elif name == "warmup_cosine":
        return LinearWarmupCosineDecay(
            optimizer, warmup_steps=int(0.1 * total_steps), total_steps=total_steps
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Config ----
    input_dim = 20
    hidden = 128
    output_dim = 5
    epochs = 80
    base_lr = 1e-3
    seed = 42

    # ---- Dataset ----
    train_loader = make_dataset(1024, input_dim, output_dim, seed)
    val_loader = make_dataset(256, input_dim, output_dim, seed + 1)

    scheduler_names = ["constant", "step", "cosine", "warmup_cosine"]
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    results = {}

    plt.figure(figsize=(10, 6))

    for name, color in zip(scheduler_names, colors):
        set_seed(seed)
        model = SimpleMLP(input_dim, hidden, output_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
        scheduler = get_scheduler(name, optimizer, total_steps=epochs)

        losses = []
        for epoch in range(epochs):
            loss = train_one_epoch(model, optimizer, scheduler, train_loader, device)
            losses.append(loss)

        results[name] = losses
        label = {
            "constant": "Constant LR",
            "step": "Step Decay (gamma=0.5, step=15)",
            "cosine": "Cosine Annealing",
            "warmup_cosine": "Linear Warmup + Cosine Decay",
        }[name]
        plt.plot(losses, label=label, color=color, linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Learning Rate Schedules — Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("compare.png", dpi=150)
    print("Saved plot to compare.png")

    # ---- Print final losses ----
    print("\nFinal training losses:")
    for name, losses in results.items():
        print(f"  {name:>15}: {losses[-1]:.4f}")

    plt.show()


if __name__ == "__main__":
    main()
