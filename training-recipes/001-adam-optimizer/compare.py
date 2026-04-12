"""
compare.py — Compare our from-scratch Adam against torch.optim.Adam.

We run identical training runs (same model, same data, same seed) with both
optimizers and plot the loss curves. They should overlap perfectly.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from adam import Adam as AdamFromScratch


class SimpleNet(nn.Module):
    """Tiny MLP for a classification task."""

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


def train_one_epoch(model, optimizer, loader, device):
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

        total_loss += loss.item() * batch_x.size(0)
        count += batch_x.size(0)

    return total_loss / count


def set_seed(seed: int = 42):
    """Seed everything for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Hyperparameters ----
    input_dim = 20
    hidden = 64
    output_dim = 5
    batch_size = 64
    epochs = 30
    lr = 1e-3
    seed = 42

    # ---- Synthetic dataset ----
    set_seed(seed)
    n_train = 1024
    n_val = 256

    X_train = torch.randn(n_train, input_dim)
    y_train = torch.randint(0, output_dim, (n_train,))
    X_val = torch.randn(n_val, input_dim)
    y_val = torch.randint(0, output_dim, (n_val,))

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )

    # ---- Train with our Adam ----
    set_seed(seed)
    model_scratch = SimpleNet(input_dim, hidden, output_dim).to(device)
    opt_scratch = AdamFromScratch(model_scratch.parameters(), lr=lr)

    losses_scratch = []
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model_scratch, opt_scratch, train_loader, device)
        losses_scratch.append(loss)

    # ---- Train with PyTorch Adam ----
    set_seed(seed)
    model_torch = SimpleNet(input_dim, hidden, output_dim).to(device)
    opt_torch = torch.optim.Adam(model_torch.parameters(), lr=lr)

    losses_torch = []
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model_torch, opt_torch, val_loader, device)
        losses_torch.append(loss)

    # ---- Plot ----
    plt.figure(figsize=(8, 5))
    plt.plot(losses_scratch, label="Our Adam (from scratch)", linewidth=2)
    plt.plot(losses_torch, label="torch.optim.Adam", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Custom Adam vs PyTorch Adam — Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("compare.png", dpi=150)
    print("Saved plot to compare.png")

    # ---- Verify numerical match ----
    max_diff = max(
        abs(a - b) for a, b in zip(losses_scratch, losses_torch)
    )
    print(f"\nMax loss difference between implementations: {max_diff:.2e}")
    if max_diff < 1e-4:
        print("Match: losses are essentially identical.")
    else:
        print("WARNING: losses diverge — investigate.")

    plt.show()


if __name__ == "__main__":
    main()
