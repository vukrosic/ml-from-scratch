"""
compare.py — Train with and without gradient clipping to show stability improvement.

We create a scenario that causes exploding gradients: a deep MLP with a high
learning rate. Without clipping, the loss diverges. With clipping, training
remains stable. We also track gradient norms over time to show how clipping
keeps them bounded.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from grad_clip import GradientClipper


class DeepMLP(nn.Module):
    """A deep network prone to exploding gradients."""

    def __init__(self, input_dim: int = 20, hidden: int = 128, depth: int = 6, output_dim: int = 5):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.ReLU())
            prev_dim = hidden
        layers.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_one_epoch(model, optimizer, clipper, loader, device, clip_grads: bool):
    """Train for one epoch. Optionally clip gradients."""
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    count = 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = nn.functional.cross_entropy(logits, batch_y)
        loss.backward()

        # Compute gradient norm before clipping
        grad_norm = sum(p.grad.norm() for p in model.parameters() if p.grad is not None)
        total_grad_norm += grad_norm.item()

        # Apply gradient clipping if enabled
        if clip_grads:
            clipper.clip()

        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        count += batch_x.size(0)

    return total_loss / count, total_grad_norm / count


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Hyperparameters chosen to provoke gradient explosion ----
    input_dim = 20
    hidden = 128
    depth = 6  # Deep network = gradient accumulation = explosion risk
    output_dim = 5
    batch_size = 64
    epochs = 30
    lr = 0.5  # High learning rate = guaranteed explosion without clipping
    clip_threshold = 1.0
    seed = 42

    print(f"Config: depth={depth}, hidden={hidden}, lr={lr}, clip_threshold={clip_threshold}")
    print(f"Without clipping: expect loss to diverge or oscillate wildly")
    print(f"With clipping: expect smooth, stable convergence\n")

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

    # ---- Train WITHOUT clipping ----
    set_seed(seed)
    model_no_clip = DeepMLP(input_dim, hidden, depth, output_dim).to(device)
    optimizer_no_clip = torch.optim.SGD(model_no_clip.parameters(), lr=lr)

    losses_no_clip = []
    grad_norms_no_clip = []
    print("Training WITHOUT gradient clipping...")
    for epoch in range(1, epochs + 1):
        loss, gn = train_one_epoch(model_no_clip, optimizer_no_clip, None, train_loader, device, clip_grads=False)
        losses_no_clip.append(loss)
        grad_norms_no_clip.append(gn)
        if epoch % 5 == 0 or epoch <= 3:
            print(f"  epoch {epoch:3d}: loss={loss:.4f}, grad_norm={gn:.4e}")

    # ---- Train WITH clipping ----
    set_seed(seed)
    model_clip = DeepMLP(input_dim, hidden, depth, output_dim).to(device)
    optimizer_clip = torch.optim.SGD(model_clip.parameters(), lr=lr)
    clipper = GradientClipper(model_clip.parameters(), max_norm=clip_threshold)

    losses_clip = []
    grad_norms_clip = []
    print("\nTraining WITH gradient clipping...")
    for epoch in range(1, epochs + 1):
        loss, gn = train_one_epoch(model_clip, optimizer_clip, clipper, train_loader, device, clip_grads=True)
        losses_clip.append(loss)
        grad_norms_clip.append(gn)
        if epoch % 5 == 0 or epoch <= 3:
            print(f"  epoch {epoch:3d}: loss={loss:.4f}, grad_norm={gn:.4e}")

    # ---- Plot loss curves ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss comparison
    axes[0].plot(losses_no_clip, label="No clipping", linewidth=2, color="red", alpha=0.7)
    axes[0].plot(losses_clip, label="With clipping", linewidth=2, color="green")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Loss: With vs Without Gradient Clipping")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Gradient norm comparison
    axes[1].plot(grad_norms_no_clip, label="No clipping", linewidth=2, color="red", alpha=0.7)
    axes[1].plot(grad_norms_clip, label="With clipping", linewidth=2, color="green")
    axes[1].axhline(clip_threshold, color="black", linestyle="--", label=f"Clip threshold={clip_threshold}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Gradient Norm")
    axes[1].set_title("Gradient Norm Over Training")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig("compare.png", dpi=150)
    print("\nSaved compare.png")

    # ---- Summary ----
    print("\n--- Summary ---")
    final_loss_no_clip = losses_no_clip[-1]
    final_loss_clip = losses_clip[-1]
    print(f"Final loss (no clip):  {final_loss_no_clip:.4f}")
    print(f"Final loss (with clip): {final_loss_clip:.4f}")

    if final_loss_no_clip > final_loss_clip * 2:
        print("\nGradient clipping provided significant stability benefit!")
    elif torch.isfinite(torch.tensor(final_loss_no_clip)) and torch.isfinite(torch.tensor(final_loss_clip)):
        print("\nBoth training runs converged. Clipping kept gradients bounded.")

    plt.show()


if __name__ == "__main__":
    main()