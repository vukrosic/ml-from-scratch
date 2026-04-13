"""
Placeholder loss curves comparing precision modes.

Trains a 4-layer MLP in FP32, FP16, and BF16 and saves loss trajectories.
Generates plots comparing convergence across precision modes.

Run: python convergence.py [--hidden-size N] [--batch-size N] [--epochs N]
"""

import argparse
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x):
        return self.net(x)


def train_model(precision, hidden_size, batch_size, epochs, device):
    """Train model in given precision and return loss curve."""
    model = MLP(hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    if precision == "fp16":
        scaler = amp.GradScaler()
    else:
        scaler = None

    losses = []
    torch.manual_seed(42)

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for _ in range(100):  # Fixed batches per epoch
            inputs = torch.randn(batch_size, 784, device=device)
            targets = torch.randint(0, 10, (batch_size,), device=device)

            optimizer.zero_grad()

            dtype = torch.bfloat16 if precision == "bf16" else torch.float16
            with amp.autocast(dtype=dtype):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"  {precision} epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Configuration: hidden={args.hidden_size}, batch={args.batch_size}, epochs={args.epochs}")
    print()

    results = {}
    for precision in ["fp32", "fp16", "bf16"]:
        print(f"Training {precision}...")
        losses = train_model(precision, args.hidden_size, args.batch_size, args.epochs, device)
        results[precision] = losses
        torch.cuda.empty_cache()

    # Plot
    plt.figure(figsize=(10, 6))
    epochs = range(1, args.epochs + 1)

    plt.plot(epochs, results["fp32"], label="FP32", linewidth=2)
    plt.plot(epochs, results["fp16"], label="FP16 + GradScaler", linewidth=2, linestyle="--")
    plt.plot(epochs, results["bf16"], label="BF16 + GradScaler", linewidth=2, linestyle=":")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Convergence: FP32 vs FP16 vs BF16")
    plt.legend()
    plt.grid(True)
    plt.savefig("convergence.png", dpi=150)
    print()
    print("Saved convergence plot to convergence.png")


if __name__ == "__main__":
    main()
