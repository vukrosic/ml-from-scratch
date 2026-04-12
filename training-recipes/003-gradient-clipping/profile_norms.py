"""
profile_norms.py — Track gradient norm over training and visualize where clips occur.

We record:
1. Pre-clip gradient norm (what the gradient norm would be without clipping)
2. Post-clip gradient norm (after applying gradient clipping)
3. Per-layer gradient norms

The difference between pre and post clip shows exactly when and where clipping activates.
This helps understand the dynamics of gradient explosion and how clipping affects training.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from grad_clip import GradientClipper


class MLP(nn.Module):
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


def compute_grad_norms(model):
    """Compute gradient norm for each parameter and total norm."""
    total_norm = 0.0
    per_layer = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            layer_norm = p.grad.norm(2).item()
            per_layer[name] = layer_norm
            total_norm += layer_norm ** 2
    total_norm = total_norm ** 0.5
    return total_norm, per_layer


def train_with_logging(model, optimizer, clipper, loader, device, clip_enabled: bool,
                        num_epochs: int = 50):
    """
    Train a model while recording gradient norms before and after clipping.

    Returns
    -------
    history : dict
        Dictionary with lists: pre_clip_norms, post_clip_norms, losses, per_layer_norms
    """
    history = {
        "pre_clip_norms": [],
        "post_clip_norms": [],
        "losses": [],
        "per_layer_norms": [],
    }

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_pre = []
        epoch_post = []
        per_layer_accum = {}

        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = nn.functional.cross_entropy(logits, batch_y)
            loss.backward()

            # Record pre-clip norm
            pre_norm, _ = compute_grad_norms(model)
            epoch_pre.append(pre_norm)

            # Apply clipping if enabled
            if clip_enabled:
                clipper.clip()

            # Record post-clip norm
            post_norm, per_layer = compute_grad_norms(model)
            epoch_post.append(post_norm)

            # Accumulate per-layer norms
            for name, ln in per_layer.items():
                if name not in per_layer_accum:
                    per_layer_accum[name] = []
                per_layer_accum[name].append(ln)

            optimizer.step()
            epoch_loss += loss.item()

        history["pre_clip_norms"].append(sum(epoch_pre) / len(epoch_pre))
        history["post_clip_norms"].append(sum(epoch_post) / len(epoch_post))
        history["losses"].append(epoch_loss / len(loader))
        history["per_layer_norms"].append(per_layer_accum)

    return history


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ---- Setup: create conditions for gradient explosion ----
    input_dim = 20
    hidden = 128
    depth = 5
    output_dim = 5
    batch_size = 64
    epochs = 40
    lr = 0.8  # High LR to provoke gradient explosion
    clip_threshold = 2.0
    seed = 42

    print(f"Config: depth={depth}, hidden={hidden}, lr={lr}, clip_threshold={clip_threshold}\n")

    set_seed(seed)
    n_train = 1024
    X = torch.randn(n_train, input_dim)
    y = torch.randint(0, output_dim, (n_train,))

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    # ---- Train WITHOUT clipping ----
    set_seed(seed)
    model_no_clip = MLP(input_dim, hidden, output_dim).to(device)
    optimizer_no_clip = torch.optim.SGD(model_no_clip.parameters(), lr=lr)

    print("Training WITHOUT gradient clipping...")
    history_no_clip = train_with_logging(
        model_no_clip, optimizer_no_clip, None, loader, device,
        clip_enabled=False, num_epochs=epochs
    )

    # ---- Train WITH clipping ----
    set_seed(seed)
    model_clip = MLP(input_dim, hidden, output_dim).to(device)
    optimizer_clip = torch.optim.SGD(model_clip.parameters(), lr=lr)
    clipper = GradientClipper(model_clip.parameters(), max_norm=clip_threshold)

    print("Training WITH gradient clipping...")
    history_clip = train_with_logging(
        model_clip, optimizer_clip, clipper, loader, device,
        clip_enabled=True, num_epochs=epochs
    )

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss comparison
    axes[0, 0].plot(history_no_clip["losses"], label="No clip", linewidth=2, alpha=0.7)
    axes[0, 0].plot(history_clip["losses"], label="With clip", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Loss: With vs Without Clipping")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Pre-clip vs Post-clip norm (with clipping)
    axes[0, 1].plot(history_clip["pre_clip_norms"], label="Pre-clip norm", linewidth=2,
                    color="red", alpha=0.7)
    axes[0, 1].plot(history_clip["post_clip_norms"], label="Post-clip norm", linewidth=2,
                    color="green")
    axes[0, 1].axhline(clip_threshold, color="black", linestyle="--",
                      label=f"Clip threshold={clip_threshold}")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Gradient Norm")
    axes[0, 1].set_title("Gradient Norm Over Time (With Clipping)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. No-clip gradient norm (showing explosion)
    axes[1, 0].plot(history_no_clip["pre_clip_norms"], linewidth=2, color="red", alpha=0.7)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Gradient Norm")
    axes[1, 0].set_title("Gradient Norm Without Clipping (Explosion)")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Per-layer norms at a specific epoch (e.g., last)
    last_layer_norms = history_clip["per_layer_norms"][-1]
    layer_names = list(last_layer_norms.keys())
    layer_norms = [last_layer_norms[name] for name in layer_names]

    # Get final per-layer norms for clipped vs no-clip
    set_seed(seed)
    model_final_clip = MLP(input_dim, hidden, output_dim).to(device)
    optimizer_final_clip = torch.optim.SGD(model_final_clip.parameters(), lr=lr)
    clipper_final = GradientClipper(model_final_clip.parameters(), max_norm=clip_threshold)

    set_seed(seed)
    model_final_no_clip = MLP(input_dim, hidden, output_dim).to(device)
    optimizer_final_no_clip = torch.optim.SGD(model_final_no_clip.parameters(), lr=lr)

    # Run one batch with both to get per-layer norms
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # No clip
        model_final_no_clip.zero_grad()
        loss = nn.functional.cross_entropy(model_final_no_clip(batch_x), batch_y)
        loss.backward()
        _, no_clip_layers = compute_grad_norms(model_final_no_clip)

        # With clip
        model_final_clip.zero_grad()
        loss = nn.functional.cross_entropy(model_final_clip(batch_x), batch_y)
        loss.backward()
        clipper_final.clip()
        _, clip_layers = compute_grad_norms(model_final_clip)
        break

    x_pos = range(len(layer_names))
    width = 0.35
    axes[1, 1].bar([p - width/2 for p in x_pos],
                   [no_clip_layers[n] for n in layer_names],
                   width, label="No clip", color="red", alpha=0.7)
    axes[1, 1].bar([p + width/2 for p in x_pos],
                   [clip_layers[n] for n in layer_names],
                   width, label="With clip", color="green", alpha=0.7)
    axes[1, 1].set_xlabel("Layer")
    axes[1, 1].set_ylabel("Gradient Norm")
    axes[1, 1].set_title("Per-Layer Gradient Norms (Final Epoch)")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([n.split('.')[1] if '.' in n else n for n in layer_names],
                               rotation=45, fontsize=8)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("profile_norms.png", dpi=150)
    print("\nSaved profile_norms.png")

    # ---- Summary stats ----
    print("\n--- Gradient Clipping Statistics ---")
    clipped_steps = sum(
        1 for pre, post in zip(history_clip["pre_clip_norms"], history_clip["post_clip_norms"])
        if pre > post * 1.01
    )
    print(f"Epochs where clipping was active: {clipped_steps}/{epochs}")
    print(f"Max pre-clip norm (no clip):  {max(history_no_clip['pre_clip_norms']):.4f}")
    print(f"Max pre-clip norm (with clip): {max(history_clip['pre_clip_norms']):.4f}")
    print(f"Final loss (no clip):  {history_no_clip['losses'][-1]:.4f}")
    print(f"Final loss (with clip): {history_clip['losses'][-1]:.4f}")

    plt.show()


if __name__ == "__main__":
    main()