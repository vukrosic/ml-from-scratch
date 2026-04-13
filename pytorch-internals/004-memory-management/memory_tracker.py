"""Track GPU memory allocation at every layer of a multi-layer MLP."""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helper: read the two numbers PyTorch exposes
# ---------------------------------------------------------------------------

def mem_stats(label: str) -> None:
    """Print allocated and reserved VRAM after each step.

    allocated  — bytes currently held by live tensors
    reserved   — bytes the caching allocator has claimed from the OS
                 (reserved >= allocated because freed blocks stay pooled)
    """
    allocated = torch.cuda.memory_allocated() / 1024**2  # MiB
    reserved  = torch.cuda.memory_reserved()  / 1024**2  # MiB
    print(f"{label:<35}  allocated={allocated:7.2f} MiB  reserved={reserved:7.2f} MiB")


# ---------------------------------------------------------------------------
# Model: a deep MLP with named layers so we can hook each one
# ---------------------------------------------------------------------------

class TrackedMLP(nn.Module):
    """Five-layer MLP where every activation is stored (no checkpointing)."""

    def __init__(self, dim: int = 2048):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(5)
        ])
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mem_stats("after input")
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            mem_stats(f"after layer {i}")
        return x


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda"
    dim    = 2048
    batch  = 256

    model = TrackedMLP(dim).to(device)

    # Weights alone already occupy memory — show the baseline
    mem_stats("after model.to(device)")

    x = torch.randn(batch, dim, device=device)
    mem_stats("after input tensor created")

    # Forward pass — activations accumulate because we'll call .backward()
    loss = model(x).mean()
    mem_stats("after forward + loss")

    loss.backward()
    mem_stats("after backward")

    # Deleting the output graph frees activation memory
    del loss
    torch.cuda.empty_cache()
    mem_stats("after del + empty_cache")
