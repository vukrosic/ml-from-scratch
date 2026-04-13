"""Gradient checkpointing from scratch — trade compute for memory.

Key idea
--------
During a normal forward pass PyTorch keeps every intermediate activation alive
so it can use them in the backward pass.  For a deep network that is a lot of
VRAM.

Checkpointing discards those activations after each segment and re-runs the
forward pass *again* during the backward pass to reconstruct them.  Cost: one
extra forward pass.  Benefit: activation memory drops from O(layers) to O(1).

We implement the core mechanism manually before showing torch.utils.checkpoint.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Shared model definition
# ---------------------------------------------------------------------------

def make_block(dim: int) -> nn.Sequential:
    """Single transformer-style feed-forward block."""
    return nn.Sequential(
        nn.Linear(dim, dim * 4),
        nn.GELU(),
        nn.Linear(dim * 4, dim),
    )


class DeepMLP(nn.Module):
    """Stack of n feed-forward blocks."""

    def __init__(self, dim: int = 1024, n_blocks: int = 12):
        super().__init__()
        self.blocks = nn.ModuleList([make_block(dim) for _ in range(n_blocks)])

    def forward_plain(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward — all activations kept in memory."""
        for block in self.blocks:
            x = x + block(x)          # residual connection
        return x

    def forward_checkpointed(self, x: torch.Tensor) -> torch.Tensor:
        """Checkpointed forward — activations are recomputed on backward."""
        for block in self.blocks:
            # checkpoint re-runs `block` during backward instead of storing
            # its outputs.  use_reentrant=False is the modern API.
            x = x + checkpoint(block, x, use_reentrant=False)
        return x


# ---------------------------------------------------------------------------
# Memory measurement helper
# ---------------------------------------------------------------------------

def peak_mem_mib(fn, *args) -> float:
    """Run fn(*args), return peak allocated MiB (resets stats first)."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    out = fn(*args)
    out.sum().backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


# ---------------------------------------------------------------------------
# Manual checkpoint (no library) — shows the raw idea
# ---------------------------------------------------------------------------

class ManualCheckpoint(torch.autograd.Function):
    """Re-run fn during backward to get the activations we threw away."""

    @staticmethod
    def forward(ctx, fn, *inputs):
        ctx.fn = fn
        # Save only the inputs, not the (potentially large) outputs
        ctx.save_for_backward(*inputs)
        with torch.no_grad():
            return fn(*inputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs = ctx.saved_tensors
        # Re-run forward with grad tracking to get a fresh computation graph
        with torch.enable_grad():
            inputs_detached = tuple(x.detach().requires_grad_(x.requires_grad) for x in inputs)
            output = ctx.fn(*inputs_detached)
        torch.autograd.backward(output, grad_outputs)
        # Return None for fn, then the input grads
        return (None,) + tuple(x.grad if x.requires_grad else None for x in inputs_detached)


def manual_checkpoint(fn, *inputs):
    return ManualCheckpoint.apply(fn, *inputs)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device    = "cuda"
    dim       = 1024
    n_blocks  = 12
    batch     = 64

    model = DeepMLP(dim=dim, n_blocks=n_blocks).to(device)
    x     = torch.randn(batch, dim, device=device, requires_grad=True)

    # --- plain forward ---
    peak_plain = peak_mem_mib(model.forward_plain, x.clone().requires_grad_(True))

    # --- checkpointed forward ---
    peak_ckpt  = peak_mem_mib(model.forward_checkpointed, x.clone().requires_grad_(True))

    print(f"Peak memory — plain forward:        {peak_plain:.1f} MiB")
    print(f"Peak memory — checkpointed forward: {peak_ckpt:.1f} MiB")
    print(f"Memory saving:                      {peak_plain - peak_ckpt:.1f} MiB  "
          f"({100*(peak_plain-peak_ckpt)/peak_plain:.0f}% reduction)")

    # --- manual checkpoint smoke-test ---
    block = model.blocks[0]
    xi    = x[:4].clone().detach().requires_grad_(True)
    out   = manual_checkpoint(block, xi)
    out.sum().backward()
    print(f"\nManual checkpoint output shape: {out.shape}  grad OK: {xi.grad is not None}")
