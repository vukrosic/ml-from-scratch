"""Visualize the CUDA caching allocator state using torch.cuda.memory_snapshot().

The snapshot is a list of segment dicts.  Each segment maps to one cudaMalloc
call.  Inside each segment are one or more blocks — some active (holding a live
tensor), some free (still reserved but reusable without a new cudaMalloc).

We walk the snapshot and print a compact summary + an ASCII bar chart so you
can see fragmentation at a glance.
"""

import torch
import torch.nn as nn
from collections import defaultdict


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def parse_snapshot() -> dict:
    """Return a summary dict from the current allocator snapshot."""
    snapshot = torch.cuda.memory_snapshot()

    total_reserved  = 0
    total_allocated = 0
    total_free      = 0
    n_segments      = len(snapshot)
    n_blocks        = 0

    pool_stats: dict[str, dict] = defaultdict(lambda: {"reserved": 0, "allocated": 0, "free": 0, "segments": 0})

    for seg in snapshot:
        pool = seg.get("segment_pool_id", "unknown")
        pool_key = str(pool)

        seg_reserved = seg["total_size"]
        seg_alloc    = sum(b["size"] for b in seg["blocks"] if b["state"] == "active_allocated")
        seg_free     = seg_reserved - seg_alloc

        total_reserved  += seg_reserved
        total_allocated += seg_alloc
        total_free      += seg_free
        n_blocks        += len(seg["blocks"])

        pool_stats[pool_key]["reserved"]  += seg_reserved
        pool_stats[pool_key]["allocated"] += seg_alloc
        pool_stats[pool_key]["free"]      += seg_free
        pool_stats[pool_key]["segments"]  += 1

    return {
        "n_segments":      n_segments,
        "n_blocks":        n_blocks,
        "total_reserved":  total_reserved,
        "total_allocated": total_allocated,
        "total_free":      total_free,
        "pool_stats":      pool_stats,
        "raw":             snapshot,
    }


def mib(n_bytes: int) -> str:
    return f"{n_bytes / 1024**2:.2f} MiB"


# ---------------------------------------------------------------------------
# ASCII bar chart
# ---------------------------------------------------------------------------

BAR_WIDTH = 50

def bar_chart(label: str, allocated: int, reserved: int) -> str:
    """One line: [████░░░░] label  allocated/reserved."""
    if reserved == 0:
        return f"  {label:<30}  (empty)"
    ratio = allocated / reserved
    filled = int(ratio * BAR_WIDTH)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    pct = ratio * 100
    return (f"  {label:<30}  [{bar}]  "
            f"{mib(allocated)} / {mib(reserved)}  ({pct:.0f}% used)")


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

def print_snapshot_report() -> None:
    info = parse_snapshot()

    print("=" * 80)
    print("  CUDA Allocator Snapshot")
    print("=" * 80)
    print(f"  Segments : {info['n_segments']}")
    print(f"  Blocks   : {info['n_blocks']}")
    print(f"  Reserved : {mib(info['total_reserved'])}")
    print(f"  Allocated: {mib(info['total_allocated'])}")
    print(f"  Free pool: {mib(info['total_free'])}")
    frag = info['total_free'] / max(info['total_reserved'], 1) * 100
    print(f"  Fragmentation: {frag:.1f}%  (free / reserved)")
    print()

    # One bar per pool
    print("  Per-pool breakdown (allocated / reserved):")
    for pool_key, ps in sorted(info["pool_stats"].items()):
        label = f"pool {pool_key} ({ps['segments']} segs)"
        print(bar_chart(label, ps["allocated"], ps["reserved"]))

    print()


# ---------------------------------------------------------------------------
# Build some tensors, then snapshot
# ---------------------------------------------------------------------------

class SmallMLP(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    device = "cuda"
    dim    = 512
    batch  = 128

    model = SmallMLP(dim).to(device)
    x     = torch.randn(batch, dim, device=device)

    # --- snapshot 1: weights loaded, no activations yet ---
    print(">>> Snapshot 1: model weights on device, no forward pass\n")
    print_snapshot_report()

    # --- forward pass ---
    out  = model(x)
    loss = out.mean()

    print(">>> Snapshot 2: after forward pass (activations live for backward)\n")
    print_snapshot_report()

    # --- backward ---
    loss.backward()

    print(">>> Snapshot 3: after backward (grad buffers allocated)\n")
    print_snapshot_report()

    # --- free activations ---
    del out, loss
    torch.cuda.empty_cache()

    print(">>> Snapshot 4: after del + empty_cache (only weights + grads remain)\n")
    print_snapshot_report()
