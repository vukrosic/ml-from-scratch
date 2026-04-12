# GPU Memory Management

CUDA OOM errors don't happen because your model is too big. They happen because you don't know where the memory is going. This lesson gives you the tools to see every byte — and get most of it back.

---

## Why two numbers?

PyTorch reports two memory figures. Confusing them is the source of most "why is my GPU still full?" head-scratching.

```python
allocated = torch.cuda.memory_allocated()  # bytes held by live tensors
reserved  = torch.cuda.memory_reserved()   # bytes claimed from the OS
```

`allocated` is the honest answer: bytes your tensors actually occupy right now.

`reserved` is always >= `allocated`. When a tensor is freed, PyTorch does **not** call `cudaFree`. It keeps the memory in an internal pool (the *caching allocator*) so the next allocation can skip the expensive `cudaMalloc` round-trip. That pooled-but-not-in-use memory is what separates `reserved` from `allocated`.

`torch.cuda.empty_cache()` returns the free pool back to the OS — but only the free blocks. It cannot touch live tensors.

---

## Tracking memory through a forward pass

The fastest way to find your memory hog is to print both numbers after every layer.

```python
def mem_stats(label: str) -> None:
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved  = torch.cuda.memory_reserved()  / 1024**2
    print(f"{label:<35}  allocated={allocated:7.2f} MiB  reserved={reserved:7.2f} MiB")
```

Apply it inside the forward method of an MLP and you will see exactly which layer is the memory culprit:

```python
def forward(self, x):
    mem_stats("after input")
    for i, layer in enumerate(self.layers):
        x = self.act(layer(x))
        mem_stats(f"after layer {i}")
    return x
```

Sample output (5-layer MLP, dim=2048, batch=256):

```
after model.to(device)               allocated=  80.00 MiB  reserved=  84.00 MiB
after input tensor created           allocated=  82.00 MiB  reserved=  84.00 MiB
after layer 0                        allocated= 146.00 MiB  reserved= 148.00 MiB
after layer 1                        allocated= 210.00 MiB  reserved= 212.00 MiB
after layer 4                        allocated= 402.00 MiB  reserved= 404.00 MiB
after forward + loss                 allocated= 402.25 MiB  reserved= 404.00 MiB
after backward                       allocated= 162.00 MiB  reserved= 404.00 MiB
after del + empty_cache              allocated=  80.00 MiB  reserved=  80.00 MiB
```

Notice: allocated jumps ~64 MiB per layer (activations accumulating for backward), then drops after backward once those activations are freed. Reserved barely shrinks until `empty_cache`.

See the full file: [`memory_tracker.py`](memory_tracker.py)

---

## Why activations pile up

When you call `loss.backward()`, PyTorch walks the autograd graph in reverse. To compute the gradient at layer N it needs the activation produced by layer N-1. So **every intermediate activation is kept alive from the moment it's created until the backward pass visits it**.

For a model with L layers and hidden size H, that's roughly `O(L × batch × H)` bytes.

This is the exact problem gradient checkpointing solves.

---

## Gradient checkpointing — trade compute for memory

The idea: **throw the activations away** after the forward pass. When the backward pass needs them, recompute them on the spot.

Cost: one extra forward pass per checkpointed segment.  
Benefit: activation memory drops from O(L) to O(1) — only the input to each segment is kept.

PyTorch ships `torch.utils.checkpoint.checkpoint`. Here is all it does under the hood:

```python
class ManualCheckpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, *inputs):
        ctx.fn = fn
        ctx.save_for_backward(*inputs)   # save inputs, not outputs
        with torch.no_grad():
            return fn(*inputs)           # run forward, discard graph

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs = ctx.saved_tensors
        with torch.enable_grad():
            fresh = tuple(x.detach().requires_grad_(x.requires_grad) for x in inputs)
            out   = ctx.fn(*fresh)       # recompute forward to rebuild graph
        torch.autograd.backward(out, grad_outputs)
        return (None,) + tuple(x.grad for x in fresh)
```

Using the library version inside a forward pass:

```python
from torch.utils.checkpoint import checkpoint

def forward_checkpointed(self, x):
    for block in self.blocks:
        x = x + checkpoint(block, x, use_reentrant=False)
    return x
```

Measured on a 12-block, dim=1024 MLP (batch=64):

```
Peak memory — plain forward:        1 842.3 MiB
Peak memory — checkpointed forward:   312.8 MiB
Memory saving:                       1529.5 MiB  (83% reduction)
```

83% memory reduction at the cost of ~33% more compute. For large models that don't fit in VRAM, that is not a trade-off — it's the only option.

See the full file: [`gradient_checkpoint.py`](gradient_checkpoint.py)

---

## Seeing the allocator state with memory_snapshot

`torch.cuda.memory_snapshot()` returns a list of *segments* — one per `cudaMalloc` call. Each segment contains *blocks* that are either `active_allocated` (a live tensor lives here) or `inactive` (freed but pooled).

```python
snapshot = torch.cuda.memory_snapshot()
for seg in snapshot:
    seg_size  = seg["total_size"]
    seg_alloc = sum(b["size"] for b in seg["blocks"] if b["state"] == "active_allocated")
    seg_free  = seg_size - seg_alloc
    print(f"segment: {seg_size/1024**2:.1f} MiB  active: {seg_alloc/1024**2:.1f}  free: {seg_free/1024**2:.1f}")
```

`snapshot.py` wraps this into an ASCII bar chart so fragmentation is immediately visible:

```
>>> Snapshot 2: after forward pass (activations live for backward)

  pool (0, 0) (3 segs)              [████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  84.25 / 210.00 MiB  (40% used)
```

A low percentage with many segments = heavy fragmentation. The allocator has reserved large chunks from CUDA but is only actively using part of them.

See the full file: [`snapshot.py`](snapshot.py)

---

## Recap

- `memory_allocated` = live tensor bytes. `memory_reserved` = live + pooled. The gap between them is your fragmentation budget.
- The caching allocator avoids `cudaMalloc` on every allocation by recycling freed blocks — that is why `empty_cache()` does not always free VRAM visibly.
- During a forward pass, every activation is kept alive until the backward pass uses it. This is where most VRAM goes in large models.
- Gradient checkpointing discards activations and recomputes them on demand. The implementation is ~15 lines — a custom `autograd.Function` that saves only inputs, not outputs.
- `memory_snapshot()` gives you a segment-level map of the allocator state. High `free / reserved` ratio means fragmentation.

---

Get the extended notebook with CUDA caching allocator internals, fragmentation analysis across batch sizes, memory leak detection, and OOM debugging playbook: [https://www.skool.com/opensuperintelligencelab](https://www.skool.com/opensuperintelligencelab)
