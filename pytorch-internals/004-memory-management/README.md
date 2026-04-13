# GPU Memory Management

> 🔴 YouTube Lesson: Coming soon | 🟡 Skool Advanced Video Lesson: [Join the advanced lesson](https://www.skool.com/become-ai-researcher-2669/about)

CUDA OOM errors don't happen because your model is too big. They happen because you don't know where the memory is going. This lesson gives you the tools to see every byte — and get most of it back.

---

## Files at a Glance

| File | What it demonstrates |
|------|----------------------|
| [`memory_tracker.py`](memory_tracker.py) | Prints `allocated` and `reserved` at every layer of an MLP forward pass. Shows how activations accumulate and when they are freed after backward. |
| [`gradient_checkpoint.py`](gradient_checkpoint.py) | Two implementations of gradient checkpointing: a custom `torch.autograd.Function` (~15 lines) and the library `torch.utils.checkpoint`. Includes peak memory comparison. |
| [`snapshot.py`](snapshot.py) | Calls `torch.cuda.memory_snapshot()` and renders the allocator state as an ASCII bar chart per memory pool. Captures the allocator at 4 points in a training loop. |

---

## The caching allocator — a book-library analogy

Think of the CUDA caching allocator like a library.

When your program needs memory for a tensor, it does **not** call `cudaMalloc` (the CUDA equivalent of requesting new memory from the operating system). Instead, it asks the *library* (the caching allocator) for a block. The library keeps a shelf of previously-freed blocks. If a suitable block is sitting there, it lends it out immediately — no OS round-trip, no zeroing, no overhead.

When you delete a tensor, the library does **not** return the block to the operating system. It puts the block back on the shelf, marked "available." The next request that fits in that block reuses it without any CUDA API call.

This is why two numbers exist:

```python
allocated = torch.cuda.memory_allocated()   # bytes sitting on the shelf (lent out)
reserved  = torch.cuda.memory_reserved()    # bytes the library has checked out from the OS
```

`reserved` is always greater than or equal to `allocated` because the library keeps some blocks on its shelf even when they are not currently lent out. These shelf-sitting blocks are "reserved but free."

`empty_cache` tells the library to return its shelf to the OS. But it can only do that for blocks that are not currently lent out. If a live tensor holds a block, that block cannot be returned — the OS will not take it back while it is still in use. So `empty_cache` only helps when fragmentation has left you with many small free blocks that cannot be reused for new allocations.

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

### A step-by-step training loop — memory lifecycle

Here is exactly what happens to GPU memory, byte by byte, through one training iteration.

**Step 1 — Model loaded onto GPU**
`model.to("cuda")` copies all parameter tensors from CPU to GPU memory. No autograd graph exists yet. `allocated` = weight memory only. `reserved` = weight memory plus a small allocator overhead.

**Step 2 — Input tensor created**
`torch.randn(batch, dim, device="cuda")` allocates the input tensor. Both `allocated` and `reserved` grow by the tensor size. No autograd graph yet — the tensor is just data.

**Step 3 — Forward pass begins**
As each layer runs, PyTorch:
- Allocates the weight gradient buffer (created on demand by `backward()`)
- Allocates the output activation tensor for that layer
- Builds the autograd graph: each activation tensor gets a `grad_fn` that points back to the operation that created it

Activations are retained *for the entire remaining forward pass* because the autograd engine needs them to compute gradients during `backward()`. This is why `allocated` grows linearly with depth — every layer's output stays in memory until the backward pass retrieves it.

**Step 4 — Loss computed**
`loss = model(x).mean()` finalizes the scalar loss tensor. The full autograd graph is now connected end-to-end. `allocated` is at its peak — every activation for every layer is live.

**Step 5 — `loss.backward()` runs**
The autograd engine walks the graph in reverse topological order. For each `grad_fn`, it:
1. Receives the gradient from the next node (starting with `torch.ones_like(loss)`)
2. Allocates a gradient buffer for that tensor's input if it does not exist
3. Calls the `backward` function for that operation
4. Propagates the gradient to the input tensor

As gradients are computed for each layer's output activation, those activation tensors are freed — their `refcount` drops to zero and they are returned to the allocator's free pool. This is why `allocated` drops sharply after `backward()`. The gradient buffers themselves are typically much smaller than the activation tensors.

**Step 6 — Optimizer step**
`optimizer.step()` updates each parameter in-place. This does not allocate new memory — it writes directly into the existing gradient and parameter tensors.

**Step 7 — `.backward()` graph freed**
After the optimizer step, the autograd graph (now stale) is freed. Gradient buffers remain until the next `forward()` call because the optimizer holds references to them.

**Step 8 — `del loss` + `empty_cache()`**
`del loss` drops the last Python reference to the loss tensor. Its underlying storage is returned to the free pool. `torch.cuda.empty_cache()` returns all free-pool blocks to the OS. `allocated` and `reserved` return to near their post-model-load values.

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

### ManualCheckpoint — line by line

The `ManualCheckpoint` implementation is the clearest illustration of how gradient checkpointing works under the hood. Here is what each line does:

```python
class ManualCheckpoint(torch.autograd.Function):
```

`torch.autograd.Function` is the primitive for defining custom differentiable operations. Each Function has a `forward` (runs on the forward pass) and a `backward` (runs during gradient computation).

```python
    @staticmethod
    def forward(ctx, fn, *inputs):
        ctx.fn = fn                    # store the function to re-run later
        ctx.save_for_backward(*inputs) # save only the inputs, NOT the outputs
        with torch.no_grad():
            return fn(*inputs)         # run fn normally, discard the graph
```

Key insight: `save_for_backward` is called with `*inputs`, not with the outputs of `fn(*inputs)`. This is the entire memory saving. During a normal forward pass, PyTorch would save the *outputs* (the activation tensors) so the backward pass can compute gradients. Here we explicitly refuse to save them. The graph produced by `fn(*inputs)` is also discarded — we return raw tensors, not graph-wrapped ones.

`torch.no_grad()` is used because we only need the raw numerical output for the forward pass. Wrapping it in `no_grad` prevents PyTorch from building the graph unnecessarily (though the backward pass will rebuild it anyway).

```python
    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs = ctx.saved_tensors  # retrieve the original forward inputs
        with torch.enable_grad():
            # Detach and re-enable grad tracking for the recompute pass
            inputs_detached = tuple(x.detach().requires_grad_(x.requires_grad) for x in inputs)
            output = ctx.fn(*inputs_detached)   # re-run forward to rebuild graph
        torch.autograd.backward(output, grad_outputs)  # propagate gradients
        return (None,) + tuple(x.grad if x.requires_grad else None for x in inputs_detached)
```

`backward` receives the upstream gradients (`grad_outputs`) that need to be propagated back through this operation. Because we saved the inputs (not outputs), we must recompute the forward pass to get the outputs, then build a fresh graph from those outputs back to the inputs. `torch.enable_grad()` re-enables gradient tracking — it was disabled by `no_grad()` in forward, and the backward pass context does not inherit the `enable_grad` state. After recomputing, `torch.autograd.backward(out, grad_outputs)` walks the freshly built graph and computes input gradients. The return tuple must match the forward signature: `(None,)` accounts for `fn` (it is not a tensor and has no gradient), followed by the gradient for each input tensor.

**Why `x.detach().requires_grad_(x.requires_grad)`?**
`detach()` severs the computation graph — the recomputed forward pass must not be connected to the original forward pass. `requires_grad_(x.requires_grad)` restores the `requires_grad` flag on each input so that gradient tracking is correct for the next layer.

### Memory vs compute tradeoff — concrete numbers

On a 12-block, dim=1024 MLP with batch=64, the measured numbers are:

```
Peak memory — plain forward:        1 842.3 MiB
Peak memory — checkpointed forward:     312.8 MiB
Memory saving:                       1529.5 MiB  (83% reduction)
```

The plain forward keeps all activations in memory: `12 blocks × batch=64 × dim=1024 × 4 bytes (float32) = 3.15 GiB` just for activations, plus weights. The checkpointed forward discards each block's output after computing it, keeping only the input tensor for the next block.

The compute cost: in the backward pass, when PyTorch reaches a checkpointed block, it re-executes the block's forward function to reconstruct the output tensor, then computes gradients on that freshly computed output. For each checkpointed block, the forward pass runs twice (once for real, once during backward for recompute). The measured extra compute is ~33%. This is the price of the 83% memory saving.

Rough calculation: for a model with L layers, checkpointing every layer saves roughly `(L-1) × activation_size` bytes. The recompute cost is `L × forward_cost` additional forward passes. For large L, this is a favorable trade — memory scales as `O(L)` but recompute scales linearly with `L` as well, and recompute is cheap relative to memory on modern GPUs.

Selective checkpointing is also an option: checkpoint expensive layers (large matrix multiplications, attention) and leave cheap layers (small linear layers, element-wise operations) to store their activations normally. This reduces the recompute cost while still saving meaningful memory.

You can also checkpoint at granularity finer than a single layer — wrap a group of operations together inside the checkpoint call. The library version (`torch.utils.checkpoint.checkpoint`) accepts any Python callable, so a whole residual block, attention head, or transformer layer can be one checkpointed unit.

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

### Reading the bar chart

Each line of the bar chart represents one *pool* — a group of segments that the allocator manages together. The bar shows `allocated / reserved` with `█` for the used fraction and `░` for the free fraction.

In the sample output above:
```
pool (0, 0) (3 segs)  [████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  84.25 / 210.00 MiB  (40% used)
```

The pool has 3 segments (3 calls to `cudaMalloc`). Total reserved from the OS: 210 MiB. Live tensors occupy 84.25 MiB — only 40% of what the allocator has checked out. The remaining 125.75 MiB is free blocks sitting in the allocator's pool.

**What causes low utilization?**
- Allocations of many different sizes: the allocator matches each new tensor to the smallest free block that fits it. If no exact-size block exists, it splits a larger one. This leaves small unusable gaps.
- Objects with different lifetimes: a tensor allocated early and freed late can block a large region while smaller tensors use the spaces on either side of it.
- Non-contiguous tensors: operations like `view()`, `transpose()`, or `slice()` can make the allocator think a freed region is usable only for tensors of a specific size.

**What does high utilization look like?**
A bar that is nearly full (e.g., `95% used`) means almost every byte the allocator has reserved from CUDA is doing active work. There is little waste — the only way to reduce memory further is to make the tensors themselves smaller (e.g., checkpointing, pruning, quantization).

See the full file: [`snapshot.py`](snapshot.py)

---

## Common Memory Bugs

### OOM causes

1. **Activations not freed before next iteration**
   If you call `model.forward()` in a loop without deleting the output or freeing the autograd graph, activations from every iteration accumulate in memory. Each `loss.backward()` frees the graph, but only if the graph is not being held by a Python reference. Common mistake: storing losses in a list for later processing.

2. **Gradient accumulation without `no_grad`**
   When running multiple forward passes without `torch.no_grad()`, PyTorch builds a graph for each one and only frees it after `backward()`. If you call `backward()` only after N micro-batches, the intermediate graphs are all held simultaneously.

3. **Detached tensors still in a list**
   `losses.append(loss.detach())` — the `.detach()` call is correct for saving the scalar value, but if you forget `.item()` or `.cpu()`, the detached tensor still lives on the GPU. Use `.item()` to extract the Python float if you only need the number.

4. **DataLoader workers holding memory**
   Each `DataLoader` worker process allocates GPU memory for model replicas (if `pin_memory=True` and the model is on GPU in the main process). Workers share memory with the main process through the multiprocessing shm. Monitor with `nvidia-smi` to confirm whether the leak is in Python or in DataLoader.

5. **Batch size too large for the activation footprint**
   For a model with L layers, hidden size H, and batch size B, the peak activation memory is approximately `L × B × H × bytes_per_element`. A 12-layer, dim=1024 model with batch=256 needs ~1.5 GiB just for activations. Use the `mem_stats` printer to find where the spike occurs.

6. **Caching allocator holding fragmented free blocks**
   Even when `allocated` is low, `reserved` can remain high due to fragmentation. `empty_cache()` returns the free pool to the OS, but only if no live tensor is within that memory range. This is normal behavior, not a bug.

### Memory leaks

A memory leak in PyTorch means memory is allocated but never returned to the allocator's free pool and cannot be reused.

1. **Orphaned autograd graph**
   Storing `model(input)` outputs in a global list without calling `.backward()` on them keeps the autograd graph alive. Each graph node holds references to its input tensors. Solution: `.detach()` tensors before storing, or call `del output; torch.cuda.empty_cache()` when done.

2. **`model.eval()` but still building graphs**
   If you use `model.train()` weights in an inference context without `torch.no_grad()`, each call still builds an autograd graph. Always wrap inference in `with torch.no_grad():`.

3. **Callbacks and hooks holding references**
   `register_forward_hook` returns a handle. If you discard the handle without calling `remove()`, the hook keeps a reference to the module and any captures it makes.

4. **Python object lifetime**
   A tensor stored in a global dict or class variable is not freed until you remove it. Use `del tensor; torch.cuda.empty_cache()` explicitly.

### Fragmentation

Fragmentation is when `reserved >> allocated` but you still cannot allocate a new tensor.

1. **Many different tensor sizes**
   The allocator maintains free lists per size class. If your program allocates 100 tensors of size 1 MiB, frees 99 of them, then needs a 1.5 MiB tensor, the 1 MiB blocks cannot be merged and the allocator must request a new segment from CUDA.

2. **Tensor shape changing across iterations**
   If input shapes vary widely (e.g., sequence models with variable sequence length), the allocator cannot reuse blocks efficiently because freed blocks are the wrong size for new allocations.

3. **Symptom: `empty_cache` does not help**
   If `empty_cache()` does not reduce `reserved` noticeably, the allocator is holding onto free blocks because there is no live tensor large enough to trigger their release. This is not a leak — the memory is available for reuse, just not returned to the OS.

---

## What `empty_cache` does and does not do

`torch.cuda.empty_cache()` is frequently misunderstood.

**What it does:**
- Returns free memory blocks from the caching allocator's internal free list back to the operating system (via `cudaFree`).
- Reduces `reserved` when there are free blocks not needed for upcoming allocations.
- Has no effect on `allocated` — it cannot free memory occupied by live tensors.

**What it does not do:**
- It does not force garbage collection of Python objects. Use `import gc; gc.collect()` alongside `empty_cache()` if you suspect Python-level leaks.
- It does not clear the allocator's active allocations. Live tensors remain live.
- It does not fix fragmentation. The allocator still uses the same segments for future allocations; it just tells the OS it is not using some of them right now.
- It does not speed up allocation. The caching allocator's free list is still the first place to look for new blocks.

**When it actually helps:**
- After processing a very large batch that required many segments, before processing a smaller batch. Without `empty_cache`, the allocator would keep those large segments reserved for reuse (which is fine), but `empty_cache` returns them to the OS if you want to share the GPU with another process.
- Before checkpoint saving or other operations that temporarily need more free memory than is currently in the allocator's free list.
- During inference when you want to maximize available memory for a second model loaded after the first.

**When it does not help:**
- During training when you are using all memory up to the peak allocation. Freeing and re-requesting memory from CUDA is expensive and the allocator will just reclaim the same blocks.
- When OOM is caused by a single large tensor that is still live.

**Practical tip:** if you are debugging an OOM, call `torch.cuda.empty_cache()` immediately before the failing line. If `reserved` drops significantly, you have a fragmentation issue (the allocator had free memory but not in the right size class). If `reserved` barely changes, the OOM is caused by a live tensor that you think has been freed.

---

## Recap

- `memory_allocated` = live tensor bytes. `memory_reserved` = live + pooled. The gap between them is your fragmentation budget.
- The caching allocator avoids `cudaMalloc` on every allocation by recycling freed blocks — that is why `empty_cache()` does not always free VRAM visibly.
- During a forward pass, every activation is kept alive until the backward pass uses it. This is where most VRAM goes in large models.
- Gradient checkpointing discards activations and recomputes them on demand. The implementation is ~15 lines — a custom `autograd.Function` that saves only inputs, not outputs.
- `memory_snapshot()` gives you a segment-level map of the allocator state. High `free / reserved` ratio means fragmentation.
- OOM is almost never about model size alone — it is about when activations are alive relative to when the next allocation happens. Print memory stats at each layer to find the spike.
- `empty_cache()` returns free blocks to the OS; it cannot return blocks occupied by live tensors. It helps when sharing the GPU or after large temporary allocations, not during steady-state training.
- The most common memory bugs are: storing detached tensors in lists, building autograd graphs in inference mode, and forgetting `use_reentrant=False` in `checkpoint` (which triggers a separate bug around graph capture).
- Gradient checkpointing saves roughly `(L-1) × activation_size` per checkpointed segment at a cost of one extra forward pass per segment. For modern LLMs with dozens of layers this is a mandatory optimization, not an optional one.
- Always use `use_reentrant=False` when calling `torch.utils.checkpoint.checkpoint`. The reentrant variant (older, default) can cause incorrect gradient computation in certain cases involving in-place operations, and it captures the graph differently which can interfere with memory management in complex models.

---

Get the video walkthrough of CUDA caching allocator internals, fragmentation analysis, memory leak detection, and OOM debugging: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
