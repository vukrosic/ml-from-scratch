# Advanced GPU Memory Management

This document goes beyond the basics in [README.md](README.md). It covers the internal mechanics of the CUDA caching allocator, the full structure of memory snapshots, memory fragmentation diagnosis, OOM debugging strategies, and gradient checkpointing tradeoffs.

---

## Which File to Run For What

| Goal | File | What you will see |
|---|---|---|
| Identify which layer is hogging memory during a forward pass | [`memory_tracker.py`](memory_tracker.py) | `allocated` and `reserved` MiB printed after every layer |
| Understand how checkpointing reduces memory vs. compute cost | [`gradient_checkpoint.py`](gradient_checkpoint.py) | Peak memory comparison between `forward_plain` and `forward_checkpointed`; also a manual `ManualCheckpoint` implementation |
| Visualize the caching allocator's internal segment/block state | [`snapshot.py`](snapshot.py) | ASCII bar charts showing allocated vs. reserved per pool, segment counts, and fragmentation % |

Run any file directly with `python <file>.py`. All three require a CUDA-capable GPU.

---

## How the CUDA Caching Allocator Works Internally

### The Problem CUDA Malloc/Free Solve

`cudaMalloc` and `cudaFree` are extremely expensive — they communicate with the GPU device driver and can take hundreds of microseconds to milliseconds. Allocating a temporary buffer for every intermediate activation would make training prohibitively slow.

### The Caching Allocator's Solution

PyTorch wraps `cudaMalloc`/`cudaFree` with an in-process **caching allocator** (sometimes called the "arena"). The allocator maintains:

1. **A block pool per stream** — memory blocks are tied to CUDA streams. Blocks from one stream cannot be reused by another unless streams are synchronized.
2. **Free blocks** — memory that has been `cudaFree`'d by PyTorch but not returned to the OS. These are kept in a doubly-linked list sorted by address (to coalesce adjacent free blocks).
3. **Allocation pools** — blocks are bucketed by size class. When you request N bytes, the allocator finds the smallest free block >= N.

### The Allocation Path

When PyTorch needs memory for a new tensor:

```
tensor allocation request
        |
        v
caching allocator checks free list
        |
   [block found] --> carve it, return pointer   (fast, no CUDA call)
        |
   [no block]  --> cudaMalloc from GPU OS heap   (slow, ~100-1000 µs)
        |
   [cubaMalloc returns segment] --> split into blocks, cache remaining
```

### The Free Path

When a tensor is freed (Python object goes out of scope, `del tensor`, etc.):

```
tensor freed
        |
        v
caching allocator marks block as "inactive" (adds to free list)
        |
        v
block stays in pool  ←── cudaFree is NOT called yet
```

The freed block is cached and reused for the next allocation. This is why `memory_reserved` stays high even when `memory_allocated` is low.

### When `empty_cache` Does NOT Free Memory

`torch.cuda.empty_cache()` only returns **completely free segments** to the OS. A segment is free only when **all its blocks are inactive**. If any block within a segment is still allocated (a live tensor), the entire segment is retained.

```python
# empty_cache is essentially:
for segment in all_segments:
    if all(block.state == "inactive" for block in segment.blocks):
        cudaFree(segment.ptr)   # return to OS
        # segment removed from allocator's view
```

This is the most common source of confusion: you delete your tensors, call `empty_cache()`, and `reserved` barely moves. It means some block is still pinned — likely a cached empty tensor, a gradient that hasn't been freed, or a CuDNN/AMP buffer.

### cudaMalloc vs cudaFree Behavior Summary

| Call | Cost | Effect |
|---|---|---|
| `cudaMalloc` | ~100-1000 µs | Requests new memory from GPU OS heap; increases `reserved` |
| `cudaFree` | ~100-1000 µs | Returns memory to GPU OS heap; decreases `reserved` |
| PyTorch block free | ~nanoseconds | Marks block inactive in allocator pool; `reserved` unchanged |
| `empty_cache` | ~1-10 ms | Returns all fully-free segments to GPU OS; decreases `reserved` |

---

## memory_snapshot() Data Structure

`torch.cuda.memory_snapshot()` returns a Python list of **segment** dictionaries. This is the allocator's internal state, captured at a point in time.

### Top-Level Structure

```python
snapshot = torch.cuda.memory_snapshot()
# [
#   { "total_size": 16777216, "blocks": [...], "segment_pool_id": 0, ... },
#   { "total_size": 65536,    "blocks": [...], "segment_pool_id": 1, ... },
#   ...
# ]
```

### Segment Fields

| Field | Type | Meaning |
|---|---|---|
| `total_size` | int | Bytes allocated via a single `cudaMalloc` call. All blocks in this segment come from this one allocation. |
| `blocks` | list[dict] | All blocks within this segment (see below). |
| `segment_pool_id` | int or str | Which pool this segment belongs to. PyTorch uses multiple pools (e.g., for different size classes). |
| `active_size` | int | Sum of bytes in `active_allocated` blocks (live tensors). |

### Block Fields

Each block in `segment["blocks"]`:

| Field | Type | Meaning |
|---|---|---|
| `size` | int | Total bytes of this block. |
| `state` | str | See block states below. |
| `ptr` | int | Raw GPU pointer address. |
| `history` | list | Optional, contains allocation history for debugging leaks. |

### Block State Meanings

| State | Meaning |
|---|---|
| `active_allocated` | This block holds a live tensor. Its bytes are counted in `memory_allocated()`. |
| `active_pending_free` | Block was freed but is waiting for GPU stream to finish before it can be reused. |
| `inactive` | Block is freed. Memory is still reserved (belongs to the segment) but the block is on the free list for reuse. |

### Walking a Snapshot

```python
snapshot = torch.cuda.memory_snapshot()

for seg in snapshot:
    seg_total    = seg["total_size"]
    seg_alloc    = sum(b["size"] for b in seg["blocks"] if b["state"] == "active_allocated")
    seg_inactive = sum(b["size"] for b in seg["blocks"] if b["state"] == "inactive")
    seg_pending  = sum(b["size"] for b in seg["blocks"] if b["state"] == "active_pending_free")

    print(f"Segment {seg['segment_pool_id']}: "
          f"total={seg_total/1024**2:.1f} MiB, "
          f"active={seg_alloc/1024**2:.1f}, "
          f"inactive={seg_inactive/1024**2:.1f}, "
          f"pending={seg_pending/1024**2:.1f}")
```

The `parse_snapshot()` function in [`snapshot.py`](snapshot.py) implements exactly this walk and aggregates stats per pool.

---

## Detecting and Diagnosing Memory Fragmentation

### What Fragmentation Looks Like

**Fragmentation** occurs when the allocator has reserved plenty of memory (`reserved` is high) but cannot satisfy new allocations without calling `cudaMalloc` again (`allocated` is much lower than `reserved`).

The key indicator in the snapshot output:

```
  Fragmentation: 62.3%  (free / reserved)
```

A high fragmentation percentage means many segments have large free blocks that cannot be coalesced — often because they are pinned by small active allocations in the middle.

### Snapshot Data That Signals Fragmentation

**Signal 1: Many segments, low utilization per segment**

```python
# From snapshot.py output — a fragmented state looks like:
#   pool 0 (15 segs)    [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  40.12 / 210.00 MiB  (19% used)
#   pool 1 (8 segs)     [██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  18.00 / 150.00 MiB  (12% used)
```

The allocator had to request many segments from `cudaMalloc` because it could not find a single contiguous block large enough.

**Signal 2: Large free blocks surrounded by small active blocks**

```python
for seg in torch.cuda.memory_snapshot():
    for b in seg["blocks"]:
        if b["state"] == "inactive" and b["size"] > 10 * 1024**2:  # > 10 MiB free
            # Check neighboring blocks in the same segment
            seg_idx = seg["blocks"].index(b)
            # If the surrounding blocks are small active_allocated blocks,
            # they prevent this segment from being returned to the OS
```

**Signal 3: `memory_allocated` is low but `cudaMalloc` fails**

If `reserved` is far below your GPU's total memory but you still get OOM, fragmentation is preventing the allocator from finding a contiguous block. Use `torch.cuda.mem_get_info()` to see total and available:

```python
total, free = torch.cuda.mem_get_info()
print(f"Total: {total/1024**2:.0f} MiB, Free: {free/1024**2:.0f} MiB")
```

### Common Causes of Fragmentation

1. **Variable-sized tensor creation** — Allocating tensors of many different shapes (common in NLP with varying sequence lengths) scatters active blocks across segments.
2. **Failing to delete temporary tensors** — Intermediate tensors from each forward pass accumulate if not explicitly freed before the next iteration.
3. **Memory pinned by CuDNN / cuFFT / cuBLAS** — These libraries allocate their own workspaces that are not visible in PyTorch's snapshot but are counted against total VRAM.
4. **Gradient accumulation without clearing** — Running many accumulation steps without clearing the grad buffer fragments the grad segment.

### Defragmentation Strategies

- Call `torch.cuda.empty_cache()` between phases that allocate very different tensor shapes.
- Pre-allocate a fixed-size workspace and reuse it instead of allocating new tensors each iteration.
- Use `torch.cuda.reset_peak_memory_stats()` to force the allocator to reset its heuristics.
- In severe cases, restart the process (fragmentation is not fully recoverable at runtime).

---

## OOM Debugging

### Step 1: Confirm It Is Actually an OOM

Oft-reported "OOM" is actually a different error:

```python
# WRONG — catches all CUDA errors including kernel failures
try:
    ...
except RuntimeError as e:
    print(e)   # might say "CUDA out of memory" but the real cause could be a bug
```

### Step 2: Print the Allocator State Right Before the Crash

```python
print(f"allocated : {torch.cuda.memory_allocated()/1024**2:.1f} MiB")
print(f"reserved  : {torch.cuda.memory_reserved()/1024**2:.1f} MiB")
print(f"max ever  : {torch.cuda.max_memory_allocated()/1024**2:.1f} MiB")
total, free = torch.cuda.mem_get_info()
print(f"GPU total : {total/1024**2:.0f} MiB, free: {free/1024**2:.0f} MiB")
```

### Step 3: Identify the Culprit Layer

Insert `mem_stats()` calls inside your model's forward pass (as shown in [`memory_tracker.py`](memory_tracker.py)):

```python
def forward(self, x):
    mem_stats("input")
    for i, layer in enumerate(self.layers):
        x = layer(x)
        mem_stats(f"after layer {i}")
    return x
```

The layer with the largest `allocated` jump is the culprit. Common offenders:
- Large embedding lookups with padding
- Attention weight matrices before compression
- Cross-attention layers in encoder-decoder models (two large inputs at once)

### Step 4: Check for Memory Leaks

A memory leak is when `memory_allocated` grows across iterations even though no new parameters are created:

```python
for step, batch in enumerate(dataloader):
    mem_before = torch.cuda.memory_allocated()
    output = model(batch)
    loss = output.mean()
    loss.backward()
    mem_after = torch.cuda.memory_allocated()
    print(f"step {step}: alloc grew by {(mem_after-mem_before)/1024**2:.2f} MiB")

    # After a few steps, growth should flatten
    # If it keeps growing every step → leak
```

Leaks usually come from:
- Custom `autograd.Function` that saves tensors without releasing them
- Hooks attached to parameters that hold references
- `nn.Module` that stores activations as plain attributes (not parameters but still kept)

### Step 5: Common OOM Causes and Fixes

| Cause | Symptom | Fix |
|---|---|---|
| Activations not freed before next iteration | `allocated` grows across batch loop | Ensure `del loss, output` and call `empty_cache()` after each iteration |
| Gradient checkpointing not applied | Very high `allocated` after forward | Wrap expensive blocks with `checkpoint()` |
| Batch size too large for hidden size | `allocated` jumps massively at first layer | Reduce batch size; use gradient accumulation |
| Mixed precision training not cleaning up scaler state | `reserved` grows gradually | Call `scaler.clear()` each step in AMP training |
| CuDNN benchmark mode allocating large workspace | OOM even with small model | Set `torch.backends.cudnn.benchmark = False` or limit `cudnn workspace` size |
| Inference running in training mode | BatchNorm buffers, dropout state kept alive | Call `model.eval()` |
| Largeattention mask or padding | Padding tokens contribute N tokens but are never used | Use `torch.masked_fill` orattention implementation that skips padding |

### Step 6: Using snapshot.py to Pinpoint the Leak

Run [`snapshot.py`](snapshot.py) before and after a loop iteration. If `n_segments` grows and `total_free` shrinks, the allocator is repeatedly calling `cudaMalloc` because it cannot reuse freed blocks — a sign of a leak or severe fragmentation:

```python
info1 = parse_snapshot()
# ... run one training step ...
info2 = parse_snapshot()

print(f"Segments: {info1['n_segments']} → {info2['n_segments']}")
print(f"Reserved: {info1['total_reserved']/1024**2:.1f} → {info2['total_reserved']/1024**2:.1f} MiB")
```

---

## Gradient Checkpointing Tradeoffs

### The Core Tradeoff

Checkpointing a segment saves `O(activations_per_segment)` memory at the cost of `O(1)` extra forward compute per checkpointed segment (the activation is recomputed during backward).

For a model with `L` layers split into `C` checkpointed segments:

| Without Checkpointing | With Checkpointing |
|---|---|
| Memory: O(L) activations | Memory: O(C) inputs to segments |
| Compute: 1 forward + 1 backward | Compute: 1 forward + 1 backward + C partial forwards |

### Choosing Checkpoint Granularity

**Finer granularity (checkpoint each layer individually)**
- Maximum memory savings: each layer's activations are discarded immediately after the next layer runs.
- Maximum compute cost: recomputes one layer per backward step for every layer.

**Coarser granularity (checkpoint multiple layers per segment)**
- Less memory savings: activations for all layers in the segment accumulate until the segment completes.
- Less compute cost: recomputes fewer times (once per segment).

A good default is checkpointing each **transformer block** or each **residual stage** — the natural architectural boundary. In [`gradient_checkpoint.py`](gradient_checkpoint.py), each `block` in `DeepMLP` is checkpointed individually.

### Measuring Compute Overhead

Use `torch.cuda.Event` to time the backward pass:

```python
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

start.record()
loss = model(x).mean()
loss.backward()
end.record()
torch.cuda.synchronize()
print(f"Backward time: {start.elapsed_time(end):.1f} ms")
```

Typical overhead for checkpointed models: 20-40% extra compute for 60-85% memory reduction.

### When Checkpointing Is Not Worth It

- Small models that fit in memory without it — the extra forward pass overhead is pure cost.
- Memory-bound training where the compute overhead of recomputation is negligible compared to the memory savings.
- Inference — never checkpoint during inference (no backward pass needed).

---

## Quick Reference: torch.cuda.memory_* API Cheatsheet

### Query Functions

| API | Returns | Units |
|---|---|---|
| `torch.cuda.memory_allocated()` | Bytes currently held by live tensors | bytes |
| `torch.cuda.memory_reserved()` | Bytes claimed from OS (live + pooled free) | bytes |
| `torch.cuda.max_memory_allocated()` | Peak bytes ever allocated (lifetime max) | bytes |
| `torch.cuda.max_memory_reserved()` | Peak bytes ever reserved | bytes |
| `torch.cuda.memory_allocated(device)` | Per-device allocated (default: current) | bytes |
| `torch.cuda.memory_reserved(device)` | Per-device reserved | bytes |
| `torch.cuda.mem_get_info()` | `(total, free)` bytes on device | bytes |
| `torch.cuda.memory_stats()` | Detailed allocator stats as dict | mixed |
| `torch.cuda.memory_summary()` | Human-readable multi-line summary | string |

### Reset Functions

| API | Effect |
|---|---|
| `torch.cuda.reset_peak_memory_stats()` | Reset the peak alloc/reserved counters |
| `torch.cuda.empty_cache()` | Return fully-free segments to the OS |
| `torch.cuda.reset_accumulated_memory_stats()` | Reset accumulated stats counters |

### Snapshot and History

| API | Returns |
|---|---|
| `torch.cuda.memory_snapshot()` | List of segment dicts (described above) |
| `torch.cuda.memory_history()` | Global allocation history (requires `PYTORCH_NO_CUDA_MEMORY_CACHE`) |

### Memory Stats Keys (from `torch.cuda.memory_stats()`)

```
allocated_bytes.all.current       # current live bytes
allocated_bytes.all.peak          # peak live bytes
allocated_bytes.all.allocated     # total bytes ever allocated (cumulative)
allocated_bytes.all.freed         # total bytes ever freed (cumulative)

reserved_bytes.all.current        # current reserved bytes
reserved_bytes.all.peak           # peak reserved bytes

segment.all.current               # number of segments
segment.all.peak                  # peak number of segments
segment.all.allocated             # cumulative segment allocations
segment.all.free                  # cumulative segment frees

active_bytes.all.current          # active_allocated bytes
active_bytes.all.inactive         # inactive (pooled free) bytes
active_bytes.all.pending          # pending_free bytes
```

### Key Constants

```python
MiB = 1024 ** 2
GiB = 1024 ** 3
```

### Diagnostic One-Liners

```python
# Print a compact memory summary
print(torch.cuda.memory_summary())

# Check if memory is leaking (call before and after a step)
print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MiB")

# See total GPU memory
total, free = torch.cuda.mem_get_info()
print(f"GPU {total/1024**3:.1f} GiB total, {free/1024**2:.0f} MiB free")

# Force reset before measuring peak for a specific run
torch.cuda.reset_peak_memory_stats()
# ... run model ...
print(f"Peak: {torch.cuda.max_memory_allocated()/1024**2:.1f} MiB")
```
