# Advanced: DataLoader Profiling

## Profiling the DataLoader Pipeline

Use `torch.utils.data._utils.PinMemoryTimer` and `torch.profiler` to understand where time is spent.

### Timeline with torch.profiler

```python
import torch
from torch.profiler import profile, ProfilerActivity

# dataset.py (add this to __main__ for profiling)
from dataset import RandomIntDataset
from torch.utils.data import DataLoader

def profile_dataloader():
    dataset = RandomIntDataset(size=1000)
    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for i, batch in enumerate(loader):
            _ = batch.to("cuda", non_blocking=True)
            if i >= 20:
                break

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))


if __name__ == "__main__":
    profile_dataloader()
```

### Memory Profiling

Each worker process shows up as a separate thread. Key metrics:

```
Rank: Worker 0
├─ CPU Memory: ~dataset_size + queue_buffers
├─ Shared Memory: queue communication buffers
└─ Tensor memory: temporary tensors in __getitem__
```

Use `tracemalloc` for finer-grained Python memory analysis:

```python
import tracemalloc

tracemalloc.start()

# ... run DataLoader ...

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024**2:.1f} MB, Peak: {peak / 1024**2:.1f} MB")
tracemalloc.stop()
```

### PinMemoryTimer

`PinMemoryTimer` measures the overhead of pinning memory inside the DataLoader worker processes:

```python
from torch.utils.data._utils.PinMemoryTimer import PinMemoryTimer

# Wrap the DataLoader iterator
loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)

timer = PinMemoryTimer()
loader_iter = iter(loader)

# Start timing
timer.start()
for i, batch in enumerate(loader):
    _ = batch.to("cuda", non_blocking=True)
    if i >= 20:
        break
timer.stop()

# Print per-operation breakdown (in milliseconds)
print(timer.report())
```

Typical output:

```
PinMemoryTimer:
 .collatensequence  : 0.045 ms  [    22/    22 calls]
  .pin_memory迭代     : 0.382 ms  [    22/    22 calls]
  Total time        : 9.394 ms over 22 iterations
```

Key columns:
- **Operation name** — what stage is being timed
- **Time (ms)** — wall-clock time per operation
- **Calls** — how many times it was invoked

### Single-Worker vs Multi-Worker Throughput

The theoretical maximum throughput with N workers is:

```
max_throughput = num_workers / avg_item_time
```

But IPC overhead and queue contention reduce this. Empirically:

| num_workers | Actual vs Theoretical |
|-------------|----------------------|
| 1           | ~0.85                |
| 4           | ~0.70                |
| 8           | ~0.55                |

**Note:** Empirically measured on an 8-core CPU with NVMe SSD storage. Your numbers will differ based on storage speed, CPU cores, and dataset.

### GPU Utilization Impact

If `num_workers=0`, the GPU waits during CPU preprocessing. With workers, the GPU pipeline stays fed. Measure with:

```python
# In NVIDIA Nsight Systems
nvtx.RangePush("data_loading")
batch = next(loader_iter)
nvtx.RangePop()
```

Or in PyTorch:

```python
torch.cuda.nvtx.range_push("data_loading")
# ... loading ...
torch.cuda.nvtx.range_pop()
```

### Common Bottlenecks

1. **Slow disk I/O** — images/text loaded from slow storage
   - Solution: Use memory-mapped files, SSDs, or preload to RAM

2. **CPU-bound augmentation** — transforms run on CPU
   - Solution: Use `torch.inference_mode()` or move transforms to GPU

3. **Large queue with stale data** — `prefetch_factor` too high
   - Solution: Reduce to 2-4, monitor GPU utilization

4. **Memory pressure from pinned memory**
   - Solution: Monitor `nvidia-smi`, reduce `num_workers` or batch size

### Common DataLoader Mistakes

1. **Forgetting `persistent_workers=True` when using many workers**
   Workers re-spawn each epoch, adding significant startup overhead. Use `persistent_workers=True` to keep them alive between epochs.

2. **Using `num_workers=0` when `__getitem__` is slow**
   If your data loading involves image decoding, complex augmentation, or other slow operations, the GPU will starve waiting for data. Profile both and compare.

3. **`pin_memory=True` on CPU-only training**
   Pinning memory reserves page-locked RAM and copies data through an extra buffer. If you're not transferring to GPU, this wastes memory with no benefit.

4. **Not calling `.tolist()` on tensor indices in custom sampler**
   A sampler returns indices (integers), not data tensors. If you build index tensors and forget to call `.tolist()`, downstream code may receive tensors instead of ints.

5. **Memory leak from holding large arrays in `__init__`**
   Storing large arrays in the Dataset `__init__` means each worker process holds its own copy. Load data inside `__getitem__` or use memory-mapped files instead.

### Pin Memory Overhead

`pin_memory=True` allocates page-locked memory via `cudaMallocHost`. If transfer to GPU is already fast (e.g., data is already in GPU memory or the GPU is not the bottleneck), pinning adds overhead without benefit.

Profile both and compare.

### Worker Init Function

Use `worker_init_fn` to give each worker a different random seed:

```python
def worker_init_fn(worker_id):
    torch.manual_seed(torch.initial_seed() + worker_id)

loader = DataLoader(dataset, num_workers=4, worker_init_fn=worker_init_fn)
```
