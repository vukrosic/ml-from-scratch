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
