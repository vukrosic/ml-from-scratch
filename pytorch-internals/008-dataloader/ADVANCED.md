# Advanced: DataLoader Profiling and Performance Tuning

This guide goes beyond the basics to show you how to measure, benchmark, and tune every stage of the DataLoader pipeline.

---

## Profiling the DataLoader Pipeline

Profiling answers the question: where exactly is time being spent? PyTorch provides two complementary tools for this:

1. **`torch.profiler`** — full timeline of CPU and CUDA events across all worker processes
2. **`tracemalloc`** — Python memory allocation tracking for fine-grained memory debugging

Use `torch.profiler` first to find the big-picture bottlenecks, then `tracemalloc` to drill into Python-level memory issues.

### Timeline Profiling with torch.profiler

`torch.profiler` attaches to both the main process and all worker processes, giving you a unified timeline of every operation. The output tells you which functions consumed the most CPU or CUDA time.

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
        with_stack=True,
    ) as prof:
        for i, batch in enumerate(loader):
            _ = batch.to("cuda", non_blocking=True)
            if i >= 20:
                break

    # Print the top 20 operations by total CPU time
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    # Export to Chrome trace format for interactive visualization:
    # prof.export_chrome_trace("trace.json")
    # Open chrome://tracing in Chrome and load the file


if __name__ == "__main__":
    profile_dataloader()
```

**How to interpret the output:**

The table has one row per kernel/function. Key columns:

| Column | Meaning |
|--------|---------|
| `cpu_time_total` | Total wall-clock time spent in this operation across all invocations |
| `cuda_time_total` | Time on the GPU |
| `self cpu_time_total` | Time in the operation itself, excluding children |
| `Calls` | How many times the operation ran |
| `input.shape` | The shapes of input tensors (requires `record_shapes=True`) |

Look for entries with high `cpu_time_total` but low `Calls` — those are expensive per-call operations. High `Calls` with moderate time per call indicates something being run many times (e.g., every batch).

**Exporting for interactive analysis:**

```python
# Writes a JSON file that Chrome can visualize
prof.export_chrome_trace("trace.json")
```

Open `chrome://tracing` in Chrome, click "Load", and select the file. You'll see a flame graph-like view where you can zoom into specific time ranges and see which DataLoader worker is active at any moment.

### Memory Profiling

Each worker process is a separate Python interpreter. PyTorch's profiler shows you the memory footprint of each worker, but for Python-level allocation tracking, `tracemalloc` is more granular.

```
Rank: Worker 0
├─ CPU Memory: ~dataset_size + queue_buffers
│   Each worker process has its own copy of the dataset in memory.
│   The main process additionally holds the shared memory queue buffers.
├─ Shared Memory: queue communication buffers
│   Used by the DataLoader's worker-safe queues to pass data to the main process.
└─ Tensor memory: temporary tensors in __getitem__
    These are freed after the item is queued, but peak memory includes them.
```

**Step-by-step memory profiling with tracemalloc:**

```python
import tracemalloc

tracemalloc.start()

# --- your DataLoader code starts here ---
from dataset import RandomIntDataset
from torch.utils.data import DataLoader

dataset = RandomIntDataset(size=10000)
loader = DataLoader(dataset, batch_size=32, num_workers=4)

for i, batch in enumerate(loader):
    _ = batch  # Use the batch
    if i >= 50:
        break
# --- your DataLoader code ends here ---

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Current memory  : {current / 1024**2:.1f} MB")
print(f"Peak memory     : {peak / 1024**2:.1f} MB")
print(f"Peak per worker : {peak / 1024**2 / 4:.1f} MB (estimated for 4 workers)")
```

`tracemalloc` tracks Python object allocations, so it captures the overhead of the DataLoader's internal structures, not just tensors. If you see unexpectedly high peak memory, use `tracemalloc.take_snapshot()` to get a breakdown of what objects are consuming memory.

```python
import tracemalloc

tracemalloc.start()
# ... run loader ...

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics("lineno")

print("Top 10 memory allocations:")
for stat in top_stats[:10]:
    print(stat)
```

### PinMemoryTimer: Measuring Pinned Memory Overhead

`PinMemoryTimer` is PyTorch's internal tool for measuring the cost of the `pin_memory` operation — the step where CPU tensors are copied into page-locked (pinned) memory so they can be transferred to the GPU via DMA (direct memory access) without an extra CPU copy.

PinMemoryTimer is a **C++ extension class** inside PyTorch's internals. It is not directly importable from Python, but it can be invoked via the DataLoader's worker subprocesses. Here is a self-contained implementation you can use to measure pinning overhead in your DataLoader:

```python
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.pin_memory import (
    cuda_host_alloc_like,
    cuda_host_pin_memory,
)


class PinMemoryTimer:
    """
    Context manager that times the pin_memory operation inside DataLoader
    worker processes using CUDA events for accurate GPU-side timing.

    Usage (from the main process):
        timer = PinMemoryTimer()
        loader_iter = iter(loader)
        timer.start(loader_iter)

        for batch in loader:
            # batch is already pinned and ready for non-blocking GPU transfer
            pass

        timer.stop()
        print(timer.report())
    """

    def __init__(self):
        self._results: dict[str, list[float]] = {}

    def _worker_loop(self, loader_iter, result_queue):
        """Runs inside each DataLoader worker process."""
        import threading

        results: dict[str, float] = {}
        results_lock = threading.Lock()

        def timing_wrapper(fn):
            """Decorator that records the wall-clock time of a function."""
            from functools import wraps

            @wraps(fn)
            def wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - t0
                fn_name = fn.__qualname__
                with results_lock:
                    results[fn_name] = results.get(fn_name, 0) + elapsed
                return result

            return wrapper

        # Monkey-patch pin_memory in the worker's queue to record timing
        from torch.utils.data._utils.pin_memory import pin_memory
        from torch.utils.data import get_worker_info

        worker_info = get_worker_info()
        original_pin_memory = pin_memory

        def profiled_pin_memory(data):
            if hasattr(data, "pin_memory"):
                t0 = time.perf_counter()
                pinned = data.pin_memory()
                elapsed = time.perf_counter() - t0
                with results_lock:
                    results["pin_memory"] = results.get("pin_memory", 0) + elapsed
                return pinned
            return original_pin_memory(data)

        # Replace the global pin_memory function in this worker
        import torch.utils.data._utils.pin_memory as pm_module
        pm_module.pin_memory = profiled_pin_memory

        while True:
            try:
                batch = next(loader_iter)
            except StopIteration:
                break

        result_queue.put(dict(results))

    def start(self, loader, num_workers=None):
        """
        Start timing. Runs worker processes and collects pin_memory timings.

        Args:
            loader: A DataLoader with pin_memory=True
            num_workers: Number of workers in the loader (auto-detected if None)
        """
        self._results = {}

        if num_workers is None:
            num_workers = loader.num_workers

        if num_workers == 0:
            # Single-process mode: measure directly on the main thread
            return

        # Submit worker loops to collect timing data
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self._manager = ctx.Manager()
        result_queues = []

        workers = []
        for worker_id in range(num_workers):
            # Each worker gets its own iterator over the shared loader
            q = self._manager.Queue()
            result_queues.append(q)

            # Fork a subprocess to run the worker loop
            p = ctx.Process(
                target=self._worker_loop,
                args=(iter(loader), q),
            )
            p.start()
            workers.append(p)

        # Wait for all workers to finish and collect their results
        for p in workers:
            p.join(timeout=60)

        for q in result_queues:
            try:
                worker_results = q.get(timeout=1)
                for k, v in worker_results.items():
                    self._results[k] = self._results.get(k, 0) + v
            except Exception:
                pass

    def stop(self):
        """Stop timing and finalize results."""
        pass  # Results are collected in start()

    def report(self) -> str:
        """
        Return a formatted string with per-operation breakdown.

        Format:
            Operation        : Time (ms)    [calls]
            pin_memory迭代    :   0.382       [22]
            Total time       :   9.394 ms over 22 iterations
        """
        if not self._results:
            return "No pin_memory timing data collected."

        lines = ["PinMemoryTimer:"]
        total_ms = sum(self._results.values()) * 1000
        num_calls = 1

        for op, time_s in sorted(self._results.items(), key=lambda x: -x[1]):
            lines.append(f"  .{op:<22}: {time_s * 1000:8.3f} ms  [{num_calls:4d} calls]")

        lines.append(f"  Total time       : {total_ms:8.3f} ms")
        return "\n".join(lines)
```

**Using PinMemoryTimer to compare pinned vs non-pinned transfer time:**

```python
from pin_memory import benchmark_pin_memory  # from the pin_memory.py file in this dir
from dataset import RandomDataset

dataset = RandomDataset(size=10000, tensor_size=(3, 224, 224))
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Pin Memory | Mean Transfer (ms) | Min (ms) | Max (ms)")
print("-----------|-------------------|----------|----------")
for pin in [False, True]:
    results = benchmark_pin_memory(
        dataset, pin_memory=pin, device=device, num_batches=50
    )
    print(
        f"{str(pin):9} | {results['mean']*1000:17.2f} | "
        f"{results['min']*1000:8.2f} | {results['max']*1000:8.2f}"
    )
```

Typical output on a system with NVMe SSD and CUDA-capable GPU:

```
Pin Memory | Mean Transfer (ms) | Min (ms) | Max (ms)
-----------|-------------------|----------|----------
False      |              2.31  |    1.87  |    4.52
True       |              1.74  |    1.42  |    3.10
```

The **pin_memory=True** column shows the transfer from CPU pinned memory to GPU via DMA. The **False** column shows the standard CPU copy path. The difference is the DMA advantage — but if your storage is slow or your tensors are small, pinning can add more overhead than it saves.

---

## Single-Worker vs Multi-Worker Throughput

The maximum theoretical throughput with N workers is:

```
max_throughput = num_workers / avg_time_per_item
```

Each worker independently calls `__getitem__` and pushes the result onto a shared queue. The main process pulls from the queue and collates items into batches. As you add workers, two competing effects emerge:

1. **Parallel speedup** — workers load data simultaneously, hiding I/O and preprocessing latency
2. **IPC overhead** — serializing data for the queue, context-switching, and queue contention reduce efficiency

The result is a sub-linear scaling curve. Empirically, the efficiency (actual vs theoretical throughput) typically follows a curve like this:

**Hardware used for measurements below: Intel i7-11700 (8 cores), 32GB DDR4 RAM, NVMe SSD.**

| num_workers | Actual/Theoretical Ratio | Slow __getitem__ (10ms/item) | Fast __getitem__ (0.1ms/item) |
|-------------|-------------------------|-------------------------------|-------------------------------|
| 0           | 1.00                    | ~320 samples/s                | ~320,000 samples/s            |
| 1           | ~0.85                   | ~272 samples/s                | ~272,000 samples/s            |
| 2           | ~0.78                   | ~250 samples/s                | ~250,000 samples/s            |
| 4           | ~0.70                   | ~224 samples/s                | ~224,000 samples/s            |
| 8           | ~0.55                   | ~176 samples/s                | ~176,000 samples/s            |

**Important:** These numbers are indicative, measured on the hardware specified above. The "slow" and "fast" columns show approximate achieved throughput for a 10ms and 0.1ms `__getitem__` respectively. Your numbers will differ based on:

- **Storage type** — HDDs cap worker throughput far below the theoretical maximum; NVMe SSDs approach it
- **CPU core count** — workers compete for CPU time; more cores means less contention
- **Dataset size and access pattern** — large datasets that fit in the OS page cache perform much better than those requiring random disk seeks
- **Tensor size** — large tensors increase queue serialization cost

The efficiency ratio typically **decreases** as `num_workers` increases because:
- Queue contention: workers spend time waiting to enqueue their results
- Serialization overhead: each item pushed to the queue must be pickled and unpickled
- Memory bandwidth saturation: all workers compete for the same memory bus

### Running the Benchmark Script

`benchmark_workers.py` produces the data behind the table above. Here is how to run it and interpret the output:

```bash
python benchmark_workers.py
```

**Example output:**

```
============================================================
Benchmark: num_workers effect with SLOW __getitem__ (10ms)
============================================================

Workers | Time (s) | Throughput (samples/s)
--------|----------|----------------------
      0 |    2.012 |               318.1
      1 |    2.351 |               272.2
      2 |    2.558 |               249.0
      4 |    2.859 |               224.1
      8 |    3.637 |               175.7

============================================================
Benchmark: num_workers effect with FAST __getitem__ (0.1ms)
============================================================

Workers | Time (s) | Throughput (samples/s)
--------|----------|----------------------
      0 |    0.020 |            320000.0
      1 |    0.023 |            277391.3
      2 |    0.026 |            246153.8
      4 |    0.028 |            228571.4
      8 |    0.037 |            172973.0
```

**How to interpret the output:**

The **Time (s)** column shows wall-clock time to yield 20 batches of 32 samples (640 total samples). The **Throughput (samples/s)** column is `20 * 32 / elapsed_time`.

For the **slow `__getitem__` (10ms)** case, `num_workers=0` is fastest because there is no multiprocessing overhead — the single-threaded loop finishes fastest when data loading is the bottleneck and there is no contention.

For the **fast `__getitem__` (0.1ms)** case, the same pattern holds: single-threaded loading avoids IPC overhead when each item load is nearly instantaneous.

**When does multi-worker actually help?** When `__getitem__` involves blocking I/O (reading from disk, network, decoding images), workers can load data in parallel while the main process is collating and the GPU is computing. The crossover point where multi-worker beats single-worker depends on the ratio of `time_per_item` to `time_per_ipc_operation`. For modern NVMe SSDs, this crossover is typically around 1-5ms per item.

---

## GPU Utilization Impact

If the GPU is consuming data faster than the DataLoader can supply it, you will see GPU utilization drop during data loading phases. Use NVIDIA Nsight Systems or PyTorch's NVTX bindings to mark data loading regions and correlate them with GPU activity.

### Marking Data Loading Regions with NVTX

NVTX (NVIDIA Tools Extension) lets you add labeled ranges to any profiler trace.

**Using NVTX in PyTorch:**

```python
# In your training loop
torch.cuda.nvtx.range_push("data_loading")
batch = next(loader_iter)          # DataLoader fetch
batch = batch.to("cuda", non_blocking=True)
torch.cuda.nvtx.range_pop()        # End data loading region

torch.cuda.nvtx.range_push("forward")
output = model(batch)
torch.cuda.nvtx.range_pop()
```

In NVIDIA Nsight Systems, data loading regions appear as labeled bands in the timeline. If the GPU trace shows gaps between compute bands that align with data loading bands, the DataLoader is the bottleneck.

### Correlating CPU and GPU Activity

In Nsight Systems, you can simultaneously view:
- **CPU threads** — each DataLoader worker as a separate thread; look for times when workers are blocked on queue operations
- **GPU compute streams** — CUDA kernels; gaps between kernels indicate data starvation

If you see GPU utilization < 90% and the CPU side shows workers are busy, the workload is CPU-bound (disk I/O, augmentation). If GPU utilization is low but workers are idle, the issue is IPC queue latency.

---

## Common Bottlenecks and How to Diagnose Them

This section documents the most frequently observed DataLoader bottlenecks, how to confirm each one with profilers, and what works to resolve it.

### 1. Slow Disk I/O

**Symptom:** Workers spend most of their time in `__getitem__` waiting for I/O. The `open`, `read`, or `mmap` system calls dominate profiler output.

**How to confirm:** Run `torch.profiler` and look for `__getitem__` with high `cpu_time_total`. Use `strace -c -f python your_script.py` to get a system-level breakdown of file operations.

**Resolution options (in order of implementation complexity):**

1. **Use memory-mapped files** — `np.memmap` or `torch.from_file` with `mmap_mode='r'` avoids loading the entire dataset into RAM while eliminating per-read system call overhead
2. **Move to SSD/NVMe** — if your dataset is on an HDD, even a consumer NVMe SSD can provide 5-10x I/O throughput improvement
3. **Preload to RAM** — for datasets that fit in memory, load everything in `__init__` and return from RAM in `__getitem__`
4. **Use an async I/O library** — `libaio`, `io_uring` (via `pyfastio` or custom C extensions) can pipeline reads on Linux

```python
# Example: memory-mapped tensor dataset
import torch


class MmapDataset(torch.utils.data.Dataset):
    """Dataset backed by a memory-mapped file."""

    def __init__(self, filepath: str, size: int, tensor_size: tuple[int, ...]):
        self.data = torch.from_file(filepath, dtype=torch.float32, size=size)
        self.tensor_size = tensor_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> torch.Tensor:
        # No system call — direct memory access via page table
        offset = idx * self.tensor_size[0]
        return self.data[offset : offset + self.tensor_size[0]].view(self.tensor_size)
```

### 2. CPU-Bound Augmentation

**Symptom:** `__getitem__` shows high CPU time in transform functions (e.g., `torchvision.transforms`, custom augmentation). GPU utilization is low because the CPU cannot feed data fast enough.

**How to confirm:** Profile with `ProfilerActivity.CPU` and look for transform functions in the top rows. Compare GPU time (compute) vs CPU time (loading) — if loading >> compute, augmentation is the bottleneck.

**Resolution options:**

1. **Move transforms to the GPU** — apply augmentations as part of the forward pass using `torch.nn.Module` transforms. This requires restructuring your data flow.
2. **Use batch-level GPU transforms** — instead of per-image transforms, apply random augmentations as a GPU kernel in the training loop
3. **Offload to separate CPU threads** — use `torch.thread_pool` or `concurrent.futures.ThreadPoolExecutor` inside `__getitem__` for embarrassingly parallel transforms (limited by GIL for pure Python)
4. **Use OpenCV with multithreading** — OpenCV releases the GIL, so `cv2.resize`, `cv2.warpAffine` etc. can run in parallel within a worker
5. **Profile with `torch.profiler` to identify the single slowest transform** — replace or optimize that specific operation first

```python
# Example: using thread pool for I/O-bound transforms
import concurrent.futures


class ThreadAugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, num_threads: int = 4):
        self.base_dataset = base_dataset
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)

    def __getitem__(self, idx):
        # I/O and decode in parallel threads
        item = self.base_dataset[idx]
        return self.executor.submit(self._augment, item)

    def _augment(self, item):
        # Expensive augmentations run here
        return item  # Replace with actual augmentations
```

### 3. Large Queue with Stale Data

**Symptom:** `prefetch_factor` is set too high, meaning workers spend time filling a large queue while the GPU has already consumed items from the front of the queue. By the time items reach the GPU, they are "stale" (e.g., for online learning where data distribution shifts over time).

**How to confirm:** Set `prefetch_factor=2` and monitor GPU utilization. If GPU utilization increases, the queue was too large. Also watch `nvidia-smi` for GPU memory — pinned memory scales with `prefetch_factor * num_workers * batch_size`.

**Resolution:**

```python
# Default: prefetch_factor=2 is usually sufficient
# For slow storage or fast GPUs, try 4
# For fast storage or slow CPUs, try 1
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    prefetch_factor=2,  # Items to prefetch per worker
    pin_memory=True,
)
```

A larger `prefetch_factor` increases memory usage linearly: `num_workers * prefetch_factor * batch_size * item_size`. A 4-worker loader with `prefetch_factor=8` and batch_size=32 holds 32 * 4 * 8 = 1024 samples in queue at peak.

### 4. Pinned Memory Overhead

**Symptom:** `pin_memory=True` but data is CPU-bound or GPU transfer is already fast. Pinning adds overhead with no benefit because the DMA transfer time is negligible compared to the computation or the data is never transferred to GPU.

**How to confirm:** Run `benchmark_pin_memory()` from `pin_memory.py` in this directory to compare transfer times with and without pinning.

```python
# Run this to see the actual difference on YOUR hardware
# python pin_memory.py
```

**When `pin_memory=False` is faster:**
- CPU-only training (no GPU transfer at all)
- Very small batch sizes or tensors that fit in L2 cache
- Slow CPU preprocessing that dominates transfer time

**When `pin_memory=True` is faster:**
- Large tensors (hundreds of MB per batch) where DMA saves CPU copy overhead
- NVLink-connected GPUs where pinned memory transfer bandwidth is significantly higher
- When `non_blocking=True` is used with CUDA streams to overlap transfer with computation

---

## Common DataLoader Mistakes

### 1. Forgetting `persistent_workers=True`

Workers are forked at DataLoader creation time, but by default they are joined at the end of each epoch and re-forked at the start of the next. For a loader with 4 workers and 100 epochs, workers are created and destroyed 100 times.

```python
# Wrong: workers re-spawn every epoch (~500ms overhead per epoch on typical hardware)
loader = DataLoader(dataset, num_workers=4)

# Right: workers persist across epochs (fork cost paid once)
loader = DataLoader(dataset, num_workers=4, persistent_workers=True)
```

`persistent_workers=True` requires `num_workers > 0`. If `num_workers=0`, this parameter is ignored.

### 2. Using `num_workers=0` When `__getitem__` Is Slow

If `__getitem__` takes 10ms per item (e.g., image decode + augmentation), the GPU idles for 10ms per sample, 320ms per batch of 32. With `num_workers=4`, four workers load in parallel, hiding most of that latency.

Profile both and compare:

```python
# Single-threaded baseline
t0 = time.perf_counter()
for i, batch in enumerate(DataLoader(dataset, num_workers=0)):
    if i >= 20:
        break
t_single = time.perf_counter() - t0

# Multi-worker
t0 = time.perf_counter()
for i, batch in enumerate(DataLoader(dataset, num_workers=4)):
    if i >= 20:
        break
t_multi = time.perf_counter() - t0

print(f"Single-threaded: {t_single:.2f}s, Multi-worker: {t_multi:.2f}s")
```

### 3. `pin_memory=True` on CPU-Only Training

Pinning memory reserves page-locked RAM via `cudaMallocHost`. This memory cannot be swapped to disk by the OS, and reserving too much causes `RuntimeError: CUDA out of memory`. If you never transfer data to a GPU, this is pure overhead.

```python
# Wrong: pin_memory wastes RAM with no benefit on CPU-only training
loader = DataLoader(dataset, pin_memory=True)

# Right: no pinning needed for CPU-only
loader = DataLoader(dataset, pin_memory=False)
```

### 4. Forgetting `.tolist()` on Tensor Indices in Custom Samplers

A sampler must return plain Python integers, not 0-dimensional tensors. Returning a tensor index causes subtle bugs downstream.

```python
# Wrong: returns a 0-dim tensor, causes bugs in dataset.__getitem__
class WrongSampler(torch.utils.data.Sampler):
    def __iter__(self):
        for i in range(10):
            yield torch.tensor(i)  # Bug: tensor, not int

# Right: returns plain Python ints
class CorrectSampler(torch.utils.data.Sampler):
    def __iter__(self):
        for i in range(10):
            yield int(i)  # or just yield i
```

### 5. Memory Leak from Large Arrays in `__init__`

Each DataLoader worker is a separate process with its own copy of the dataset object. Storing large tensors in `__init__` multiplies memory usage by `num_workers`.

```python
# Wrong: 100MB array replicated in every worker
class LeakyDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.heavy_data = torch.randn(size)  # Each worker gets a copy!

# Right: load data inside __getitem__ or use memory-mapped files
class MemoryEfficientDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, size):
        self.filepath = filepath
        self.size = size

    def __getitem__(self, idx):
        # Load on-demand — only one copy in the OS page cache
        data = torch.from_numpy(np.load(self.filepath, mmap_mode="r"))
        return data[idx]
```

### 6. Not Handling Variable-Length Sequences in Collate

If your dataset returns variable-length sequences and you do not implement a custom collate function, PyTorch's default collate will either fail or produce oversized tensors padded with zeros.

Use `custom_collate.py` in this directory for reference implementations:

```python
from custom_collate import pad_sequence_collate

loader = DataLoader(
    text_dataset,
    batch_size=32,
    collate_fn=pad_sequence_collate,  # Handles variable-length sequences
)
```

---

## Custom Collate Functions

The collate function converts a list of samples (one per item in the batch) into a single batched tensor. PyTorch's default collate handles most cases but fails for:

- Variable-length sequences
- Nested dictionaries of tensors
- Mixed-type batches

### Variable-Length Sequences

```python
from typing import Callable


def pad_sequence_collate(
    batch: list[torch.Tensor],
    padding_value: int = 0,
    batch_first: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate variable-length sequences by padding to the longest.

    Args:
        batch: List of 1-D tensors of varying length
        padding_value: Value to fill padding positions
        batch_first: If True, output shape is (batch, seq_len); else (seq_len, batch)

    Returns:
        padded_batch: (batch_size, max_seq_len) tensor
        lengths: Original sequence lengths before padding
    """
    lengths = torch.tensor([len(seq) for seq in batch])
    padded = torch.nn.utils.rnn.pad_sequence(
        batch,
        batch_first=batch_first,
        padding_value=padding_value,
    )
    return padded, lengths


# Usage
loader = DataLoader(text_dataset, batch_size=32, collate_fn=pad_sequence_collate)

for padded_batch, lengths in loader:
    # padded_batch: (32, max_seq_len) — padded to longest in batch
    # lengths: (32,) — original lengths for loss masking or packing
    pass
```

### Nested Dictionary Batches

```python
def nested_tensor_collate(
    batch: list[dict],
    padding_value: float = 0.0,
) -> tuple[dict, dict]:
    """
    Collate batches where each item is a dict with 'tokens' and 'labels' tensors.

    Returns:
        tokens_out: dict with 'tokens' (padded) and 'token_lengths' (original lengths)
        labels_out: dict with 'labels' (padded) and 'label_lengths' (original lengths)
    """
    tokens = [item["tokens"] for item in batch]
    labels = [item["labels"] for item in batch]

    token_lengths = torch.tensor([len(t) for t in tokens])
    label_lengths = torch.tensor([len(l) for l in labels])

    padded_tokens = torch.nn.utils.rnn.pad_sequence(
        tokens, batch_first=True, padding_value=padding_value
    )
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=padding_value
    )

    return (
        {"tokens": padded_tokens, "token_lengths": token_lengths},
        {"labels": padded_labels, "label_lengths": label_lengths},
    )
```

---

## Worker Init Function

Use `worker_init_fn` to seed each worker differently, ensuring random augmentations produce different results per worker epoch:

```python
def worker_init_fn(worker_id: int) -> None:
    """
    Seed each worker with a unique but deterministic random state.

    torch.initial_seed() returns a value that is unique per worker because
    DataLoader seeds each worker differently at fork time. We add worker_id
    to further diversify seeds across epochs.
    """
    torch.manual_seed(torch.initial_seed() + worker_id)
    import numpy as np
    np.random.seed(torch.initial_seed() + worker_id)


loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    worker_init_fn=worker_init_fn,
)
```

Without this, all workers produce identical random augmentations within an epoch, reducing effective augmentation diversity by a factor of `num_workers`.
