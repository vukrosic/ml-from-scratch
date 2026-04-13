# DataLoader From Scratch

You need to feed data to your model efficiently. DataLoader is the pipe, and understanding its internals lets you avoid the bottlenecks that slow training.

---

## What Problem Does DataLoader Solve?

Training a deep learning model means looping over your dataset many times (epochs). Doing this naively loads data on the main thread, blocks GPU waiting for CPU, and offers no parallelism.

DataLoader solves this with a multi-process pipeline:

```
Main Process          Worker 0           Worker 1
    |                    |                  |
    |--request batch--->|                  |
    |                   |--load sample--->  |
    |                   |                  |--load sample--->|
    |<--return batch----|<--return batch---|                  |
    |                    |                  |                  |
```

---

## torch.utils.data.Dataset

DataLoader consumes a `Dataset`. There are two styles.

### Map-Style Dataset

A map-style dataset is a sequence—you index it with `[i]` and it returns one item.

```python
# dataset.py
import torch
from torch.utils.data import Dataset


class RandomIntDataset(Dataset):
    """Map-style dataset yielding random integers."""

    def __init__(self, size: int, min_val: int = 0, max_val: int = 100):
        self.size = size
        self.min_val = min_val
        self.max_val = max_val

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.randint(self.min_val, self.max_val, (1,))


class TextSequenceDataset(Dataset):
    """Map-style dataset yielding variable-length token sequences."""

    def __init__(self, sequences: list[list[int]], vocab_size: int = 1000):
        self.sequences = sequences
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.long)


# Example usage
if __name__ == "__main__":
    # Fixed-size random ints
    int_ds = RandomIntDataset(size=100)
    print(f"Dataset size: {len(int_ds)}")
    print(f"Sample: {int_ds[0]}")

    # Variable-length sequences
    seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10], [11]]
    text_ds = TextSequenceDataset(seqs)
    print(f"Dataset size: {len(text_ds)}")
    print(f"Sample 0: {text_ds[0]}")
    print(f"Sample 1: {text_ds[1]}")
```

Key methods:
- `__len__(self)` — returns the number of items
- `__getitem__(self, idx)` — returns item at index `idx`

The dataset itself holds no batch dimension—it returns a single item per index.

### Iterable-Style Dataset

An iterable-style dataset is a stream—you iterate it directly and it handles its own indexing logic. Use this for large datasets that don't fit in memory, or for data from a generator or database cursor.

```python
# dataset.py (continued)
class InfiniteRandomDataset(torch.utils.data.IterableDataset):
    """Iterable-style dataset that generates infinite random data."""

    def __init__(self, batch_size: int = 32, dim: int = 10):
        self.batch_size = batch_size
        self.dim = dim

    def __iter__(self):
        while True:
            yield torch.randn(self.batch_size, self.dim)


class FileStreamDataset(torch.utils.data.IterableDataset):
    """Iterable-style dataset that reads lines from a file."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def __iter__(self):
        with open(self.filepath, "r") as f:
            for line in f:
                yield line.strip()


# Example usage
if __name__ == "__main__":
    # Infinite dataset - must be used with limit or DataLoader worker_ctrl+c
    inf_ds = InfiniteRandomDataset(batch_size=16, dim=5)
    samples_taken = 0
    for batch in inf_ds:
        samples_taken += 1
        if samples_taken >= 3:
            break
    print(f"Got {samples_taken} batches from infinite dataset")
```

Iterable-style datasets have no `__len__` and no indexed `__getitem__`. The data source decides when iteration stops.

---

## Sampler — Controlling Data Order

Samplers decide which indices to return and in what order.

```python
# dataset.py (continued)
from torch.utils.data import Sampler


class AlternatingSampler(Sampler):
    """Sampler that alternates between two halves of the dataset."""

    def __init__(self, data_source: Dataset, seed: int = 42):
        self.data_source = data_source
        self.seed = seed
        self.rng = torch.Generator()

    def __iter__(self):
        self.rng.manual_seed(self.seed)
        n = len(self.data_source)
        indices = torch.randperm(n, generator=self.rng).tolist()
        first_half = indices[:n // 2]
        second_half = indices[n // 2:]
        # Interleave: first from each half alternately
        interleaved = []
        for i in range(max(len(first_half), len(second_half))):
            if i < len(first_half):
                interleaved.append(first_half[i])
            if i < len(second_half):
                interleaved.append(second_half[i])
        return iter(interleaved)

    def __len__(self) -> int:
        return len(self.data_source)


if __name__ == "__main__":
    ds = RandomIntDataset(size=20)
    sampler = AlternatingSampler(ds)
    print(f"Sampler order: {list(sampler)}")
```

PyTorch provides built-in samplers:
- `SequentialSampler` — returns indices 0, 1, 2, ...
- `RandomSampler` — returns shuffled indices
- `SubsetRandomSampler` — samples from a specific subset of indices
- `WeightedRandomSampler` — samples with given class weights

---

## collate_fn — How It Batches Variable-Length Sequences

When you pass a list of tensors with different shapes to `DataLoader`, it calls `collate_fn` to combine them into a single batched tensor.

The default `collate_fn` fails on variable-length sequences. You need a custom collate function to pad them.

```python
# custom_collate.py
import torch
from torch.utils.data import DataLoader
from typing import Callable


def pad_sequence_collate(
    batch: list[torch.Tensor],
    padding_value: int = 0,
    batch_first: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate variable-length sequences by padding them to the longest sequence.

    Returns:
        padded_batch: Tensor of shape (batch_size, max_seq_len, ...) 
        lengths: Tensor of original sequence lengths
    """
    lengths = torch.tensor([len(seq) for seq in batch])
    padded = torch.nn.utils.rnn.pad_sequence(
        batch,
        batch_first=batch_first,
        padding_value=padding_value,
    )
    return padded, lengths


def nested_tensor_collate(
    batch: list[dict],
    padding_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate a batch of dictionaries with 'tokens' and 'labels' keys.
    Each dictionary contains variable-length sequences.
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


# Demonstration
if __name__ == "__main__":
    # Variable-length sequences
    sequences = [
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5]),
        torch.tensor([6, 7, 8, 9]),
        torch.tensor([10]),
    ]

    batch, lengths = pad_sequence_collate(sequences)
    print(f"Padded batch shape: {batch.shape}")
    print(f"Padded batch:\n{batch}")
    print(f"Lengths: {lengths}")

    # Dictionary collate
    dict_batch = [
        {"tokens": torch.tensor([1, 2]), "labels": torch.tensor([0, 1])},
        {"tokens": torch.tensor([3, 4, 5, 6]), "labels": torch.tensor([1, 0, 1, 0])},
        {"tokens": torch.tensor([7, 8, 9]), "labels": torch.tensor([1, 1])},
    ]
    tokens_out, labels_out = nested_tensor_collate(dict_batch)
    print(f"\nTokens batch shape: {tokens_out['tokens'].shape}")
    print(f"Labels batch shape: {labels_out['labels'].shape}")
```

`collate_fn` receives a list of samples (each from one `__getitem__` call) and returns a batched tensor or dictionary of tensors.

---

## num_workers — Why 0 Is Sometimes Faster

`num_workers` controls how many subprocesses load data in parallel.

- `num_workers=0` — everything happens on the main thread
- `num_workers=N` — N worker processes prefetch data

More workers seem better, but they add overhead:

```python
# benchmark_workers.py
import time
import torch
from torch.utils.data import DataLoader, Dataset


class TimedDataset(Dataset):
    """Dataset with artificial load delay to simulate expensive __getitem__."""

    def __init__(self, size: int, load_time: float = 0.01):
        self.size = size
        self.load_time = load_time

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        time.sleep(self.load_time)  # Simulate I/O or preprocessing
        return torch.randn(10)


def benchmark_workers(
    dataset: Dataset,
    num_workers: int,
    batch_size: int = 32,
    num_batches: int = 10,
) -> float:
    """Time how long it takes to fetch num_batches batches."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    start = time.perf_counter()
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        _ = batch.shape  # Force CUDA transfer if pin_memory is used
    elapsed = time.perf_counter() - start
    return elapsed


if __name__ == "__main__":
    dataset = TimedDataset(size=1000, load_time=0.01)  # 10ms per sample

    print("Workers | Time (s) | Throughput (samples/s)")
    print("--------|----------|----------------------")

    for workers in [0, 1, 2, 4, 8]:
        t = benchmark_workers(dataset, num_workers=workers, num_batches=20)
        throughput = (20 * 32) / t
        print(f"{workers:7d} | {t:8.3f} | {throughput:21.1f}")
```

Typical output:

```
Workers | Time (s) | Throughput (samples/s)
--------|----------|----------------------
      0 |    6.421 |                 99.4
      1 |    3.351 |                190.9
      2 |    1.891 |                338.5
      4 |    1.102 |                580.6
      8 |    0.918 |                696.9
```

When `load_time` is very small (microseconds), the multiprocessing overhead dominates. Try with `load_time=0.0001` (0.1ms):

```
Workers | Time (s) | Throughput (samples/s)
--------|----------|----------------------
      0 |    0.064 |              9985.4
      1 |    0.891 |               714.2
      2 |    1.102 |               580.3
      4 |    1.445 |               443.1
      8 |    2.103 |               304.8
```

**Why 0 can be faster:**
- No process spawn overhead
- No IPC (inter-process communication) for each batch
- No memory copy between processes
- The GIL is released during the actual data operation in `__getitem__`

**Rule of thumb:** If your `__getitem__` is fast (< 1ms), try `num_workers=0` first. If it's slow (image decoding, complex augmentation), more workers help.

---

## pin_memory — Why It Speeds Up Host-to-GPU Transfers

`pin_memory=True` allocates page-locked (non-paginated) memory on the host. This enables direct memory access (DMA) between CPU and GPU without an intermediate CPU copy.

```python
# pin_memory.py
import time
import torch
from torch.utils.data import DataLoader, Dataset


class RandomDataset(Dataset):
    """Simple dataset producing random tensors."""

    def __init__(self, size: int, tensor_size: tuple[int, int, int]):
        self.size = size
        self.tensor_size = tensor_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.randn(self.tensor_size)


def benchmark_pin_memory(
    dataset: Dataset,
    pin_memory: bool,
    batch_size: int = 32,
    num_batches: int = 50,
    device: str = "cuda",
) -> dict[str, float]:
    """Benchmark DataLoader with and without pinned memory."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=4,
        persistent_workers=True,
    )

    # Warmup
    for i, batch in enumerate(loader):
        if i >= 3:
            break
        _ = batch.to(device, non_blocking=True)

    torch.cuda.synchronize() if device.startswith("cuda") else None

    times = []
    for i, batch in enumerate(loader):
        start = time.perf_counter()
        _ = batch.to(device, non_blocking=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
    }


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU benchmarks")

    dataset = RandomDataset(size=10000, tensor_size=(3, 224, 224))

    print("Pin Memory | Transfer Time (ms)")
    print("-----------|--------------------")
    for pin in [False, True]:
        results = benchmark_pin_memory(dataset, pin_memory=pin)
        print(f"{pin:9} | {results['mean']*1000:8.2f} ± {results['max']*1000:.2f}")
```

Typical speedup on a PCIe Gen3 x16 link:

```
Pin Memory | Transfer Time (ms)
-----------|--------------------
      False |     2.31 ± 0.45
       True |     0.89 ± 0.12
```

**Why it works:**
- Normal memory can be paged to disk; the OS must page it back before DMA
- Pinned memory stays in RAM; DMA controller transfers directly to GPU
- `non_blocking=True` on `.to(device)` overlaps transfer with computation (if streams are used correctly)

**Trade-offs:**
- Pinned memory is a limited resource (set by `cuda::Nvidia::DriverAPI::getLimit()`)
- Too much pinned memory can cause OOM or slow down other processes
- Only helps when transferring to CUDA

---

## prefetch_factor — Controlling Pipeline Depth

`prefetch_factor=N` means each worker loads `N` batches ahead of time into a queue. The total pipeline depth is `num_workers * prefetch_factor`.

```python
# Demonstrate prefetch_factor effect
if __name__ == "__main__":
    ds = RandomDataset(size=1000, tensor_size=(3, 224, 224))

    print("Prefetch Factor | Worker | Time (s)")
    print("----------------|--------|----------")

    for prefetch in [2, 4, 8]:
        for workers in [0, 4]:
            if workers == 0 and prefetch > 2:
                continue  # prefetch_factor only matters with workers > 0
            t = benchmark_workers(
                ds, num_workers=workers, num_batches=20
            )
            print(f"{prefetch:16d} | {workers:6d} | {t:8.3f}")
```

Increasing `prefetch_factor` fills the queue faster, reducing the chance the GPU stalls. But a too-large queue increases memory usage and can make your batches stale if training order matters (e.g., for Curriculum Learning).

---

## Memory: What a DataLoader Worker Actually Holds

Each DataLoader worker process holds its own copy of:

1. **The Dataset object** — Python object with `__getitem__` logic
2. **The model (if passed to the worker)** — not done by default, but if you use custom worker_init_fn
3. **Shared memory buffers** — for the queue communication with the main process

Worker memory = Dataset memory + queue buffers + per-batch intermediate tensors.

**Memory leak scenario:**

```python
# WARNING: This is a pattern that causes memory leaks
class BadDataset(Dataset):
    def __init__(self, large_array):
        self.large_array = large_array  # Holds reference!

    def __getitem__(self, idx):
        return self.large_array[idx]  # Returns view, not copy
```

Each worker holds a reference to `large_array`. If `large_array` is 1GB and you have 8 workers, you use 8GB just for datasets in addition to your model.

**Best practice:** Keep Dataset objects small. Load data inside `__getitem__` or use memory-mapped files (e.g., `numpy.memmap`, `torch.from_file`).

---

## Full DataLoader Configuration Example

```python
from torch.utils.data import DataLoader, Dataset

loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=True,
    persistent_workers=True,  # Keep workers alive between epochs
    drop_last=True,            # Drop final incomplete batch
    timeout=0,                 # 0 = no timeout (wait forever)
)
```

---

## Recap

- Map-style `Dataset` uses `__len__` and `__getitem__`; iterable-style uses `__iter__`
- Samplers control index order; built-ins handle sequential, random, and weighted sampling
- `collate_fn` combines samples into batches; custom collate handles variable-length sequences
- `num_workers=0` avoids multiprocessing overhead when `__getitem__` is fast
- `pin_memory=True` enables DMA for faster CPU-to-GPU transfers
- `prefetch_factor` controls how many batches each worker prefetches
- Each worker holds its own Dataset copy—keep datasets small to avoid memory bloat

---

## Going Further

For real benchmark numbers across num_workers values, pin_memory speedup measurements, prefetch_factor scaling, and memory profiling of worker processes — see [ADVANCED.md](./ADVANCED.md).

---

Get the video walkthrough of num_workers tuning, pin_memory profiling, and worker memory analysis: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
