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
    print("=" * 60)
    print("Benchmark: num_workers effect with SLOW __getitem__ (10ms)")
    print("=" * 60)
    dataset = TimedDataset(size=1000, load_time=0.01)  # 10ms per sample

    print("\nWorkers | Time (s) | Throughput (samples/s)")
    print("--------|----------|----------------------")

    for workers in [0, 1, 2, 4, 8]:
        t = benchmark_workers(dataset, num_workers=workers, num_batches=20)
        throughput = (20 * 32) / t
        print(f"{workers:7d} | {t:8.3f} | {throughput:21.1f}")

    print("\n" + "=" * 60)
    print("Benchmark: num_workers effect with FAST __getitem__ (0.1ms)")
    print("=" * 60)
    dataset_fast = TimedDataset(size=1000, load_time=0.0001)  # 0.1ms per sample

    print("\nWorkers | Time (s) | Throughput (samples/s)")
    print("--------|----------|----------------------")

    for workers in [0, 1, 2, 4, 8]:
        t = benchmark_workers(dataset_fast, num_workers=workers, num_batches=20)
        throughput = (20 * 32) / t
        print(f"{workers:7d} | {t:8.3f} | {throughput:21.1f}")
