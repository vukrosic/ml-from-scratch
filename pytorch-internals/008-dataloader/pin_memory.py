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

    if device.startswith("cuda"):
        torch.cuda.synchronize()

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
        print("Running CPU-only demonstration instead.")
        device = "cpu"
    else:
        device = "cuda"

    dataset = RandomDataset(size=10000, tensor_size=(3, 224, 224))

    print("Pin Memory | Transfer Time (ms)")
    print("-----------|--------------------")
    for pin in [False, True]:
        results = benchmark_pin_memory(dataset, pin_memory=pin, device=device)
        print(f"{pin:9} | {results['mean']*1000:8.2f} ± {results['max']*1000:.2f}")
