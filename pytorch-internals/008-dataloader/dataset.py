import torch
from torch.utils.data import Dataset, IterableDataset


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


class InfiniteRandomDataset(IterableDataset):
    """Iterable-style dataset that generates infinite random data."""

    def __init__(self, batch_size: int = 32, dim: int = 10):
        self.batch_size = batch_size
        self.dim = dim

    def __iter__(self):
        while True:
            yield torch.randn(self.batch_size, self.dim)


class FileStreamDataset(IterableDataset):
    """Iterable-style dataset that reads lines from a file."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def __iter__(self):
        with open(self.filepath, "r") as f:
            for line in f:
                yield line.strip()


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
