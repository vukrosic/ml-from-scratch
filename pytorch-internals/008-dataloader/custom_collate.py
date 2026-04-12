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
) -> tuple[dict, dict]:
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
