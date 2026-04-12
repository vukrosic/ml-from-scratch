"""
Causal Mask utilities.

The causal mask ensures that during autoregressive decoding, token i can only
attend to tokens 0..i. This prevents information leakage from future tokens.

We create a lower-triangular mask using torch.triu with diagonal=1, which sets
all elements above the main diagonal (and the diagonal itself) to zero.
"""

import torch


def create_causal_mask(seq_len, device='cpu'):
    """
    Create a causal (lower-triangular) mask.

    Returns a boolean mask where True = position is masked (cannot attend).
    Shape: (seq_len, seq_len)

    Example for seq_len=4:
        [[False, True,  True,  True],   # pos 0 can only attend to 0
         [False, False, True,  True],   # pos 1 can attend to 0, 1
         [False, False, False, True],  # pos 2 can attend to 0, 1, 2
         [False, False, False, False]] # pos 3 can attend to all

    torch.triu with diagonal=1 shifts the 1s up one diagonal, so the
    main diagonal itself is included in the masked region — meaning each
    token cannot attend to itself in the strict "strictly causal" version.
    """
    # torch.triu: upper triangle (including diagonal) becomes 1
    # We invert to get True = masked
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    return mask


def apply_causal_mask(scores, mask):
    """
    Apply a causal mask to attention scores before softmax.

    Args:
        scores: attention scores of shape (batch, n_heads, seq_len, seq_len)
        mask: boolean mask of shape (seq_len, seq_len) where True = masked

    Returns:
        masked scores with -inf in positions that are masked

    The mask is broadcast across batch and heads dimensions automatically.
    """
    # Fill masked positions with -inf so softmax sends their weight to ~0
    scores = scores.masked_fill(mask, float('-inf'))
    return scores


if __name__ == "__main__":
    seq_len = 5
    mask = create_causal_mask(seq_len)
    print("Causal mask (True = masked):")
    print(mask.int())  # Print as 0/1 for readability

    print("\nMask shape:", mask.shape)

    # Demo: applying mask to random scores
    batch, n_heads = 2, 4
    scores = torch.randn(batch, n_heads, seq_len, seq_len)

    masked_scores = apply_causal_mask(scores, mask)
    print("\nOriginal scores (first head, first batch):")
    print(scores[0, 0])
    print("\nMasked scores (first head, first batch):")
    print(masked_scores[0, 0])
    print("\nNotice: diagonal and upper triangle are -inf.")
