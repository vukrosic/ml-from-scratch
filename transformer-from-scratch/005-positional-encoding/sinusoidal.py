"""
Sinusoidal Positional Encoding — from the original Attention Is All You Need paper.

Implements:
- compute_freqs(): power-of-10000 terms for each dimension
- sinusoidal_pe(): analytical PE matrix (pure Python)
- sinusoidal_pe_vectorized(): vectorised version using torch

The formula from the paper:
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
"""

import math
import torch


# ---------------------------------------------------------------------------
# Pure-Python reference implementation
# ---------------------------------------------------------------------------

def compute_freqs(d_model):
    """
    Return a list of frequency terms, one per dimension.
    Even i  ->  10000^(-i/d_model)
    Odd i   ->  10000^(-(i-1)/d_model)
    """
    freqs = []
    for i in range(d_model):
        if i % 2 == 0:
            freqs.append(1.0 / (10000 ** (i / d_model)))
        else:
            freqs.append(1.0 / (10000 ** ((i - 1) / d_model)))
    return freqs


def sinusoidal_pe(seq_len, d_model):
    """
    Build a (seq_len, d_model) sinusoidal positional encoding matrix.
    Even columns: sin. Odd columns: cos.
    """
    freqs = compute_freqs(d_model)
    pe = []
    for pos in range(seq_len):
        row = [
            math.sin(pos * freq) if i % 2 == 0 else math.cos(pos * freq)
            for i, freq in enumerate(freqs)
        ]
        pe.append(row)
    return pe


# ---------------------------------------------------------------------------
# Vectorised torch implementation
# ---------------------------------------------------------------------------

def sinusoidal_pe_vectorized(seq_len, d_model):
    """
    Vectorised sinusoidal PE. Returns a (seq_len, d_model) torch.Tensor.
    Even dims: sin. Odd dims: cos.
    """
    positions = torch.arange(seq_len).unsqueeze(1)           # (seq, 1)
    freqs_idx = torch.arange(0, d_model, 2).unsqueeze(0)     # (1, d_model//2)
    freqs = torch.exp(-math.log(10000.0) * freqs_idx / d_model)  # (1, d_model//2)
    angles = positions * freqs                                # (seq, d_model//2)

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seq_len, d_model = 8, 16

    # Pure Python
    pe_py = sinusoidal_pe(seq_len, d_model)
    print(f"Python PE shape: {len(pe_py)} x {len(pe_py[0])}")

    # Vectorised torch
    pe_t = sinusoidal_pe_vectorized(seq_len, d_model)
    print(f"Torch PE shape: {pe_t.shape}")

    # Verify they match
    assert torch.allclose(pe_t.float(), torch.tensor(pe_py, dtype=torch.float32), atol=1e-5)
    print("Vectorised matches analytical version.")

    # Inspect the matrix
    print("\nPE matrix (first 4 positions):")
    print(torch.round(pe_t[:4], decimals=3))

    # Show that position 0 and position 7 are distinct
    print(f"\n||PE[0] - PE[7]|| = {(pe_t[0] - pe_t[7]).abs().sum():.4f}")
    print("Positions are unique vectors — nearby positions are somewhat similar.")
