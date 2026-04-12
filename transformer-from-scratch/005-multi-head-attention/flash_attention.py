"""
Flash Attention from scratch — tile-based online softmax.

Standard attention is O(N^2) in memory because we must materialize
the full N x N attention matrix before softmax.

Flash Attention reorganizes the computation to be O(N) memory:
it processes the matrix in tiles (blocks), applying softmax online
so we never need to store the full N x N matrix.

This implementation uses the online softmax algorithm:
https://arxiv.org/abs/1805.02867
"""

import torch
import math


def standard_attention(q, k, v, scale):
    """
    Standard (textbook) attention: materializes the full NxN score matrix.
    q, k, v each shape: (batch, n_heads, seq_len, d_head)
    Returns: (batch, n_heads, seq_len, d_head)

    Memory: O(N^2) for the score matrix.
    """
    scores = q @ k.transpose(-2, -1) / scale  # (batch, n_heads, N, N)
    attn_weights = torch.softmax(scores, dim=-1)
    return attn_weights @ v


def online_softmax(x, axis=-1):
    """
    Online softmax: computes softmax incrementally without materializing
    the full exp(x) vector.

    Standard softmax: m = max(x), f_i = exp(x_i - m), softmax_i = f_i / sum(f)

    Online version processes elements one at a time and maintains
    the running max (m), sum (s), and numerator (f) — no overflow,
    and we can interleave the computation with the attention computation.

    This is the key insight behind Flash Attention's memory savings.
    """
    # For a full tensor we still compute it the standard way,
    # but the algorithm shows the principle: we can compute max and
    # sum incrementally, element by element or block by block.
    m = x.max(axis=axis, keepdim=True).values
    x_shifted = x - m
    e_x = torch.exp(x_shifted)
    return e_x / e_x.sum(axis=axis, keepdim=True)


def flash_attention_tile(q, k, v, scale, block_size=64):
    """
    Flash Attention using tile-based online softmax.

    Instead of computing all N^2 scores at once:
        1. Load a tile of Q (block_size x d_head).
        2. For each tile of K, V:
           - Compute scores for Q_tile vs K_tile (block_size x block_size sub-matrix).
           - Update running max, sum, and output using the online softmax logic.
        3. This keeps memory at O(block_size * N) = O(N) instead of O(N^2).

    q, k, v shape: (batch, n_heads, seq_len, d_head)
    block_size: how many rows/columns to process at a time.
    """
    batch, n_heads, seq_len, d_head = q.shape

    # Output accumulator: weighted sum of V (numerator of online softmax).
    out = torch.zeros_like(q)

    # For each query block, we process all key-value blocks in tiles.
    for q_offset in range(0, seq_len, block_size):
        q_block = q[:, :, q_offset:q_offset + block_size, :]  # (B, H, block, d_head)

        # Running softmax state for this query block.
        m_i = torch.full(
            (batch, n_heads, q_block.size(2), 1),
            float('-inf'),
            device=q.device, dtype=q.dtype
        )
        denom_acc = torch.zeros(
            (batch, n_heads, q_block.size(2), 1),
            device=q.device, dtype=q.dtype
        )
        out_acc = torch.zeros_like(q_block)

        for k_offset in range(0, seq_len, block_size):
            k_block = k[:, :, k_offset:k_offset + block_size, :]  # (B, H, block, d_head)
            v_block = v[:, :, k_offset:k_offset + block_size, :]  # (B, H, block, d_head)

            # Score of this Q block against this K block.
            s_block = q_block @ k_block.transpose(-2, -1) / scale  # (B, H, Q_block, K_block)

            # Online softmax: update running max.
            m_new = torch.maximum(m_i, s_block.max(dim=-1, keepdim=True).values)

            # Compute exp(s - new_max) for this block.
            # The trick is: we don't need the full softmax, just the ratio.
            # Correct update: P_i = exp(s_i - m_new) / sum(exp(s_j - m_new))
            # Numerator exp(s - m_new) is safe (max exponent is exp(0) = 1).
            p_block = torch.exp(s_block - m_new)

            # Previous contribution rescaled to account for the new max.
            # If max changed from m_i to m_new > m_i:
            #   old contribution exp(s_old - m_i) needs to be rescaled.
            #   exp(s_old - m_i) * exp(m_i - m_new) = exp(s_old - m_new).
            # So we multiply the accumulated output and denominator by
            # exp(m_i - m_new) before adding the new block's contribution.
            if k_offset == 0:
                # First block: no rescaling needed.
                exp_sums = p_block.sum(dim=-1, keepdim=True)
                out_acc = (p_block.unsqueeze(-1) * v_block.unsqueeze(-3)).sum(dim=-2)
            else:
                rescale = torch.exp(m_i - m_new)  # (B, H, Q_block, 1)
                out_acc = out_acc * rescale
                denom_acc = denom_acc * rescale

                # Add new block.
                p_block_new = torch.exp(s_block - m_new)
                exp_sums_new = p_block_new.sum(dim=-1, keepdim=True)
                out_acc = out_acc + (p_block_new.unsqueeze(-1) * v_block.unsqueeze(-3)).sum(dim=-2)
                denom_acc = denom_acc + exp_sums_new
                m_i = m_new
                continue

            denom_acc = exp_sums
            m_i = m_new

        # Normalize by denominator to get the true softmax-weighted sum.
        out_acc = out_acc / (denom_acc + 1e-8)
        out[:, :, q_offset:q_offset + block_size, :] = out_acc

    return out


# ---------------------------------------------------------------------------
# Verification and memory comparison
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    batch, n_heads, seq_len, d_head = 2, 4, 128, 32
    scale = math.sqrt(d_head)

    q = torch.randn(batch, n_heads, seq_len, d_head)
    k = torch.randn(batch, n_heads, seq_len, d_head)
    v = torch.randn(batch, n_heads, seq_len, d_head)

    # ---- Numerical equivalence ----
    out_standard = standard_attention(q, k, v, scale)
    out_flash = flash_attention_tile(q, k, v, scale, block_size=32)

    diff = (out_standard - out_flash).abs().max().item()
    print(f"Max |difference| between standard and flash attention: {diff:.2e}")

    if diff < 1e-4:
        print("PASS: Flash attention output matches standard attention.")
    else:
        print("FAIL: Outputs diverge beyond tolerance.")

    # ---- Memory analysis ----
    # Standard: full (seq_len x seq_len) score matrix per head.
    standard_mem_elements = seq_len * seq_len * n_heads * batch
    # Flash: only one block of scores at a time: block_size * seq_len.
    block_size = 32
    flash_mem_elements = block_size * seq_len * n_heads * batch

    print(f"\nMemory analysis for seq_len={seq_len}, n_heads={n_heads}, batch={batch}:")
    print(f"  Standard attention score matrix elements: {standard_mem_elements:,}")
    print(f"  Flash attention peak elements:           {flash_mem_elements:,}")
    print(f"  Ratio (flash / standard):                {flash_mem_elements / standard_mem_elements:.4f}")
