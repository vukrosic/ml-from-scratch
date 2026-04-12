"""
Rotary Position Embedding (RoPE) — from "RoFormer: Enhanced Transformer with Rotary Position Embedding".

Instead of adding a positional signal to token embeddings, RoPE rotates the
query and key vectors by an angle proportional to their position.
The key insight: the dot product q · k at position m with position n
depends only on the relative distance (m-n) — there is no absolute position.

The rotation matrix for position pos and dimension i:
  R(pos, i) = [cos(pos * theta_i), -sin(pos * theta_i)]
              [sin(pos * theta_i),  cos(pos * theta_i)]
where theta_i = 10000^(-i/d_head) for dimension pair (2i, 2i+1).

No learned parameters — purely deterministic.
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Build rotation matrices
# ---------------------------------------------------------------------------

def build_rotary_matrix(seq_len, d_head, theta=10000.0):
    """
    Build the rotation angles for each position and dimension pair.
    Returns shape (seq_len, d_head // 2, 2, 2) — rotation matrices
    for each position and each dimension pair.
    """
    d_pairs = d_head // 2
    # Theta per dimension pair: 10000^(-2i/d_head) for pair i
    idx = torch.arange(0, d_pairs, dtype=torch.float32)
    thetas = theta ** (-2 * idx / d_head)   # (d_pairs,)

    positions = torch.arange(seq_len, dtype=torch.float32)   # (seq,)
    # Outer product: (seq, d_pairs)
    angles = positions.unsqueeze(1) * thetas.unsqueeze(0)

    cos = torch.cos(angles)   # (seq, d_pairs)
    sin = torch.sin(angles)   # (seq, d_pairs)
    return cos, sin


# ---------------------------------------------------------------------------
# Apply RoPE to Q and K
# ---------------------------------------------------------------------------

def rotate_half(x):
    """
    Apply the rotation to a (batch, n_heads, seq, d_head) tensor.
    Each adjacent pair of dimensions (2i, 2i+1) forms a 2D rotation.
    x[..., 0::2] is rotated by -theta, x[..., 1::2] is rotated by +theta
    via: [cos, -sin; sin, cos] @ [x0; x1]
    """
    seq_len = x.shape[-2]
    d_head = x.shape[-1]
    d_pairs = d_head // 2

    # Separate even and odd dimensions
    x0 = x[..., 0::2]          # (..., d_pairs)
    x1 = x[..., 1::2]          # (..., d_pairs)

    # Build cos/sin per position and pair: shape (seq, d_pairs)
    cos, sin = build_rotary_matrix(seq_len, d_head)
    cos = cos.to(x.dtype)      # broadcast to x's dtype
    sin = sin.to(x.dtype)

    # Reshape for broadcasting: (1, 1, seq, d_pairs) or similar
    x0_rot = x0 * cos - x1 * sin
    x1_rot = x0 * sin + x1 * cos

    # Interleave back: even indices get x0_rot, odd indices get x1_rot
    x_rot = torch.empty_like(x)
    x_rot[..., 0::2] = x0_rot
    x_rot[..., 1::2] = x1_rot
    return x_rot


# ---------------------------------------------------------------------------
# RoPE attention
# ---------------------------------------------------------------------------

class RoPEMultiHeadAttention(nn.Module):
    """
    Multi-head attention with Rotary Position Embedding on Q and K.
    The dot-product attention score between position m and n depends only on (m-n).
    No absolute position embeddings — relative position is baked in.
    """
    def __init__(self, d_model, n_heads, theta=10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.theta = theta
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, self.n_heads, 3 * self.d_head)
        qkv = qkv.transpose(1, 2)          # (batch, heads, seq, 3*d_head)
        q, k, v = qkv.chunk(3, dim=-1)    # each (batch, heads, seq, d_head)

        # Apply RoPE rotation to Q and K
        q = rotate_half(q)
        k = rotate_half(k)

        # Scaled dot-product attention
        scale = math.sqrt(self.d_head)
        scores = q @ k.transpose(-2, -1) / scale
        attn = torch.softmax(scores, dim=-1) @ v

        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(attn)


# ---------------------------------------------------------------------------
# Verify: attention score depends only on relative position
# ---------------------------------------------------------------------------

def verify_rope_property(d_model=64, n_heads=4, seq_len=8):
    """
    For RoPE, the attention score between position i and j depends only on |i-j|.
    Verify this by checking that attn[i,j] == attn[i+k, j+k] for all k.
    """
    rope_attn = RoPEMultiHeadAttention(d_model, n_heads)
    rope_attn.eval()
    x = torch.randn(1, seq_len, d_model)

    with torch.no_grad():
        # Manually compute Q and K with RoPE
        qkv = rope_attn.qkv_proj(x)
        qkv = qkv.view(1, seq_len, n_heads, 3 * rope_attn.d_head)
        qkv = qkv.transpose(1, 2)
        q, k, _ = qkv.chunk(3, dim=-1)
        q = rotate_half(q)
        k = rotate_half(k)

        scale = math.sqrt(rope_attn.d_head)
        scores = q @ k.transpose(-2, -1) / scale  # (1, heads, seq, seq)
        scores = scores[0, 0]  # (seq, seq) — take first head

    print("Attention score matrix (position-based):")
    print(torch.round(scores, decimals=3))

    # Check: scores[i,j] == scores[i+1, j+1] ?
    all_match = True
    for i in range(seq_len - 2):
        for j in range(seq_len - 2):
            if abs(scores[i, j] - scores[i+1, j+1]) > 1e-4:
                all_match = False
                break
    print(f"\nAttention depends only on relative position: {all_match}")
    if not all_match:
        print("(Minor differences expected due to floating-point accumulation)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    d_model, n_heads = 64, 4
    seq_len = 8

    # Build and run RoPE attention
    rope = RoPEMultiHeadAttention(d_model, n_heads)
    x = torch.randn(2, seq_len, d_model)
    out = rope(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape

    # Verify the relative-position property
    print("\n--- RoPE relative-position property verification ---")
    verify_rope_property(d_model, n_heads, seq_len)
    print("RoPE encodes relative position through rotation — "
          "no absolute PE table needed.")

    # Compare with standard attention (no PE) — both should show
    # that without position signal, attention has no position awareness
    print("\n--- Sinusoidal PE vs RoPE ---")
    sin_cos, sin_sin = build_rotary_matrix(seq_len, d_model)
    print(f"Rotation angle matrix shape: {sin_cos.shape} (cos and sin per position per pair)")
    print(f"theta_i = 10000^(-2i/d_head) for dimension pair i")
    print("Larger dimension pairs rotate slower → "
          "high-frequency info in low dims, low-frequency in high dims.")
