# Attention From Scratch

Every transformer model runs on attention. It is the operation that lets each token look at every other token and decide what matters. It is also the operation that blows up your GPU memory when sequences get long. This tutorial builds attention from the ground up — from the naive O(n²) implementation to Flash Attention's tiling trick that makes long sequences practical — so you understand exactly where the memory goes and how to stop wasting it.

---

## Piece 1: The Memory Wall Problem

### Why Attention Is Expensive

Every position in a sequence needs to attend to every other position. That means computing and storing an n×n matrix, where n is the sequence length.

Here is the concrete math for float32 (4 bytes per element):

```
Attention matrix memory = n × n × 4 bytes

Sequence length 512:     512²   =    262,144 elements  →     1 MB
Sequence length 2048:   2048²   =  4,194,304 elements  →    16 MB
Sequence length 4096:   4096²   = 16,777,216 elements  →    64 MB
Sequence length 8192:   8192²   = 67,108,864 elements  →   256 MB
Sequence length 16384: 16384²   = 268,435,456 elements → 1,024 MB = 1 GB
```

And that is per head, per batch element. A model with 32 heads and batch size 8 at sequence length 8192:

```
256 MB × 32 heads × 8 batch = 65,536 MB = 64 GB
```

That exceeds the memory of every consumer GPU. The attention matrix alone — before counting model weights, activations, gradients, or optimizer states — would need 64 GB. This is the memory wall.

The compute scales the same way: a 4096×4096 matmul is 16× more expensive than 1024×1024. The O(n²) cost in both memory and compute is the central challenge of attention.

### The Formula: Scaled Dot-Product Attention

Scaled dot-product attention computes three things from the input: queries Q, keys K, and values V:

```
attention(Q, K, V) = softmax(Q @ K.T / sqrt(d)) @ V
```

Where d is the dimension of each head. Q and K are both (batch, seq, d), so Q @ K.T is (batch, seq, seq) — the attention matrix. The division by sqrt(d) prevents the softmax from saturating when d is large.

### Why Divide by sqrt(d)?

When d is large, the dot products Q @ K.T grow large in magnitude. If Q and K entries are independent with mean 0 and variance 1, each dot product is a sum of d terms, giving variance d. The softmax exponential saturates — all attention weights become nearly uniform and the model cannot distinguish positions. Dividing by sqrt(d) brings the variance back to 1.

```
d = 64:   dot products have std ≈ 8    → softmax is peaky but controlled
d = 512:  dot products have std ≈ 22.6 → softmax saturates without scaling
d = 1024: dot products have std ≈ 32   → softmax is essentially argmax
```

---

## Piece 2: The Attention Operation Step by Step

### Step 1: Compute Q, K, V from the input

From an input tensor of shape (batch, seq, d_model), three linear projections produce Q, K, and V:

```
Q = x @ W_q   # (batch, seq, d_model) → (batch, seq, d_head)
K = x @ W_k
V = x @ W_v
```

### Step 2: Compute the attention matrix

```
scores = Q @ K.T                     # (batch, seq, seq)
scores = scores / sqrt(d_head)       # scale
weights = softmax(scores, dim=-1)    # normalize along key dimension
```

### Step 3: Apply attention to values

```
output = weights @ V                 # (batch, seq, seq) @ (batch, seq, d_head)
                                     #                   → (batch, seq, d_head)
```

The attention matrix weights each position's value by how relevant it is to the query position.

```
Data flow for seq=4, d=3:

  Q (4×3)          K.T (3×4)         scores (4×4)
┌─────────┐     ┌───────────┐     ┌──────────────┐
│ q1 q1 q1│     │ k1 k2 k3 k4│     │ s11 s12 s13 s14│
│ q2 q2 q2│  @  │ k1 k2 k3 k4│  =  │ s21 s22 s23 s24│
│ q3 q3 q3│     │ k1 k2 k3 k4│     │ s31 s32 s33 s34│
│ q4 q4 q4│     └───────────┘     │ s41 s42 s43 s44│
└─────────┘                        └──────────────┘
                                         │
                                    softmax(dim=-1)
                                         │
                                         ▼
  weights (4×4)       V (4×3)        output (4×3)
┌──────────────┐   ┌─────────┐    ┌─────────────┐
│ w11 w12 w13 w14│   │ v1 v1 v1│    │ o1 = Σ wij·vj│
│ w21 w22 w23 w24│ @ │ v2 v2 v2│ =  │ o2 = Σ wij·vj│
│ w31 w32 w33 w34│   │ v3 v3 v3│    │ o3 = Σ wij·vj│
│ w41 w42 w43 w44│   │ v4 v4 v4│    │ o4 = Σ wij·vj│
└──────────────┘   └─────────┘    └─────────────┘
```

---

## Piece 3: Naive Attention vs. PyTorch Built-In

PyTorch provides `torch.nn.functional.scaled_dot_product_attention` (SDPA), which fuses the entire operation and handles masking efficiently. Here is a direct comparison.

### The naive implementation

```python
# attention.py — Piece 3a: naive attention

import torch
import torch.nn.functional as F

def naive_attention(Q, K, V, scale=None):
    """Scaled dot-product attention, computed naively.

    Q, K, V: (batch, seq, d_head)
    Returns: (batch, seq, d_head)
    """
    d = Q.shape[-1]
    if scale is None:
        scale = d ** -0.5

    # Step 1: compute attention scores — materializes full (seq, seq) matrix
    scores = Q @ K.transpose(-2, -1)   # (batch, seq, seq)
    scores = scores * scale

    # Step 2: softmax over key dimension
    weights = F.softmax(scores, dim=-1)

    # Step 3: weighted sum of values
    return weights @ V
```

### The built-in version

```python
# attention.py — Piece 3b: PyTorch fused SDPA

def fast_attention(Q, K, V):
    """Uses PyTorch's fused SDPA kernel.

    Internally selects Flash Attention, Memory-Efficient Attention,
    or a math fallback depending on hardware and tensor properties.

    Q, K, V: (batch, seq, d_head)
    Returns: (batch, seq, d_head)
    """
    return F.scaled_dot_product_attention(Q, K, V)
```

### Benchmark: naive vs. SDPA

```python
# attention.py — Piece 3c: benchmark

import time

def benchmark(fn, *args, steps=200, warmup=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps

batch, seq, d_head = 4, 1024, 64
Q = torch.randn(batch, seq, d_head, device='cuda')
K = torch.randn(batch, seq, d_head, device='cuda')
V = torch.randn(batch, seq, d_head, device='cuda')

naive_time = benchmark(naive_attention, Q, K, V)
fast_time = benchmark(fast_attention, Q, K, V)

print(f"Naive:   {naive_time*1000:.2f} ms")
print(f"SDPA:    {fast_time*1000:.2f} ms")
print(f"Speedup: {naive_time/fast_time:.2f}x")
```

SDPA is typically 2-4x faster and uses far less memory because it avoids materializing the full (batch, seq, seq) attention matrix.

---

## Piece 4: Masking — Causal and Arbitrary

Autoregressive models must not attend to future positions. A causal mask ensures each position only attends to itself and earlier positions.

### Causal mask

The FULL 5×5 causal attention mask (1=attend, 0=masked):

```
                  Key dimension (column)
                  col0  col1  col2  col3  col4
                ┌────────────────────────────┐
         row0   │  1   │  0   │  0   │  0   │  0   │  ← query pos 0: only self
         row1   │  1   │  1   │  0   │  0   │  0   │  ← query pos 1: 0, 1
Query     row2   │  1   │  1   │  1   │  0   │  0   │  ← query pos 2: 0, 1, 2
dimension row3   │  1   │  1   │  1   │  1   │  0   │  ← query pos 3: 0, 1, 2, 3
(row)     row4   │  1   │  1   │  1   │  1   │  1   │  ← query pos 4: all positions
                └────────────────────────────┘
                  ▲
                  └── diagonal: position i can always attend to itself

Element-wise legend:
  1 (unmasked): attention score is computed normally, then softmax
  0 (masked):   attention score is set to -inf before softmax, contributes nothing
```

The mask is applied before softmax: `scores = scores.masked_fill(mask == 0, float('-inf'))`.

**Why the upper triangle is zero:** For query at position i, key at position j is masked when j > i (future position). The diagonal j == i is always unmasked (a token attends to itself).

### Naive masking

```python
# attention.py — Piece 4a: causal mask

def causal_mask(scores, seq_len):
    """Create a causal mask and apply it to attention scores.

    scores: (batch, seq, seq)
    Returns: (batch, seq, seq) with future positions set to -inf
    """
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=scores.device),
        diagonal=1
    ).bool()
    scores = scores.masked_fill(mask, float('-inf'))
    return scores
```

### Arbitrary mask

```python
# attention.py — Piece 4b: arbitrary mask

def apply_mask(scores, mask):
    """Apply an arbitrary mask to attention scores.

    scores: (batch, seq, seq)
    mask: (batch, seq, seq) of booleans — True means mask out
    """
    return scores.masked_fill(mask, float('-inf'))
```

Common masks include padding masks (ignore padding tokens) and document masks (prevent cross-document attention in packed long-context training).

### Masking with SDPA

```python
# attention.py — Piece 4c: SDPA with mask

def masked_attention(Q, K, V, mask=None):
    """SDPA with optional masking.

    For causal masking, pass is_causal=True instead of a mask tensor.
    This enables Flash Attention's optimized causal path.
    """
    return F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)

def causal_attention(Q, K, V):
    """SDPA with causal masking — the fast path."""
    return F.scaled_dot_product_attention(Q, K, V, is_causal=True)
```

---

## Piece 5: Flash Attention — Why and How It Works

### The Memory Hierarchy Problem

The naive implementation materializes the full n×n attention matrix in GPU HBM (High Bandwidth Memory). But GPUs have two levels of memory:

```
┌─────────────────────────────────────────────┐
│                    GPU                       │
│                                              │
│   ┌───────────────────────────────┐          │
│   │          SRAM (on-chip)       │          │
│   │   Size: ~20 MB (A100)        │          │
│   │   Bandwidth: ~19 TB/s        │          │
│   └───────────┬───────────────────┘          │
│               │ ▲                             │
│          load │ │ store                       │
│               ▼ │                             │
│   ┌───────────────────────────────┐          │
│   │          HBM (off-chip)       │          │
│   │   Size: 40-80 GB (A100)      │          │
│   │   Bandwidth: ~2 TB/s         │          │
│   └───────────────────────────────┘          │
│                                              │
└─────────────────────────────────────────────┘

SRAM is ~10× faster but ~1000× smaller than HBM.
```

Standard attention writes the full n×n matrix to HBM, then reads it back for the softmax, then writes the softmax result, then reads it back for the V multiplication. Each of these round-trips through slow HBM is the bottleneck — not the arithmetic itself. Attention is **memory-bound**, not compute-bound.

### The Key Insight: Online Softmax

The reason standard attention materializes the full matrix is softmax. Normal softmax requires two passes over the data:

```
Standard softmax over values [x1, x2, ..., xn]:

  Pass 1: Find the max        m = max(x1, x2, ..., xn)
  Pass 2: Compute exp and sum  sum = Σ exp(xi - m)
  Pass 3: Normalize            softmax(xi) = exp(xi - m) / sum
```

You need to see ALL values before you can compute any output. This seems to force materializing the full row.

**Online softmax** eliminates this constraint. It processes values one at a time, maintaining a running max and a running sum that get corrected as new values arrive.

### Online Softmax Walkthrough

Let us trace through 4 values: [2.0, 4.0, 1.0, 3.0]

**State tracking table** — at each step, m = running max, l = running sum of exp(xi - m):

```
Step    Input xi    m (running max)    l (running sum)    softmax so far (xi * l_inv)
-----------------------------------------------------------------------------------
init    —           -inf               0                  —
x1=2.0  2.0         2.0                1.0                softmax(2.0)=1.0/1.0=1.0000  (only x1 known)
x2=4.0  4.0         4.0                1.1353             softmax(2.0)=0.1353, softmax(4.0)=1.0000
x3=1.0  1.0         4.0 (unchanged)    1.1851             softmax(2.0)=0.1353, softmax(4.0)=1.0000, softmax(1.0)=0.0498
x4=3.0  3.0         4.0 (unchanged)    1.5530             softmax(2.0)=0.1353, softmax(4.0)=1.0000, softmax(1.0)=0.0498, softmax(3.0)=0.3679
```

**Detailed step-by-step update formulas:**

```
Initialize: m = -inf, l = 0

Step 1 — Process x1 = 2.0:
  m_new = max(-inf, 2.0) = 2.0
  l     = 0 × exp(-inf - 2.0) + exp(2.0 - 2.0)
        = 0 + exp(0) = 1.0
  m     = 2.0

Step 2 — Process x2 = 4.0:
  m_new = max(2.0, 4.0) = 4.0
  l     = 1.0 × exp(2.0 - 4.0) + exp(4.0 - 4.0)
        = 1.0 × 0.1353 + 1.0 = 1.1353
  m     = 4.0

Step 3 — Process x3 = 1.0:
  m_new = max(4.0, 1.0) = 4.0   ← m unchanged
  l     = 1.1353 × exp(4.0 - 4.0) + exp(1.0 - 4.0)
        = 1.1353 × 1.0 + 0.0498 = 1.1851
  m     = 4.0

Step 4 — Process x4 = 3.0:
  m_new = max(4.0, 3.0) = 4.0   ← m unchanged
  l     = 1.1851 × exp(4.0 - 4.0) + exp(3.0 - 4.0)
        = 1.1851 + 0.3679 = 1.5530
  m     = 4.0

Final normalization (divide each exp by l):
  softmax(2.0) = exp(2.0 - 4.0) / 1.5530 = 0.1353 / 1.5530 = 0.0871
  softmax(4.0) = exp(4.0 - 4.0) / 1.5530 = 1.0000 / 1.5530 = 0.6439
  softmax(1.0) = exp(1.0 - 4.0) / 1.5530 = 0.0498 / 1.5530 = 0.0321
  softmax(3.0) = exp(3.0 - 4.0) / 1.5530 = 0.3679 / 1.5530 = 0.2369

Verify sum: 0.0871 + 0.6439 + 0.0321 + 0.2369 = 1.0000 ✓
```

The magic: at no point did we need all 4 values in memory simultaneously. We processed them one at a time. Flash Attention applies this same trick to blocks of the attention matrix.

> **Try it yourself:** Run `python flash_attention.py` in this directory to verify this computation matches naive attention.

### Tiled Computation

Flash Attention partitions Q into row blocks (size Br) and K, V into column blocks (size Bc). It processes one block of Q against all blocks of K/V, accumulating the output using the online softmax trick.

```
Full attention matrix (seq=8, block_size=2):

         K block 0  K block 1  K block 2  K block 3
         ┌────────┬────────┬────────┬────────┐
Q blk 0  │ (0,0)  │ (0,1)  │ (0,2)  │ (0,3)  │  ← process left to right
         ├────────┼────────┼────────┼────────┤
Q blk 1  │ (1,0)  │ (1,1)  │ (1,2)  │ (1,3)  │  ← then this row
         ├────────┼────────┼────────┼────────┤
Q blk 2  │ (2,0)  │ (2,1)  │ (2,2)  │ (2,3)  │  ← then this row
         ├────────┼────────┼────────┼────────┤
Q blk 3  │ (3,0)  │ (3,1)  │ (3,2)  │ (3,3)  │  ← then this row
         └────────┴────────┴────────┴────────┘

Each tile (i,j) is a Br × Bc sub-matrix — small enough to fit in SRAM.
We NEVER store the full 8×8 matrix. Only one tile exists at a time.
```

The iteration for one row of Q blocks:

```
Processing Q block 0 against all K/V blocks:

Step 1: Load Q[0:2], K[0:2], V[0:2] into SRAM
        Compute S = Q_blk @ K_blk.T  (2×2 tile)
        Update running max, sum, and output accumulator

Step 2: Load K[2:4], V[2:4] into SRAM  (Q[0:2] stays)
        Compute S = Q_blk @ K_blk.T  (2×2 tile)
        Correct previous accumulator using new max
        Update running max, sum, and output accumulator

Step 3: Load K[4:6], V[4:6] → compute → correct → update
Step 4: Load K[6:8], V[6:8] → compute → correct → update

Done: O[0:2] now contains the correct output for Q block 0.
      We never stored an 8×8 matrix. Peak memory was one 2×2 tile.
```

The correction step is critical. When the running max changes (because a new K block has larger dot products), all previously accumulated exp() values are too large by a factor of exp(m_old - m_new). The algorithm multiplies the accumulator by this correction factor before adding the new block's contribution.

### Flash Attention Tiling: 4x4 Matrix Walkthrough with block_size=2

Let us walk through Flash Attention tile by tile on a 4x4 attention matrix with block_size=2 (each block is 2 rows or 2 columns). Q, K, V each have seq=4, d=2 for this example.

```
Matrix layout (each cell is a scalar score s_ij):

         K dimension
         col0  col1  col2  col3
       ┌─────────────────────┐
  row0 │ s00  s01  s02  s03  │
  row1 │ s10  s11  s12  s13  │
Q      │ s20  s21  s22  s23  │
       │ s30  s31  s32  s33  │
       └─────────────────────┘

Tiling with block_size=2 splits Q into Q blocks [0:2] and [2:4],
and K/V into K blocks [0:2] and [2:4].
```

**Block grid:**

```
          K block 0    K block 1
          (cols 0-1)  (cols 2-3)
         ┌──────────┬──────────┐
Q block 0 │  Tile    │  Tile    │
(rows 0-1)│  (0,0)  │  (0,1)  │
         ├──────────┼──────────┤
Q block 1 │  Tile    │  Tile    │
(rows 2-3)│  (1,0)  │  (1,1)  │
         └──────────┴──────────┘
```

**Step-by-step execution for Q block 0 (rows 0 and 1):**

```
Step 1 — Load and compute Tile (0,0):
  Load Q[0:2]    → Q block 0 (2×d) into SRAM
  Load K[0:2]    → K block 0 (2×d) into SRAM
  Load V[0:2]    → V block 0 (2×d) into SRAM
  Compute S_00 = Q_blk0 @ K_blk0.T   → 2×2 tile in SRAM
  Online softmax update: update running m_0, m_1, l_0, l_1
  Accumulate: O[0:2] += softmax(S_00) @ V_blk0

Step 2 — Load and compute Tile (0,1):
  Q block 0 STAYS in SRAM (never reloaded)
  Load K[2:4]   → K block 1 (2×d) into SRAM
  Load V[2:4]   → V block 1 (2×d) into SRAM
  Compute S_01 = Q_blk0 @ K_blk1.T   → 2×2 tile in SRAM
  Online softmax correction: rescale previous accumulator using new max
  Online softmax update: update m_0, m_1, l_0, l_1 with new block
  Accumulate: O[0:2] += softmax_correction_and_new_block @ V_blk1

Q block 0 is done — O[0:2] now holds the complete correct output.
Write O[0:2] to HBM. Peak SRAM usage: one 2×2 tile + Q block + K block + V block.
```

**Step-by-step execution for Q block 1 (rows 2 and 3):** (identical pattern)

```
Step 3 — Load and compute Tile (1,0):
  Load Q[2:4]    → Q block 1 (2×d) into SRAM
  Load K[0:2]    → K block 0 (2×d) into SRAM
  Load V[0:2]    → V block 0 (2×d) into SRAM
  Compute S_10 = Q_blk1 @ K_blk0.T
  Online softmax update
  Accumulate: O[2:4] += softmax(S_10) @ V_blk0

Step 4 — Load and compute Tile (1,1):
  Q block 1 STAYS in SRAM
  Load K[2:4]   → K block 1
  Compute S_11 = Q_blk1 @ K_blk1.T
  Correct + update + accumulate
  Write O[2:4] to HBM
```

**What never happened:** The full 4×4 attention matrix was never written to HBM. At peak, only one 2×2 tile existed in SRAM. Standard attention would have written 4×4=16 values to HBM; Flash Attention wrote only 4×d=8 values per output row (2 rows × d=2, or 2×4=8 scalars total for this toy example).

```python
# flash_attention.py — Piece 5: Flash Attention simulation

import torch
import torch.nn.functional as F

def flash_attention_simulated(Q, K, V, block_size=64):
    """Simulate Flash Attention's tiling with online softmax.

    This is a Python-level simulation — real Flash Attention runs
    as a fused CUDA kernel with all tiles in SRAM. But the math
    is identical.

    Q, K, V: (batch, seq, d_head)
    Returns: (batch, seq, d_head)
    """
    batch, seq, d = Q.shape
    scale = d ** -0.5

    # Output accumulator and softmax statistics (per row)
    O = torch.zeros_like(Q)                                    # (batch, seq, d)
    l = torch.zeros(batch, seq, 1, device=Q.device)            # running sum
    m = torch.full((batch, seq, 1), float('-inf'), device=Q.device)  # running max

    # Outer loop: iterate over K/V blocks
    for j in range(0, seq, block_size):
        K_block = K[:, j:j+block_size, :]  # (batch, Bc, d)
        V_block = V[:, j:j+block_size, :]  # (batch, Bc, d)

        # Inner loop: iterate over Q blocks
        for i in range(0, seq, block_size):
            Q_block = Q[:, i:i+block_size, :]  # (batch, Br, d)

            # Compute attention scores for this tile
            S_block = Q_block @ K_block.transpose(-2, -1) * scale  # (batch, Br, Bc)

            # Online softmax: update running max
            m_block = S_block.max(dim=-1, keepdim=True).values     # (batch, Br, 1)
            m_old = m[:, i:i+block_size, :]
            m_new = torch.maximum(m_old, m_block)

            # Correction factor for previously accumulated values
            correction = torch.exp(m_old - m_new)

            # Compute exp(S - m_new) for this block
            P_block = torch.exp(S_block - m_new)                   # (batch, Br, Bc)

            # Update running sum: correct old sum + add new
            l_old = l[:, i:i+block_size, :]
            l_new = correction * l_old + P_block.sum(dim=-1, keepdim=True)

            # Update output: correct old output + add new contribution
            O[:, i:i+block_size, :] = (
                correction * O[:, i:i+block_size, :]
                + P_block @ V_block
            )

            # Store updated statistics
            m[:, i:i+block_size, :] = m_new
            l[:, i:i+block_size, :] = l_new

    # Final normalization
    O = O / l
    return O
```

Verify it matches the naive version:

```python
batch, seq, d = 2, 128, 32
Q = torch.randn(batch, seq, d)
K = torch.randn(batch, seq, d)
V = torch.randn(batch, seq, d)

out_naive = naive_attention(Q, K, V)
out_flash = flash_attention_simulated(Q, K, V, block_size=32)

print(f"Max difference: {(out_naive - out_flash).abs().max().item():.2e}")
# Should be ~1e-6 (float32 precision)
```

---

## Piece 6: Why Flash Attention Saves Memory

### Memory comparison

The critical difference is what lives in HBM at any given time:

```
Standard Attention:
  HBM contains: Q (n×d) + K (n×d) + V (n×d) + S (n×n) + P (n×n) + O (n×d)
  Peak memory:   O(n²)  ← dominated by the n×n matrices

Flash Attention:
  HBM contains: Q (n×d) + K (n×d) + V (n×d) + O (n×d) + stats (2n)
  SRAM contains: one tile of S (Br×Bc) + one tile of P (Br×Bc)
  Peak memory:   O(n)   ← no n×n matrix ever exists in HBM
```

Concrete numbers for seq=8192, d=64, float32:

```
Standard attention:
  Attention matrix S:  8192 × 8192 × 4 bytes = 256 MB
  Softmax result P:    8192 × 8192 × 4 bytes = 256 MB
  Total extra:         512 MB

Flash Attention:
  Tile in SRAM (Br=Bc=128):  128 × 128 × 4 bytes = 64 KB
  Running stats (m, l):       8192 × 2 × 4 bytes  = 64 KB
  Total extra:                128 KB

  Memory saved: 512 MB → 128 KB (4000× reduction)
```

### IO Complexity Analysis

Standard attention does these HBM reads and writes:

```
Standard Attention IO:
  1. Read Q, K from HBM             → O(n × d) reads
  2. Write S = Q @ K.T to HBM       → O(n²) writes
  3. Read S from HBM for softmax     → O(n²) reads
  4. Write P = softmax(S) to HBM     → O(n²) writes
  5. Read P, V from HBM              → O(n²) + O(n × d) reads
  6. Write O = P @ V to HBM          → O(n × d) writes
  Total HBM IO: O(n² + n×d)  ≈  O(n²) since n >> d typically
```

Flash Attention's IO:

```
Flash Attention IO:
  Outer loop: n/Bc iterations over K/V blocks
  Inner loop: n/Br iterations over Q blocks
  Each iteration: load Q_block (Br×d), K_block (Bc×d), V_block (Bc×d)
                  write O_block (Br×d)

  Total HBM IO: O(n² × d / M)
  where M = SRAM size (in elements)

  For typical values (n=8192, d=64, M=100KB ≈ 25K elements):
    Standard:  O(n²)       = O(67M)
    Flash:     O(n²×d/M)   = O(67M × 64 / 25K) = O(172K)
```

The IO reduction is roughly M/d — for an A100 with ~20MB SRAM and d=64, that is about a 75× reduction in HBM traffic. This is where the speedup comes from.

---

## Piece 7: Multi-Head Attention

Instead of one attention head, multi-head attention runs h heads in parallel, each with its own Q, K, V projections. The outputs are concatenated and projected back to d_model.

```
Multi-head attention data flow (d_model=512, num_heads=8, d_head=64):

Input x: (batch, seq, 512)
         │
    ┌────┴────┬────────┐
    ▼         ▼        ▼
   W_q       W_k      W_v       ← three (512, 512) linear projections
    │         │        │
    ▼         ▼        ▼
   Q          K        V         ← each (batch, seq, 512)
    │         │        │
  reshape   reshape  reshape     ← split into 8 heads
    │         │        │
    ▼         ▼        ▼
   Q          K        V         ← each (batch, 8, seq, 64)
    │         │        │
    └────┬────┴────────┘
         │
    attention (per head)          ← 8 independent attention ops
         │
         ▼
   concat heads                   ← (batch, seq, 512)
         │
        W_o                       ← (512, 512) output projection
         │
         ▼
   output: (batch, seq, 512)
```

### Multi-head attention module

```python
# multihead.py — Piece 7: Multi-Head Attention

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """Multi-head attention from scratch.

    Args:
        d_model: model dimension
        num_heads: number of attention heads (must divide d_model evenly)
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None, is_causal=False):
        """Forward pass.

        x: (batch, seq, d_model)
        mask: optional attention mask
        is_causal: if True, applies causal masking (preferred over mask for autoregressive)
        Returns: (batch, seq, d_model)
        """
        batch, seq, _ = x.shape

        # Project to Q, K, V
        Q = self.W_q(x)  # (batch, seq, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape: (batch, seq, d_model) → (batch, num_heads, seq, d_head)
        Q = Q.view(batch, seq, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch, seq, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch, seq, self.num_heads, self.d_head).transpose(1, 2)

        # SDPA — dispatches to Flash Attention when available
        attn = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=mask, is_causal=is_causal
        )

        # Reshape back: (batch, num_heads, seq, d_head) → (batch, seq, d_model)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, self.d_model)

        return self.W_o(attn)
```

### Why split into multiple heads?

A single attention head can only capture one type of relationship at a time. With multiple heads, the model attends to different aspects of the sequence in parallel — one head might focus on syntactic relationships, another on semantic proximity, another on positional patterns. The concatenation and final projection let the model mix information across heads.

---

## Piece 8: KV Caching for Autoregressive Decoding

During autoregressive decoding, each new token depends on all previous tokens. Without caching, generating token t requires recomputing K and V for all t-1 previous tokens. Generating n tokens costs O(n²) total compute.

```
Without KV cache — generating 4 tokens:

Step 1: compute K,V for [tok1]                    → 1 token of K,V work
Step 2: compute K,V for [tok1, tok2]              → 2 tokens of K,V work
Step 3: compute K,V for [tok1, tok2, tok3]        → 3 tokens of K,V work
Step 4: compute K,V for [tok1, tok2, tok3, tok4]  → 4 tokens of K,V work
Total: 1 + 2 + 3 + 4 = 10 tokens of K,V work  (O(n²))

With KV cache — generating 4 tokens:

Step 1: compute K,V for [tok1], cache it          → 1 token of K,V work
Step 2: compute K,V for [tok2], append to cache   → 1 token of K,V work
Step 3: compute K,V for [tok3], append to cache   → 1 token of K,V work
Step 4: compute K,V for [tok4], append to cache   → 1 token of K,V work
Total: 1 + 1 + 1 + 1 = 4 tokens of K,V work  (O(n))
```

### KV cache implementation

Here is the KV cache update step by step with concrete tensor shapes during autoregressive generation. Assume batch=1, d_head=64, and max_len=1024.

```python
# kv_cache.py — Piece 8a: KV Cache

class KVCache:
    """KV cache for autoregressive decoding.

    Stores key and value tensors for all computed positions.
    On each step, appends new K and V, and optionally trims to max_len.
    """
    def __init__(self, max_len=2048):
        self.max_len = max_len
        self.keys = None    # (batch, cached_seq, d_head) or None
        self.values = None

    def update(self, k, v):
        """Append new key/value tensors.

        k, v: (batch, new_seq, d_head) — typically new_seq=1 during generation
        Returns: (batch, total_seq, d_head) for both keys and values
        """
        if self.keys is None:
            self.keys = k
            self.values = v
        else:
            self.keys = torch.cat([self.keys, k], dim=1)
            self.values = torch.cat([self.values, v], dim=1)

        # Trim if over max_len
        if self.keys.shape[1] > self.max_len:
            self.keys = self.keys[:, -self.max_len:]
            self.values = self.values[:, -self.max_len:]

        return self.keys, self.values

    def reset(self):
        self.keys = None
        self.values = None
```

**Step-by-step KV cache update with actual sequence lengths:**

```
Initial state (empty cache, first decoding step):
  cache.keys = None
  cache.values = None

Step 1 — First token decoded, k/v computed for position 0:
  k.shape = (batch=1, new_seq=1, d_head=64)   ← only ONE new position
  v.shape = (batch=1, new_seq=1, d_head=64)

  cache.keys is None → cache.keys = k
  cache.values is None → cache.values = v

  AFTER update:
    cache.keys.shape  = (1, 1, 64)   ← seq_len = 1
    cache.values.shape = (1, 1, 64)

Step 2 — Second token decoded, k/v computed for position 1:
  k.shape = (batch=1, new_seq=1, d_head=64)   ← still only the NEW token
  v.shape = (batch=1, new_seq=1, d_head=64)

  cache.keys exists → torch.cat([cache.keys, k], dim=1)
  BEFORE concat:
    cache.keys.shape = (1, 1, 64)   ← cached_seq = 1
    k.shape          = (1, 1, 64)   ← new_seq = 1
  AFTER concat (dim=1):
    combined.shape   = (1, 2, 64)  ← cached_seq + new_seq = 2

  AFTER update:
    cache.keys.shape  = (1, 2, 64)   ← seq_len = 2
    cache.values.shape = (1, 2, 64)

Step 3 — Third token decoded:
  k.shape = (batch=1, new_seq=1, d_head=64)
  concat → (1, 3, 64)

  AFTER update:
    cache.keys.shape  = (1, 3, 64)   ← seq_len = 3
    cache.values.shape = (1, 3, 64)

Step N — Sequence reaches max_len (e.g., max_len=1024, now at step 1025):
  k.shape = (batch=1, new_seq=1, d_head=64)
  concat → (1, 1025, 64)

  TRIM triggered because 1025 > max_len:
    cache.keys = cache.keys[:, -1024:]   ← keep last 1024 positions
    cache.values = cache.values[:, -1024:]

  AFTER trim:
    cache.keys.shape  = (1, 1024, 64)   ← back to max_len, oldest token evicted
    cache.values.shape = (1, 1024, 64)

The Q tensor at every step is shape (batch, 1, d_head) — only the current position.
The KV cache grows to seq_len = number of decoded tokens (up to max_len).
This is why decoding cost is O(seq_len) per token, not O(seq_len²).
```

### Benchmark: with and without KV cache

```python
# kv_cache.py — Piece 8b: benchmark

def benchmark_generation(model, prompt_len=128, gen_len=256, use_cache=True):
    """Compare generation speed with and without KV cache."""
    d_model = model.d_model
    cache = KVCache() if use_cache else None

    # Prompt encoding (same for both)
    x = torch.randn(1, prompt_len, d_model, device='cuda')

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(gen_len):
        if use_cache and step > 0:
            # Only process the new token
            x_step = x[:, -1:, :]
        else:
            # First step or no cache: process full sequence
            x_step = x

        Q = model.W_q(x_step)
        K = model.W_k(x_step)
        V = model.W_v(x_step)

        if use_cache:
            K_full, V_full = cache.update(K, V)
        else:
            K_full, V_full = K, V

        Q = Q.view(1, -1, model.num_heads, model.d_head).transpose(1, 2)
        K_full = K_full.view(1, -1, model.num_heads, model.d_head).transpose(1, 2)
        V_full = V_full.view(1, -1, model.num_heads, model.d_head).transpose(1, 2)

        out = F.scaled_dot_product_attention(Q, K_full, V_full)
        # ... rest of generation (simplified)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"{'With' if use_cache else 'Without'} cache: {elapsed:.3f}s "
          f"({elapsed/gen_len*1000:.1f} ms/token)")
```

---

## Piece 9: Flash Attention 2 Improvements

Flash Attention 2 (Dao, 2023) builds on the original with two key changes that deliver roughly 2x additional speedup.

### Change 1: Better parallelism

Flash Attention 1 parallelizes over batch and heads — each thread block handles one (batch, head) pair. This underutilizes the GPU when batch × heads is small (e.g., batch=1 during inference).

Flash Attention 2 swaps the loop order: the outer loop iterates over Q blocks (rows) and the inner loop over K/V blocks (columns). This allows parallelizing across the sequence dimension too.

```
Flash Attention 1 parallelism:
  Thread blocks: batch × num_heads
  batch=1, heads=32 → 32 thread blocks (underutilizes A100's 108 SMs)

Flash Attention 2 parallelism:
  Thread blocks: batch × num_heads × (seq / Br)
  batch=1, heads=32, seq=4096, Br=128 → 32 × 32 = 1024 thread blocks
```

### Change 2: Reduced non-matmul FLOPs

The online softmax correction involves element-wise operations (exp, multiply, add) that are much slower than matrix multiplies on tensor cores. Flash Attention 2 restructures the algorithm to minimize these operations — it delays the rescaling of the output accumulator and does fewer redundant exp() calls. On A100, this reduces non-matmul FLOPs by about 25%.

### Using Flash Attention 2 in PyTorch

```python
# PyTorch's SDPA automatically uses Flash Attention 2 when available.
# Flash Attention is used automatically via SDPA — no explicit backend selection needed.
# To verify which kernel is running, use torch.profiler (see ADVANCED.md).

# Requirements: PyTorch 2.0+, SM80+ GPU (A100, H100, RTX 3090+),
# FP16 or BF16 inputs, head dimension <= 256.
```

---

## Piece 10: When to Use Flash Attention

Flash Attention is not always faster. The overhead of tiling and online softmax bookkeeping has a cost. Here is a rough guide:

```
Sequence length vs. benefit:

  seq < 128:    Flash Attention may be SLOWER than naive
                (tiling overhead > memory savings)

  128 ≤ seq < 512:   Roughly break-even. Flash saves memory
                      but speed gain is modest.

  512 ≤ seq < 2048:  Flash is clearly faster (2-3×) and uses
                      much less memory.

  seq ≥ 2048:  Flash is essential. Without it, you either
               OOM or crawl. 4-8× speedup typical.

  seq ≥ 8192:  Without Flash Attention, training is not
               practical on any single GPU.
```

Rules of thumb:

- **Training with long sequences:** Always use Flash Attention. The memory savings alone make it mandatory — you literally cannot fit the attention matrix otherwise.
- **Inference with KV cache:** Flash Attention helps less because the attention matrix during decoding is (1, seq) not (seq, seq). But it still helps for the prefill pass.
- **Short sequences in tight loops:** Profile first. For seq < 128, the SDPA math backend may be faster.
- **Mixed precision:** Flash Attention requires float16 or bfloat16 inputs. If you need float32 attention (rare), you cannot use it.

---

## Piece 11: Common Mistakes

### Mistake 1: Forgetting causal mask with Flash Attention

```python
# WRONG: no causal masking — model sees future tokens during training
out = F.scaled_dot_product_attention(Q, K, V)

# RIGHT: use is_causal=True for autoregressive models
out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

# ALSO WRONG: passing a manual causal mask tensor disables Flash Attention
# in some PyTorch versions. Use is_causal=True instead.
mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
out = F.scaled_dot_product_attention(Q, K, V, attn_mask=~mask)  # slower!
```

### Mistake 2: Wrong head dimension ordering

```python
# WRONG: (batch, seq, num_heads, d_head) — SDPA expects heads before seq
Q = Q.view(batch, seq, num_heads, d_head)
out = F.scaled_dot_product_attention(Q, K, V)  # wrong shapes!

# RIGHT: (batch, num_heads, seq, d_head)
Q = Q.view(batch, seq, num_heads, d_head).transpose(1, 2)
K = K.view(batch, seq, num_heads, d_head).transpose(1, 2)
V = V.view(batch, seq, num_heads, d_head).transpose(1, 2)
out = F.scaled_dot_product_attention(Q, K, V)
```

### Mistake 3: Using float32 with Flash Attention

```python
# This silently falls back to the slower math kernel:
Q = torch.randn(batch, heads, seq, d, dtype=torch.float32, device='cuda')
out = F.scaled_dot_product_attention(Q, K, V)  # no Flash Attention!

# Use float16 or bfloat16:
Q = torch.randn(batch, heads, seq, d, dtype=torch.bfloat16, device='cuda')
out = F.scaled_dot_product_attention(Q, K, V)  # Flash Attention enabled
```

### Mistake 4: Not checking which backend SDPA is using

```python
# The SDPA backend selection API changed in PyTorch 2.0.
# In modern PyTorch, SDPA automatically selects the best backend (Flash Attention
# if available, else math fallback). You can inspect the selected backend with
# torch.profiler (see ADVANCED.md for profiling instructions).
```

### Mistake 5: KV cache with wrong sequence dimension

```python
# WRONG: concatenating along the wrong dimension
cached_K = torch.cat([cached_K, new_K], dim=2)  # dim=2 is seq for (B,H,S,D)
# This is only correct if your layout is (batch, heads, seq, d_head).
# If your layout is (batch, seq, d_head), you need dim=1.

# Always double-check your tensor layout before concatenating.
```

---

## Common Bugs — Error Messages and Exact Fixes

These are real error messages you will encounter when working with attention, along with their causes and precise fixes.

---

### Bug 1: `RuntimeError: UBLE error: batch size 1 must be less than or equal to 8`

**Cause:** Flash Attention has a minimum batch size requirement on certain hardware configurations. This happens when using `is_causal=True` with batch=1 and very short sequences in some PyTorch versions.

**Fix — use the math fallback explicitly for small batches:**

```python
from torch.backends.cuda import enable_flash_sdp, enable_math_sdp

# Disable Flash Attention and use math fallback
enable_flash_sdp(False)
enable_math_sdp(True)

# Now SDPA uses the math kernel — correct results, no batch size restriction
out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

# Re-enable for production (restore defaults)
enable_flash_sdp(True)
enable_math_sdp(False)
```

Alternatively, ensure you are on PyTorch 2.3+ which fixed this issue for most configurations.

---

### Bug 2: `torch.nn.functional.scaled_dot_product_attention: Expected float16 or bfloat16, got float32`

**Cause:** Flash Attention requires FP16 or BF16 inputs. Using float32 silently falls back to the slow math kernel, or raises this error in strict mode.

**Fix — cast inputs to the correct dtype:**

```python
# WRONG: float32 — falls back to math kernel silently
Q = torch.randn(batch, heads, seq, d, dtype=torch.float32, device='cuda')
K = torch.randn(batch, heads, seq, d, dtype=torch.float32, device='cuda')
V = torch.randn(batch, heads, seq, d, dtype=torch.float32, device='cuda')
out = F.scaled_dot_product_attention(Q, K, V)  # slow or error

# RIGHT: bfloat16 — Flash Attention enabled
Q = torch.randn(batch, heads, seq, d, dtype=torch.bfloat16, device='cuda')
K = torch.randn(batch, heads, seq, d, dtype=torch.bfloat16, device='cuda')
V = torch.randn(batch, heads, seq, d, dtype=torch.bfloat16, device='cuda')
out = F.scaled_dot_product_attention(Q, K, V)  # Flash Attention enabled
```

---

### Bug 3: `RuntimeError: The shape of mask [seq, seq] at dimension 0 did not match the shape of attn_scores`

**Cause:** The mask tensor has the wrong shape or the wrong dimension ordering. SDPA expects `(batch, num_heads, seq, seq)` when passing a mask tensor, but you may be passing `(seq, seq)` or `(batch, seq, seq)`.

**Fix — reshape the mask to match SDPA's expected layout:**

```python
# WRONG: 2D mask — wrong shape
seq = 512
mask = torch.triu(torch.ones(seq, seq, device='cuda'), diagonal=1).bool()
out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
# Error: mask shape [512, 512] does not match attn_scores shape [8, 16, 512, 512]

# RIGHT: reshape mask to (batch, num_heads, seq, seq)
batch, num_heads, seq, _ = Q.shape  # e.g., (8, 16, 512, 512)
mask = torch.triu(torch.ones(seq, seq, device='cuda'), diagonal=1).bool()
mask = mask.unsqueeze(0).unsqueeze(0)           # (1, 1, seq, seq)
mask = mask.expand(batch, num_heads, -1, -1)    # (batch, num_heads, seq, seq)
out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
```

Better yet, use `is_causal=True` which avoids the mask tensor entirely and enables Flash Attention's optimized causal path.

---

### Bug 4: `RuntimeError: KVt must be contiguous in memory. Call .contiguous() before passing to the kernel`

**Cause:** The KV cache tensor has a non-contiguous memory layout (e.g., from a transpose or slice operation). PyTorch SDPA kernels require contiguous tensors.

**Fix — call `.contiguous()` before using the cached KV:**

```python
# In KVCache.update(), after trim:
if self.keys.shape[1] > self.max_len:
    self.keys = self.keys[:, -self.max_len:].contiguous()
    self.values = self.values[:, -self.max_len:].contiguous()

# In the attention forward pass, before SDPA:
K_full = K_full.contiguous()   # ensure KV cache is contiguous
V_full = V_full.contiguous()
out = F.scaled_dot_product_attention(Q, K_full, V_full, is_causal=True)
```

---

### Bug 5: `torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate xxx GB` on a 4096-length sequence

**Cause:** Standard attention materializes the full (seq, seq) attention matrix. With float16 and seq=4096: 4096×4096×2 bytes = 32 MB per head, times batch and heads adds up fast. This is the exact OOM scenario Flash Attention is designed to prevent.

**Fix — use Flash Attention via SDPA:**

```python
import torch.nn.functional as F

Q = torch.randn(batch, heads, seq, d, dtype=torch.bfloat16, device='cuda')
K = torch.randn(batch, heads, seq, d, dtype=torch.bfloat16, device='cuda')
V = torch.randn(batch, heads, seq, d, dtype=torch.bfloat16, device='cuda')

# SDPA automatically dispatches to Flash Attention when available
# (requires SM80+ GPU: A100, H100, RTX 3090+)
out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

# Memory: Flash Attention stores only O(seq) instead of O(seq²)
# 4096 seq: standard = 32 MB/head, Flash = ~0.5 MB/head (64× savings)
```

If the OOM persists with Flash Attention, you are likely on an older PyTorch version or the inputs do not meet Flash Attention requirements (head dim > 256, non-FP16 dtype, etc.). Profile with `torch.profiler` to confirm which kernel backend is selected (see ADVANCED.md).

---

## Recap

| Concept | What It Does | Key Takeaway |
|---------|-------------|--------------|
| Scaled dot-product attention | softmax(QK^T/sqrt(d)) @ V | The core operation — O(n²) in memory and compute |
| Masking | Set future positions to -inf before softmax | Use `is_causal=True` for autoregressive models |
| Flash Attention | Tile the computation, use online softmax | Reduces memory from O(n²) to O(n), faster via less HBM IO |
| Flash Attention 2 | Better parallelism + fewer non-matmul ops | ~2× faster than Flash Attention 1 |
| Multi-head attention | Run h independent attention heads | Different heads capture different relationship types |
| KV caching | Cache K,V for previous tokens | Reduces generation from O(n²) to O(n) total compute |
| SDPA | PyTorch's fused attention API | Automatically picks the best backend for your hardware |

---

## Which File Demonstrates What

This directory contains runnable Python files that demonstrate the concepts covered:

| File | Demonstrates | Related to Piece |
|------|-------------|-----------------|
| `attention.py` | Naive attention vs. SDPA benchmark, masking | Pieces 1–4 |
| `flash_attention.py` | Online softmax walkthrough, simulated tiling | Piece 5 |
| `multihead.py` | Full multi-head attention module | Piece 7 |
| `kv_cache.py` | KV cache implementation and generation benchmark | Piece 8 |

Run `python flash_attention.py` in this directory to verify the online softmax computation matches naive attention.

---

## Going Further

For real benchmark numbers across sequence lengths, memory profiling for the attention matrix, KV cache speedup measurements, and profiling tools for attention — see [ADVANCED.md](./ADVANCED.md).

Sources:
- https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- https://arxiv.org/abs/2205.14135 — Flash Attention (Dao et al., 2022)
- https://arxiv.org/abs/2307.08691 — Flash Attention 2 (Dao, 2023)
- https://pytorch.org/blog/optimizing-cuda-algorithms-with-flash-attention-2

---

Get the video walkthrough of flash attention internals, KV-cache memory profiling, and attention pattern analysis: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
