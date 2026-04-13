# Attention Mechanisms — Advanced

This document covers the IO complexity of attention, Flash Attention internals (with Triton pseudocode), Flash Attention 2 and 3 improvements, causal masking without materialization, ring attention for multi-GPU contexts, and real benchmarks.

---

## IO Complexity of Attention

The performance bottleneck of attention on modern GPUs is not compute but **memory bandwidth**. Understanding why requires modeling the memory hierarchy.

### GPU Memory Hierarchy

```
Level          Capacity     Bandwidth          Latency
-------------------------------------------------------------
Registers      ~256 KB      N/A (instant)      ~0 cycles
SRAM (shared)  ~20 MB       ~19 TB/s (A100)    ~30 cycles
HBM (DRAM)     40-80 GB     ~2 TB/s (A100)     ~400 cycles
```

The ratio matters: SRAM is ~10x faster than HBM but ~1000x smaller. Any algorithm that keeps data in SRAM wins.

### Standard Attention: HBM Read/Write Analysis

Standard attention computes Q @ K.T -> softmax -> @ V in three separate kernels:

```
Step 1: S = Q @ K.T
  Read:  Q (N x d) + K (N x d) from HBM  = 2Nd
  Write: S (N x N) to HBM                 = N^2
  HBM access: 2Nd + N^2

Step 2: P = softmax(S)
  Read:  S (N x N) from HBM               = N^2
  Write: P (N x N) to HBM                 = N^2
  HBM access: 2N^2

Step 3: O = P @ V
  Read:  P (N x N) + V (N x d) from HBM   = N^2 + Nd
  Write: O (N x d) to HBM                 = Nd
  HBM access: N^2 + 2Nd

Total HBM access: 4N^2 + 4Nd = O(N^2)
Memory stored: N x N attention matrix = O(N^2)
```

For N=4096, d=128: the attention matrix alone is 4096^2 * 4 bytes = 64 MB per head. With 32 heads, that is 2 GB just for one layer's forward pass.

### Flash Attention: HBM Read/Write Analysis

Flash Attention tiles the computation so the N x N matrix never materializes in HBM:

```
Tiling: split Q into blocks of size B_r, K/V into blocks of size B_c
  B_r, B_c chosen so that Q_block, K_block, V_block, S_block all fit in SRAM

For each Q block (T_r = ceil(N / B_r) blocks):
  For each KV block (T_c = ceil(N / B_c) blocks):
    Read:  Q_block (B_r x d) + K_block (B_c x d) + V_block (B_c x d)
    Compute: S_block = Q_block @ K_block.T  (B_r x B_c, stays in SRAM)
             P_block = softmax(S_block)      (in SRAM, online algorithm)
             O_block += P_block @ V_block    (accumulate in SRAM)
    Write: nothing to HBM per inner iteration

  Write: O_block (B_r x d) to HBM after all KV blocks processed

Total HBM access:
  Read Q:  T_c * Nd    (each Q block read T_c times)
  Read K:  T_r * Nd    (each KV block read T_r times)
  Read V:  T_r * Nd
  Write O: Nd

  = Nd * (T_c + 2*T_r + 1)
  = O(N^2 * d / M)    where M = SRAM size

  Since d << N and M >> d^2:
  HBM access ~ O(N^2 * d^2 / M) which is much less than O(N^2)

Memory stored: O(N) — only the output and logsumexp for backward pass
```

### The IO Complexity Proof (Theorem from Dao et al.)

**Theorem**: Any exact attention algorithm that computes the N x N attention matrix must make at least Omega(N^2 * d^2 / M) HBM accesses, where M is SRAM size.

**Proof sketch**:
1. The output O = softmax(QK^T)V depends on all N^2 entries of QK^T
2. SRAM can hold M elements, so each "pass" processes at most M entries
3. Computing a B_r x B_c block of QK^T requires B_r * d elements of Q and B_c * d elements of K
4. To fit both in SRAM: B_r * d + B_c * d <= M, so B_r * B_c <= M^2 / (4d^2)
5. We need N^2 / (B_r * B_c) >= N^2 * 4d^2 / M^2 passes
6. Each pass reads at least B_r * d + B_c * d >= 2d * sqrt(B_r * B_c) elements
7. Total reads >= N^2 * d^2 / M (by AM-GM and substitution)

Flash Attention achieves this lower bound to within constant factors. It is **IO-optimal**.

---

## Triton Kernel Pseudocode for Flash Attention

This is a simplified but structurally accurate Triton kernel for the forward pass:

```python
@triton.jit
def flash_attention_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    stride_qh, stride_qm, stride_qd,   # strides (similar for K, V, O)
    stride_kh, stride_kn, stride_kd,
    stride_vh, stride_vn, stride_vd,
    stride_oh, stride_om, stride_od,
    N_CTX: tl.constexpr, D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    sm_scale,
):
    off_bh = tl.program_id(0)   # batch * head index
    off_m = tl.program_id(1)    # Q block index
    m_range = off_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_range = tl.arange(0, D_HEAD)

    # Load Q block: stays in SRAM for all KV iterations
    q = tl.load(Q_ptr + off_bh*stride_qh + m_range[:,None]*stride_qm
                + d_range[None,:]*stride_qd, mask=m_range[:,None] < N_CTX)

    # Online softmax accumulators
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        n_range = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K_ptr + off_bh*stride_kh + n_range[:,None]*stride_kn
                     + d_range[None,:]*stride_kd, mask=n_range[:,None] < N_CTX)
        s = tl.dot(q, tl.trans(k)) * sm_scale      # (BLOCK_M, BLOCK_N) in SRAM

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(V_ptr + off_bh*stride_vh + n_range[:,None]*stride_vn
                     + d_range[None,:]*stride_vd, mask=n_range[:,None] < N_CTX)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]   # final normalization
    tl.store(O_ptr + off_bh*stride_oh + m_range[:,None]*stride_om
             + d_range[None,:]*stride_od, acc.to(tl.float16), mask=m_range[:,None] < N_CTX)
    tl.store(LSE_ptr + off_bh*N_CTX + m_range, m_i + tl.log(l_i), mask=m_range < N_CTX)
```

### Key Points About the Kernel

1. **Q stays in registers/SRAM** for the entire inner loop. Only K and V are streamed from HBM.
2. **Online softmax** (Milakov & Gimelshein, 2018) computes softmax without a second pass. The running max `m_i` and sum `l_i` are updated incrementally.
3. **The NxN attention matrix never exists in HBM**. The BLOCK_M x BLOCK_N tile `s` lives only in SRAM.
4. **Logsumexp is the only extra storage** -- O(N) per head, used for the backward pass to recompute softmax without storing the full attention matrix.

---

## Flash Attention 2 Improvements

Flash Attention 2 (Dao, 2023) achieves approximately 2x speedup over Flash Attention 1 through three changes.

### 1. Reduced Non-Matmul FLOPs

In Flash Attention 1, the online softmax rescaling requires element-wise operations (exp, multiply, divide) at every inner loop iteration. These non-matmul FLOPs are bottlenecked by SRAM bandwidth, not tensor core throughput.

Flash Attention 2 restructures the computation to defer normalization:

```
Flash Attention 1 (per inner iteration):
  p = exp(s - m_new)              # BLOCK_M x BLOCK_N element-wise ops
  acc = acc * alpha[:, None]      # BLOCK_M x D_HEAD element-wise ops
  acc += p @ V                    # matmul

Flash Attention 2:
  p = exp(s - m_new)              # same
  acc += p @ V                    # matmul (NO rescaling of acc here)
  # Rescaling deferred to end of inner loop or when max changes
  # Only rescale when m_new != m_old (which happens rarely)
```

This reduces the ratio of non-matmul to matmul FLOPs from ~3:1 to ~1:3, keeping tensor cores fed.

### 2. Sequence-Level Parallelism (Swapping Loop Order)

Flash Attention 1 parallelizes over batch and heads. Flash Attention 2 adds parallelism over the sequence dimension:

```
Flash Attention 1: parallelize over (batch, head)
  Grid: (batch * n_heads,)
  Each thread block processes one full row of Q blocks
  Problem: with batch=1, head=32, only 32 thread blocks -> 40% A100 occupancy

Flash Attention 2: parallelize over (batch, head, q_block)
  Grid: (batch * n_heads, ceil(N / BLOCK_M))
  Each thread block processes one Q block against all KV blocks
  With N=4096, BLOCK_M=128: 32 * 32 = 1024 thread blocks -> full occupancy
```

This is critical for inference with batch=1 and for long-context training where N is large but batch may be small.

### 3. Optimized Warp Partitioning

Flash Attention 1 splits the K/V dimension across warps within a thread block (each warp computes a slice of the dot product, then they reduce). Flash Attention 2 splits the Q dimension instead:

```
FA1: 4 warps split K/V columns -> need cross-warp reduction after each step
FA2: 4 warps split Q rows -> no cross-warp communication needed

Each warp independently processes its Q rows against all K/V blocks.
Eliminates shared memory synchronization between warps.
```

### Flash Attention 2 Performance

```
Hardware: A100 80GB
                         FA1          FA2          Speedup
Forward pass (N=2048):   143 TFLOPs   230 TFLOPs   1.6x
Forward pass (N=4096):   151 TFLOPs   245 TFLOPs   1.6x
Forward pass (N=8192):   155 TFLOPs   257 TFLOPs   1.7x
Backward pass (N=4096):   96 TFLOPs   158 TFLOPs   1.6x

FA2 achieves ~72% of A100 theoretical FP16 peak (312 TFLOPs).
```

---

## Flash Attention 3 (Hopper Architecture)

Flash Attention 3 (Dao et al., 2024) exploits H100-specific hardware features for another 1.5-2x speedup.

### Warp Specialization on Hopper

H100 GPUs can run different warps in specialized roles simultaneously using the Tensor Memory Accelerator (TMA) and asynchronous warp groups:

```
Producer warps:  Issue TMA loads (K, V blocks from HBM to shared memory)
Consumer warps:  Execute tensor core matmuls (Q @ K.T, P @ V)

These run concurrently via Hopper's asynchronous pipeline:
  Cycle 0: Producer loads K_block[0], Consumer idle
  Cycle 1: Producer loads K_block[1], Consumer computes with K_block[0]
  Cycle 2: Producer loads K_block[2], Consumer computes with K_block[1]
  ...
  Memory latency is fully hidden behind compute.
```

### FP8 Attention

H100 tensor cores support FP8 (E4M3 and E5M2), doubling throughput vs FP16:

```
Precision   H100 Tensor Core Throughput
FP16/BF16:  990 TFLOPs
FP8:        1979 TFLOPs (2x)
```

Flash Attention 3 supports FP8 with careful handling of precision:

```python
# FA3 FP8 strategy:
# Q, K stored in FP8 (E4M3) -> S = Q @ K.T computed in FP8
# But softmax(S) is computed in FP32 (precision-critical)
# P stored in FP8 (E5M2, larger exponent range for post-softmax values)
# V stored in FP8 (E4M3) -> O = P @ V computed in FP8
# Output O accumulated in FP32, then cast to BF16/FP16
```

The mixed-precision strategy preserves model quality while getting 2x compute throughput. Empirically, FP8 Flash Attention 3 shows less than 0.1% perplexity degradation on LLaMA fine-tuning.

### Ping-Pong Scheduling

Flash Attention 3 uses "ping-pong" scheduling between two warp groups to maximize tensor core utilization:

```
Warp Group 0:  GEMM_0  wait  GEMM_1  wait  GEMM_2  ...
Warp Group 1:  wait    GEMM_0  wait  GEMM_1  wait   ...

One group computes while the other waits for data.
Tensor cores are never idle.
```

### Flash Attention 3 Performance

```
Hardware: H100 SXM5 80GB
                         FA2 (H100)   FA3 (H100)   Speedup
Forward FP16 (N=4096):   390 TFLOPs   580 TFLOPs   1.5x
Forward FP16 (N=16k):    420 TFLOPs   620 TFLOPs   1.5x
Forward FP8 (N=4096):    N/A          920 TFLOPs   (2.4x vs FA2 FP16)
Forward FP8 (N=16k):     N/A         1010 TFLOPs   (2.4x vs FA2 FP16)

FA3 FP16 achieves ~62% of H100 theoretical FP16 peak.
FA3 FP8 achieves ~51% of H100 theoretical FP8 peak.
```

---

## Causal Masking Without Materialization

Causal (autoregressive) masking requires that position i only attends to positions <= i. Naive masking creates the full NxN matrix and sets upper-triangle entries to -inf before softmax. Flash Attention avoids this entirely.

### Tile-Level Masking

When tiling Q into row blocks and K into column blocks, some tiles fall entirely above the diagonal (all masked), entirely below (no masking needed), or on the diagonal (partial masking):

```
Q blocks (rows) vs K blocks (cols) for N=8, BLOCK=2:

        K0    K1    K2    K3
  Q0  [ diag  skip  skip  skip ]
  Q1  [ full  diag  skip  skip ]
  Q2  [ full  full  diag  skip ]
  Q3  [ full  full  full  diag ]

full = no masking needed (all positions visible)
skip = fully masked (skip entirely -- no HBM reads, no compute)
diag = partial masking (apply mask within the tile)
```

### Implementation in the Kernel

```python
# Inside the Flash Attention inner loop:
for start_n in range(0, N_CTX, BLOCK_N):
    # Skip fully masked blocks (Q rows < K cols means fully causal-masked)
    if IS_CAUSAL:
        # If the entire K block is after the Q block, skip it
        if start_n > off_m * BLOCK_M + BLOCK_M - 1:
            continue

    # ... load K, compute S = Q @ K.T ...

    if IS_CAUSAL:
        # Check if this is a diagonal block (partial masking needed)
        if start_n + BLOCK_N > off_m * BLOCK_M:
            # Apply causal mask: mask positions where col > row
            causal_mask = (m_range[:, None] >= n_range[None, :])
            s = tl.where(causal_mask, s, float('-inf'))

    # ... continue with softmax and V accumulation ...
```

### Performance Impact

Skipping fully masked tiles makes causal attention roughly 2x faster than full attention for long sequences. For a sequence of length N, about half the NxN tiles are fully masked:

```
Full attention: T_r x T_c = (N/B)^2 tiles
Causal attention: ~(N/B)^2 / 2 tiles (lower triangle only)

With Flash Attention 2 causal:
  N=2048:  1.8x faster than non-causal
  N=4096:  1.9x faster than non-causal
  N=16k:   2.0x faster than non-causal (approaches theoretical 2x)
```

---

## Ring Attention for Multi-GPU Million-Token Contexts

Ring Attention (Liu et al., 2023) distributes attention computation across multiple GPUs to handle sequences that exceed single-GPU memory. Each GPU holds a shard of the sequence and passes KV blocks around a ring.

### The Ring Topology

```
GPU 0 has Q[0:N/P], K[0:N/P], V[0:N/P]
GPU 1 has Q[N/P:2N/P], K[N/P:2N/P], V[N/P:2N/P]
...
GPU P-1 has Q[(P-1)N/P:N], K[(P-1)N/P:N], V[(P-1)N/P:N]

Ring steps:
Step 0: Each GPU computes local attention (its Q against its KV)
Step 1: Send KV to next GPU, receive KV from previous GPU
        Each GPU computes attention with the received KV block
Step 2: Send KV to next GPU again
        ...
Step P-1: Each GPU has seen all KV blocks

Total: P-1 communication steps, overlapped with computation
```

### Overlapping Communication and Compute

The key insight: while GPU i computes attention with the current KV block, it simultaneously sends/receives the next KV block via NVLink or InfiniBand:

```python
def ring_attention(q_local, kv_blocks, rank, world_size, comm):
    """Each GPU holds its Q shard, iterates over KV shards from all GPUs."""
    o = torch.zeros_like(q_local)
    lse = torch.full((q_local.shape[0],), float('-inf'))
    kv = kv_blocks[rank]

    for step in range(world_size):
        # Overlap: async send/recv KV while computing attention
        if step < world_size - 1:
            kv_next = torch.empty_like(kv)
            send_op = comm.isend(kv, dst=(rank+1) % world_size)
            recv_op = comm.irecv(kv_next, src=(rank-1) % world_size)

        o_blk, lse_blk = flash_attention_block(q_local, kv)
        # Merge via logsumexp trick (mathematically exact)
        o, lse = merge_attention_outputs(o, lse, o_blk, lse_blk)

        if step < world_size - 1:
            send_op.wait(); recv_op.wait()
            kv = kv_next
    return o

def merge_attention_outputs(o1, lse1, o2, lse2):
    lse_max = torch.maximum(lse1, lse2)
    w1, w2 = torch.exp(lse1 - lse_max), torch.exp(lse2 - lse_max)
    return (w1[:,None]*o1 + w2[:,None]*o2) / (w1+w2)[:,None], lse_max + torch.log(w1+w2)
```

### Ring Attention with Causal Masking

For causal attention, many ring steps produce fully masked tiles. Smart scheduling skips them:

```
With 8 GPUs, step s:
  GPU i computes attention for Q shard i against KV shard (i-s) mod 8
  If shard (i-s) mod 8 > shard i, the entire block is above the diagonal -> skip

Result: GPU i only needs ceil((i+1) * P / N) steps instead of P steps
Average across GPUs: ~P/2 compute steps (but all P communication steps)
```

### Scaling Numbers

```
Ring attention scaling (A100 80GB GPUs, LLaMA-7B, seq_len per GPU = 32K):
  1 GPU:   32K context   (standard Flash Attention 2)
  8 GPUs:  256K context  (92% scaling efficiency)
  32 GPUs: 1M context    (87% scaling efficiency)
  64 GPUs: 2M context    (83% scaling efficiency)

Communication overhead per step:
  KV block size: 2 * 32K * 128 * 2 bytes = 16 MB per head
  32 heads: 512 MB per step
  NVLink 4 bandwidth: 900 GB/s -> transfer time: 0.57 ms
  Flash Attention compute for 32K x 32K: ~12 ms
  Overlap ratio: 95% (communication is 5% of compute)
```

---

## Benchmarks: Training Speedup and Memory Reduction

### Training Throughput (LLaMA-7B on A100 80GB)

```
Sequence    Standard Attention    Flash Attention 2    Speedup    Memory Saved
Length      (tokens/s)            (tokens/s)
--------    ------------------    -----------------    -------    ------------
512         12,400                14,800               1.2x       1.8x
1024        10,200                13,900               1.4x       2.5x
2048         6,800                12,600               1.9x       4.2x
4096         3,100                11,200               3.6x       8.1x
8192         OOM                  9,400                inf        16x+
16384        OOM                  6,800                inf        64x+
```

### Memory Usage Comparison (Forward Pass, per Layer)

```
Sequence    Standard Attention    Flash Attention 2    Reduction
Length      Memory                Memory
--------    ------------------    -----------------    ---------
512         8 MB                  4.5 MB               1.8x
1024        32 MB                 5 MB                 6.4x
2048        128 MB                6 MB                 21x
4096        512 MB                8 MB                 64x
8192        2048 MB               12 MB                170x
16384       8192 MB               20 MB                410x

Standard: stores full NxN attention matrix (O(N^2))
Flash: stores only output + logsumexp (O(N))
```

### Key Takeaways from Benchmarks

1. **Below 512 tokens**, Flash Attention gives modest speedups (1.2-1.5x).
2. **At 2048+ tokens**, Flash Attention becomes essential -- standard attention either OOMs or is 2-4x slower.
3. **Memory savings scale quadratically** with sequence length (O(N^2) vs O(N) storage).
4. **Training throughput** improves not just from faster kernels but from enabling larger batch sizes with the memory savings.
5. **The backward pass** benefits less (1.5-2x vs 2-4x forward) because Flash Attention recomputes attention rather than storing it.

---

## Flash Attention Availability in PyTorch

PyTorch 2.0+ uses Flash Attention automatically via `scaled_dot_product_attention`. Control backends with:

```python
torch.backends.cuda.enable_flash_sdp(True)         # Flash Attention (Dao)
torch.backends.cuda.enable_mem_efficient_sdp(True)  # xFormers memory-efficient
torch.backends.cuda.enable_math_sdp(True)           # Naive math fallback
```

Requirements: SM80+ GPU (A100, H100, RTX 3090+), head dim <= 256, FP16 or BF16 inputs. Use `torch.profiler` with `ProfilerActivity.CUDA` to verify which kernel backend is selected at runtime.

---

Get the video walkthrough: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
