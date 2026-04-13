# Distributed Training From Scratch

> 🔴 YouTube Lesson: Coming soon | 🟡 Skool Advanced Video Lesson: [Join the advanced lesson](https://www.skool.com/become-ai-researcher-2669/about)

Multi-GPU training speeds up your job proportionally to the number of GPUs — until communication overhead eats your gains. Understanding how PyTorch actually moves data between processes is what lets you push beyond toy benchmarks and scale to real workloads. This is how `torch.distributed` actually works.

---

## What Problem Distributed Training Solves

### The Problem: One GPU Is Not Enough

A large model and a large dataset mean training takes weeks on a single GPU. You need multiple GPUs working together. But every GPU needs to see the full model and the full dataset — and they need to agree on gradients after each step.

Naive approaches fail fast. If GPU 0 holds the model and computes gradients, then sends them to GPU 1, GPU 1 sits idle during the entire compute phase. Parallelism without synchronization is just sequential execution with extra steps.

### The Fix: Run the Same Model on Every GPU, Split the Data

Each GPU holds a full copy of the model. Each GPU processes a different slice of the batch. After the backward pass, all GPUs communicate their gradients — they all compute the same averaged gradient, then all take the same optimizer step. Every GPU stays busy computing while the others are communicating.

```
GPU 0: forward(batch_slice_0) → backward → all_reduce gradients
GPU 1: forward(batch_slice_1) → backward → all_reduce gradients
...
```

This is DistributedDataParallel. Every GPU computes, then they sync.

---

## How torch.distributed Init Works

Before any communication can happen, every process needs to find its peers. PyTorch supports three init methods:

### env:// — The Standard Launcher Method

`torchrun` (the recommended launcher) sets environment variables that tell each process how to connect. With `env://`, PyTorch reads those variables automatically:

```python
dist.init_process_group(backend="nccl", init_method="env://")
```

`torchrun` sets `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, and `RANK` for every process. You do not configure anything manually.

### tcp:// — Manual Address Specification

Specify an IP address and port that all processes can reach. Useful when `torchrun` is not available:

```python
dist.init_process_group(backend="nccl", init_method="tcp://192.168.1.100:29500")
```

Every process needs the same address. One machine acts as the coordinator.

### file:// — Shared Filesystem Required

All processes communicate through a shared filesystem. Simple but slow — do not use this for multi-node training:

```python
dist.init_process_group(backend="nccl", init_method="file:///tmp/shared_store")
```

---

## The Backend: NCCL vs Gloo

**NCCL** (NVIDIA Collective Communications Library) is the fast path for NVIDIA GPUs. It uses NVLink and GPUDirect RDMA for peer-to-peer communication without touching CPU memory. Always use NCCL when you have NVIDIA GPUs.

**Gloo** is the open-source fallback. It works on CPUs and any GPU via CUDA but is significantly slower than NCCL. Only use Gloo for debugging or when NCCL is not available.

Rule: use `backend="nccl"` for GPU training, `backend="gloo"` for CPU.

---

## Collective Operations

Distributed training is built from five primitive operations. Each has specific semantics and cost.

### all_reduce — Sum Across All Ranks

Every rank has a tensor. After `all_reduce`, every rank has the sum of all tensors. Here is the exact state before and after with 2 ranks and real values:

```
BEFORE all_reduce:
  rank0 tensor:  [1.0, 2.0]
  rank1 tensor:  [3.0, 4.0]

all_reduce operation (element-wise sum across all ranks):
  element 0: 1.0 + 3.0 = 4.0
  element 1: 2.0 + 4.0 = 6.0

AFTER all_reduce (identical on every rank):
  rank0 tensor:  [4.0, 6.0]
  rank1 tensor:  [4.0, 6.0]
```

This is how gradient synchronization works. Each rank computes its gradient contribution, then all_reduce sums them so every rank has the identical averaged gradient. DDP then divides by world size to get the mean.

### all_gather — Collect Tensors From All Ranks

Every rank contributes a tensor. After `all_gather`, every rank has the concatenation of all tensors:

```
BEFORE all_gather:
  rank0 tensor:  [1.0, 2.0]
  rank1 tensor:  [3.0, 4.0]

all_gather operation (concatenate along rank dimension):
  result: [rank0 tensor, rank1 tensor] = [1.0, 2.0, 3.0, 4.0]

AFTER all_gather (identical on every rank):
  rank0 tensor:  [1.0, 2.0, 3.0, 4.0]
  rank1 tensor:  [1.0, 2.0, 3.0, 4.0]
```

This appears in pipeline parallelism when a later stage needs the full input.

### broadcast — One-to-All Distribution

One rank has a tensor. After `broadcast`, all ranks have a copy:

```
BEFORE broadcast from rank0:
  rank0 tensor:  [1.0, 2.0]
  rank1 tensor:  [?, ?]   (uninitialized or garbage)

AFTER broadcast from rank0 (rank0 is the source):
  rank0 tensor:  [1.0, 2.0]   (unchanged — it was the source)
  rank1 tensor:  [1.0, 2.0]   (now matches rank0)
```

Used to distribute model parameters or configuration from rank 0 to everyone else.

### reduce — All-to-One Sum

Every rank contributes a tensor. After `reduce`, only the root rank has the sum:

```
BEFORE reduce to rank0:
  rank0 tensor:  [1.0, 2.0]
  rank1 tensor:  [3.0, 4.0]

reduce operation (element-wise sum, delivered only to rank0):
  element 0: 1.0 + 3.0 = 4.0
  element 1: 2.0 + 4.0 = 6.0

AFTER reduce to rank0:
  rank0 tensor:  [4.0, 6.0]   (contains the aggregated sum)
  rank1 tensor:  [3.0, 4.0]   (unchanged — no data received)
```

Useful when only one rank needs the aggregated result — saves communication for non-essential ranks.

### barrier — Synchronization Point

Every rank waits until all ranks have arrived. Not a data movement operation — it only synchronizes. Useful for timing sections of code across ranks:

```
rank0: does work  →  barrier  →  does more work
rank1: does work  →  barrier  →  does more work
rank2: does work  →  barrier  →  does more work
        ↑                    ↑
  All ranks wait here   All ranks wait here
  until everyone        until everyone
  arrives                arrives
```

No tensors are exchanged — only a synchronization signal.

---

## DistributedDataParallel: How It Shards Gradients

DDP wraps a model and handles gradient synchronization automatically. Here is what actually happens during a training step:

### What DDP Actually Does

When you wrap a model with `DDP`, every rank has a full copy of the model. During the backward pass, each rank computes gradients for its portion of the batch. These gradients are local — they only reflect that rank's data, not the full batch.

Before the optimizer step, DDP runs `all_reduce` on every parameter's gradient. Every rank contributes its local gradient; every rank receives the averaged gradient. Then all ranks take the same optimizer step on identical gradients.

```
Step 1: rank0 forward(batch0), backward → grad0 (local gradient)
        rank1 forward(batch1), backward → grad1 (local gradient)

Step 2: DDP all_reduce(grad0) + all_reduce(grad1) → averaged_grad (same on every rank)

Step 3: optimizer.step() with identical averaged_grad on every rank
```

The key insight: no data is transferred during forward or backward. Only gradients are synchronized — and only once per training step, after all compute is done.

### Gradient Synchronization in Detail

Gradient averaging is the core of DDP's correctness guarantee. The algorithm:

1. Each rank computes `loss.backward()` which fills `.grad` with local gradients
2. DDP hooks detect when `.grad` is ready and trigger `all_reduce` asynchronously
3. `all_reduce` sums all local gradients and divides by world size — yielding the correct average
4. The optimizer reads the averaged gradient and updates parameters

```
rank0 grad:  [2.0, 4.0]
rank1 grad:  [4.0, 6.0]

all_reduce sum:  [6.0, 10.0]
divide by 2:     [3.0, 5.0]  ← same on every rank
```

Without DDP, each rank would apply its own local gradient and the models would diverge.

### Gradient Sync Verification: Correct vs Incorrect with 2 Ranks

Use `all_gather` to collect each rank's local gradient and compare them element-wise. Here is what correct synchronization looks like versus a bug where synchronization was skipped.

**Setup**: Suppose we have a single parameter with gradient tensor `[g0, g1]` on each rank. There is a bug in the code (e.g., DDP was not wrapped correctly, or `optimizer.zero_grad()` was called after backward but before the sync).

**Step-by-step verification code:**

```python
import torch.distributed as dist

# Assume model is a DDP-wrapped model
for name, param in model.named_parameters():
    if param.grad is None:
        continue  # skip parameters without gradients

    # Clone the local gradient — we will gather it from all ranks
    local_grad = param.grad.clone()          # shape: (param_size,)
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Prepare output buffer — one slot per rank
    all_grads = [torch.zeros_like(local_grad) for _ in range(world_size)]

    # Collect local gradients from every rank into all_grads
    # After this call, all ranks have the same list of tensors
    dist.all_gather(all_grads, local_grad)

    # Compare all ranks to rank 0's gradient
    if rank == 0:
        rank0_grad = all_grads[0].cpu().numpy()

        for r in range(1, world_size):
            rank_r_grad = all_grads[r].cpu().numpy()

            if torch.allclose(rank0_grad, rank_r_grad, rtol=1e-5, atol=1e-5):
                print(f"{name}: SYNC OK — all ranks have identical gradients")
                print(f"  rank0 grad: {rank0_grad}")
                print(f"  rank{r} grad: {rank_r_grad}")
            else:
                print(f"{name}: SYNC MISMATCH — rank 0 and rank {r} differ!")
                print(f"  rank0 grad: {rank0_grad}")
                print(f"  rank{r} grad: {rank_r_grad}")
                print(f"  diff: {rank0_grad - rank_r_grad}")
```

**Correct behavior — DDP properly wrapping the model:**

```
BEFORE backward():
  rank0 grad:  uninitialized
  rank1 grad:  uninitialized

AFTER loss.backward() (DDP hooks fire automatically):
  rank0 local grad:  [2.0, 4.0]   (computed from batch_slice_0)
  rank1 local grad:  [4.0, 6.0]   (computed from batch_slice_1)

DDP all_reduce happens here (automatic, triggered by backward hooks):
  rank0 receives: [4.0, 6.0] from rank1
  rank1 receives: [2.0, 4.0] from rank0
  Both compute: (local + received) / 2

AFTER DDP sync (param.grad is now the averaged gradient on EVERY rank):
  rank0 param.grad:  [3.0, 5.0]   ← averaged
  rank1 param.grad:  [3.0, 5.0]   ← identical
  Verification output: "fc1.weight: SYNC OK — all ranks have identical gradients"
```

**Incorrect behavior — no DDP wrapper, or zero_grad called after backward:**

```
rank0 local grad:  [2.0, 4.0]   (computed from batch_slice_0)
rank1 local grad:  [4.0, 6.0]   (computed from batch_slice_1)

NO synchronization — each rank keeps its own local gradient.

rank0 param.grad:  [2.0, 4.0]   ← NOT averaged
rank1 param.grad:  [4.0, 6.0]   ← NOT averaged
  Verification output: "fc1.weight: SYNC MISMATCH — rank 0 and rank 1 differ!"
  diff: [-2.0, -2.0]

Consequence: optimizer.step() updates rank0 and rank1 with different
parameters. After one step, the models have diverged. After N steps,
they are completely different models — DDP training is broken.
```

The mismatch is a clear signal that gradient synchronization is not happening. Common causes: wrapping with `DDP` after moving to GPU but not using `device_ids`, calling `optimizer.zero_grad()` between `backward()` and `step()`, or using `DataParallel` instead of `DDP`.

---

## torchrun: The Standard Launcher

`torchrun` replaces manual process spawning. It launches N processes on your machine (or cluster), sets environment variables correctly, and handles graceful shutdown.

Basic usage:

```bash
torchrun --nproc_per_node=2 train.py
```

This launches 2 processes on the local node. `torchrun` sets `RANK=0` and `RANK=1`, `WORLD_SIZE=2`, and a shared `MASTER_ADDR` and `MASTER_PORT`.

For multi-node training:

```bash
# On node 0:
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=192.168.1.100 train.py

# On node 1:
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=192.168.1.100 train.py
```

The `--master_addr` must be reachable from all nodes. Usually the first node's IP address.

### torchrun Flags: Exactly What Each One Sets

Each flag directly maps to an environment variable that PyTorch reads:

```
--nproc_per_node=2
  Sets: WORLD_SIZE=2
  Meaning: Total number of processes across the entire run.
           With 2 GPUs per node and 2 nodes, WORLD_SIZE=4.

--nnodes=2
  Sets: WORLD_SIZE (recomputed as nproc_per_node × nnodes)
  Meaning: Total number of compute nodes participating in the job.

--node_rank=0
  Sets: NODE_RANK=0
  Meaning: Which node this process is running on.
           Node 0 is typically the master node (hosts master_addr).

--master_addr=192.168.1.100
  Sets: MASTER_ADDR=192.168.1.100
  Meaning: IP address of the "master" node.
           All processes connect to this address to find each other.
           Must be reachable from every node (use IP, not hostname).

--master_port=29500
  Sets: MASTER_PORT=29500
  Meaning: TCP port on master_addr used for initial process discovery.
           Must be free on the master node. Use a different port if 29500 is busy.
```

Environment variables torchrun sets automatically (you do not set these yourself):

| Env Variable | Set By | Meaning |
|---|---|---|
| `RANK` | torchrun | Global rank of this process (0 to WORLD_SIZE-1) |
| `LOCAL_RANK` | torchrun | Local rank within this node (0 to nproc_per_node-1) |
| `WORLD_SIZE` | torchrun | Total number of processes |
| `LOCAL_WORLD_SIZE` | torchrun | Number of processes on this node |
| `NODE_RANK` | torchrun | Index of the current node |
| `MASTER_ADDR` | torchrun (or --master_addr) | IP of the master node |
| `MASTER_PORT` | torchrun (or --master_port) | Port on master node for rendezvous |

Your script reads these via `dist.get_rank()`, `dist.get_world_size()`, etc. — you never import or set them directly.

---

## torchrun Sets Everything Up Automatically

When you use `torchrun`, you do not configure rank or world size manually. The launcher injects the correct values via environment variables. Your script reads them:

```python
import torch.distributed as dist

# torchrun sets these environment variables automatically
dist.init_process_group(backend="nccl", init_method="env://")

rank = dist.get_rank()      # 0, 1, 2, ...  N-1
world_size = dist.get_world_size()  # total number of processes
```

Your training code stays the same regardless of how many GPUs you use.

---

## Minimal End-to-End Example

### Piece 1: The model

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        return self.fc2(x)
```

### Piece 2: Init distributed

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# torchrun sets rank, world_size, and master address automatically
dist.init_process_group(backend="nccl", init_method="env://")
rank = dist.get_rank()
world_size = dist.get_world_size()
```

### Piece 3: Wrap with DDP

```python
model = SimpleModel().cuda(rank)
model = DDP(model, device_ids=[rank])
```

### Piece 4: Training loop

Here is every line of the DDP training loop explained step by step:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for inputs, targets in dataloader:
    # Step 1: Move this rank's batch slice to its GPU.
    # Each rank has a different slice of the data — rank 0 gets batch_slice_0,
    # rank 1 gets batch_slice_1, etc.
    inputs = inputs.cuda(rank)

    # Step 2: Move the corresponding targets to the same GPU.
    # Targets must be on the same device as the inputs for cross_entropy.
    targets = targets.cuda(rank)

    # Step 3: Clear stale gradients from the previous step.
    # Without this, new gradients would accumulate on top of old ones.
    optimizer.zero_grad()

    # Step 4: Forward pass — only this rank's slice of the batch is used.
    # The model is a DDP-wrapped copy; gradients produced here are local.
    output = model(inputs)

    # Step 5: Compute loss between predictions and this rank's targets.
    # The loss is also local — it only reflects this rank's data.
    loss = nn.functional.cross_entropy(output, targets)

    # Step 6: Backward pass — fills .grad on each parameter with LOCAL gradients.
    # At this point, grad on rank 0 differs from grad on rank 1.
    # DDP hooks detect this and schedule async all_reduce across all ranks.
    loss.backward()

    # Step 7: DDP all_reduce completes here (or earlier, async).
    # Every rank now has IDENTICAL gradients — the average of all local grads.
    # Step 8: Optimizer uses these identical gradients to update parameters.
    # All ranks make the same weight updates, staying in sync.
    optimizer.step()
```

The key insight: steps 1-6 are purely local computation. Step 7 (gradient sync) is the only collective communication, and it happens exactly once per training step, after all compute is finished.

### Piece 5: Launch

```bash
torchrun --nproc_per_node=2 train.py
```

Run with 4 GPUs:

```bash
torchrun --nproc_per_node=4 train.py
```

The same script scales to any number of GPUs.

---

## Why DDP Beats DataParallel

PyTorch also has `DataParallel` (DP). It is simpler but slower:

| | DDP | DataParallel |
|--|-----|--------------|
| Communication | Once per step (all_reduce) | Every batch split (gather + broadcast) |
| Memory | One copy per GPU | Replicated on master GPU |
| Multi-GPU | All GPUs compute | One GPU coordinates |
| Multi-node | Works | Does not work |

DDP communicates less, uses memory more efficiently, and scales to multiple nodes. With NCCL on two GPUs, DDP typically achieves ~1.8x speedup vs single GPU for compute-bound workloads. Measured on 2x A100, per the benchmark in ADVANCED.md. Always prefer DDP.

---

## Recap

- **Init methods:** `env://` with `torchrun` is the standard setup. `tcp://` for manual multi-node. `file://` for single-node shared filesystem.
- **Backend:** Use NCCL for NVIDIA GPUs, Gloo as the CPU fallback.
- **Collective ops:** `all_reduce` sums across ranks (gradient sync), `all_gather` collects tensors, `broadcast` distributes from root, `reduce` aggregates to root, `barrier` synchronizes.
- **DDP:** Every rank has a full model copy. After backward, gradients are all_reduced so all ranks have identical averaged gradients before the optimizer step.
- **torchrun:** Launches processes, sets environment variables, handles cleanup. Use `--nproc_per_node` to control GPU count.
- **Scaling:** DDP scales near-linearly (~Nx speedup on N GPUs) for compute-bound workloads. Small models or slow interconnects bottleneck on communication.

---

## Which File to Run For What

| File | What It Demonstrates |
|------|----------------------|
| `ddp.py` | End-to-end DDP training with torchrun — gradient sync, multi-GPU training loop |
| `all_reduce.py` | Manual all_reduce gradient averaging — shows the mechanics behind DDP |
| `collective_ops.py` | All five collective ops (all_reduce, all_gather, broadcast, reduce, barrier) with timing |
| `init.py` | Different init methods (env://, tcp://, file://) and how torchrun sets up environment |

---

## Going Further

For real benchmark numbers across GPU counts, NCCL vs Gloo bandwidth tables, common error messages and fixes, multi-node setup walkthrough, and profiling distributed training with PyTorch Profiler — see [ADVANCED.md](./ADVANCED.md).

Sources:
- https://pytorch.org/tutorials/intermediate/dist_tuto.html
- https://pytorch.org/docs/stable/distributed.html
- https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html

---

Get the video walkthrough of NCCL vs Gloo benchmarks, multi-node setup, and distributed profiling: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
