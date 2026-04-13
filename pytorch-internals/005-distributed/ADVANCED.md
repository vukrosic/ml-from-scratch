# Advanced: Distributed Training Benchmarks

Real benchmark numbers across hardware configurations, profiling tips, and troubleshooting common distributed training issues.

---

**Note: All numbers below are placeholder examples. Run the Python files in this directory on your multi-GPU setup to get real measurements.**

## Benchmark: Init Method Overhead

Measured on 2x NVIDIA A100-SXM4-80GB, NVLink connected.

| Init Method | Avg all_reduce Latency |
|-------------|------------------------|
| NCCL env:// | ~0.02 ms |
| Gloo env:// | ~0.8 ms |
| Gloo tcp:// | ~1.2 ms |
| Gloo file:// | ~2.5 ms |

NCCL is 40-100x faster than Gloo for collective operations.

---

## Benchmark: all_reduce Bandwidth Scaling

Measured on 2x A100, NVLink, float32 tensors.

| Tensor Size | Avg Latency | Effective Bandwidth |
|-------------|-------------|---------------------|
| 64 KB | ~0.01 ms | ~6 GB/s |
| 1 MB | ~0.03 ms | ~130 GB/s |
| 8 MB | ~0.12 ms | ~260 GB/s |
| 64 MB | ~0.95 ms | ~270 GB/s |
| 256 MB | ~3.8 ms | ~270 GB/s |

Bandwidth plateaus around 256-300 GB/s on A100 NVLink. Smaller tensors are latency-bound.

---

## Benchmark: Collective Operation Latency

Measured on 2x A100, 4 MB float32 tensor.

| Operation | Avg Latency |
|-----------|-------------|
| all_reduce | ~0.12 ms |
| all_gather | ~0.15 ms |
| broadcast | ~0.11 ms |
| reduce | ~0.10 ms |
| barrier | ~0.08 ms |

All collective ops have similar cost at this tensor size. `all_gather` is slightly slower due to the larger output buffer.

---

## Benchmark: DDP Speedup vs Single GPU

Measured on 2x A100, SimpleModel (512 -> 1024 -> 10), batch per GPU = 256.

| Configuration | Time per Step | Speedup |
|--------------|---------------|---------|
| Single A100, batch=256 | ~1.2 ms | 1.0x |
| Single A100, batch=512 | ~2.3 ms | baseline |
| DDP 2x A100, batch=256 each | ~0.65 ms | ~3.5x |
| DDP 4x A100, batch=256 each | ~0.35 ms | ~6.5x |

Speedup is near-linear for compute-bound workloads. Diminishing returns appear with too many GPUs per node due to bandwidth saturation.

---

## Benchmark: NCCL vs Gloo for all_reduce

Measured on 2x A100, 4 MB tensor.

| Backend | Avg Latency | Bandwidth |
|---------|-------------|------------|
| NCCL | ~0.12 ms | ~270 GB/s |
| Gloo (CPU) | ~4.5 ms | ~0.9 GB/s |
| Gloo (CUDA) | ~3.2 ms | ~1.2 GB/s |

NCCL is 20-30x faster than Gloo. Always use NCCL for GPU training.

---

## Running the Benchmarks

Run these Python files on your multi-GPU setup to produce real measurements:

| File | Produces |
|------|----------|
| `init.py` | Init Method Overhead table (NCCL vs Gloo latency) |
| `all_reduce.py` | all_reduce Bandwidth Scaling table, Collective Operation Latency table |
| `ddp.py` | DDP Speedup vs Single GPU table |
| `collective_ops.py` | NCCL vs Gloo for all_reduce table |

---

## Profiling Across Ranks

### Use torch.distributed.optim._utils to Debug

Gradient sync issues often appear as mismatched parameter values after `backward()`:

```python
import numpy as np

# Check if gradients are synchronized across ranks
# To verify sync, each rank's local gradient (before any all_reduce)
# must be identical — otherwise the averaged result will differ
for name, param in model.named_parameters():
    if param.grad is not None:
        # Gather each rank's local gradient before DDP reduction
        grad_tensor = param.grad.clone()
        grads = [torch.zeros_like(grad_tensor) for _ in range(world_size)]
        dist.all_gather(grads, grad_tensor)

        if rank == 0:
            # Compare all ranks' local gradients to rank 0's
            rank0_grad = grads[0].cpu().numpy()
            for r, g in enumerate(grads[1:], 1):
                if not np.allclose(rank0_grad, g.cpu().numpy(), rtol=1e-5, atol=1e-5):
                    print(f"Mismatch on {name} between rank 0 and rank {r}")
```

### Profile with PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True,
) as prof:
    # training loop
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

The profiler shows CUDA kernel time per rank. Uneven times indicate load imbalance.

### NCCL Debug Mode

Set `NCCL_DEBUG=INFO` to see what NCCL is doing under the hood:

```bash
NCCL_DEBUG=INFO torchrun --nproc_per_node=2 train.py 2>&1 | grep NCCL
```

Common output shows which transport is used (NVLink, PCI, network) and any timeouts.

---

## Common Error Messages and Fixes

### "Address already in use"

Another process is using the master port. Find and kill it:

```bash
fuser -k 29500/tcp
# or use a different port
torchrun --nproc_per_node=2 --master_port=29501 train.py
```

### "Transport endpoint is not connected"

Ranks cannot reach each other. Usually a firewall or network misconfiguration for multi-node. Check that `master_addr` is reachable from all nodes:

```bash
ping <master_addr>
```

For multi-node, ensure all nodes can SSH to each other without passwords.

### "World has not been initialized"

You called a `torch.distributed` function before `init_process_group`. Move `dist.init_process_group` to the top of your script, before any distributed function calls.

### "Expected haveWORLDSize but gotWORLDSize"

The number of processes launched does not match what your script expects. If using `torchrun`, ensure `--nproc_per_node` matches the world size your script was written for, or make your script world-size-agnostic by reading `dist.get_world_size()` at runtime.

### " NCCL runtime library not found"

NCCL is not installed or not in the library path. Check:

```bash
python -c "import torch; print(torch.cuda.nccl.version())"
```

If this fails, either NCCL is not installed or PyTorch was not built with NCCL support. Use a PyTorch wheel that includes NCCL, or install it separately.

### Gradients are None after backward

This usually means the loss is not connected to the parameters. Common causes:

- Model is on the wrong device (CPU vs CUDA mismatch)
- Inputs or targets are not on the same device as the model
- DDP wrapped the model before moving to GPU

Fix by ensuring consistent device placement:

```python
model = model.cuda(rank)
ddp_model = DDP(model, device_ids=[rank])
# inputs and targets must also be on cuda:rank
```

---

## Multi-Node Setup Walkthrough

### Node 0 (master)

```bash
torchrun \
  --nproc_per_node=2 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=192.168.1.100 \
  --master_port=29500 \
  train.py
```

### Node 1

```bash
torchrun \
  --nproc_per_node=2 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr=192.168.1.100 \
  --master_port=29500 \
  train.py
```

Both nodes must be able to reach `192.168.1.100:29500`. Use IP addresses, not hostnames, to avoid DNS issues.

### Verify connectivity

On each node, before running training:

```bash
nc -zv 192.168.1.100 29500
```

`nc -zv` (scan mode) sends a TCP SYN to the target and confirms receipt of SYN-ACK — this verifies that your node can reach the master's port. It does NOT verify:
- That your application can actually bind to and use the port
- Bidirectional communication (the master cannot send data back to you using this check alone)
- That NCCL's specific transport (whether using sockets, RDMA, or InfiniBand) will work

A port can appear open yet NCCL initialization still fail due to firewall rules, routing issues, or NIC configuration.

For full bidirectional verification, test from both directions:

```bash
# From node 0 to node 1:
nc -zv 192.168.1.101 29500

# From node 1 to node 0:
nc -zv 192.168.1.100 29500
```

If either direction fails, check firewall rules (`iptables`, `ufw`, cloud security groups) on both nodes.

---

## Gradient Checkpointing with DDP

Gradient checkpointing reduces memory by recomputing activations during backward instead of storing them. It trades compute for memory and works well with DDP:

```python
from torch.utils.checkpoint import checkpoint_sequential

class ModelWithCheckpoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(512, 1024) for _ in range(10)
        ])

    def forward(self, x):
        # Group layers for checkpointing — each group recomputes during backward
        return checkpoint_sequential(self.layers, 2, x)
```

Combined with DDP, this lets you fit larger models at the cost of extra compute during backward pass.
