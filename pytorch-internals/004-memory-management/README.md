# GPU Memory Management

Where your VRAM goes and how to get it back.

## What we cover
- `torch.cuda.memory_allocated()` vs `memory_reserved()`
- Memory fragmentation and the caching allocator
- Gradient checkpointing to trade compute for memory
- `torch.cuda.memory_snapshot()` visualization

## Files (planned)
- `memory_tracker.py` — log memory at each layer
- `gradient_checkpoint.py` — with vs without
- `snapshot.py` — visualize allocator state
