# Agent Instructions — How To Build Lessons

This file is the single source of truth for any agent (human or AI) producing content for this repo. Read this fully before building anything.

---

## The Plan

One ML tutorial per day, published on YouTube. All code is free on GitHub. Extended paid materials on Skool ($49 tier only, no $9 tier).

**Three outputs per lesson:**
1. **GitHub** — free code + README article (this repo)
2. **YouTube** — video walkthrough of the blog article
3. **Skool $49** — extended notebook with deeper content NOT available for free

---

## Rules

### Code style
- Every concept gets its own small code block (3–8 lines). NEVER dump 50-line monoliths.
- Each code block must be preceded by a plain-English explanation of what it does and why.
- All code must be runnable standalone. No hidden imports, no "see other file" dependencies between lessons.
- Files are `lowercase_snake.py`. One concern per file.
- Include a `if __name__ == "__main__":` block in every file so the user can run it immediately.

### README (the article)
- Each lesson folder has one `README.md`. This is the article text.
- Structure: Hook → Concepts (each with small code block) → Recap → CTA
- Hook: 2–3 sentences explaining why this matters. No fluff.
- Recap: 3–5 bullet points of what the reader now understands.
- CTA: always end with `"Get the extended notebook with [specific extras]: [Skool link]"`
- No meta/production notes in the README. It's viewer-facing only.

### Folder structure
- Series folders: lowercase-kebab, no numbers, NEVER rename once created
- Lesson folders: `NNN-slug/`, append-only, NEVER reorder or renumber
- To add a lesson between 002 and 003, use the next available number at the end (e.g. 011)

### What goes on GitHub (FREE)
- Core concept code: the "from scratch" implementation
- Benchmark script: compare our implementation vs PyTorch built-in
- A clear, well-written README explaining everything

### What goes on Skool $49 (PAID) — NOT in this repo
- Extended Jupyter notebook (.ipynb) with all code runnable in one place
- Content that goes BEYOND the free lesson (more architectures, more comparisons, production-grade versions)
- The $49 asset must be worth $49 — it's not a copy of the free code in a notebook

### No $9 tier
- We do not sell a $9 tier. Skip it. Only free (GitHub) and $49 (Skool).

---

## Skool $49 — What To Include

The $49 asset extends the free lesson with depth and breadth the video doesn't cover. Here's what "extended" means for different lesson types:

### Pattern: "From Scratch" lessons (softmax, attention, autograd, etc.)

**Free (GitHub):**
- Build the thing from scratch
- Compare against PyTorch built-in
- One benchmark

**$49 (Skool):**
- All free code in one Jupyter notebook, runnable top-to-bottom
- 3–5 additional model architectures or use cases (not just 1)
- Numerical stability deep-dive or edge cases
- Gradient/Jacobian derivation and verification
- Performance profiling (memory + speed) across batch sizes
- Visual explanations (plots, heatmaps) that don't fit in a video

### Pattern: "Internals" lessons (torch.compile, autograd, memory, etc.)

**Free (GitHub):**
- Simplified toy version showing the concept
- 1–2 real PyTorch examples

**$49 (Skool):**
- Full profiling notebook across 5+ model architectures
- Annotated real source code walkthrough (e.g. actual Triton kernels, actual autograd tape)
- Automated diagnostic script (e.g. graph break scanner, memory leak finder)
- Comparison table across modes/settings with benchmarks
- Real-world examples from popular repos (HuggingFace, timm) showing the concept in production

### Pattern: "Research Paper" lessons (attention paper, GPT, etc.)

**Free (GitHub):**
- Core model implementation matching the paper
- Train on a small dataset
- Show it works

**$49 (Skool):**
- Full reproduction notebook with training curves
- Ablation studies (what happens if you remove component X?)
- Comparison against modern improvements
- Annotated excerpts from the actual paper explaining key equations
- Hyperparameter sensitivity analysis

---

## Examples

### Example 1: torch.compile (pytorch-internals/001-torch-compile)

**GitHub (free):**
- `tracer.py` — TracedTensor class, records operations
- `fusion.py` — merge matmul+relu into one op
- `codegen.py` — generate Python from optimized graph
- `benchmark.py` — eager vs compiled on MLP
- `graph_breaks.py` — .item() breaks, torch.where fixes
- `README.md` — article walking through the full pipeline

**Skool $49:**
- Jupyter notebook with profiling across 5 architectures (MLP, CNN, Transformer, attention, U-Net)
- Annotated Inductor-generated Triton code line by line
- Automated graph break scanner script
- `mode="default"` vs `"reduce-overhead"` vs `"max-autotune"` benchmarks
- Batch size sweep: at what point does compile win?
- Operator compatibility table

### Example 2: softmax (transformer-from-scratch/001-softmax)

**GitHub (free):**
- `softmax.py` — naive, stable, and temperature-scaled versions
- `jacobian.py` — compute the Jacobian of softmax
- `benchmark.py` — compare against torch.softmax
- `README.md` — article

**Skool $49:**
- Jupyter notebook with all code + inline plots
- Log-softmax derivation and implementation
- Softmax across 5 dtypes (float16, bfloat16, float32, float64, int8-approximation) — numerical behavior comparison
- Attention weight visualization using softmax with real sentence embeddings
- Temperature scaling interactive exploration (different T values, entropy plots)
- Gumbel-softmax for differentiable sampling

### Example 3: KV-Cache (inference/001-kv-cache)

**GitHub (free):**
- `naive.py` — generation without cache
- `kv_cache.py` — generation with cache
- `benchmark.py` — tokens/sec comparison
- `README.md` — article

**Skool $49:**
- Jupyter notebook with full walkthrough
- Memory usage analysis: cache size vs sequence length vs model size
- Multi-head KV-cache with grouped-query attention (GQA)
- Paged attention explanation (vLLM-style)
- Cache eviction strategies for long sequences
- Profiling: where time is actually spent with and without cache

### Example 4: Attention Is All You Need (research-papers/001-attention-is-all-you-need)

**GitHub (free):**
- `model.py` — full encoder-decoder transformer
- `train.py` — training loop on small translation task
- `data.py` — data loading
- `evaluate.py` — BLEU scoring
- `README.md` — article

**Skool $49:**
- Full reproduction notebook with training curves matching paper Figure 4
- Ablation: remove positional encoding, reduce heads, remove residual connections — measure impact
- Attention weight visualizations per layer per head
- Comparison: original sinusoidal vs learned vs RoPE positional encoding
- Annotated key equations from the paper with code implementing each one
- Scaling experiment: vary d_model and n_heads, plot convergence

---

## Delivery Format for Skool $49

Every $49 asset is delivered as:
1. **One Jupyter notebook (.ipynb)** — the main deliverable, runnable top-to-bottom
2. **Markdown notes** — for non-code deep-dives (paper annotations, theory)
3. **Standalone .py scripts** — for anything that should be run from terminal (benchmarks, profilers)

The notebook is the star. It must flow like a tutorial, not a code dump. Every cell has a markdown cell above it explaining what's next and why.

---

## YouTube Description Template

```
{title} — ML From Scratch

Code (free): https://github.com/vukrosic/ml-from-scratch/tree/main/{series}/{NNN-slug}/
Blog: https://vukrosic.vercel.app/blog/{slug}/
Extended notebook ($49): https://www.skool.com/opensuperintelligencelab

All code is free and open source forever.
```

## YouTube Pinned Comment Template

```
Get the extended notebook with [specific extras for this lesson]: [Skool link]
```
