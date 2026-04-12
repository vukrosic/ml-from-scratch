# Agent Instructions — How To Build Lessons

This file is the single source of truth for any agent (human or AI) producing content for this repo. Read this fully before building anything.

---

## The Plan

One ML tutorial per day, published on YouTube. **ALL code is free on GitHub — including extended/deep-dive code.** The paid product on Skool ($49) is **video explanations** of the extended code, not the code itself.

**Three outputs per lesson:**
1. **GitHub** — ALL code (core + extended) + README article (this repo, free forever)
2. **YouTube** — free video walkthrough of the core article/README
3. **Skool $49** — paid video walkthroughs of the extended code and deep-dive topics that the free YouTube video doesn't cover

**No $9 tier.** Only free (GitHub + YouTube) and $49 (Skool video explanations).

---

## Monetization Model

```
FREE (GitHub repo):     All .py files, all README articles, all code
FREE (YouTube):         Video walkthrough of the core README/article
PAID (Skool $49):       Video explanations of the extended code + deep-dive topics
```

The code is the hook. The videos are the product. People can read and run everything for free. They pay for the guided walkthrough of the hard parts.

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
- CTA: always end with `"Get the video walkthrough of [specific extended topics]: [Skool link]"`
- No meta/production notes in the README. It's viewer-facing only.

### Folder structure
- Series folders: lowercase-kebab, no numbers, NEVER rename once created
- Lesson folders: `NNN-slug/`, append-only, NEVER reorder or renumber
- To add a lesson between 002 and 003, use the next available number at the end (e.g. 011)

### What goes in each lesson folder
Every lesson folder contains:

**Core files (covered in free YouTube video):**
- `README.md` — the article
- Core `.py` files — the "from scratch" implementation
- `benchmark.py` — compare our implementation vs PyTorch built-in

**Extended files (covered in paid Skool video):**
- Additional `.py` files that go deeper (more architectures, profiling, ablations, scanners)
- These files are FREE to read and run, but the video explaining them is on Skool $49

**No `skool_extended.md` files.** Extended content descriptions go in the README's CTA section, not in a separate planning doc. Keep planning docs out of the public repo.

### .gitignore
Never commit: `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `.env`

---

## Extended Code — What To Include Per Lesson Type

### Pattern: "From Scratch" lessons (softmax, attention, autograd, etc.)

**Core (free YouTube video covers these):**
- Build the thing from scratch
- Compare against PyTorch built-in
- One benchmark

**Extended (free code, paid Skool video explains these):**
- 3–5 additional model architectures or use cases
- Numerical stability deep-dive or edge cases
- Gradient/Jacobian derivation and verification
- Performance profiling (memory + speed) across batch sizes
- Visual explanations (plots, heatmaps)

### Pattern: "Internals" lessons (torch.compile, autograd, memory, etc.)

**Core:**
- Simplified toy version showing the concept
- 1–2 real PyTorch examples

**Extended:**
- Full profiling across 5+ model architectures
- Annotated real source code walkthrough (e.g. actual Triton kernels)
- Automated diagnostic scripts (e.g. graph break scanner)
- Comparison table across modes/settings with benchmarks
- Real-world examples from popular repos (HuggingFace, timm)

### Pattern: "Research Paper" lessons (attention paper, GPT, etc.)

**Core:**
- Core model implementation matching the paper
- Train on a small dataset
- Show it works

**Extended:**
- Ablation studies (what happens if you remove component X?)
- Comparison against modern improvements
- Hyperparameter sensitivity analysis
- Training curves and reproduction verification

---

## Examples

### Example 1: torch.compile (pytorch-internals/001-torch-compile)

**Core files:**
- `tracer.py` — TracedTensor class, records operations
- `fusion.py` — merge matmul+relu into one op
- `codegen.py` — generate Python from optimized graph
- `benchmark.py` — eager vs compiled on MLP
- `graph_breaks.py` — .item() breaks, torch.where fixes

**Extended files:**
- `profile_architectures.py` — profiling across 5 architectures (MLP, CNN, Transformer, attention, U-Net)
- `graph_break_scanner.py` — automated scanner that finds all graph breaks in a model
- `benchmark_modes.py` — `mode="default"` vs `"reduce-overhead"` vs `"max-autotune"`

### Example 2: softmax (transformer-from-scratch/001-softmax)

**Core files:**
- `softmax.py` — naive, stable, and temperature-scaled versions
- `jacobian.py` — compute the Jacobian of softmax
- `benchmark.py` — compare against torch.softmax

**Extended files:**
- Log-softmax derivation and implementation
- Softmax across 5 dtypes — numerical behavior comparison
- Temperature scaling entropy plots
- Gumbel-softmax for differentiable sampling

### Example 3: KV-Cache (inference/001-kv-cache)

**Core files:**
- `naive.py` — generation without cache
- `kv_cache.py` — generation with cache
- `benchmark.py` — tokens/sec comparison

**Extended files:**
- Memory usage analysis across sequence lengths
- Multi-head KV-cache with grouped-query attention (GQA)
- Paged attention explanation (vLLM-style)
- Cache eviction strategies

### Example 4: Attention Is All You Need (research-papers/001-attention-is-all-you-need)

**Core files:**
- `model.py` — full encoder-decoder transformer
- `train.py` — training loop on small translation task
- `data.py` — data loading
- `evaluate.py` — BLEU scoring

**Extended files:**
- Ablation scripts (remove positional encoding, reduce heads, etc.)
- Attention weight visualizations per layer per head
- Positional encoding comparison (sinusoidal vs learned vs RoPE)
- Scaling experiments

---

## YouTube Description Template

```
{title} — ML From Scratch

All code (free): https://github.com/vukrosic/ml-from-scratch/tree/main/{series}/{NNN-slug}/
Blog: https://vukrosic.vercel.app/blog/{slug}/
Extended video walkthroughs: https://www.skool.com/opensuperintelligencelab

All code is free and open source forever.
```

## YouTube Pinned Comment Template

```
Get the video walkthrough of [specific extended topics for this lesson]: [Skool link]
```
