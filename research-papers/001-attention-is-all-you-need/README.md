# Attention Is All You Need — Paper Reproduction

Reproduce the original transformer paper. Encoder-decoder, multi-head attention, positional encoding.

## What we build
- Full encoder-decoder transformer matching the paper
- Train on small translation task
- Reproduce Figure 3 (architecture) and Table 2 (BLEU scores on small scale)

## Files (planned)
- `model.py` — full transformer
- `train.py` — training loop
- `data.py` — small parallel corpus
- `evaluate.py` — BLEU scoring
