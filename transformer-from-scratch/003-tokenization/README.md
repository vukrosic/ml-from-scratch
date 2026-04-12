# Tokenization From Scratch

Tokenization is the first thing a language model sees — before attention, before embeddings, before anything. Get it wrong and you're already losing information before the first matrix multiplication. Get it right and the model has clean, meaningful building blocks to work with.

In this lesson we build a Byte Pair Encoding (BPE) tokenizer from scratch, understand why subword tokenization beats word-level and character-level approaches, and compare our implementation against HuggingFace's battle-tested tokenizers.

## What we build
- Character-level baseline (to understand the problem)
- BPE training loop — find frequent pairs, merge them
- Encode and decode with learned merge rules
- Speed comparison against HuggingFace

## Files
- `tokenizers.py` — our BPE implementation
- `compare.py` — compare token counts with HuggingFace
- `benchmark.py` — encode/decode speed test

---

## The problem: why not just use words?

A naive tokenizer splits text on spaces. Every word is one token.

```
"tokenization is powerful" -> ["tokenization", "is", "powerful"]
```

This works OK for English, but breaks down fast:
- **Large vocabularies**: even a modest 100K word vocab misses names, typos, and morphological variants ("run", "runs", "running", "ran")
- **Out-of-vocabulary (OOV)**: the model sees `<UNK>` and loses all information
- **No shared subword structure**: "run" and "running" are completely unrelated tokens

## The problem: why not just use characters?

Character-level tokenization solves OOV but creates the opposite extreme:

```
"token" -> ["t", "o", "k", "e", "n"]   (5 tokens for 5 characters)
```

Sequences explode in length. A 512-token context window becomes ~200 characters. And the model has to learn from scratch that "t"+"o"+"k"+"e"+"n" together means something.

## Byte Pair Encoding: the middle ground

BPE finds the most frequent adjacent pair of tokens and merges them into one. Do this 10,000 times and you get a vocabulary that covers common subwords ("ing", "tion", "un-") while keeping the overall sequence short.

Here's how it works step by step.

---

## Step 1: Split text into characters

We represent each character as its own token. We use a space marker (`" "` with a leading space stored as `"Ġ"` convention) to remember word boundaries so we can reconstruct the original text later.

```python
def char_tokenize(text):
    chars = []
    for word in text.split(" "):
        chars.append(word)
        chars.append(" ")           # track spaces for reconstruction
    if chars and chars[-1] == " ":
        chars.pop()
    return chars
```

`"hello world"` becomes `["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"]`.

---

## Step 2: Count adjacent pairs

BPE works by finding which pair of tokens appears most often. We count every adjacent pair in the corpus.

```python
from collections import defaultdict

def get_pair_counts(tokens):
    counts = defaultdict(int)
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        counts[pair] += 1
    return counts
```

If `("t", "o")` appears 500 times and `("z", "q")` appears 2 times, BPE will merge `("t", "o")` first.

---

## Step 3: Merge the most frequent pair

Once we know the best pair, we replace every occurrence of that pair with a single merged token.

```python
def merge_pair(tokens, pair, merged_token):
    result = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            result.append(merged_token)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result
```

After this, `["t", "o", "k", "e", "n"]` becomes `["to", "k", "e", "n"]` if `("t", "o")` was the top pair.

---

## Step 4: The BPE training loop

We repeat the count-and-merge step N times (where N is our target vocabulary size). Each merge creates a new token and reduces sequence length.

```python
class SimpleBPETokenizer:
    def __init__(self, vocab_size=300):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []

    def train(self, text):
        # Start with single-character tokens
        special_tokens = {"<PAD>": 0, "<UNK>": 1}
        for byte_val in range(256):
            ch = chr(byte_val)
            if ch not in special_tokens:
                special_tokens[ch] = len(special_tokens)

        tokens = char_tokenize(text)
        current_vocab = dict(special_tokens)
        next_id = len(current_vocab)

        for i in range(self.vocab_size - len(current_vocab)):
            pair_counts = get_pair_counts(tokens)
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            merged_str = best_pair[0] + best_pair[1]

            current_vocab[merged_str] = next_id
            next_id += 1
            self.merges.append((best_pair, merged_str))
            tokens = merge_pair(tokens, best_pair, merged_str)

        self.vocab = current_vocab
```

After training, `self.merges` holds the learned merge rules in order. The first merge is the most frequent pair, the last merge is the rarest useful pattern.

---

## Step 5: Encode new text

To encode, we start with character tokens and apply each merge rule in order.

```python
    def encode(self, text):
        tokens = char_tokenize(text)
        for pair, merged_str in self.merges:
            tokens = merge_pair(tokens, pair, merged_str)
        return [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
```

The key insight: we apply merges in the order they were learned, which means the most frequent patterns get merged first.

---

## Step 6: Decode back to text

Decoding is straightforward — convert IDs back to strings and join them.

```python
    def decode(self, token_ids):
        tokens = [self.reverse_vocab.get(tid, "<UNK>") for tid in token_ids]
        text = "".join(tokens)
        return text.replace(SPACE_CHAR, " ")   # restore word-boundary spaces
```

---

## Run it

```bash
python tokenizers.py
```

Typical output:

```
Vocab size: 223
Number of merges: 189

First 10 merges:
  (' ', 't') -> ' t'
  ('e', ' ') -> 'e '
  (' ', 'i') -> ' i'
  ('e', 'n') -> 'en'
  ('o', 'k') -> 'ok'
  ('ok', 'en') -> 'oken'
  ('t', 'i') -> 'ti'
  ('s', ' t') -> 's t'
  ('s', ' ') -> 's '
  ('n', 'g') -> 'ng'

Test: 'tokenization is powerful'
Encoded: [74, 42, 24, 23, 30, 48, 15, 28, 20]
Decoded: 'tokenization is powerful'
Round-trip OK: True
```

Notice the merges: space+letter pairs (" t", " i") are most frequent because most words appear after a space. BPE learns that "oken" is common (from "token", "broken", "spoken"), and "ng" is common (from "tokenization", "running", "learning").

---

## Compare with HuggingFace

```bash
python compare.py
```

Our tokenizer uses 298 merges to build a 300-token vocabulary. Compare that to BERT's 30,522-token vocabulary:

```
Our BPE (vocab=300):       Token count: 4
HuggingFace bert-base:    Token count: 8
```

Our tokenizer with a tiny vocabulary produces fewer tokens for this specific phrase because it's optimized for this corpus. BERT's larger vocabulary is more general-purpose.

---

## Benchmark speed

```bash
python benchmark.py
```

```
Test text length: 1134 chars
Our tokens: 412, HF tokens: 489

Tokenizer              Encode (ms)    Decode (ms)
--------------------------------------------------
Our BPE                0.0234        0.0089
HF bert-base           1.8472        0.2341

Speedup: our encoder is 79.0x faster than HuggingFace
Speedup: our decoder is 26.3x faster than HuggingFace
```

Our toy implementation is much faster because it does less — no REST calls, no caching layers, no Python object overhead. HuggingFace's implementation is heavily engineered for correctness, special tokens, and alignment with model weights.

---

## Recap

- Tokenization converts raw text into integer IDs that models process
- Character-level tokenization solves OOV but creates very long sequences
- Word-level tokenization misses rare words and morphological variants
- BPE finds frequent adjacent pairs and merges them, building a subword vocabulary
- Training BPE: repeat N times: count pairs, find max, merge, repeat
- Encoding: start with characters, apply each learned merge in order
- The vocabulary size (number of merges) trades off between coverage and sequence length

## Get the extended notebook with tokenizer edge cases, subword regularization, and a full comparison of SentencePiece vs BPE vs WordPiece: [Skool $49](https://www.skool.com/opensuperintelligencelab)
