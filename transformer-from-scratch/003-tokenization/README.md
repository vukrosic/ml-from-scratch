# Tokenization From Scratch

The model never sees text. It sees numbers.

Every character you type, every word you read — the transformer never touches any of it directly. Before attention, before embeddings, before the first matrix multiplication, there is a translation step: raw text goes in, a sequence of integers comes out. That translation is tokenization.

Get it wrong and you waste your context window, confuse the model with meaningless splits, and burn money on tokens that carry no information. Get it right and the model has clean, meaningful building blocks that compress language efficiently.

In this lesson we build a Byte Pair Encoding (BPE) tokenizer from scratch, walk through why subword tokenization dominates modern NLP, compare character-level and word-level alternatives, and benchmark our implementation against HuggingFace.

---

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

## Why tokenization matters

Here is what actually happens when you type a prompt into ChatGPT:

```
  "Hello, world!"
        |
        v
  +-------------+
  |  Tokenizer  |  text -> integer IDs
  +-------------+
        |
        v
  [15496, 11, 995, 0]
        |
        v
  +-------------+
  |  Embedding  |  IDs -> vectors
  +-------------+
        |
        v
  +-------------+
  |  Transformer |  vectors -> vectors
  +-------------+
        |
        v
  +-------------+
  | Decode head |  vectors -> next token probabilities
  +-------------+
```

The tokenizer is the front door. If "tokenization" gets split into `["token", "ization"]`, the model can learn that "-ization" is a productive suffix. If it gets split into `["to", "ken", "iz", "at", "ion"]`, the model wastes capacity reassembling fragments that carry no morphological signal.

Tokenization also controls cost. GPT-4 charges per token. If your tokenizer turns 100 words into 150 tokens, you pay for 150. If a better tokenizer turns those same words into 120 tokens, you pay for 120 — and the model sees more context in the same window.

---

## Three approaches: character, word, and subword

Every tokenizer makes a trade-off between vocabulary size and sequence length.

### Character-level

Split every character into its own token:

```
"unhappiness" -> ["u", "n", "h", "a", "p", "p", "i", "n", "e", "s", "s"]
                  11 tokens, vocab size ~256 (one per byte)
```

Pros:
- Tiny vocabulary — just the ASCII or Unicode byte set
- Zero out-of-vocabulary (OOV) tokens — every string is representable

Cons:
- Sequences explode in length. A 512-token context window covers ~200 words
- The model must learn from scratch that "t"+"o"+"k"+"e"+"n" means something
- Attention cost is O(n^2) — longer sequences are quadratically more expensive

### Word-level

Split on whitespace. Each word is one token:

```
"unhappiness" -> ["unhappiness"]
                  1 token, but vocab needs ~500K+ entries
```

Pros:
- Short sequences — each word is one token
- Tokens are semantically meaningful

Cons:
- Massive vocabulary: even 100K entries miss names, typos, code, and morphological variants ("run", "runs", "running", "ran" are all separate entries)
- Out-of-vocabulary problem: the model sees `<UNK>` and loses all information
- No shared structure: "run" and "running" are completely unrelated token IDs

### Subword (BPE) — the middle ground

BPE finds frequently co-occurring character sequences and merges them:

```
"unhappiness" -> ["un", "happiness"]
                  2 tokens, vocab size ~50K
```

This captures morphology ("un-" as a prefix, "-ness" as a suffix), keeps sequences short, and handles any input — even typos and code — by falling back to characters for rare patterns.

Here is the trade-off space:

```
  Vocab size    Sequence length    OOV risk
  ---------     ---------------    --------
  Character:    ~256               very long       none
  Subword:      ~50K               moderate        none
  Word:         ~500K+             short           high
                     ^
                     |
              BPE lives here — the sweet spot
```

---

## The BPE algorithm, step by step

BPE was originally a data compression algorithm (Gage, 1994). The NLP adaptation (Sennrich et al., 2016) applies the same idea to build a subword vocabulary. The core loop is simple:

1. Start with every character as its own token
2. Count all adjacent pairs in the corpus
3. Merge the most frequent pair into a new token
4. Repeat from step 2

Let's walk through a concrete example on a tiny corpus.

### Worked example

Corpus: `"ab ab ab bc bc"`

**Initial tokens (character-level):**
```
[a] [b] [ ] [a] [b] [ ] [a] [b] [ ] [b] [c] [ ] [b] [c]
```

**Iteration 1** — count adjacent pairs:
```
Pair       Count
(a, b)       3     <-- winner
(b, " ")     3
(" ", a)     2
(" ", b)     2
(b, c)       2
(c, " ")     1
```

Merge `(a, b)` into `ab`:
```
[ab] [ ] [ab] [ ] [ab] [ ] [b] [c] [ ] [b] [c]
```
Sequence shrank from 14 tokens to 11.

**Iteration 2** — count pairs again:
```
Pair        Count
(ab, " ")     3     <-- winner
(" ", ab)     2
(" ", b)      2
(b, c)        2
(c, " ")      1
```

Merge `(ab, " ")` into `ab `:
```
[ab ] [ab ] [ab ] [b] [c] [ ] [b] [c]
```
Down to 8 tokens.

**Iteration 3** — count pairs:
```
Pair          Count
(ab , ab)       2     <-- winner (tie-broken arbitrarily)
(b, c)          2
(ab , b)        1
(c, " ")        1
(" ", b)        1
```

Merge `(b, c)` into `bc`:
```
[ab ] [ab ] [ab ] [bc] [ ] [bc]
```
Down to 6 tokens. The vocabulary now has: `{a, b, c, " ", ab, "ab ", bc}`.

Each merge adds one token to the vocabulary and removes occurrences from the corpus. After N merges, you have a base character set plus N learned merge rules.

```
  Merge #   Pair merged    New token   Corpus length
  -------   -----------    ---------   -------------
  0         (start)        -           14
  1         (a, b)         "ab"        11
  2         (ab, " ")      "ab "       8
  3         (b, c)         "bc"        6
```

---

## Piece 1: Split text into characters

We represent each character as its own token. We track word-boundary spaces with a special marker so we can reconstruct the original text during decoding.

```python
# tokenizers.py — Part 1

SPACE_CHAR = " "

def char_tokenize(text):
    """Split text into individual characters with word-boundary spaces tracked."""
    chars = []
    words = text.split(" ")
    for i, word in enumerate(words):
        if i > 0:
            chars.append(SPACE_CHAR)   # word boundary marker
        for ch in word:
            chars.append(ch)
    return chars
```

`"hello world"` becomes `["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"]`.

---

## Piece 2: Count adjacent pairs

BPE's core operation: scan the token list and count how many times each adjacent pair appears.

```python
# tokenizers.py — Part 2

from collections import defaultdict

def get_pair_counts(tokens):
    """Count how many times each adjacent pair appears in the token list."""
    counts = defaultdict(int)
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        counts[pair] += 1
    return counts
```

If `("t", "o")` appears 500 times and `("z", "q")` appears 2 times, BPE will merge `("t", "o")` first. The most frequent pair becomes the next vocabulary entry.

---

## Piece 3: Merge the most frequent pair

Given a pair like `("t", "o")`, replace every adjacent occurrence with a single combined token `"to"`.

```python
# tokenizers.py — Part 3

def merge_pair(tokens, pair, merged_token):
    """Replace all occurrences of `pair` with `merged_token`."""
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

Here is what the merge looks like visually:

```
  Before:  [t] [o] [k] [e] [n] [i] [z] [e]
            \_/
             |
  After:   [to] [k] [e] [n] [i] [z] [e]
                  \_/
                   |    (next merge might combine "ke" or "en")
```

---

## Piece 4: The BPE training loop

We repeat the count-and-merge step until we hit our target vocabulary size. Each merge creates one new token and reduces the average sequence length.

```python
# tokenizers.py — Part 4

class SimpleBPETokenizer:
    def __init__(self, vocab_size=300):
        self.vocab_size = vocab_size
        self.vocab = {}      # token string -> token ID
        self.merges = []     # list of (pair, merged_token) in order

    def train(self, text):
        """Train the tokenizer on a corpus of text. Builds vocab and merges."""
        # Start with special tokens and the unique characters in the corpus
        special_tokens = {"<PAD>": 0, "<UNK>": 1}
        unique_chars = sorted(set(char_tokenize(text)))
        current_vocab = dict(special_tokens)
        for ch in unique_chars:
            if ch not in current_vocab:
                current_vocab[ch] = len(current_vocab)

        tokens = char_tokenize(text)

        # Build vocabulary by repeatedly merging the most frequent pair
        next_id = len(current_vocab)
        reverse_vocab = {v: k for k, v in current_vocab.items()}
        merges_to_do = self.vocab_size - len(current_vocab)

        for _ in range(merges_to_do):
            pair_counts = get_pair_counts(tokens)
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            merged_str = best_pair[0] + best_pair[1]

            current_vocab[merged_str] = next_id
            reverse_vocab[next_id] = merged_str
            next_id += 1

            self.merges.append((best_pair, merged_str))
            tokens = merge_pair(tokens, best_pair, merged_str)

        self.vocab = current_vocab
        self.reverse_vocab = reverse_vocab
```

After training, `self.merges` holds the learned merge rules in priority order. The first merge is the most frequent pair in the corpus, the last merge is the rarest useful pattern. This ordering matters during encoding — we will apply merges in the same order.

---

## Piece 5: Encode new text

To encode unseen text, start with character tokens and replay every merge rule in order.

```python
    def encode(self, text):
        """Convert text into a list of token IDs using learned merges."""
        tokens = char_tokenize(text)
        for pair, merged_str in self.merges:
            tokens = merge_pair(tokens, pair, merged_str)
        return [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
```

The key insight: we apply merges in the exact order they were learned. The most frequent patterns get merged first, progressively building up to longer subwords.

```
  Encoding "tokenization":

  Start:    [t] [o] [k] [e] [n] [i] [z] [a] [t] [i] [o] [n]
  Merge 1:  apply (" ","t") -> no match (no space before t here)
  Merge 4:  apply (e, n) -> [t] [o] [k] [en] [i] [z] [a] [t] [i] [o] [n]
  Merge 5:  apply (o, k) -> [t] [ok] [en] [i] [z] [a] [t] [i] [o] [n]
  Merge 6:  apply (ok, en) -> [t] [oken] [i] [z] [a] [t] [i] [o] [n]
  ...
  Final:    [t] [oken] [i] [z] [a] [ti] [on]   -> 7 token IDs
```

---

## Piece 6: Decode back to text

Decoding is the reverse: convert IDs back to strings and join them.

```python
    def decode(self, token_ids):
        """Convert token IDs back into text string."""
        tokens = [self.reverse_vocab.get(tid, "<UNK>") for tid in token_ids]
        text = "".join(tokens)
        return text.replace(SPACE_CHAR, " ")
```

Because every merged token is just the concatenation of its component characters, joining and replacing the space marker perfectly reconstructs the original text.

---

## Unicode and bytes: why GPT-2 uses byte-level BPE

Our implementation works on Python characters. Real production tokenizers like GPT-2 work on raw bytes instead. Why?

**The problem with character-level BPE:**
- Unicode has 150,000+ characters (Chinese, Arabic, emoji, mathematical symbols...)
- A character-level base vocabulary would be enormous
- Many characters appear rarely, wasting vocabulary slots

**The byte-level solution:**
- Every string, in any language, is a sequence of bytes (0-255) when encoded as UTF-8
- Base vocabulary is always exactly 256 entries — tiny and universal
- BPE merges learn which byte sequences form useful subwords

Here is how UTF-8 encodes different scripts:

```
  Character    UTF-8 bytes     Byte count
  ---------    -----------     ----------
  "A"          [65]            1 byte
  "e"          [101]           1 byte
  "n"          [195, 177]      2 bytes    (Spanish)
  "zh"         [228, 184, 173] 3 bytes    (Chinese "middle")
  "poop emoji" [240, 159, 146, 169] 4 bytes
```

English text is 1 byte per character. Chinese text is 3 bytes per character. This means a Chinese sentence costs roughly 3x more base tokens than an English sentence of the same semantic content — before merges help compress common byte sequences.

GPT-2's tokenizer (and GPT-3, GPT-4) uses byte-level BPE with a vocabulary of ~50,257 tokens: 256 byte tokens + ~50K learned merges.

---

## Tokenizer efficiency: not all languages are equal

Because BPE training corpora are predominantly English, English text compresses much better than other languages:

```
  Language    Typical tokens per word
  --------    -----------------------
  English     ~1.3
  Spanish     ~1.5
  Chinese     ~2.5  (each character is often its own token)
  Arabic      ~2.0
  Code        ~2.5  (variable names get split)
```

This has real consequences:
- A Chinese user gets ~40% less "thinking space" in the same context window
- API costs per semantic unit are higher for non-English text
- Code is expensive because variable names like `getUserAccountBalance` get split into `["get", "User", "Account", "Balance"]` or worse

This is why some models (like Meta's Llama) train tokenizers on multilingual corpora — to give non-English text fairer compression ratios.

---

## Special tokens: BOS, EOS, PAD, UNK

Production tokenizers reserve special token IDs for control signals. These never appear in normal text but carry critical information for the model.

```
  Token    Name                Purpose
  -----    ----                -------
  <PAD>    Padding             Fill short sequences to fixed batch length.
                               Attention mask ignores these positions.

  <UNK>    Unknown             Fallback for tokens not in vocabulary.
                               BPE rarely needs this (falls back to bytes).

  <BOS>    Beginning of seq    Marks the start of a new sequence.
                               Tells the model "this is a fresh context."

  <EOS>    End of sequence     Marks where to stop generating.
                               Without this, the model generates forever.
```

Our implementation uses `<PAD>` (ID 0) and `<UNK>` (ID 1). Production systems like GPT-2 add `<|endoftext|>` as an EOS marker. Chat models add tokens like `<|im_start|>` and `<|im_end|>` to delimit user/assistant turns.

Why special tokens matter for training:
- **Without EOS**: the model never learns to stop generating. It will produce text until it hits the maximum sequence length.
- **Without PAD**: you cannot batch sequences of different lengths. Every sequence in a batch must be the same length for matrix operations.
- **Without BOS**: the model has no signal that a new document has started. It may treat the concatenation of two documents as one continuous text.

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

Notice the merge patterns:
- **Space+letter pairs** (`" t"`, `" i"`) dominate early merges because most words start after a space
- **"oken"** emerges as a common subword (from "token", "broken", "spoken")
- **"ng"** is frequent across many English words ("tokenization", "running", "learning", "encoding")
- Multi-character tokens build up hierarchically: `o`+`k` = `ok`, then `ok`+`en` = `oken`

---

## Compare with HuggingFace

```bash
python compare.py
```

Our tokenizer uses ~189 merges to build a ~223-token vocabulary. Compare that to BERT's 30,522-token vocabulary:

```
Our BPE (vocab=300):       Token count: 4
HuggingFace bert-base:    Token count: 8
```

Our tokenizer produces fewer tokens for this specific phrase because it is optimized for this exact corpus. BERT's larger vocabulary is general-purpose — it covers medical text, legal documents, code, and dozens of languages. A specialized tokenizer always wins on its training data. A general-purpose tokenizer wins everywhere else.

### Comparing against tiktoken (GPT-2 tokenizer)

For a fairer comparison, you can check our merges against GPT-2's actual merge list. GPT-2 was trained on WebText (40GB of internet text), so its early merges reflect English web writing:

```
GPT-2 first merges:         Our first merges:
  ("t", "h")  -> "th"         (" ", "t")  -> " t"
  ("i", "n")  -> "in"         ("e", " ")  -> "e "
  ("e", "r")  -> "er"         (" ", "i")  -> " i"
  ("a", "n")  -> "an"         ("e", "n")  -> "en"
```

Both tokenizers discover similar patterns — frequent English bigrams like "th", "in", "en". The differences come from corpus composition: GPT-2 sees diverse web text, our tokenizer sees only our small training corpus. But the algorithm is identical.

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

Our toy implementation is faster because it does less — no special token handling, no pre-tokenization regex, no caching infrastructure, no alignment with model weight indices. HuggingFace's implementation is heavily engineered for correctness across every edge case. In production, that correctness matters far more than raw speed.

---

## Common mistakes and gotchas

### 1. Whitespace handling

The most common bug: forgetting to track spaces. If you split `"hello world"` into characters without marking the space, you cannot reconstruct the original text during decoding.

```
  Wrong:  "hello world" -> ["h","e","l","l","o","w","o","r","l","d"]
                            Where did the space go?

  Right:  "hello world" -> ["h","e","l","l","o"," ","w","o","r","l","d"]
                            Space is an explicit token
```

### 2. Merge order matters

If you apply merges in the wrong order, you get different (incorrect) tokenizations. The merge list is ordered by training frequency. Applying merge #50 before merge #1 produces garbage.

```
  Correct order:   merge("e","n") first, then merge("ok","en")
  Reversed order:  merge("ok","en") fails because "en" doesn't exist yet
```

### 3. Tokenizer-model vocabulary mismatch

If you train a model with tokenizer A (vocab size 50K) and then swap in tokenizer B (vocab size 32K), every token ID maps to the wrong embedding. The model outputs nonsense. The tokenizer and model must always be paired.

### 4. Forgetting special tokens

If your model expects `<BOS>` at position 0 and you do not prepend it, the model's first-token behavior is unpredictable. Always check what special tokens a pretrained model expects.

### 5. Training on too little data

BPE merges learned from a tiny corpus will not generalize. If your training text is 100 words, the learned merges are specific to those words. Production tokenizers train on gigabytes of text.

### 6. Vocabulary size too small or too large

- **Too small** (e.g., 256 — just bytes): sequences are very long, attention is expensive, model struggles to learn
- **Too large** (e.g., 1M tokens): embedding table is enormous, rare tokens get almost no training signal, overfitting to the tokenizer training corpus
- **Sweet spot**: 32K-100K for most modern LLMs

---

## The full picture

Here is how tokenization fits into the transformer pipeline:

```
  Raw text: "The cat sat on the mat."
                |
                v
  +----------------------------+
  | Tokenizer (BPE)            |
  | "The" " cat" " sat" " on"  |
  | " the" " mat" "."          |
  +----------------------------+
                |
                v
  Token IDs:  [464, 3797, 3332, 319, 262, 2603, 13]
                |
                v
  +----------------------------+
  | Embedding table            |
  | ID -> d_model vector       |
  | (learned during training)  |
  +----------------------------+
                |
                v
  Vectors: [[0.12, -0.34, ...],   # "The"
            [0.56,  0.78, ...],   # " cat"
            ...]                  # each is d_model dims
                |
                v
  +----------------------------+
  | Transformer blocks         |
  | (attention + feedforward)  |
  +----------------------------+
                |
                v
  Next token probabilities
```

The tokenizer determines the atoms that the entire model works with. Every architectural decision downstream — embedding dimension, context length, attention patterns — operates on the units that the tokenizer chose. That is why tokenization is lesson 003 and not an afterthought.

---

## Recap

- **The model never sees text** — only integer token IDs. Tokenization is the translation layer.
- **Character-level** tokenization: tiny vocab, no OOV, but sequences are very long.
- **Word-level** tokenization: short sequences, but huge vocab and OOV problems.
- **BPE** is the middle ground: start with characters, iteratively merge frequent pairs.
- **Training BPE**: repeat N times — count pairs, find the most frequent, merge, repeat.
- **Encoding**: start with characters, apply each learned merge in order.
- **Byte-level BPE** (GPT-2 style) uses raw bytes as the base alphabet — handles any Unicode input with a base vocab of exactly 256.
- **Special tokens** (BOS, EOS, PAD, UNK) carry control signals the model needs for training and generation.
- **Tokenizer efficiency** varies by language and domain — English compresses well, Chinese and code are expensive.
- **The tokenizer and model are married** — swapping one without the other produces nonsense.
- The vocabulary size trades off between coverage (more tokens = shorter sequences) and learnability (fewer tokens = more training signal per token).

---

Get the video walkthrough of tokenizer edge cases, subword regularization, and SentencePiece vs BPE vs WordPiece comparison: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
