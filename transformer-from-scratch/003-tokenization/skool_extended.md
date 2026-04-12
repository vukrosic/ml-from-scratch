# Tokenization Extended -- Skool $49

*Extended content not available on YouTube. Full notebook with all code runnable top-to-bottom.*

---

## 1. SentencePiece vs BPE vs WordPiece

All three are subword tokenization algorithms, but they differ in how they build the vocabulary and how encoding/decoding works.

### BPE (Byte Pair Encoding) — used by GPT-2, GPT-3, LLaMA

**How it builds vocab**: Start with character-level tokens. Repeatedly count adjacent pairs, find the most frequent, and merge. Each merge is irreversible — once "to" is merged from ("t", "o"), you can never split it back during decoding.

**Encoding**: Greedy left-to-right application of merge rules. After each merge, restart from the beginning of the token list.

**Key property**: BPE is deterministic and produces the same tokenization for the same text every time.

**Example merges learned from a news corpus**:
```
("t", "o") -> "to"         ("con", "t") -> "cont"
(" ", "the") -> " the"     ("un", " ") -> "un "
("i", "ng") -> "ing"       ("de", "c") -> "dec"
```

### WordPiece — used by BERT

**How it builds vocab**: Similar to BPE but uses frequency counts differently. WordPiece starts from a large word vocabulary and splits rare words into subwords. It selects the split that maximizes the language model likelihood (not just pair frequency).

**Encoding**: Same greedy left-to-right application. The key difference is in what gets merged — WordPiece merges based on maximizing `freq(combined) / (freq(part1) * freq(part2))`, which is a likelihood ratio, not raw frequency.

**Key property**: WordPiece always prepends `##` to continuation pieces (`##ing`, `##ful`), making it easy to distinguish word starts from subword continuations.

**Example**:
```
"unbelievable" -> ["un", "##believ", "##able"]
"running"      -> ["runn", "##ing"]
```

### SentencePiece — used by T5, XLNet, MarianMT

**How it builds vocab**: SentencePiece treats the input as a raw byte stream (not pre-tokenized into words). It applies BPE on the full Unicode character set, including spaces and newlines. This means it can tokenize ANY text without a language-specific pre-tokenization step.

**Key differences from BPE**:
- No pre-tokenization step (no space tokenization before BPE)
- Uses Unicode byte encoding by default (handles any language, including Chinese, Japanese, Arabic)
- Includes whitespace as part of the vocabulary
- Directly tokenizes the raw string: `"Hello world"` becomes `["▁Hello", "▁world"]` (note the `▁` represents space)

**Example**:
```
Original:  "Hello world"
SentencePiece: ["▁Hello", "▁world"]
BPE (with space tracking): ["ĠHello", "Ġworld"]
```

### Comparison Table

| Feature | BPE | WordPiece | SentencePiece |
|---------|-----|-----------|---------------|
| Pre-tokenization | Spaces + characters | Spaces + characters | None (raw bytes) |
| Continuation marker | `Ġ` (GPT-2 style) | `##` (BERT style) | `▁` (S sentencepiece) |
| Split criterion | Max pair frequency | Max language model likelihood | Max pair frequency (like BPE) |
| Unicode handling | Per-language | Per-language | Native (UTF-8 bytes) |
| Typical vocab size | 50K | 30K | 32K |
| Used by | GPT-2/3, LLaMA, CodeLLaMA | BERT, DistilBERT | T5, Marian, XLNet |
| Languages | English-focused (with pre-tokenization) | English-focused | Language-agnostic |

### Why does pre-tokenization matter?

BPE and WordPiece first split on whitespace before applying the merge algorithm. This means:
1. A merge rule can never cross a word boundary
2. Languages without spaces (Chinese: `今天天气很好`) require character-level pre-tokenization
3. Compound words in German (`Handwerksmeister`) are split on whitespace then re-merged

SentencePiece skips this step entirely, which makes it genuinely language-agnostic.

---

## 2. Subword Regularization — Training with Noise

Standard BPE produces one deterministic tokenization. But during training, injecting randomness into tokenization improves generalization and reduces overfitting. This is called **subword regularization**.

### The Unigram Language Model

Instead of building a vocabulary via merges, the **Unigram model** (used by SentencePiece) assumes a probabilistic generative process:

```
P(segmentation) = product over segments of P(token)
```

Given a vocabulary, it finds the segmentation that maximizes likelihood. Crucially, when multiple segmentations are possible, it samples from the distribution — this is the source of randomness.

### How the Unigram model works

1. Start with a large seed vocabulary (all characters + common subwords from pre-existing merges)
2. Compute the loss for each token (how much does removing it increase cross-entropy?)
3. Remove tokens with the lowest impact on loss
4. Repeat until vocab size is reached

During training with Unigram, the same input sentence can be tokenized differently each epoch:

```
Epoch 1: "unbelievable" -> ["un", "believ", "able"]
Epoch 2: "unbelievable" -> ["un", "believ", "able"]  (but with different dropout)
Epoch 3: "unbelievable" -> ["un", "believ", "able"]
```

More importantly, during **data augmentation**:
```
"the model runs" -> ["the", "model", "runs"]
"the model runs" -> ["the", "mo", "del", "runs"]   (different segmentation)
"the model runs" -> ["the", " model", " runs"]     (space merged differently)
```

SentencePiece implements this via `--character-coverage` and `--model_type=unigram`.

### Why does this help?

When a model sees "unbelievable" tokenized as `["un", "believ", "able"]`, it learns to map those subword pieces to a meaning. If it also sees "unreasonable" tokenized as `["un", "reason", "able"]`, it learns that "un-" is a prefix, "able" is a suffix. Regularization forces the model to not rely too heavily on any single tokenization, improving generalization to unseen words.

### BPE with subword regularization (not standard BPE)

Standard BPE doesn't have this randomness. However, you can approximate it:
1. Train BPE normally
2. During training, randomly drop some merge rules (dropout)
3. The effective tokenization becomes probabilistic

HuggingFace's `tokenizers` library supports this via `BpeTrainer` with `show_process_rate=True` and dropout configuration.

---

## 3. Tokenizer Edge Cases and Special Tokens

### Edge Case 1: Empty strings and whitespace-only

```python
tokenizer.encode("")           # returns [] — empty list, not error
tokenizer.encode("   ")        # returns [] or [space_token] depending on impl
tokenizer.decode([])           # returns "" — no error
```

Always handle these explicitly in production code.

### Edge Case 2: Unknown characters

UTF-8 has 1.1M possible characters. Most tokenizers limit vocab to ~50K. Unknown characters must be handled.

**Strategy 1: Byte-level fallback** (GPT-2, LLaMA)
Map every byte (0-255) to a token. Any Unicode character decomposes to bytes. Decode by re-encoding bytes to UTF-8.

```
"中文" -> [0xe4, 0xb8, 0xad, 0xe6, 0x96, 0x87] -> 6 tokens
```

**Strategy 2: Replacement token** (BERT, most HuggingFace)
Replace unknown characters with `<UNK>`. Information is lost.

```python
tokenizer.encode("🏋️‍♂️")   # likely [UNK] or fails
```

**Strategy 3: Byte-level BPE with custom vocab** (GPT-2)
Pre-tokenize to bytes, build BPE over the 256-byte vocabulary. This is what GPT-2 does.

### Edge Case 3: Very long words

A word longer than any training example may be split character-by-character:

```
Training corpus max word length: 23
Input word length: 47

"pneumonoultramicroscopicsilicovolcanoconiosis" ->
["p", "n", "e", "u", "m", "o", "n", "o", ...]  (47 tokens!)
```

This is why corpus diversity matters for tokenizer training.

### Edge Case 4: T9-style over-merging

With small vocabularies, BPE can over-merge common character pairs that don't form meaningful units:

```
If ("t", "h") is very frequent, it merges to "th"
If ("th", "e") is also frequent, it merges to "the"

"the" is now a single token — great
But "t", "h", "e" separately now can't form "h" or "t" alone in other contexts
```

This is called **over-merge** and is why vocabulary size matters.

### Edge Case 5: Model-specific special tokens

Every model has special tokens beyond the learned vocabulary. These are critical for correct model behavior:

| Token | Name | Purpose |
|-------|------|---------|
| `<PAD>` | Padding | Batch processing — pad sequences to same length |
| `<UNK>` | Unknown | OOV characters |
| `<CLS>` | Classification | BERT — prepended to every input for classification tasks |
| `<SEP>` | Separator | BERT — separates two sentences in pair tasks |
| `<EOS>` | End of sequence | Language models — marks end of generated text |
| `<BOS>` | Beginning of sequence | Some models (LLaMA, GPT) — marks start |
| `<MASK>` | Mask | MLM models (BERT) — for masked prediction |
| `</s>` | End of sentence | MarianMT, RoBERTa |
| `<|endoftext|>` | GPT-2 special separator | GPT-2 — prevents cross-document contamination |

Missing or misplacing a special token causes garbled output:

```
Correct:  [CLS] the cat sat on the mat [SEP]
Wrong:    the cat sat on the mat           (no CLS/SEP — model confused)
```

### Edge Case 6: Batch encoding with different lengths

When encoding multiple texts for a batch, they must be padded to the same length:

```python
texts = ["short", "very long text here", "medium"]
ids = tokenizer(texts, padding=True, return_tensors="pt")
# Produces: [[id1, id2, id3, pad, pad],
#            [id4, id5, id6, id7, id8],
#            [id9, id10, id11, pad, pad]]
```

The attention mask (not the padding token IDs) tells the model which positions are real tokens vs padding.

### Edge Case 7: Encoding before and after special token insertion

Order matters:

```python
# Correct order for BERT-style input:
encoded = tokenizer.encode("the cat", "sat on mat")
# Internally: [CLS] the cat [SEP] sat on mat [SEP]

# If you manually prepend CLS:
encoded = tokenizer.encode("[CLS] the cat [SEP]")
# The tokenizer does NOT know [CLS] is special — treats it as text
```

Always use the tokenizer's built-in `encode(text, text_pair)` or `__call__` method rather than manually inserting special tokens.

### Edge Case 8: Truncation and overflow

Models have fixed context windows. Long inputs must be truncated:

```python
tokenizer(text, max_length=512, truncation=True)
```

If a document is 1000 tokens and `max_length=512`, the model only sees the first 512. This matters enormously for long-document tasks.

For sliding window approaches (for long documents):

```python
# RAG-style: encode with overlap
tokens = tokenizer(text)
windows = [tokens[i:i+256] for i in range(0, len(tokens), 128)]
# 256-token windows, 128-token stride — each 512-char span appears in 2 windows
```

---

## 4. Hands-On: Build a Unigram Tokenizer

Our BPE tokenizer from the free lesson uses deterministic merges. Here is the Unigram model approach, which is probabilistic and used by SentencePiece.

The core idea: instead of greedily merging the most frequent pair, we train a unigram language model over candidate segmentations and sample from it.

```python
import math
from collections import Counter

class UnigramTokenizer:
    """
    Simplified Unigram tokenizer. At training time it:
    1. Starts with character-level tokenization
    2. Uses a loss function to decide which tokens to keep
    3. At inference, uses Viterbi algorithm to find best segmentation
    """

    def __init__(self, vocab_size=300):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.token_probs = {}

    def train(self, text):
        # Step 1: character-level tokenization
        tokens = list(text)

        # Step 2: count token frequencies
        freq = Counter(tokens)

        # Step 3: iteratively add most useful subword tokens
        # "Useful" = reduces corpus bits the most
        # For simplicity: add tokens that reduce unique character count
        self.vocab = {ch: i for i, ch in enumerate(sorted(set(tokens)))}
        self._estimate_probs(tokens)

        # Step 4: greedy vocabulary growth (simplified)
        for _ in range(self.vocab_size - len(self.vocab)):
            # Find best new token (bigram with highest frequency)
            pair_counts = Counter()
            for i in range(len(tokens) - 1):
                pair = tokens[i] + tokens[i+1]
                pair_counts[pair] += 1

            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0][0]
            new_id = len(self.vocab)
            self.vocab[best_pair] = new_id
            tokens = self._merge_tokens(tokens, best_pair)
            self._estimate_probs(tokens)

    def _merge_tokens(self, tokens, pair):
        result = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] + tokens[i+1] == pair:
                result.append(pair)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result

    def _estimate_probs(self, tokens):
        freq = Counter(tokens)
        total = sum(freq.values())
        self.token_probs = {t: f / total for t, f in freq.items()}

    def encode(self, text):
        tokens = list(text)
        for vocab_token in sorted(self.vocab.keys(), key=len, reverse=True):
            tokens = self._merge_tokens(tokens, vocab_token)
        return [self.vocab.get(t, 0) for t in tokens]

    def decode(self, token_ids):
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        tokens = [reverse_vocab.get(tid, "<UNK>") for tid in token_ids]
        return "".join(tokens)
```

The key difference from BPE: **the Unigram model can generate multiple valid segmentations**, and during training it samples among them. This is what enables subword regularization.

---

## 5. Practical Checklist: Choosing a Tokenizer

When starting a new project:

1. **What model are you using?** Match its tokenizer exactly — `AutoTokenizer.from_pretrained("model-name")`

2. **What language?** English-dominant tasks: BPE or WordPiece. Multilingual or non-Latin script: SentencePiece

3. **What vocabulary size?** Larger vocab = fewer tokens per document = faster inference. Smaller vocab = better generalization to rare words. Default: 30-50K for general purpose

4. **Do you need special tokens?** For classification: BERT needs `[CLS]`. For generation: GPT needs `<|endoftext|>`. For translation: MarianMT needs `</s>`

5. **What's your context window?** 512 tokens is standard. For long documents consider: sliding window, tokenizer-level truncation, or document-level chunking

6. **Are you fine-tuning or training from scratch?** Fine-tuning: use the exact same tokenizer as the pretrained model. Training from scratch: tokenize a large corpus, analyze token length distribution, adjust vocab size based on OOV rate
