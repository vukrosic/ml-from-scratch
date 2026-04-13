# Tokenization: Advanced Topics

Beyond BPE basics — the algorithms, failure modes, and engineering tricks that determine how well your model handles real-world text.

The core lesson covers byte-pair encoding from scratch. This companion explores alternative tokenization algorithms (Unigram, WordPiece, byte-level), multilingual efficiency problems, adversarial attacks on tokenizers, training custom tokenizers, and token healing at generation boundaries.

---

## 1. Top-down vs bottom-up: Unigram vs BPE

BPE builds a vocabulary **bottom-up**: start with characters, repeatedly merge the most frequent pair. Unigram works **top-down**: start with a huge vocabulary, prune the least useful tokens.

```
BPE (bottom-up):
  Characters: [t, h, e, _, c, a, t]
  Step 1: merge (t,h) -> [th, e, _, c, a, t]
  Step 2: merge (c,a) -> [th, e, _, ca, t]
  Step 3: merge (th,e) -> [the, _, ca, t]
  ...keep merging until vocab_size reached

Unigram (top-down):
  Start: [the, th, he, t, h, e, cat, ca, at, c, a, ...]  (huge seed vocab)
  Step 1: score each token by likelihood loss if removed
  Step 2: remove bottom 20% least useful tokens
  Step 3: repeat until vocab_size reached
```

### Unigram model: the math

Unigram treats tokenization as a probabilistic model. Each token has a probability, and the best segmentation of a sentence maximizes the product of token probabilities.

```python
import math
from collections import defaultdict

class UnigramTokenizer:
    def __init__(self, vocab_with_scores):
        """vocab_with_scores: dict of {token: log_probability}"""
        self.vocab = vocab_with_scores

    def tokenize(self, text):
        """Viterbi algorithm: find highest-probability segmentation."""
        n = len(text)
        # best_score[i] = best log-prob for text[:i]
        best_score = [-float('inf')] * (n + 1)
        best_score[0] = 0.0
        best_edge = [None] * (n + 1)  # backpointer

        for end in range(1, n + 1):
            for start in range(end):
                token = text[start:end]
                if token in self.vocab:
                    score = best_score[start] + self.vocab[token]
                    if score > best_score[end]:
                        best_score[end] = score
                        best_edge[end] = start

        # Backtrack to recover tokens
        tokens = []
        pos = n
        while pos > 0:
            start = best_edge[pos]
            tokens.append(text[start:pos])
            pos = start
        return list(reversed(tokens))
```

The key advantage: Unigram can output **multiple possible segmentations** with probabilities, which acts as data augmentation during training. SentencePiece calls this "subword regularization."

```python
# During training, sample different segmentations
def sample_tokenization(self, text, temperature=1.0, n_best=10):
    """Sample from n-best segmentations instead of always using Viterbi best."""
    segmentations = self._nbest_viterbi(text, n_best)
    # Weight by probability, sample one
    probs = [math.exp(score / temperature) for _, score in segmentations]
    total = sum(probs)
    probs = [p / total for p in probs]
    idx = random.choices(range(len(probs)), weights=probs, k=1)[0]
    return segmentations[idx][0]
```

### Unigram vocabulary pruning

SentencePiece's Unigram training loop:

```
1. Seed vocabulary: all substrings up to max length (typically 16 chars)
   from the corpus, filtered to top ~1M candidates by frequency
2. Run EM algorithm to assign probabilities to each token
3. For each token, compute: "how much would total corpus likelihood
   decrease if we removed this token?"
4. Sort by loss impact. Remove bottom 20% (keep characters always)
5. Repeat from step 2 until vocab_size reached
```

```python
def compute_removal_loss(token, corpus_segmentations, vocab):
    """How much worse does the corpus get if we remove this token?"""
    loss = 0.0
    for sentence in corpus_segmentations:
        if token in sentence.tokens:
            # Re-segment without this token
            reduced_vocab = {k: v for k, v in vocab.items() if k != token}
            old_score = sentence.log_prob
            new_score = viterbi_score(sentence.text, reduced_vocab)
            loss += old_score - new_score  # always negative (worse)
    return loss
```

Tokens that are easily replaced by sub-tokens have low removal cost and get pruned first. Tokens that are the only way to represent some text have high removal cost and survive.

---

## 2. WordPiece: BERT's tokenizer

WordPiece looks similar to BPE but uses a different merge criterion. Instead of merging the most frequent pair, it merges the pair that maximizes the language model likelihood.

```
BPE merge score:     count(AB)
WordPiece merge:     count(AB) / (count(A) * count(B))
```

This is essentially pointwise mutual information (PMI). A pair that occurs together more than chance would predict gets a high score.

```python
def wordpiece_merge_score(pair, pair_counts, token_counts, corpus_size):
    """WordPiece uses PMI-like scoring instead of raw frequency."""
    a, b = pair
    count_ab = pair_counts[pair]
    count_a = token_counts[a]
    count_b = token_counts[b]
    # Likelihood ratio: observed co-occurrence vs expected under independence
    return count_ab / (count_a * count_b)

# BPE would just do:
def bpe_merge_score(pair, pair_counts):
    return pair_counts[pair]
```

WordPiece also uses the `##` prefix convention to mark continuation subwords:

```
BPE:       "unhappiness" -> ["un", "happiness"]
WordPiece: "unhappiness" -> ["un", "##happi", "##ness"]
```

The `##` prefix tells you "this token continues the previous word" — useful for tasks like named entity recognition where word boundaries matter.

```python
def wordpiece_tokenize(word, vocab, max_token_len=200):
    """Greedy left-to-right longest match (BERT's actual algorithm)."""
    tokens = []
    start = 0
    while start < len(word):
        end = min(start + max_token_len, len(word))
        found = False
        while end > start:
            substr = word[start:end]
            if start > 0:
                substr = "##" + substr
            if substr in vocab:
                tokens.append(substr)
                found = True
                break
            end -= 1
        if not found:
            tokens.append("[UNK]")
            start += 1
        else:
            start = end
    return tokens
```

Important: BERT's WordPiece uses **greedy left-to-right matching** at inference time, not Viterbi. This is faster but can produce suboptimal segmentations compared to Unigram.

---

## 3. Byte-level models: ByT5 and byte fallback

Instead of learning subword tokens, just feed raw bytes. Every possible input is representable with a vocabulary of 259 tokens (256 bytes + 3 special tokens).

```
Token-based:  "Hello" -> [15496]                     # 1 token
Byte-level:   "Hello" -> [72, 101, 108, 108, 111]    # 5 tokens (ASCII codes)

Token-based:  "こんにちは" -> [46036, 230, 109]       # 3 tokens (maybe)
Byte-level:   "こんにちは" -> [227,129,147, 227,...]  # 15 tokens (3 bytes each, UTF-8)
```

### ByT5 architecture

ByT5 compensates for longer sequences with an asymmetric architecture:

```
Standard T5:    12 encoder layers, 12 decoder layers
ByT5:           12 encoder layers,  4 decoder layers  (fewer decoder layers)
                + encoder downsampling (local attention + stride)

Sequence length comparison for same text:
  T5 tokenized:    128 tokens
  ByT5 bytes:      512 bytes   (4x longer on English, 6-8x on CJK)
```

```python
# ByT5-style local attention downsampling
def byte_encoder_block(byte_embeddings, block_size=4):
    """Process bytes in local windows, then pool to reduce length."""
    B, L, D = byte_embeddings.shape
    # Reshape into blocks
    blocks = byte_embeddings.view(B, L // block_size, block_size, D)
    # Local attention within each block (cheap, no quadratic blowup)
    local_out = local_self_attention(blocks)  # (B, L//block_size, block_size, D)
    # Pool each block to single vector
    pooled = local_out.mean(dim=2)  # (B, L//block_size, D)
    # Now run global attention on the shorter sequence
    return global_self_attention(pooled)
```

### SentencePiece byte fallback

Modern tokenizers like LLaMA's use a hybrid: Unigram/BPE for known tokens, byte fallback for unknown characters.

```python
# LLaMA tokenizer byte fallback
# Vocabulary includes: regular tokens + 256 byte tokens like <0x00>...<0xFF>
def tokenize_with_byte_fallback(text, sp_model):
    tokens = sp_model.encode(text)
    result = []
    for token_id in tokens:
        if token_id == UNK_ID:
            # Fall back to raw bytes
            piece = sp_model.id_to_piece(token_id)
            for byte in piece.encode('utf-8'):
                result.append(BYTE_TOKEN_OFFSET + byte)  # <0x00> through <0xFF>
        else:
            result.append(token_id)
    return result
```

This eliminates the `[UNK]` problem entirely. Any UTF-8 text can be represented, even if the tokenizer never saw it during training.

---

## 4. Multilingual token costs

Tokenizers trained primarily on English are brutally inefficient for other languages. The same semantic content requires vastly different token counts.

```
Sentence: "The cat sat on the mat." (6 words, simple English)

GPT-4 tokenizer (cl100k_base):
  English:     "The cat sat on the mat."        ->  7 tokens
  Chinese:     "猫坐在垫子上。"                    -> 11 tokens
  Japanese:    "猫がマットの上に座った。"            -> 14 tokens
  Arabic:      "جلست القطة على الحصيرة."          -> 15 tokens
  Hindi:       "बिल्ली चटाई पर बैठ गई।"            -> 22 tokens

Same meaning, 3x more tokens for Hindi than English.
```

This has real costs:

```
Impact of tokenizer inefficiency:
  1. API cost:        Hindi users pay ~3x more per message
  2. Context window:  Hindi gets ~3x less content in same window
  3. Speed:           3x more tokens = 3x slower generation
  4. Training:        Model sees ~3x fewer Hindi "concepts" per batch
```

```python
# Measure tokenizer fertility (tokens per word) across languages
import tiktoken

def measure_fertility(text, lang_name, encoding_name="cl100k_base"):
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    words = text.split()
    fertility = len(tokens) / max(len(words), 1)
    chars_per_token = len(text) / max(len(tokens), 1)
    print(f"{lang_name:12s}: {len(tokens):3d} tokens, "
          f"fertility={fertility:.2f}, chars/token={chars_per_token:.1f}")
    return len(tokens)
```

### Code tokenization

Code has its own efficiency quirks. Indentation and boilerplate eat tokens:

```
Python:
  "    def forward(self, x):\n"           -> 8 tokens
  "        return self.linear(x)\n"       -> 7 tokens
  (4-space indent = 1 token, 8-space = 1-2 tokens in cl100k)

Whitespace-heavy languages (Python, YAML) use more tokens than
brace languages (C, Rust) for the same logic.
```

---

## 5. Tokenizer attacks and glitch tokens

Every tokenizer has "glitch tokens" — tokens that exist in the vocabulary but were poorly learned during training because they appeared in degenerate contexts.

### The SolidGoldMagikarp phenomenon

In early GPT models, researchers found tokens like `SolidGoldMagikarp` (a Reddit username that appeared in the training data for the tokenizer but was filtered from the LLM training data). Prompting the model with these tokens caused bizarre behavior:

```
Token ID 39802: " SolidGoldMagikarp"
- Tokenizer learned it from Reddit scrape
- LLM training data filtered Reddit usernames
- Result: token embedding was barely trained
- Model behavior when prompted: hallucination, refusal, gibberish
```

```python
# Finding potential glitch tokens
def find_undertrained_tokens(model, tokenizer, threshold=0.1):
    """Tokens with very low embedding norm may be undertrained."""
    embeddings = model.get_input_embeddings().weight  # (vocab_size, d_model)
    norms = embeddings.norm(dim=1)
    mean_norm = norms.mean()

    glitch_candidates = []
    for token_id in range(len(tokenizer)):
        if norms[token_id] < mean_norm * threshold:
            token_str = tokenizer.decode([token_id])
            glitch_candidates.append((token_id, token_str, norms[token_id].item()))

    return sorted(glitch_candidates, key=lambda x: x[2])
```

### Adversarial tokenization attacks

Attackers can exploit tokenizer behavior to bypass safety filters:

```
Safety filter checks for: "how to build a bomb"
Tokenized normally: ["how", " to", " build", " a", " bomb"]  -> BLOCKED

Attack 1: Unicode lookalikes
  "how to build a bоmb"  (Cyrillic "о" instead of Latin "o")
  Tokenized: ["how", " to", " build", " a", " b", "о", "mb"]  -> PASSES

Attack 2: Zero-width characters
  "how to build a b\u200bomb"  (zero-width space)
  Tokenized: different token IDs entirely  -> PASSES

Attack 3: Token boundary manipulation
  "how to buil" + "d a bo" + "mb"  (split across messages)
  Each fragment is innocuous  -> PASSES
```

```python
# Defense: normalize text before tokenization
import unicodedata

def normalize_for_safety(text):
    """Normalize unicode to catch lookalike attacks."""
    # NFKC normalization: maps lookalikes to canonical forms
    text = unicodedata.normalize("NFKC", text)
    # Remove zero-width characters
    text = ''.join(c for c in text if unicodedata.category(c) != 'Cf')
    # Remove variation selectors
    text = ''.join(c for c in text if not (0xFE00 <= ord(c) <= 0xFE0F))
    return text
```

---

## 6. Training a custom tokenizer

When to train your own tokenizer:
- Domain-specific vocabulary (legal, medical, code)
- Non-English-dominant data
- Efficiency gains on your specific data distribution

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

def train_bpe_tokenizer(files, vocab_size=32000, save_path="tokenizer.json"):
    """Train a BPE tokenizer from scratch using HuggingFace tokenizers."""

    # Initialize with byte-level BPE (like GPT-2)
    tokenizer = Tokenizer(models.BPE())

    # Pre-tokenization: split on whitespace and punctuation first
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder must match pre-tokenizer
    tokenizer.decoder = decoders.ByteLevel()

    # Training configuration
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,           # token must appear at least twice
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Train on files
    tokenizer.train(files, trainer=trainer)
    tokenizer.save(save_path)
    return tokenizer

# Train on your domain data
tokenizer = train_bpe_tokenizer(
    files=["corpus_part1.txt", "corpus_part2.txt"],
    vocab_size=32000
)
```

### Training a SentencePiece Unigram model

```python
import sentencepiece as spm

def train_sentencepiece(input_file, vocab_size=32000, model_type="unigram"):
    """Train SentencePiece tokenizer (used by LLaMA, T5, etc.)."""
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix="sp_tokenizer",
        vocab_size=vocab_size,
        model_type=model_type,        # "unigram" or "bpe"
        character_coverage=0.9995,     # keep 99.95% of characters
        byte_fallback=True,            # use byte tokens for OOV
        split_digits=True,             # each digit is a separate token
        num_threads=16,
        max_sentence_length=16384,
        shuffle_input_sentence=True,
        # Normalization
        normalization_rule_name="identity",  # no normalization (modern default)
        remove_extra_whitespaces=False,
        # Special tokens
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )

    # Load and test
    sp = spm.SentencePieceProcessor(model_file="sp_tokenizer.model")
    print(sp.encode("Hello world!", out_type=str))
    # Subword regularization: sample different segmentations
    for _ in range(5):
        print(sp.encode("Hello world!", out_type=str,
                        enable_sampling=True, alpha=0.1, nbest_size=-1))
    return sp
```

### Vocabulary size trade-offs

```
Vocab size    Pros                              Cons
─────────────────────────────────────────────────────────────────
   256        No tokenizer needed (bytes)       Sequences 4-6x longer
  4,000       Small embedding table             Poor compression
 32,000       Good balance (LLaMA, Mistral)     Standard choice
 50,257       GPT-2 default                     Slightly large
100,256       GPT-4 (cl100k_base)               Better multilingual
250,000+      BLOOM, multilingual models         Huge embedding table

Embedding table memory at d_model=4096:
  32K vocab:   32,000 * 4,096 * 2 bytes = 250 MB  (fp16)
  100K vocab: 100,000 * 4,096 * 2 bytes = 781 MB  (fp16)
  250K vocab: 250,000 * 4,096 * 2 bytes = 1.9 GB  (fp16)
```

---

## 7. Token healing at generation boundaries

When you split a prompt and a completion, the boundary can create tokenization artifacts.

```
Full text:     "The quick brown fox"
Tokenized:     ["The", " quick", " brown", " fox"]

But if we split generation at "brown":
  Prompt:      "The quick brown"   -> ["The", " quick", " brown"]
  Completion:  " fox"              -> [" fox"]
  Concatenated tokens: ["The", " quick", " brown", " fox"]  ✓ OK here

Problem case:
  Prompt:      "https://www."      -> ["https", "://", "www", "."]
  Completion:  "google.com"        -> ["google", ".", "com"]
  Concatenated: ["https", "://", "www", ".", "google", ".", "com"]  (7 tokens)
  
  But tokenizing the full string:
  "https://www.google.com" -> ["https", "://", "www.google", ".", "com"]  (5 tokens)

The boundary created DIFFERENT tokenization than the full string would have.
```

This matters because the model was trained on the full-string tokenization. At the boundary, it sees token sequences it never encountered during training.

### Token healing implementation

```python
def heal_tokens(prompt_tokens, tokenizer, n_backup=1):
    """Back up n tokens from the prompt end to let the model re-tokenize
    across the boundary.
    
    Used by guidance, llama.cpp, and other frameworks.
    """
    if len(prompt_tokens) < n_backup:
        return prompt_tokens, ""

    # Decode the last n tokens back to text
    backup_tokens = prompt_tokens[-n_backup:]
    backup_text = tokenizer.decode(backup_tokens)

    # Remove backup tokens from prompt
    healed_prompt_tokens = prompt_tokens[:-n_backup]

    # The backup text becomes a prefix constraint on generation:
    # the model must first generate tokens that decode to backup_text,
    # then continue freely
    return healed_prompt_tokens, backup_text

def constrained_generate(model, prompt_tokens, tokenizer, max_new_tokens=100):
    """Generate with token healing."""
    healed_prompt, prefix_constraint = heal_tokens(prompt_tokens, tokenizer)

    # Force-generate tokens matching the prefix constraint first
    generated = list(healed_prompt)
    constraint_remaining = prefix_constraint

    for _ in range(max_new_tokens):
        logits = model(torch.tensor([generated]))[:, -1, :]

        if constraint_remaining:
            # Mask logits to only allow tokens consistent with constraint
            for token_id in range(len(tokenizer)):
                token_text = tokenizer.decode([token_id])
                if not constraint_remaining.startswith(token_text[:len(constraint_remaining)]):
                    logits[0, token_id] = -float('inf')

        next_token = logits.argmax(dim=-1).item()
        generated.append(next_token)

        if constraint_remaining:
            token_text = tokenizer.decode([next_token])
            constraint_remaining = constraint_remaining[len(token_text):]

        if next_token == tokenizer.eos_token_id:
            break

    return generated
```

---

## 8. Tokenizer benchmarking

How to compare tokenizers on your data:

```python
import time
import tiktoken
import sentencepiece as spm

def benchmark_tokenizer(tokenizer_name, texts, encode_fn, decode_fn):
    """Benchmark speed, compression, and roundtrip fidelity."""
    total_tokens = 0
    total_chars = 0
    total_bytes = 0
    encode_time = 0
    decode_time = 0
    roundtrip_failures = 0

    for text in texts:
        total_chars += len(text)
        total_bytes += len(text.encode('utf-8'))

        t0 = time.perf_counter()
        tokens = encode_fn(text)
        t1 = time.perf_counter()
        decoded = decode_fn(tokens)
        t2 = time.perf_counter()

        total_tokens += len(tokens)
        encode_time += t1 - t0
        decode_time += t2 - t1

        if decoded != text:
            roundtrip_failures += 1

    print(f"Tokenizer: {tokenizer_name}")
    print(f"  Compression: {total_bytes / total_tokens:.1f} bytes/token")
    print(f"  Fertility:   {total_tokens / len(texts):.1f} tokens/sample")
    print(f"  Encode:      {total_chars / encode_time / 1e6:.1f} M chars/sec")
    print(f"  Decode:      {total_tokens / decode_time / 1e6:.1f} M tokens/sec")
    print(f"  Roundtrip failures: {roundtrip_failures}/{len(texts)}")
```

Key metrics to track:
- **Bytes per token** (compression ratio): higher is better, 3.5-4.5 typical for English
- **Fertility** (tokens per word): lower is better, 1.3-1.5 for English
- **Encoding speed**: production tokenizers should do 1M+ chars/sec
- **Roundtrip fidelity**: encode then decode must recover original text exactly

---

## 9. Practical checklist

```
Choosing a tokenizer for your project:

[ ] English-only or multilingual?
    -> Multilingual: larger vocab (64K+), train on balanced corpus
    -> English-only: 32K vocab is fine

[ ] What domains?
    -> Code: ensure whitespace tokens, split_digits=True
    -> Medical/legal: consider domain-specific training

[ ] BPE vs Unigram?
    -> BPE: deterministic, widely supported, GPT/LLaMA-style
    -> Unigram: subword regularization, slightly better for small data

[ ] Byte fallback?
    -> Always yes for modern models. Eliminates [UNK] tokens entirely.

[ ] Vocab size?
    -> 32K for standard models
    -> 64-100K for multilingual
    -> 256 for byte-level (ByT5-style, research only)

[ ] Special tokens needed?
    -> Chat: <|im_start|>, <|im_end|>, role markers
    -> Code: <|fim_prefix|>, <|fim_middle|>, <|fim_suffix|> for infilling
    -> Tool use: function call delimiters
```

---

Get the video walkthrough: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
