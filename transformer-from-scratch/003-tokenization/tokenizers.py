"""Simple BPE tokenizer built from scratch — train, encode, decode."""


# =============================================================================
# Part 1: Character-level baseline
# =============================================================================
# Why start here? To understand WHY BPE exists. Character-level tokenization
# means every byte is its own token. It's simple but produces very long sequences.
# A 10-word sentence becomes 50+ tokens. BPE fixes this.

# We use a special marker for word-boundary spaces so we can distinguish
# them from spaces that are part of merged tokens. SPACE_CHAR is inserted
# between words during char_tokenize() and replaced with a normal space
# during decode(), so we can recover word boundaries.
SPACE_CHAR = " "


def char_tokenize(text):
    """Split text into individual characters with word-boundary spaces tracked."""
    # We replace the normal space " " between words with a special SPACE_CHAR.
    # This lets us recover word boundaries during decoding.
    chars = []
    words = text.split(" ")
    for i, word in enumerate(words):
        if i > 0:
            chars.append(SPACE_CHAR)   # word boundary marker
        for ch in word:
            chars.append(ch)
    return chars


# =============================================================================
# Part 2: Count adjacent pairs
# =============================================================================
# BPE works by finding the most common adjacent pair of tokens and merging them.
# This function counts how many times each pair appears in a sequence of tokens.

from collections import defaultdict


def get_pair_counts(tokens):
    """Count how many times each adjacent pair appears in the token list."""
    counts = defaultdict(int)
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        counts[pair] += 1
    return counts


# =============================================================================
# Part 3: Merge the most frequent pair
# =============================================================================
# Given a pair like ("t", "o"), we merge it everywhere it appears by replacing
# both tokens with a single combined token "to". The merged token gets a new ID.

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


# =============================================================================
# Part 4: The BPE training loop
# =============================================================================
# Training BPE means building a sequence of merge operations.
# Start with character-level tokens, repeatedly find the best pair and merge.
# The result is a vocabulary (base chars + merged tokens) and a list of merges.

class SimpleBPETokenizer:
    def __init__(self, vocab_size=300):
        self.vocab_size = vocab_size
        self.vocab = {}      # token string -> token ID
        self.merges = []    # list of (pair, merged_token) in order

    def train(self, text):
        """Train the tokenizer on a corpus of text. Builds vocab and merges."""
        # Step 1: start with just special tokens and the characters that appear
        # in the corpus. This keeps the starting vocab small so we have room
        # for many merges (which are the actual "learned" vocabulary).
        special_tokens = {"<PAD>": 0, "<UNK>": 1}
        unique_chars = sorted(set(char_tokenize(text)))
        current_vocab = dict(special_tokens)
        for ch in unique_chars:
            if ch not in current_vocab:
                current_vocab[ch] = len(current_vocab)

        # Start with character-level tokenization
        tokens = char_tokenize(text)

        # Build vocabulary by repeatedly merging the most frequent adjacent pair
        next_id = len(current_vocab)
        reverse_vocab = {v: k for k, v in current_vocab.items()}
        merges_to_do = self.vocab_size - len(current_vocab)

        for _ in range(merges_to_do):
            pair_counts = get_pair_counts(tokens)
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            merged_str = best_pair[0] + best_pair[1]

            # Assign new token ID for the merged string
            current_vocab[merged_str] = next_id
            reverse_vocab[next_id] = merged_str
            next_id += 1

            # Record the merge operation
            self.merges.append((best_pair, merged_str))

            # Apply the merge to the entire corpus
            tokens = merge_pair(tokens, best_pair, merged_str)

        self.vocab = current_vocab
        self.reverse_vocab = reverse_vocab

    def encode(self, text):
        """Convert text into a list of token IDs using learned merges."""
        tokens = char_tokenize(text)
        # Apply each merge rule in order
        for pair, merged_str in self.merges:
            tokens = merge_pair(tokens, pair, merged_str)
        # Convert to IDs
        return [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]

    def decode(self, token_ids):
        """Convert token IDs back into text string."""
        tokens = [self.reverse_vocab.get(tid, "<UNK>") for tid in token_ids]
        text = "".join(tokens)
        # Replace our word-boundary marker with a normal space to reconstruct the text
        return text.replace(SPACE_CHAR, " ")


if __name__ == "__main__":
    corpus = (
        "tokenization is the process of converting text into tokens. "
        "a good tokenizer splits text into pieces that capture meaning well. "
        "subword tokenization is used in modern language models like BERT and GPT. "
        "byte pair encoding is a simple and effective algorithm for this task."
    )

    tokenizer = SimpleBPETokenizer(vocab_size=300)
    tokenizer.train(corpus)

    print(f"Vocab size: {len(tokenizer.vocab)}")
    print(f"Number of merges: {len(tokenizer.merges)}")

    # Show first few merges (these are the most frequent subwords learned)
    print("\nFirst 10 merges:")
    for pair, merged in tokenizer.merges[:10]:
        print(f"  {pair} -> {repr(merged)}")

    # Encode and decode a test sentence
    test = "tokenization is powerful"
    encoded = tokenizer.encode(test)
    decoded = tokenizer.decode(encoded)

    print(f"\nTest: {repr(test)}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {repr(decoded)}")
    print(f"Round-trip OK: {test == decoded}")
