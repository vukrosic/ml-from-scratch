"""Compare our BPE tokenizer against HuggingFace tokenizers on the same text."""


# =============================================================================
# Compare token counts between our tokenizer and a HuggingFace tokenizer
# =============================================================================
# This shows why tokenizer choice matters: the same sentence gets broken into
# very different numbers of tokens depending on the algorithm and vocabulary size.

from tokenizers import SimpleBPETokenizer


def load_hf_tokenizer(name="bert-base-uncased"):
    """Load a pretrained HuggingFace tokenizer by name."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def compare_tokenizers(text, our_vocab_size=300, hf_name="bert-base-uncased"):
    """Compare token counts and actual tokens between our BPE and a HuggingFace tokenizer."""
    # Train our tokenizer on the same text
    our_tokenizer = SimpleBPETokenizer(vocab_size=our_vocab_size)
    our_tokenizer.train(text)

    # Load the HuggingFace tokenizer
    hf_tokenizer = load_hf_tokenizer(hf_name)

    # Encode with both
    our_ids = our_tokenizer.encode(text)
    hf_ids = hf_tokenizer.encode(text, return_tensors="pt").tolist()[0]

    print(f"Text: {repr(text[:80])}...")
    print(f"---")
    print(f"Our BPE (vocab={our_vocab_size}):")
    print(f"  Token count: {len(our_ids)}")
    print(f"  Tokens: {our_ids[:20]}...")
    print(f"---")
    print(f"HuggingFace {hf_name}:")
    print(f"  Token count: {len(hf_ids)}")
    print(f"  First 20 tokens: {hf_ids[:20]}")
    print(f"---")
    print(f"Ratio (HF / Ours): {len(hf_ids) / len(our_ids):.2f}x")

    # Show vocabulary size comparison
    print(f"\nVocab sizes:")
    print(f"  Our vocab: {len(our_tokenizer.vocab)}")
    print(f"  HF vocab: {hf_tokenizer.vocab_size}")


if __name__ == "__main__":
    test_texts = [
        "tokenization is the process of converting text into tokens",
        "the quick brown fox jumps over the lazy dog",
        "language models like BERT and GPT use subword tokenization",
        "analyzing tokenizer performance requires understanding vocabulary size",
    ]

    print("=" * 60)
    print("Comparing our BPE tokenizer vs HuggingFace bert-base-uncased")
    print("=" * 60)

    for text in test_texts:
        compare_tokenizers(text, our_vocab_size=300, hf_name="bert-base-uncased")
        print()
