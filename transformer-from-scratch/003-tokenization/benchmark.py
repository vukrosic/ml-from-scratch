"""Benchmark encode/decode speed: our BPE vs HuggingFace tokenizers."""


# =============================================================================
# Why benchmark tokenizers?
# =============================================================================
# Tokenization happens every time you run inference or training. For a model
# processing millions of documents, a 2x speedup in tokenization saves hours.
# This script measures that difference.

import time
import random
import string


def random_text(num_words=100, seed=42):
    """Generate random text for benchmarking."""
    random.seed(seed)
    words = ["".join(random.choices(string.ascii_letters + " ", k=random.randint(1, 12)))
             for _ in range(num_words)]
    return " ".join(words).strip()


def benchmark_encode(tokenizer, text, steps=500, warmup=50):
    """Benchmark encoding speed. Returns average ms per call."""
    for _ in range(warmup):
        tokenizer.encode(text)

    t0 = time.perf_counter()
    for _ in range(steps):
        tokenizer.encode(text)
    elapsed = time.perf_counter() - t0
    return elapsed / steps * 1000


def benchmark_decode(tokenizer, token_ids, steps=500, warmup=50):
    """Benchmark decoding speed. Returns average ms per call."""
    for _ in range(warmup):
        tokenizer.decode(token_ids)

    t0 = time.perf_counter()
    for _ in range(steps):
        tokenizer.decode(token_ids)
    elapsed = time.perf_counter() - t0
    return elapsed / steps * 1000


if __name__ == "__main__":
    from tokenizers import SimpleBPETokenizer

    # Test corpus — larger for more accurate timing
    corpus = random_text(num_words=500, seed=0) * 10

    # Train our tokenizer
    our_tokenizer = SimpleBPETokenizer(vocab_size=500)
    our_tokenizer.train(corpus)

    # Load HuggingFace tokenizer
    from transformers import AutoTokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Encode a fresh text sample for benchmarking
    sample = random_text(num_words=200, seed=123)
    our_ids = our_tokenizer.encode(sample)
    hf_ids = hf_tokenizer.encode(sample, return_tensors="pt").tolist()[0]

    print(f"Test text length: {len(sample)} chars")
    print(f"Our tokens: {len(our_ids)}, HF tokens: {len(hf_ids)}")
    print()
    print(f"{'Tokenizer':<20} {'Encode (ms)':<15} {'Decode (ms)':<15}")
    print("-" * 50)

    # Benchmark our tokenizer
    enc_time = benchmark_encode(our_tokenizer, sample)
    dec_time = benchmark_decode(our_tokenizer, our_ids)
    print(f"{'Our BPE':<20} {enc_time:<15.4f} {dec_time:<15.4f}")

    # Benchmark HuggingFace
    enc_time_hf = benchmark_encode(hf_tokenizer, sample)
    dec_time_hf = benchmark_decode(hf_tokenizer, hf_ids)
    print(f"{'HF bert-base':<20} {enc_time_hf:<15.4f} {dec_time_hf:<15.4f}")

    print()
    if enc_time_hf > 0:
        print(f"Speedup: our encoder is {enc_time_hf / enc_time:.1f}x {('faster' if enc_time < enc_time_hf else 'slower')} than HuggingFace")
    if dec_time_hf > 0:
        print(f"Speedup: our decoder is {dec_time_hf / dec_time:.1f}x {('faster' if dec_time < dec_time_hf else 'slower')} than HuggingFace")
