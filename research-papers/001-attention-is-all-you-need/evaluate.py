"""
evaluate.py — BLEU score computation for the translation model.

We implement a simple n-gram BLEU score (the same metric used in the paper's
Table 2) so the user can measure translation quality without any external
dependencies.  Both a from-scratch implementation and a brief example using
the `nltk` library (which is the standard approach) are provided.

BLEU (Bilingual Evaluation Understudy) score:
  - Computes the geometric mean of modified n-gram precision scores
  - Penalises translations that are too short compared to the reference
  - Range [0, 1] — higher is better; commonly reported as a percentage (0-100)

The paper's Table 2 reports BLEU scores on WMT English->German and
English->French translation tasks.  Our tiny dataset will score much lower
(since the model and data are toy-scale), but the metric still correctly
ranks better models vs worse models.
"""

import math
import torch

from model import Transformer, make_causal_mask, make_src_mask


# ---------------------------------------------------------------------------
# 1.  From-scratch BLEU implementation
# ---------------------------------------------------------------------------
def ngram_count(sequence, n):
    """Count n-grams in a sequence.  Returns a dict {ngram: count}."""
    ngrams = []
    seq = tuple(sequence)
    for i in range(len(seq) - n + 1):
        ngrams.append(seq[i:i + n])
    from collections import Counter
    return Counter(ngrams)


def modified_precision(reference, hypothesis, n):
    """
    Modified n-gram precision (Breu et al., 2002).

    Count how many n-grams in the hypothesis also appear in the reference,
    clipped by the reference's n-gram count.
    """
    ref_ngrams  = ngram_count(reference, n)
    hyp_ngrams  = ngram_count(hypothesis, n)

    total_clip = 0
    for ngram, count in hyp_ngrams.items():
        total_clip += min(count, ref_ngrams.get(ngram, 0))
    return total_clip / max(1, sum(hyp_ngrams.values()))


def brevity_penalty(reference, hypothesis):
    """
    BP = exp(min(1 - r/c, 0))
    where r = len(reference), c = len(hypothesis)
    """
    r = len(reference)
    c = len(hypothesis)
    if c == 0:
        return 0.0
    return math.exp(min(1.0 - r / c, 0.0))


def sentence_bleu(reference, hypothesis, max_n=4):
    """
    Compute BLEU for a single sentence pair using geometric mean of
    clipped precisions across n-gram orders 1..max_n.
    """
    bp = brevity_penalty(reference, hypothesis)
    p_log_sum = 0.0
    for n in range(1, max_n + 1):
        p = modified_precision(reference, hypothesis, n)
        # Avoid log(0); use a small floor
        p = max(p, 1e-10)
        p_log_sum += math.log(p)
    return bp * math.exp(p_log_sum / max_n)


def corpus_bleu(references, hypotheses, max_n=4):
    """
    Compute BLEU over a whole corpus (list of sentence pairs).
    Returns the geometric mean of per-sentence BLEU scores.
    """
    assert len(references) == len(hypotheses)
    scores = [
        sentence_bleu(ref, hyp, max_n)
        for ref, hyp in zip(references, hypotheses)
    ]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# 2.  Greedy decoding (for use in evaluation)
# ---------------------------------------------------------------------------
def greedy_decode(model, src, src_vocab, tgt_vocab, max_len=50, device="cpu"):
    """
    Simple greedy decoding: at each step, pick the most likely next token.

    model: trained Transformer
    src:   (1, src_len) tensor of source token indices
    Returns: decoded string
    """
    model.eval()
    with torch.no_grad():
        src = src.to(device)
        src_mask = make_src_mask(src.shape[1], device).to(device)

        # Encode the source sentence
        enc_output = model.encode(src, src_mask)

        # Start with SOS token
        decoded_ids = [src_vocab.stoi["<sos>"]]
        eos_id = src_vocab.stoi["<eos>"]

        for _ in range(max_len):
            tgt = torch.tensor([[decoded_ids[-1]]], dtype=torch.long, device=device)
            tgt_mask = make_causal_mask(1, device).to(device)

            logits = model.decode(tgt, enc_output, tgt_mask=tgt_mask)
            next_token = logits.argmax(dim=-1).item()
            decoded_ids.append(next_token)

            if next_token == eos_id:
                break

        return tgt_vocab.decode(decoded_ids)


# ---------------------------------------------------------------------------
# 3.  Evaluate the trained model
# ---------------------------------------------------------------------------
def evaluate_model(model, pairs, src_vocab, tgt_vocab, device="cpu"):
    """
    Run greedy decoding on all pairs and report:
      - Corpus-level BLEU score
      - A few sample translations
    """
    model = model.to(device)
    model.eval()

    references = []
    hypotheses = []

    for src_str, tgt_str in pairs:
        src_ids = torch.tensor([[src_vocab.stoi["<sos>"]]
                               + [src_vocab.stoi.get(ch, src_vocab.stoi["<unk>"])
                                  for ch in src_str]
                               + [src_vocab.stoi["<eos>"]]], dtype=torch.long)

        hyp_str = greedy_decode(model, src_ids, src_vocab, tgt_vocab, device=device)
        ref_str = tgt_str

        references.append(ref_str)
        hypotheses.append(hyp_str)

    bleu = corpus_bleu(references, hypotheses)

    print("\n=== Sample Translations ===")
    for src_str, ref_str, hyp_str in zip(
            [s for s, _ in pairs[:10]],
            references[:10],
            hypotheses[:10]):
        print(f"  SRC : {src_str}")
        print(f"  REF : {ref_str}")
        print(f"  HYP : {hyp_str}")
        print()

    return bleu, references, hypotheses


# ---------------------------------------------------------------------------
# 4.  Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pickle

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the trained checkpoint
    checkpoint = torch.load("transformer_checkpoint.pt", map_location=device)
    src_vocab  = checkpoint["src_vocab"]
    tgt_vocab  = checkpoint["tgt_vocab"]
    cfg        = checkpoint["config"]

    model = Transformer(
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        n_layers=cfg["n_layers"],
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
    )
    model.load_state_dict(checkpoint["model_state"])

    # Evaluate on a few held-out examples
    test_pairs = [
        ("hello",         "bonjour"),
        ("goodbye",       "adieu"),
        ("i love you",    "je t'aime"),
        ("thank you",     "merci"),
        ("how are you",   "comment allez vous"),
    ]

    bleu, refs, hyps = evaluate_model(model, test_pairs, src_vocab, tgt_vocab, device)
    print(f"\n=== BLEU Score ===")
    print(f"  Corpus BLEU (from scratch): {bleu:.4f}")

    # Also show what nltk would give (informational)
    try:
        from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu
        nltk_refs = [[r.split()] for r in refs]
        nltk_hyps = [h.split() for h in hyps]
        nltk_bleu = nltk_corpus_bleu(nltk_refs, nltk_hyps)
        print(f"  Corpus BLEU (nltk):         {nltk_bleu:.4f}")
    except ImportError:
        print("  nltk not installed — install with: pip install nltk")
        print("  (our from-scratch BLEU is still valid)")
