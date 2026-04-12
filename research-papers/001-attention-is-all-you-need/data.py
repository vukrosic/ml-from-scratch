"""
data.py — Simple character-level translation dataset.

We use a tiny hand-crafted parallel corpus of English -> French phrases
and train the Transformer to translate at the character level.

Using character-level tokens means:
  - Tiny vocabularies (tens of characters, not tens of thousands of words)
  - A 512-d_model model can actually learn something useful in hundreds of steps
  - No external tokenizer dependencies — everything is self-contained
"""

import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# 1.  Tiny Parallel Corpus  (English -> French, short phrases)
# ---------------------------------------------------------------------------
# Each pair is (english_phrase, french_phrase). Keep them short so the model
# can make progress in a few hundred gradient steps.

PAIRS = [
    # Greetings
    ("hello",           "bonjour"),
    ("good morning",    "bonjour"),
    ("good evening",    "bonsoir"),
    ("good night",      "bonne nuit"),
    ("see you",         "aurevoir"),
    ("goodbye",         "adieu"),
    ("yes",             "oui"),
    ("no",              "non"),
    ("please",          "s'il vous plait"),
    ("thank you",       "merci"),
    ("thanks",          "merci"),
    ("excuse me",       "excusez moi"),
    ("sorry",           "desole"),
    # Questions
    ("how are you",     "comment allez vous"),
    ("what is your name","comment vous appelez vous"),
    ("my name is",      "je m'appelle"),
    ("where is",        "ou est"),
    ("when is",         "quand est"),
    ("who is",          "qui est"),
    ("why is",          "pourquoi"),
    ("how much",        "combien"),
    # Common phrases
    ("i love you",      "je t'aime"),
    ("i understand",   "je comprends"),
    ("i dont understand","je ne comprends pas"),
    ("i am happy",      "je suis content"),
    ("i am sad",        "je suis triste"),
    ("are you okay",    "est ce que vous allez bien"),
    ("good job",        "bien joue"),
    ("very good",       "tres bien"),
    ("of course",       "bien sur"),
    ("perhaps",         "peut etre"),
    # Places
    ("the house",       "la maison"),
    ("the car",         "la voiture"),
    ("the city",        "la ville"),
    ("the book",        "le livre"),
    ("the water",       "l'eau"),
    ("the food",        "la nourriture"),
    # Numbers
    ("one",             "un"),
    ("two",             "deux"),
    ("three",           "trois"),
    ("four",            "quatre"),
    ("five",            "cinq"),
    # Time
    ("today",           "aujourd'hui"),
    ("tomorrow",        "demain"),
    ("yesterday",       "hier"),
    ("now",             "maintenant"),
    ("later",           "plus tard"),
]

# Add more phrases by reversing (French -> English) for data augmentation
REVERSED_PAIRS = [(f, e) for e, f in PAIRS]
ALL_PAIRS = PAIRS + REVERSED_PAIRS


# ---------------------------------------------------------------------------
# 2.  Vocabulary building
# ---------------------------------------------------------------------------
class CharVocab:
    """Minimal character-level vocabulary with special tokens."""

    PAD = "<pad>"
    SOS = "<sos>"   # start-of-sequence
    EOS = "<eos>"   # end-of-sequence
    UNK = "<unk>"   # unknown

    def __init__(self, words):
        chars = set()
        for word in words:
            chars.update(word)
        self.stoi = {self.PAD: 0, self.SOS: 1, self.EOS: 2, self.UNK: 3}
        for i, ch in enumerate(sorted(chars)):
            self.stoi[ch] = i + 4
        self.itos = {v: k for k, v in self.stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def encode(self, word, add_sos=False, add_eos=False):
        """Convert a string to a list of integer token indices."""
        ids = [self.stoi.get(ch, self.stoi[self.UNK]) for ch in word]
        if add_sos:
            ids = [self.stoi[self.SOS]] + ids
        if add_eos:
            ids = ids + [self.stoi[self.EOS]]
        return ids

    def decode(self, ids):
        """Convert a list of integer indices back to a string."""
        chars = []
        for idx in ids:
            if idx == self.stoi[self.EOS]:
                break
            if idx not in (self.stoi[self.PAD], self.stoi[self.SOS]):
                chars.append(self.itos.get(idx, self.UNK))
        return "".join(chars)


def build_vocabs(pairs):
    """Build separate source and target vocabularies from the parallel corpus."""
    src_words = [src for src, _ in pairs]
    tgt_words = [tgt for _, tgt in pairs]
    src_vocab = CharVocab(src_words)
    tgt_vocab = CharVocab(tgt_words)
    return src_vocab, tgt_vocab


# ---------------------------------------------------------------------------
# 3.  Dataset
# ---------------------------------------------------------------------------
class TranslationDataset(Dataset):
    """
    Returns (src_ids, tgt_ids) where:
      src_ids = [SOS] + src_chars + [EOS]
      tgt_ids = [SOS] + tgt_chars + [EOS]
    """

    def __init__(self, pairs, src_vocab, tgt_vocab, max_len=50):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_str, tgt_str = self.pairs[idx]
        src_ids = self.src_vocab.encode(src_str, add_sos=True, add_eos=True)
        tgt_ids = self.tgt_vocab.encode(tgt_str, add_sos=True, add_eos=True)
        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )


def collate_fn(batch, pad_idx=0):
    """Pad sequences to the same length within a batch."""
    srcs, tgts = zip(*batch)
    src_lens = torch.tensor([s.shape[0] for s in srcs])
    tgt_lens = torch.tensor([t.shape[0] for t in tgts])
    src_padded = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True,
                                                  padding_value=pad_idx)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgts, batch_first=True,
                                                  padding_value=pad_idx)
    return src_padded, tgt_padded, src_lens, tgt_lens


# ---------------------------------------------------------------------------
# 4.  DataLoader factory
# ---------------------------------------------------------------------------
def build_dataloaders(batch_size=32, max_len=50):
    src_vocab, tgt_vocab = build_vocabs(ALL_PAIRS)

    dataset = TranslationDataset(ALL_PAIRS, src_vocab, tgt_vocab, max_len)

    # Simple 80/20 train/val split
    n_train = int(0.8 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn)

    return train_loader, val_loader, src_vocab, tgt_vocab


# ---------------------------------------------------------------------------
# 5.  Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    train_loader, val_loader, src_vocab, tgt_vocab = build_dataloaders(batch_size=4)

    print(f"Source vocab size : {len(src_vocab)}")
    print(f"Target vocab size : {len(tgt_vocab)}")
    print(f"Train batches     : {len(train_loader)}")
    print(f"Val batches       : {len(val_loader)}")

    src_batch, tgt_batch, src_lens, tgt_lens = next(iter(train_loader))
    print(f"\nSource batch shape: {src_batch.shape}")
    print(f"Target batch shape: {tgt_batch.shape}")
    print(f"Sample src tokens : {src_batch[0].tolist()}")
    print(f"Sample tgt tokens : {tgt_batch[0].tolist()}")
    print(f"Decoded src       : '{src_vocab.decode(src_batch[0].tolist())}'")
    print(f"Decoded tgt       : '{tgt_vocab.decode(tgt_batch[0].tolist())}'")
