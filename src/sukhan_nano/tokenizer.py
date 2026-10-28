"""Character-level BPE tokenizer for Sukhan-Nano, implemented from scratch.

Why character-level BPE (and not pure characters or a library tokenizer):
  * Urdu's Perso-Arabic alphabet is compact (~150 distinct codepoints in the
    corpus, including diacritics, digits and punctuation), so a character base
    vocabulary is small and nothing is ever out-of-vocabulary in-corpus.
  * BPE merges on top of that base learn frequent Urdu letter sequences and
    whole common words, which shortens sequences ~3x versus raw characters --
    at this model's tiny compute budget that is the difference between
    learning spelling and learning words.
  * A from-scratch implementation keeps the artifact fully reproducible with
    zero tokenizer dependencies: the merge table in tokenizer.json *is* the
    tokenizer.

Merges are learned over whitespace-delimited words weighted by frequency
(the standard efficient formulation); encoding applies merges in rank order.
Round-trip property: decode(encode(text)) == text for any text whose
characters appear in the training corpus; unseen characters map to <unk>.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

UNK = "<unk>"
_WORD_RE = re.compile(r"\S+|\s+")


class BPETokenizer:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = []          # list of (a, b) in rank order
        self.merge_rank = {}      # (a, b) -> rank
        self.vocab_size = 0

    # ------------------------------------------------------------- training
    @classmethod
    def train(cls, text, vocab_size, verbose=True):
        tok = cls()
        base_chars = sorted(set(text))
        tokens = [UNK] + base_chars
        if vocab_size <= len(tokens):
            raise ValueError("vocab_size must exceed the base alphabet size")
        tok.token_to_id = {t: i for i, t in enumerate(tokens)}
        tok.id_to_token = {i: t for t, i in tok.token_to_id.items()}

        # word -> (list of symbols, frequency)
        word_freqs = Counter(_WORD_RE.findall(text))
        words = {w: (list(w), f) for w, f in word_freqs.items()}

        # pair -> total weighted count, and pair -> set of words containing it
        def pair_counts():
            counts = Counter()
            where = defaultdict(set)
            for w, (syms, f) in words.items():
                for a, b in zip(syms, syms[1:]):
                    counts[(a, b)] += f
                    where[(a, b)].add(w)
            return counts, where

        counts, where = pair_counts()
        n_merges = vocab_size - len(tokens)
        for rank in range(n_merges):
            if not counts:
                break
            (a, b), _ = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
            new_tok = a + b
            if new_tok not in tok.token_to_id:
                new_id = len(tok.token_to_id)
                tok.token_to_id[new_tok] = new_id
                tok.id_to_token[new_id] = new_tok
            tok.merges.append((a, b))
            tok.merge_rank[(a, b)] = rank
            affected = list(where.get((a, b), ()))
            # subtract old contributions of affected words, merge, re-add
            for w in affected:
                syms, f = words[w]
                for x, y in zip(syms, syms[1:]):
                    counts[(x, y)] -= f
                    if counts[(x, y)] <= 0:
                        del counts[(x, y)]
                merged, i = [], 0
                while i < len(syms):
                    if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                        merged.append(new_tok)
                        i += 2
                    else:
                        merged.append(syms[i])
                        i += 1
                words[w] = (merged, f)
            for w in affected:
                syms, f = words[w]
                for x, y in zip(syms, syms[1:]):
                    counts[(x, y)] += f
                    where[(x, y)].add(w)
            if verbose and (rank % 500 == 0 or rank == n_merges - 1):
                print(f"  merge {rank + 1}/{n_merges}: {a!r}+{b!r} -> {new_tok!r}")
        tok.vocab_size = len(tok.token_to_id)
        return tok

    # ------------------------------------------------------------- encoding
    def _encode_word(self, word):
        syms = [ch if ch in self.token_to_id else UNK for ch in word]
        if len(syms) < 2:
            return syms
        while True:
            best, best_rank = None, None
            for a, b in zip(syms, syms[1:]):
                r = self.merge_rank.get((a, b))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = (a, b), r
            if best is None:
                return syms
            a, b = best
            merged, i = [], 0
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(syms[i])
                    i += 1
            syms = merged
            if len(syms) < 2:
                return syms

    def encode(self, text):
        ids = []
        for piece in _WORD_RE.findall(text):
            for sym in self._encode_word(piece):
                ids.append(self.token_to_id.get(sym, 0))
        return ids

    def decode(self, ids):
        return "".join(self.id_to_token.get(int(i), UNK) for i in ids)

    # ---------------------------------------------------------- persistence
    def save(self, path):
        data = {
            "format": "sukhan-nano char-bpe v1",
            "vocab": self.id_to_token,
            "merges": self.merges,
            "unk_token": UNK,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.id_to_token = {int(k): v for k, v in data["vocab"].items()}
        tok.token_to_id = {v: k for k, v in tok.id_to_token.items()}
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.merge_rank = {tuple(m): r for r, m in enumerate(tok.merges)}
        tok.vocab_size = len(tok.token_to_id)
        return tok
