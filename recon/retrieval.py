"""A small BM25 index over invoice documents.

Lexical retrieval is a deliberate choice here: invoice numbers and supplier names are
exact identifiers, which BM25 handles well and embeddings often blur together.
"""

import math
import re
import unicodedata
from collections import Counter
from typing import Callable


def normalize(text: str) -> str:
    """Lowercase and strip accents, so 'Échéance' matches 'echeance'."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def tokenize(text: str) -> list[str]:
    text = normalize(text)
    tokens = re.findall(r"[a-z0-9]+", text)
    # Keep hyphenated identifiers whole too, so "DUP-2026-0003" is one strong token.
    tokens += re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)+", text)
    return tokens


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.docs), 1)
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def score(self, query_tokens: list[str], i: int) -> float:
        doc, length = self.docs[i], self.lengths[i]
        s = 0.0
        for term in query_tokens:
            tf = doc.get(term, 0)
            if tf:
                s += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg_len))
        return s

    def search(self, query: str, k: int, keep: Callable[[int], bool] = lambda i: True) -> list[tuple[int, float]]:
        """Top-k (doc index, score) among documents that pass `keep`."""
        q = tokenize(query)
        scored = [(i, self.score(q, i)) for i in range(len(self.docs)) if keep(i)]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
