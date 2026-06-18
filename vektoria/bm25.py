"""
BM25 keyword search index — zero external dependencies.

Implements Okapi BM25 scoring with French-aware tokenization.
Designed to complement vector similarity search in a hybrid retrieval setup.
"""

import math
import re
import unicodedata
from collections import Counter


# Common French stopwords (articles, prepositions, conjunctions)
FRENCH_STOPWORDS = frozenset({
    "le", "la", "les", "de", "du", "des", "un", "une",
    "et", "ou", "en", "à", "a", "au", "aux",
    "ce", "ces", "cette", "cet",
    "il", "elle", "ils", "elles", "on", "nous", "vous",
    "je", "tu", "me", "te", "se", "ne", "pas", "plus",
    "son", "sa", "ses", "leur", "leurs", "mon", "ma", "mes",
    "ton", "ta", "tes", "notre", "nos", "votre", "vos",
    "que", "qui", "quoi", "dont", "où",
    "pour", "par", "avec", "dans", "sur", "sous", "entre",
    "est", "sont", "être", "avoir", "fait", "été",
    "mais", "donc", "car", "ni", "si",
    "tout", "tous", "toute", "toutes",
    "bien", "très", "aussi", "comme",
    "peut", "ont", "era", "sera",
    "l", "d", "n", "s", "c", "j", "y", "m", "qu",
})


def _strip_accents(text: str) -> str:
    """Remove accents from text (e.g. 'généralités' -> 'generalites')."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    """
    French-aware tokenizer.

    - Lowercase
    - Strip accents
    - Split on non-alpha characters
    - Remove stopwords and single-character tokens
    """
    text = text.lower()
    text = _strip_accents(text)
    tokens = re.findall(r"[a-z]+", text)
    return [t for t in tokens if len(t) > 1 and t not in FRENCH_STOPWORDS]


class BM25Index:
    """
    In-memory BM25 index for keyword search.

    Usage:
        idx = BM25Index()
        idx.add("doc1", "Le droit civil suisse ...")
        idx.add("doc2", "Les obligations contractuelles ...")
        results = idx.search("obligations", top_k=5)
        # returns [("doc2", 1.23), ...]
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: Term frequency saturation parameter (default 1.5).
            b:  Length normalization parameter (default 0.75).
        """
        self.k1 = k1
        self.b = b

        # doc_id -> token frequencies
        self._doc_freqs: dict[str, Counter] = {}
        # doc_id -> document length (in tokens)
        self._doc_lengths: dict[str, int] = {}
        # token -> set of doc_ids containing it
        self._inverted_index: dict[str, set[str]] = {}
        # Total number of documents
        self._num_docs: int = 0
        # Average document length
        self._avg_dl: float = 0.0

    def add(self, doc_id: str, text: str) -> None:
        """
        Tokenize and index a document.

        Args:
            doc_id: Unique document identifier.
            text: Raw text content to index.
        """
        tokens = tokenize(text)
        tf = Counter(tokens)

        # If doc already exists, remove it first
        if doc_id in self._doc_freqs:
            self.remove(doc_id)

        self._doc_freqs[doc_id] = tf
        self._doc_lengths[doc_id] = len(tokens)
        self._num_docs += 1

        # Update inverted index
        for token in tf:
            if token not in self._inverted_index:
                self._inverted_index[token] = set()
            self._inverted_index[token].add(doc_id)

        # Recalculate average document length
        self._avg_dl = (
            sum(self._doc_lengths.values()) / self._num_docs
            if self._num_docs > 0
            else 0.0
        )

    def remove(self, doc_id: str) -> None:
        """Remove a document from the index."""
        if doc_id not in self._doc_freqs:
            return

        tf = self._doc_freqs.pop(doc_id)
        self._doc_lengths.pop(doc_id)
        self._num_docs -= 1

        for token in tf:
            if token in self._inverted_index:
                self._inverted_index[token].discard(doc_id)
                if not self._inverted_index[token]:
                    del self._inverted_index[token]

        self._avg_dl = (
            sum(self._doc_lengths.values()) / self._num_docs
            if self._num_docs > 0
            else 0.0
        )

    def search(
        self, query: str, top_k: int = 10, doc_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        """
        Search for documents matching the query.

        Args:
            query: Search query text.
            top_k: Maximum number of results to return.
            doc_ids: Optional set of doc_ids to restrict search to.

        Returns:
            List of (doc_id, score) tuples sorted by descending score.
        """
        if self._num_docs == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: dict[str, float] = {}

        for token in query_tokens:
            if token not in self._inverted_index:
                continue

            matching_docs = self._inverted_index[token]
            df = len(matching_docs)

            # IDF component: log((N - df + 0.5) / (df + 0.5) + 1)
            idf = math.log((self._num_docs - df + 0.5) / (df + 0.5) + 1.0)

            for doc_id in matching_docs:
                if doc_ids is not None and doc_id not in doc_ids:
                    continue

                tf = self._doc_freqs[doc_id][token]
                dl = self._doc_lengths[doc_id]

                # BM25 TF component
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
                )

                score = idf * tf_norm
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        # Sort by score descending, take top_k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    @property
    def num_docs(self) -> int:
        """Number of indexed documents."""
        return self._num_docs
