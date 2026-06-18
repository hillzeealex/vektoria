"""
Embedding backends — turn text into the unit vectors the index stores.

Every backend returns **L2-normalized float32** vectors, so the store's cosine
similarity is a plain dot product regardless of which model produced them. This
is also where two classic footguns are handled centrally: Ollama returns
unnormalized vectors (normalized here), and the test backend uses a *stable*
hash (not Python's per-process-salted ``hash()``) so vectors survive restarts.

Backends implement the ``Embedder`` protocol: a ``dimension`` and two methods,
``embed_documents`` (batch, for ingestion) and ``embed_query`` (one query).
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from typing import Protocol, runtime_checkable

import numpy as np

_EPS = 1e-9


def _l2(arr: np.ndarray) -> np.ndarray:
    """L2-normalize along the last axis (works for 1-D and 2-D)."""
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, _EPS)


@runtime_checkable
class Embedder(Protocol):
    dimension: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class HashEmbedder:
    """Deterministic, dependency-free pseudo-embeddings for tests and demos.

    Seeded from a stable BLAKE2b digest so the same text always yields the same
    vector — across processes and restarts. Carries no semantic meaning; never
    use it in production.
    """

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def _vec(self, text: str) -> np.ndarray:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little"))
        return rng.standard_normal(self.dimension).astype(np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return _l2(np.vstack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2(self._vec(text))


class SentenceTransformerEmbedder:
    """Local HuggingFace embeddings via sentence-transformers.

    Defaults target the multilingual-e5 family, which is trained with
    ``query:`` / ``passage:`` prefixes — override them for other models.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        *,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        batch_size: int = 32,
    ):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vecs = self._model.encode(
            [self.passage_prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vec = self._model.encode(self.query_prefix + text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)


class OllamaEmbedder:
    """Embeddings from a local Ollama server. Ollama returns raw (unnormalized)
    vectors, so we normalize them here for correct cosine similarity."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        *,
        url: str = "http://localhost:11434",
        timeout: float = 30.0,
    ):
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.dimension = len(self._embed_raw("dimension probe"))

    def _embed_raw(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.model, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{self.url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())["embedding"]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return _l2(np.vstack([np.asarray(self._embed_raw(t), dtype=np.float32) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2(np.asarray(self._embed_raw(text), dtype=np.float32))


def make_embedder(backend: str, **kwargs) -> Embedder:
    """Construct an embedder by backend name: 'hash', 'sentence-transformers', 'ollama'."""
    backends = {
        "hash": HashEmbedder,
        "sentence-transformers": SentenceTransformerEmbedder,
        "ollama": OllamaEmbedder,
    }
    if backend not in backends:
        raise ValueError(f"Unknown embedding backend {backend!r}; choose from {sorted(backends)}")
    return backends[backend](**kwargs)
