"""The vector-storage seam: one ``VectorBackend`` interface, two adapters.

An index keeps an in-memory mirror of its vectors so queries never touch disk.
*How* those vectors are stored and searched is the one thing that varies between
exact and approximate retrieval — so it lives behind this interface, and the
``Index`` holds a single backend instead of branching on which one it has.

Two adapters implement it:

* ``BruteForceBackend`` — a dense float32 matrix; exact cosine via one matmul.
  The default, and the right choice up to ~1M vectors.
* ``TurboVecBackend`` (in :mod:`vektoria.ann`) — a quantized ANN index, imported
  lazily so the core install stays numpy-only.

Vectors are addressed by **row id**: their position in the index's id list.
Backends never see vector ids, metadata, or the keyword index — only rows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class VectorBackend(ABC):
    """Stores unit vectors by row id and answers nearest-neighbour queries."""

    @abstractmethod
    def add(self, vectors: np.ndarray, row_ids: list[int]) -> None:
        """Append ``vectors`` at ``row_ids`` (contiguous, at the end of the store)."""

    @abstractmethod
    def replace(self, row_id: int, vector: np.ndarray) -> None:
        """Overwrite the vector at an existing ``row_id`` in place."""

    @abstractmethod
    def keep_rows(self, keep: list[int], reload: Callable[[], np.ndarray]) -> None:
        """Compact the store down to rows ``keep`` (old indices), renumbered to
        ``0..len(keep)-1``. ``reload`` yields the surviving vectors from the
        source of truth, for backends that cannot slice themselves in place."""

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int, filtered: bool) -> list[tuple[int, float]]:
        """Return ``(row, score)`` pairs best-first. ``filtered`` signals that
        the caller will post-filter, so the backend should over-fetch."""

    @abstractmethod
    def candidate_scores(self, query: np.ndarray, candidate_k: int) -> dict[int, float]:
        """Return ``row -> vector_score`` for hybrid fusion. Exact backends score
        every row; approximate ones return their top ``candidate_k`` candidates."""


class BruteForceBackend(VectorBackend):
    """Exact cosine over a dense matrix of unit rows. ``None`` when empty."""

    def __init__(self, vectors: np.ndarray | None):
        self._matrix = vectors

    def add(self, vectors: np.ndarray, row_ids: list[int]) -> None:
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])

    def replace(self, row_id: int, vector: np.ndarray) -> None:
        self._matrix[row_id] = vector

    def keep_rows(self, keep: list[int], reload: Callable[[], np.ndarray]) -> None:
        self._matrix = self._matrix[keep] if keep else None

    def search(self, query: np.ndarray, top_k: int, filtered: bool) -> list[tuple[int, float]]:
        sims = self._matrix @ query  # cosine: rows and query are unit vectors
        fetch_k = top_k * 4 if filtered else top_k
        if len(sims) <= fetch_k:
            idxs = np.argsort(sims)[::-1]
        else:
            part = np.argpartition(sims, -fetch_k)[-fetch_k:]
            idxs = part[np.argsort(sims[part])[::-1]]
        return [(int(i), float(sims[i])) for i in idxs]

    def candidate_scores(self, query: np.ndarray, candidate_k: int) -> dict[int, float]:
        sims = self._matrix @ query
        return {i: float(sims[i]) for i in range(len(sims))}


def make_backend(name: str, dimension: int, bit_width: int,
                 vectors: np.ndarray | None) -> VectorBackend:
    """Build the backend named in an index's metadata, seeded with its vectors."""
    if name == "turbovec":
        from vektoria.ann import TurboVecBackend  # lazy: keeps core install numpy-only
        backend = TurboVecBackend(dimension, bit_width)
        if vectors is not None and len(vectors):
            backend.add(vectors, list(range(len(vectors))))
        return backend
    return BruteForceBackend(vectors)
