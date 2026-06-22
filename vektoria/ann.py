"""Optional approximate-nearest-neighbour backend, powered by TurboVec (Rust).

Brute-force exact search is the default and is the right choice up to ~1M
vectors. For larger indexes you can trade a little recall for big memory savings
(2–4-bit quantization) and sub-linear-ish search by selecting the ``turbovec``
backend on an index. This module is a thin wrapper around TurboVec's
``IdMapIndex`` implementing the shared :class:`~vektoria.backends.VectorBackend`
interface; it is imported lazily, so ``pip install vektoria`` stays light — the
engine is only needed when an index actually uses it (``vektoria[ann]``).

Vectors are keyed by an integer **row id** (their position in the index's
in-memory id list). The wrapper hides TurboVec's prepare-before-search step.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from vektoria.backends import VectorBackend


class TurboVecBackend(VectorBackend):
    def __init__(self, dim: int, bit_width: int = 4):
        self.dim = dim
        self.bit_width = bit_width
        self._new_index()

    def _new_index(self) -> None:
        import turbovec  # vektoria[ann]

        self._ix = turbovec.IdMapIndex(self.dim, self.bit_width)
        self._n = 0
        self._dirty = False

    def add(self, vectors: np.ndarray, row_ids: list[int]) -> None:
        if len(row_ids) == 0:
            return
        self._ix.add_with_ids(
            np.ascontiguousarray(vectors, dtype=np.float32),
            np.asarray(row_ids, dtype=np.uint64),
        )
        self._n += len(row_ids)
        self._dirty = True

    def remove(self, row_id: int) -> None:
        self._ix.remove(int(row_id))
        self._n -= 1
        self._dirty = True

    def replace(self, row_id: int, vector: np.ndarray) -> None:
        self.remove(row_id)
        self.add(vector.reshape(1, -1), [row_id])

    def keep_rows(self, keep: list[int], reload: Callable[[], np.ndarray]) -> None:
        # TurboVec row ids compact after a delete and its quantization is one-way,
        # so rebuild a fresh index from the surviving vectors, renumbered to 0..n-1.
        self._new_index()
        vectors = reload()
        if len(vectors):
            self.add(vectors, list(range(len(vectors))))

    def search(self, query: np.ndarray, top_k: int, filtered: bool) -> list[tuple[int, float]]:
        # Always over-fetch: ANN is approximate, so a recall margin (and room for
        # post-filtering) matters more than it does for the exact backend.
        return self._knn(query, max(top_k * 4, 50))

    def candidate_scores(self, query: np.ndarray, candidate_k: int) -> dict[int, float]:
        return dict(self._knn(query, candidate_k))

    def _knn(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to k (row_id, score) pairs, best first."""
        if self._n == 0:
            return []
        if self._dirty:
            self._ix.prepare()
            self._dirty = False
        k = min(k, self._n)
        scores, ids = self._ix.search(
            np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32), k
        )
        ids = np.asarray(ids)[0].tolist()
        scores = np.asarray(scores)[0].tolist()
        return [(int(i), float(s)) for i, s in zip(ids, scores) if i >= 0]
