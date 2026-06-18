"""
Optional approximate-nearest-neighbour backend, powered by TurboVec (Rust).

Brute-force exact search is the default and is the right choice up to ~1M
vectors. For larger indexes you can trade a little recall for big memory savings
(2–4-bit quantization) and sub-linear-ish search by selecting the ``turbovec``
backend on an index. This module is a thin wrapper around TurboVec's
``IdMapIndex``; it is imported lazily, so ``pip install vektoria`` stays light —
the engine is only needed when an index actually uses it (``vektoria[ann]``).

Vectors are keyed by an integer **row id** (their position in the index's
in-memory id list). The wrapper hides TurboVec's prepare-before-search step.
"""

from __future__ import annotations

import numpy as np


class TurboVecBackend:
    def __init__(self, dim: int, bit_width: int = 4):
        import turbovec  # vektoria[ann]

        self.dim = dim
        self.bit_width = bit_width
        self._ix = turbovec.IdMapIndex(dim, bit_width)
        self._n = 0
        self._dirty = False

    def add(self, vectors: np.ndarray, row_ids: list[int]) -> None:
        if not row_ids:
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

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
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
