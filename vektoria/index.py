"""
Vektoria index — one index = one directory with a SQLite DB as source of truth.

Storage: vectors are L2-normalized on write and persisted as float32 blobs in
SQLite alongside JSON metadata. The index also keeps an in-memory mirror — a
numpy matrix, an id list, a metadata list and a BM25 index — updated
*incrementally* on every write (no full reload), so a single upsert is O(batch),
not O(total).

Concurrency: FastAPI serves sync endpoints from a threadpool, so a cached Index
is touched by multiple threads. The connection is opened with
``check_same_thread=False`` and every operation is serialized by a per-index
``Lock``. Brute-force queries are sub-millisecond at this scale, so a single lock
is a deliberate v1 trade-off; lock-free reads over an immutable snapshot are the
natural next step if read throughput ever needs it.
"""

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vektoria.bm25 import BM25Index

SUPPORTED_METRICS = {"cosine"}
_EPS = 1e-9


@dataclass
class QueryMatch:
    """A single search hit."""
    id: str
    score: float
    metadata: dict


class Index:
    def __init__(self, path):
        self.path = Path(path)
        db_path = self.path / "index.db"
        if not db_path.exists():
            raise FileNotFoundError(f"No index at {self.path}. Use Index.create().")
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._dimension = int(self._meta_value("dimension"))
        self._metric = self._meta_value("metric")
        self._load_cache()

    @classmethod
    def create(cls, path, dimension: int, metric: str = "cosine") -> "Index":
        if metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric {metric!r}; v1 supports {sorted(SUPPORTED_METRICS)}"
            )
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(str(path / "index.db"))
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );
            """
        )
        db.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            [("dimension", str(dimension)), ("metric", metric)],
        )
        db.commit()
        db.close()
        return cls(path)

    @classmethod
    def stat(cls, path) -> dict:
        """Read an index's summary (dimension, metric, count) without loading
        vectors into memory — used for cheap listings."""
        path = Path(path)
        db = sqlite3.connect(str(path / "index.db"))
        db.row_factory = sqlite3.Row
        try:
            meta = {r["key"]: r["value"] for r in db.execute("SELECT key, value FROM meta")}
            count = db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        finally:
            db.close()
        return {"dimension": int(meta["dimension"]), "metric": meta["metric"], "count": count}

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def metric(self) -> str:
        return self._metric

    def count(self) -> int:
        with self._lock:
            return len(self._ids)

    def close(self):
        with self._lock:
            self._db.close()

    # ── writes ───────────────────────────────────────────────────────
    def upsert(self, items: list[dict]) -> int:
        if not items:
            return 0
        # Validate + normalize outside the lock (pure CPU, no shared state).
        prepared = []
        for it in items:
            values = it["values"]
            if len(values) != self._dimension:
                raise ValueError(
                    f"Vector for id={it['id']!r} has dim {len(values)}, expected {self._dimension}"
                )
            vec = np.asarray(values, dtype=np.float32)
            vec = vec / (np.linalg.norm(vec) + _EPS)
            prepared.append((it["id"], vec, it.get("metadata") or {}))

        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO vectors (id, vector, metadata) VALUES (?, ?, ?)",
                [(rid, v.tobytes(), json.dumps(m, ensure_ascii=False)) for rid, v, m in prepared],
            )
            self._db.commit()

            appended = []
            for rid, v, m in prepared:
                if rid in self._row_of:                 # in-place replace, O(1)
                    i = self._row_of[rid]
                    self._matrix[i] = v
                    self._meta[i] = m
                    self._bm25.remove(rid)
                else:                                    # new row, batched append
                    self._row_of[rid] = len(self._ids)
                    self._ids.append(rid)
                    self._meta.append(m)
                    appended.append(v)
                self._bm25.add(rid, m.get("text", ""))
            if appended:
                block = np.vstack(appended)
                self._matrix = block if self._matrix is None else np.vstack([self._matrix, block])
        return len(items)

    def delete(self, ids: list[str] | None = None, filter: dict | None = None) -> int:
        if not ids and not filter:
            return 0
        with self._lock:
            targets: set[str] = {i for i in (ids or []) if i in self._row_of}
            if filter:
                targets |= {
                    rid for rid, i in self._row_of.items()
                    if self._matches_filter(self._meta[i], filter)
                }
            if not targets:
                return 0

            self._db.executemany("DELETE FROM vectors WHERE id = ?", [(i,) for i in targets])
            self._db.commit()

            keep = [i for i, rid in enumerate(self._ids) if rid not in targets]
            self._matrix = self._matrix[keep] if keep else None
            self._ids = [self._ids[i] for i in keep]
            self._meta = [self._meta[i] for i in keep]
            self._row_of = {rid: n for n, rid in enumerate(self._ids)}
            for rid in targets:
                self._bm25.remove(rid)
            return len(targets)

    # ── reads ────────────────────────────────────────────────────────
    def query(
        self,
        vector,
        top_k: int = 5,
        filter: dict | None = None,
        hybrid: bool = False,
        alpha: float = 0.5,
        text: str | None = None,
    ) -> list[QueryMatch]:
        if hybrid and not text:
            raise ValueError("hybrid=True requires a 'text' argument for BM25")
        with self._lock:
            if self._matrix is None or not self._ids:
                return []
            q = np.asarray(vector, dtype=np.float32)
            q = q / (np.linalg.norm(q) + _EPS)
            sims = self._matrix @ q  # cosine: rows and query are unit vectors
            ordered = self._rank_hybrid(sims, text, alpha, top_k) if hybrid \
                else self._rank_vector(sims, top_k, bool(filter))
            return self._assemble(ordered, top_k, filter)

    def export(self) -> dict:
        with self._lock:
            vectors = [
                {"id": self._ids[i], "values": self._matrix[i].tolist(), "metadata": self._meta[i]}
                for i in range(len(self._ids))
            ]
            return {"dimension": self._dimension, "metric": self._metric, "vectors": vectors}

    # ── ranking helpers (assume the lock is held) ────────────────────
    def _rank_vector(self, sims, top_k, filtered) -> list[tuple[int, float]]:
        fetch_k = top_k * 4 if filtered else top_k
        if len(sims) <= fetch_k:
            idxs = np.argsort(sims)[::-1]
        else:
            part = np.argpartition(sims, -fetch_k)[-fetch_k:]
            idxs = part[np.argsort(sims[part])[::-1]]
        return [(int(i), float(sims[i])) for i in idxs]

    def _rank_hybrid(self, sims, text, alpha, top_k) -> list[tuple[int, float]]:
        candidate_k = max(top_k * 4, 50)
        vec = self._normalize({i: float(sims[i]) for i in range(len(self._ids))})
        bm25 = self._normalize({
            self._row_of[did]: s
            for did, s in self._bm25.search(text, top_k=candidate_k)
            if did in self._row_of
        })
        combined = {
            i: alpha * vec.get(i, 0.0) + (1 - alpha) * bm25.get(i, 0.0)
            for i in set(vec) | set(bm25)
        }
        return sorted(combined.items(), key=lambda kv: kv[1], reverse=True)

    def _assemble(self, ordered, top_k, filter) -> list[QueryMatch]:
        out: list[QueryMatch] = []
        for idx, score in ordered:
            if len(out) >= top_k:
                break
            meta = self._meta[idx]
            if filter and not self._matches_filter(meta, filter):
                continue
            out.append(QueryMatch(id=self._ids[idx], score=float(score), metadata=meta))
        return out

    @staticmethod
    def _normalize(scores: dict) -> dict:
        """Min-max normalize scores to [0, 1]; identical scores all map to 1.0."""
        if not scores:
            return {}
        vals = scores.values()
        lo, hi = min(vals), max(vals)
        span = hi - lo
        if span < _EPS:
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / span for k, v in scores.items()}

    @staticmethod
    def _matches_filter(meta: dict, filter: dict) -> bool:
        for key, value in filter.items():
            actual = meta.get(key)
            if isinstance(value, list):
                if actual not in value:
                    return False
            elif actual != value:
                return False
        return True

    # ── internal ─────────────────────────────────────────────────────
    def _meta_value(self, key: str) -> str:
        return self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()["value"]

    def _load_cache(self):
        """Build the in-memory mirror once, at open time."""
        rows = self._db.execute(
            "SELECT id, vector, metadata FROM vectors ORDER BY rowid"
        ).fetchall()
        self._ids: list[str] = []
        self._meta: list[dict] = []
        self._row_of: dict[str, int] = {}
        self._bm25 = BM25Index()
        vecs = []
        for r in rows:
            meta = json.loads(r["metadata"])
            self._row_of[r["id"]] = len(self._ids)
            self._ids.append(r["id"])
            self._meta.append(meta)
            vecs.append(np.frombuffer(r["vector"], dtype=np.float32))
            self._bm25.add(r["id"], meta.get("text", ""))
        self._matrix = np.vstack(vecs) if vecs else None
