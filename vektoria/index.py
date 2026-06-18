"""
Vektoria index — one index = one directory with a SQLite DB as source of truth.

Vectors are stored as float32 blobs (L2-normalized on write) alongside JSON
metadata. On open, the index builds an in-memory numpy matrix + id list + BM25
index for brute-force cosine and hybrid search.
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vektoria.bm25 import BM25Index

SUPPORTED_METRICS = {"cosine"}


@dataclass
class QueryMatch:
    """A single search hit."""
    id: str
    score: float
    metadata: dict


class Index:
    def __init__(self, path):
        self.path = Path(path)
        if not (self.path / "index.db").exists():
            raise FileNotFoundError(f"No index at {self.path}. Use Index.create().")
        self._db = sqlite3.connect(str(self.path / "index.db"))
        self._db.row_factory = sqlite3.Row
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

    def _meta(self, key: str) -> str:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"]

    @property
    def dimension(self) -> int:
        return int(self._meta("dimension"))

    @property
    def metric(self) -> str:
        return self._meta("metric")

    def count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

    def close(self):
        self._db.close()

    def upsert(self, items: list[dict]) -> int:
        if not items:
            return 0
        dim = self.dimension
        rows = []
        for it in items:
            values = it["values"]
            if len(values) != dim:
                raise ValueError(
                    f"Vector for id={it['id']!r} has dim {len(values)}, expected {dim}"
                )
            vec = np.asarray(values, dtype=np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-9)  # L2 normalize on write
            meta = json.dumps(it.get("metadata") or {}, ensure_ascii=False)
            rows.append((it["id"], vec.astype(np.float32).tobytes(), meta))

        self._db.executemany(
            "INSERT OR REPLACE INTO vectors (id, vector, metadata) VALUES (?, ?, ?)",
            rows,
        )
        self._db.commit()
        self._load_cache()  # rebuild matrix + ids + bm25 from DB (source of truth)
        return len(items)

    def delete(self, ids: list[str] | None = None, filter: dict | None = None) -> int:
        if not ids and not filter:
            return 0

        target_ids: set[str] = set(ids or [])
        if filter:
            rows = self._db.execute("SELECT id, metadata FROM vectors").fetchall()
            for r in rows:
                if self._matches_filter(json.loads(r["metadata"]), filter):
                    target_ids.add(r["id"])

        if not target_ids:
            return 0

        self._db.executemany(
            "DELETE FROM vectors WHERE id = ?", [(i,) for i in target_ids]
        )
        self._db.commit()
        self._load_cache()  # rebuild from DB → no orphaned vectors survive
        return len(target_ids)

    def _row_metadata(self, vector_id: str) -> dict:
        row = self._db.execute(
            "SELECT metadata FROM vectors WHERE id = ?", (vector_id,)
        ).fetchone()
        return json.loads(row["metadata"]) if row else {}

    def query(
        self,
        vector,
        top_k: int = 5,
        filter: dict | None = None,
        hybrid: bool = False,
        alpha: float = 0.5,
        text: str | None = None,
    ) -> list[QueryMatch]:
        if self._matrix is None or len(self._ids) == 0:
            return []

        q = np.asarray(vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = self._matrix @ q  # cosine (rows are normalized)

        fetch_k = top_k * 4 if filter else top_k
        if len(sims) <= fetch_k:
            order = np.argsort(sims)[::-1]
        else:
            part = np.argpartition(sims, -fetch_k)[-fetch_k:]
            order = part[np.argsort(sims[part])[::-1]]

        out: list[QueryMatch] = []
        for i in order:
            if len(out) >= top_k:
                break
            meta = self._row_metadata(self._ids[i])
            if filter and not self._matches_filter(meta, filter):
                continue
            out.append(QueryMatch(id=self._ids[i], score=float(sims[i]), metadata=meta))
        return out

    def _matches_filter(self, meta: dict, filter: dict) -> bool:
        for key, value in filter.items():
            actual = meta.get(key)
            if isinstance(value, list):
                if actual not in value:
                    return False
            elif actual != value:
                return False
        return True

    # in-memory cache
    def _load_cache(self):
        rows = self._db.execute(
            "SELECT id, vector, metadata FROM vectors ORDER BY rowid"
        ).fetchall()
        self._ids = [r["id"] for r in rows]
        if rows:
            self._matrix = np.vstack(
                [np.frombuffer(r["vector"], dtype=np.float32) for r in rows]
            )
        else:
            self._matrix = None
        self._bm25 = BM25Index()
        for r in rows:
            meta = json.loads(r["metadata"])
            self._bm25.add(r["id"], meta.get("text", ""))
