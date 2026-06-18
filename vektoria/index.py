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
