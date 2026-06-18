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
