"""
Custom vector store — zero cloud dependency.

Storage:
- Vectors: numpy .npy files (memory-mapped for performance)
- Metadata: SQLite database (section titles, pages, source doc)
- Index: brute-force cosine similarity (sufficient for <100k vectors)

All data stays on disk in a single directory.
"""

import os
import json
import sqlite3
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from embedder.embedder import EmbeddedChunk
from vector_store.bm25 import BM25Index


@dataclass
class SearchResult:
    """A search result with similarity score and metadata."""
    chunk_id: str
    text: str
    score: float
    heading_path: list[str]
    page_start: int
    page_end: int
    source: str
    word_count: int


class VectorStore:
    """
    Local vector store with numpy + SQLite.

    Usage:
        store = VectorStore("./data/my_store")
        store.add(embedded_chunks)
        results = store.search(query_vector, top_k=5)
    """

    def __init__(self, path: str):
        """
        Args:
            path: Directory to store all data (vectors, metadata, index).
        """
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self._vectors_path = self.path / "vectors.npy"
        self._db_path = self.path / "metadata.db"

        self._db = sqlite3.connect(str(self._db_path))
        self._db.row_factory = sqlite3.Row
        self._init_db()

        self._vectors: np.ndarray | None = None
        self._load_vectors()

        # BM25 keyword index (populated from DB on load)
        self._bm25 = BM25Index()
        self._load_bm25()

    def _init_db(self):
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                heading_path TEXT NOT NULL,
                level INTEGER NOT NULL,
                page_start INTEGER NOT NULL,
                page_end INTEGER NOT NULL,
                source TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                vector_idx INTEGER NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]',
                custom_metadata TEXT NOT NULL DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_source ON chunks(source);
            CREATE INDEX IF NOT EXISTS idx_vector_idx ON chunks(vector_idx);
        """)
        # Migrate existing databases: add new columns if missing
        self._migrate_db()
        self._db.commit()

    def _migrate_db(self):
        """Add tags and custom_metadata columns if they don't exist yet."""
        cursor = self._db.execute("PRAGMA table_info(chunks)")
        existing_cols = {row["name"] for row in cursor.fetchall()}

        if "tags" not in existing_cols:
            self._db.execute(
                "ALTER TABLE chunks ADD COLUMN tags TEXT NOT NULL DEFAULT '[]'"
            )
        if "custom_metadata" not in existing_cols:
            self._db.execute(
                "ALTER TABLE chunks ADD COLUMN custom_metadata TEXT NOT NULL DEFAULT '{}'"
            )

    def _load_vectors(self):
        if self._vectors_path.exists():
            self._vectors = np.load(str(self._vectors_path))
        else:
            self._vectors = None

    def _load_bm25(self):
        """Rebuild BM25 index from all stored chunks."""
        rows = self._db.execute("SELECT id, text FROM chunks").fetchall()
        for row in rows:
            self._bm25.add(row["id"], row["text"])

    @property
    def count(self) -> int:
        row = self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0]

    @property
    def dimension(self) -> int | None:
        if self._vectors is not None and len(self._vectors) > 0:
            return self._vectors.shape[1]
        return None

    def add(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        """
        Add embedded chunks to the store. Returns number added.
        """
        if not embedded_chunks:
            return 0

        # Build vectors array
        new_vectors = np.array(
            [ec.vector for ec in embedded_chunks], dtype=np.float32
        )

        # Determine starting index
        if self._vectors is not None and len(self._vectors) > 0:
            start_idx = len(self._vectors)
            self._vectors = np.vstack([self._vectors, new_vectors])
        else:
            start_idx = 0
            self._vectors = new_vectors

        # Save vectors
        np.save(str(self._vectors_path), self._vectors)

        # Insert metadata
        rows = []
        for i, ec in enumerate(embedded_chunks):
            c = ec.chunk
            custom_metadata = json.dumps(
                getattr(c, "metadata", None) or {}, ensure_ascii=False
            )
            rows.append((
                c.id,
                c.text,
                json.dumps(c.heading_path, ensure_ascii=False),
                c.level,
                c.page_start,
                c.page_end,
                c.source,
                c.word_count,
                start_idx + i,
                "[]",  # tags — empty by default
                custom_metadata,
            ))

        self._db.executemany(
            """INSERT OR REPLACE INTO chunks
               (id, text, heading_path, level, page_start, page_end,
                source, word_count, vector_idx, tags, custom_metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._db.commit()

        # Update BM25 index
        for ec in embedded_chunks:
            self._bm25.add(ec.chunk.id, ec.chunk.text)

        return len(embedded_chunks)

    def add_tags(self, chunk_id: str, tags: list[str]) -> None:
        """
        Add tags to a chunk. Merges with existing tags (no duplicates).

        Args:
            chunk_id: The chunk to tag.
            tags: List of tag strings to add.
        """
        row = self._db.execute(
            "SELECT tags FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Chunk not found: {chunk_id}")

        existing_tags: list[str] = json.loads(row["tags"])
        merged = list(dict.fromkeys(existing_tags + tags))  # preserve order, deduplicate

        self._db.execute(
            "UPDATE chunks SET tags = ? WHERE id = ?",
            (json.dumps(merged, ensure_ascii=False), chunk_id),
        )
        self._db.commit()

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        source_filter: str | None = None,
        filters: dict[str, str | list[str]] | None = None,
    ) -> list[SearchResult]:
        """
        Search for the most similar chunks.

        Args:
            query_vector: The query embedding (must be L2-normalized).
            top_k: Number of results to return.
            source_filter: Optional filter by document source name.
                (Kept for backward compatibility; equivalent to filters={"source": "..."})
            filters: Optional metadata filters. Supported keys:
                - source (str): exact match on source field
                - level (str): exact match on heading level (as string)
                - page_start (str): minimum page number (inclusive, as string)
                - page_end (str): maximum page number (inclusive, as string)
                - tags (list[str]): chunk must have ALL specified tags
                - Any other key is matched against custom_metadata JSON fields
        """
        if self._vectors is None or len(self._vectors) == 0:
            return []

        # Merge source_filter into filters for unified handling
        if source_filter:
            filters = dict(filters) if filters else {}
            filters.setdefault("source", source_filter)

        # Cosine similarity (vectors are L2-normalized)
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-9)
        similarities = self._vectors @ query_norm

        # We need extra candidates when filtering, since some may be excluded
        fetch_k = top_k * 4 if filters else top_k

        # Get top-k indices
        if len(similarities) <= fetch_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -fetch_k)[-fetch_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        # Fetch metadata for top results
        results: list[SearchResult] = []
        for idx in top_indices:
            if len(results) >= top_k:
                break

            score = float(similarities[idx])

            row = self._db.execute(
                "SELECT * FROM chunks WHERE vector_idx = ?", (int(idx),)
            ).fetchone()

            if row is None:
                continue

            if not self._matches_filters(row, filters):
                continue

            results.append(SearchResult(
                chunk_id=row["id"],
                text=row["text"],
                score=score,
                heading_path=json.loads(row["heading_path"]),
                page_start=row["page_start"],
                page_end=row["page_end"],
                source=row["source"],
                word_count=row["word_count"],
            ))

        return results

    def hybrid_search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int = 5,
        alpha: float = 0.5,
        filters: dict[str, str | list[str]] | None = None,
    ) -> list[SearchResult]:
        """
        Hybrid search combining vector similarity and BM25 keyword scores.

        Args:
            query_text: The raw text query (used for BM25 keyword search).
            query_vector: The query embedding vector (used for cosine similarity).
            top_k: Number of results to return.
            alpha: Interpolation weight.
                alpha=1.0 -> pure vector search
                alpha=0.0 -> pure keyword search
            filters: Optional metadata filters (same format as search()).

        Returns:
            List of SearchResult objects sorted by combined score.
        """
        if self._vectors is None or len(self._vectors) == 0:
            return []

        # --- Vector scores ---
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-9)
        similarities = self._vectors @ query_norm

        # Build chunk_id -> vector_score mapping for all chunks
        # We fetch a generous number of candidates from both sources
        candidate_k = max(top_k * 4, 50)

        if len(similarities) <= candidate_k:
            vec_top_indices = np.argsort(similarities)[::-1]
        else:
            vec_top_indices = np.argpartition(similarities, -candidate_k)[-candidate_k:]
            vec_top_indices = vec_top_indices[
                np.argsort(similarities[vec_top_indices])[::-1]
            ]

        # Collect vector candidates: chunk_id -> raw vector score
        vec_scores: dict[str, float] = {}
        chunk_rows: dict[str, sqlite3.Row] = {}

        for idx in vec_top_indices:
            row = self._db.execute(
                "SELECT * FROM chunks WHERE vector_idx = ?", (int(idx),)
            ).fetchone()
            if row is None:
                continue
            if not self._matches_filters(row, filters):
                continue
            chunk_id = row["id"]
            vec_scores[chunk_id] = float(similarities[idx])
            chunk_rows[chunk_id] = row

        # --- BM25 scores ---
        bm25_results = self._bm25.search(query_text, top_k=candidate_k)
        bm25_scores: dict[str, float] = {}
        for doc_id, score in bm25_results:
            # Check filters for BM25 candidates too
            if doc_id not in chunk_rows:
                row = self._db.execute(
                    "SELECT * FROM chunks WHERE id = ?", (doc_id,)
                ).fetchone()
                if row is None:
                    continue
                if not self._matches_filters(row, filters):
                    continue
                chunk_rows[doc_id] = row
            bm25_scores[doc_id] = score

        # --- Normalize score distributions ---
        vec_norm = self._normalize_scores(vec_scores)
        bm25_norm = self._normalize_scores(bm25_scores)

        # --- Combine scores ---
        all_candidates = set(vec_norm.keys()) | set(bm25_norm.keys())
        combined: dict[str, float] = {}
        for chunk_id in all_candidates:
            v = vec_norm.get(chunk_id, 0.0)
            b = bm25_norm.get(chunk_id, 0.0)
            combined[chunk_id] = alpha * v + (1 - alpha) * b

        # Rank and return top_k
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: list[SearchResult] = []
        for chunk_id, score in ranked:
            row = chunk_rows[chunk_id]
            results.append(SearchResult(
                chunk_id=row["id"],
                text=row["text"],
                score=score,
                heading_path=json.loads(row["heading_path"]),
                page_start=row["page_start"],
                page_end=row["page_end"],
                source=row["source"],
                word_count=row["word_count"],
            ))

        return results

    @staticmethod
    def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        """
        Min-max normalize a dict of scores to [0, 1].
        Returns empty dict if input is empty.
        """
        if not scores:
            return {}
        vals = list(scores.values())
        min_s = min(vals)
        max_s = max(vals)
        span = max_s - min_s
        if span < 1e-9:
            # All scores are effectively identical
            return {k: 1.0 for k in scores}
        return {k: (v - min_s) / span for k, v in scores.items()}

    def _matches_filters(
        self, row: sqlite3.Row, filters: dict[str, str | list[str]] | None
    ) -> bool:
        """Check if a database row matches all provided filters."""
        if not filters:
            return True

        for key, value in filters.items():
            if key == "source":
                if row["source"] != value:
                    return False

            elif key == "level":
                if str(row["level"]) != str(value):
                    return False

            elif key == "page_start":
                # Filter: chunk's page_end must be >= the requested page_start
                if row["page_end"] < int(value):
                    return False

            elif key == "page_end":
                # Filter: chunk's page_start must be <= the requested page_end
                if row["page_start"] > int(value):
                    return False

            elif key == "tags":
                # value must be a list of tags; chunk must have ALL of them
                required_tags = value if isinstance(value, list) else [value]
                chunk_tags: list[str] = json.loads(row["tags"])
                if not all(t in chunk_tags for t in required_tags):
                    return False

            else:
                # Match against custom_metadata JSON fields
                custom: dict = json.loads(row["custom_metadata"])
                if isinstance(value, list):
                    # Any of the values must match
                    if custom.get(key) not in value:
                        return False
                else:
                    if custom.get(key) != value:
                        return False

        return True

    def list_sources(self) -> list[dict]:
        """List all indexed documents with their chunk counts."""
        rows = self._db.execute(
            "SELECT source, COUNT(*) as chunks, SUM(word_count) as words "
            "FROM chunks GROUP BY source"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_source(self, source: str) -> int:
        """Remove all chunks from a specific source document."""
        # Get vector indices to remove
        rows = self._db.execute(
            "SELECT id, vector_idx FROM chunks WHERE source = ? ORDER BY vector_idx",
            (source,),
        ).fetchall()

        if not rows:
            return 0

        indices = {r["vector_idx"] for r in rows}
        chunk_ids = [r["id"] for r in rows]
        count = len(indices)

        # Remove from BM25 index
        for cid in chunk_ids:
            self._bm25.remove(cid)

        # Delete from DB
        self._db.execute("DELETE FROM chunks WHERE source = ?", (source,))
        self._db.commit()

        # Rebuild vectors without deleted indices
        if self._vectors is not None:
            mask = np.ones(len(self._vectors), dtype=bool)
            for idx in indices:
                mask[idx] = False
            self._vectors = self._vectors[mask]
            np.save(str(self._vectors_path), self._vectors)

            # Re-index remaining chunks
            self._reindex_vectors()

        return count

    def _reindex_vectors(self):
        """Re-assign vector_idx after deletion."""
        rows = self._db.execute(
            "SELECT id, vector_idx FROM chunks ORDER BY vector_idx"
        ).fetchall()

        for new_idx, row in enumerate(rows):
            self._db.execute(
                "UPDATE chunks SET vector_idx = ? WHERE id = ?",
                (new_idx, row["id"]),
            )
        self._db.commit()

    def close(self):
        self._db.close()
