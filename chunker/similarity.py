"""
Similarity-based chunker that groups consecutive paragraphs
by semantic similarity using an embedder.

Best for documents where topic boundaries don't align with
any structural markers (no headings, no consistent separators).
"""

from __future__ import annotations

import numpy as np

from .chunker import Chunk
from embedder.embedder import LocalEmbedder


class SimilarityChunker:
    """
    Group consecutive paragraphs into chunks based on embedding similarity.

    Strategy:
    - Split the text into paragraphs
    - Embed each paragraph
    - Walk through paragraphs: when cosine similarity between consecutive
      paragraphs drops below the threshold, start a new chunk
    - Enforce min/max word limits by merging or splitting as needed
    """

    def __init__(
        self,
        embedder: LocalEmbedder,
        *,
        similarity_threshold: float = 0.5,
        max_words: int = 800,
        min_words: int = 50,
    ):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.max_words = max_words
        self.min_words = min_words

    def chunk(self, text: str, *, source: str = "unknown") -> list[Chunk]:
        """
        Split raw text into semantically coherent chunks.

        Args:
            text: The full document text.
            source: Document identifier for the Chunk.source field.

        Returns:
            List of Chunk objects with heading_path=[].
        """
        text = text.strip()
        if not text:
            return []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if not paragraphs:
            return []

        if len(paragraphs) == 1:
            return [self._make_chunk(paragraphs[0], 0, source)]

        # Embed all paragraphs
        embeddings = self.embedder.embed_texts(paragraphs)

        # Find split points based on similarity drops
        groups = self._group_by_similarity(paragraphs, embeddings)

        # Enforce size constraints
        groups = self._enforce_limits(groups)

        # Build Chunk objects
        chunks: list[Chunk] = []
        for i, group in enumerate(groups):
            chunk_text = "\n\n".join(group)
            chunks.append(self._make_chunk(chunk_text, i, source))

        return chunks

    # ── Internal ─────────────────────────────────────────────────────

    def _group_by_similarity(
        self,
        paragraphs: list[str],
        embeddings: np.ndarray,
    ) -> list[list[str]]:
        """Group consecutive paragraphs; split when similarity drops."""
        groups: list[list[str]] = [[paragraphs[0]]]

        for i in range(1, len(paragraphs)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])

            if sim < self.similarity_threshold:
                # Similarity drop — start new group
                groups.append([paragraphs[i]])
            else:
                groups[-1].append(paragraphs[i])

        return groups

    def _enforce_limits(self, groups: list[list[str]]) -> list[list[str]]:
        """Merge groups that are too small, split those that are too large."""
        # Merge small groups with their neighbor
        merged: list[list[str]] = []
        for group in groups:
            word_count = sum(len(p.split()) for p in group)

            if merged and word_count < self.min_words:
                # Merge with previous group
                merged[-1].extend(group)
            else:
                merged.append(group)

        # Check last group
        if len(merged) > 1:
            last_words = sum(len(p.split()) for p in merged[-1])
            if last_words < self.min_words:
                merged[-2].extend(merged[-1])
                merged.pop()

        # Split groups that are too large
        result: list[list[str]] = []
        for group in merged:
            word_count = sum(len(p.split()) for p in group)
            if word_count <= self.max_words:
                result.append(group)
            else:
                result.extend(self._split_group(group))

        return result

    def _split_group(self, paragraphs: list[str]) -> list[list[str]]:
        """Split a group of paragraphs into sub-groups within max_words."""
        groups: list[list[str]] = []
        current: list[str] = []
        current_words = 0

        for para in paragraphs:
            para_words = len(para.split())

            if current_words + para_words > self.max_words and current:
                groups.append(current)
                current = []
                current_words = 0

            current.append(para)
            current_words += para_words

        if current:
            groups.append(current)

        return groups

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _make_chunk(self, text: str, index: int, source: str) -> Chunk:
        return Chunk(
            id=f"{_slugify(source)}_chunk_{index:04d}",
            text=text,
            heading_path=[],
            level=0,
            page_start=0,
            page_end=0,
            source=source,
            word_count=len(text.split()),
        )


def _slugify(text: str) -> str:
    """Simple slugify for IDs."""
    import re

    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:50]
