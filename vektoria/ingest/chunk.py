"""Split text into overlapping, word-bounded chunks for embedding."""

from __future__ import annotations


def chunk_text(text: str, max_words: int = 400, overlap: int = 40) -> list[str]:
    """Sliding-window chunker.

    Args:
        max_words: maximum words per chunk (> 0).
        overlap: words shared between consecutive chunks (clamped to
            ``[0, max_words - 1]`` so the window always advances).
    """
    if max_words <= 0:
        raise ValueError("max_words must be > 0")
    overlap = max(0, min(overlap, max_words - 1))

    words = text.split()
    if not words:
        return []

    step = max_words - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + max_words]))
        if start + max_words >= len(words):
            break  # last window reached the end; don't emit a trailing overlap-only chunk
    return chunks
