"""
Recursive text chunker for unstructured documents.

Splits text by trying progressively finer separators until chunks
are within the target size. Good fallback for plain .txt or any
document without clear heading structure.
"""

from .chunker import Chunk

# Separators ordered from coarsest to finest
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]


class RecursiveChunker:
    """
    Recursively split text using a hierarchy of separators.

    Strategy:
    - Try the first separator to split the text
    - If any resulting piece is still too large, recurse with the next separator
    - Overlap is applied at the word level between consecutive chunks
    """

    def __init__(
        self,
        *,
        max_words: int = 800,
        overlap_words: int = 50,
        separators: list[str] | None = None,
    ):
        self.max_words = max_words
        self.overlap_words = overlap_words
        self.separators = separators or list(DEFAULT_SEPARATORS)

    def chunk(self, text: str, *, source: str = "unknown") -> list[Chunk]:
        """
        Split raw text into chunks.

        Args:
            text: The full document text (plain or markdown).
            source: Document identifier for the Chunk.source field.

        Returns:
            List of Chunk objects with heading_path=[].
        """
        text = text.strip()
        if not text:
            return []

        raw_pieces = self._recursive_split(text, separator_idx=0)

        # Merge very small trailing pieces and apply overlap
        chunks = self._build_chunks_with_overlap(raw_pieces, source)
        return chunks

    # ── Internal ─────────────────────────────────────────────────────

    def _recursive_split(self, text: str, separator_idx: int) -> list[str]:
        """Split text recursively, moving to finer separators as needed."""
        if separator_idx >= len(self.separators):
            # No more separators — return text as-is even if too large
            return [text]

        sep = self.separators[separator_idx]
        parts = text.split(sep)

        result: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue

            if len(part.split()) <= self.max_words:
                result.append(part)
            else:
                # Still too large — recurse with next separator
                sub_parts = self._recursive_split(part, separator_idx + 1)
                result.extend(sub_parts)

        return result

    def _build_chunks_with_overlap(
        self, pieces: list[str], source: str
    ) -> list[Chunk]:
        """
        Combine small pieces into chunks up to max_words,
        then add word-level overlap between consecutive chunks.
        """
        if not pieces:
            return []

        # First pass: group pieces into chunks that fit within max_words
        grouped: list[str] = []
        current_parts: list[str] = []
        current_words = 0

        for piece in pieces:
            piece_words = len(piece.split())

            if current_words + piece_words > self.max_words and current_parts:
                grouped.append("\n\n".join(current_parts))
                current_parts = []
                current_words = 0

            current_parts.append(piece)
            current_words += piece_words

        if current_parts:
            grouped.append("\n\n".join(current_parts))

        # Second pass: apply word overlap
        chunks: list[Chunk] = []
        for i, group_text in enumerate(grouped):
            text = group_text

            # Prepend overlap from previous chunk
            if i > 0 and self.overlap_words > 0:
                prev_words = grouped[i - 1].split()
                overlap = prev_words[-self.overlap_words :]
                if overlap:
                    text = " ".join(overlap) + "\n\n" + text

            chunk = Chunk(
                id=f"{_slugify(source)}_chunk_{i:04d}",
                text=text,
                heading_path=[],
                level=0,
                page_start=0,
                page_end=0,
                source=source,
                word_count=len(text.split()),
            )
            chunks.append(chunk)

        return chunks


def _slugify(text: str) -> str:
    """Simple slugify for IDs."""
    import re

    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:50]
