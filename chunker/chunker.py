"""
Semantic chunker for legal course documents.

Splits structured markdown by section boundaries instead of fixed token count.
Each chunk preserves its hierarchical context (Part > Chapter > Section).

Designed for RAG: each chunk is self-contained with enough context for
an embedding model to understand what it's about.
"""

from dataclasses import dataclass, field
from pdf_extractor.models import StructuredDocument, Section


@dataclass
class Chunk:
    """A semantically meaningful chunk ready for embedding."""
    id: str                        # unique identifier
    text: str                      # the actual content
    heading_path: list[str]        # hierarchy: ["2. Généralités", "2.1. L'importance"]
    level: int                     # heading level of this chunk's section
    page_start: int                # first page
    page_end: int                  # last page
    source: str                    # document title
    word_count: int = 0
    metadata: dict = field(default_factory=dict)


class SemanticChunker:
    """
    Chunk a StructuredDocument by section boundaries.

    Strategy:
    - Each section with content becomes a chunk
    - The chunk text is prefixed with the heading path for context
    - Sections that are too long are split at paragraph boundaries
    - Sections that are too short are merged with adjacent siblings
    """

    def __init__(
        self,
        *,
        max_words: int = 800,
        min_words: int = 50,
        overlap_sentences: int = 2,
    ):
        self.max_words = max_words
        self.min_words = min_words
        self.overlap_sentences = overlap_sentences

    def chunk(self, doc: StructuredDocument) -> list[Chunk]:
        if not doc.sections:
            return self._chunk_flat(doc)

        raw_chunks = self._sections_to_chunks(doc)
        merged = self._merge_small_chunks(raw_chunks)
        split = self._split_large_chunks(merged)

        # Assign IDs
        for i, c in enumerate(split):
            c.id = f"{_slugify(doc.title)}_chunk_{i:04d}"
            c.word_count = len(c.text.split())

        return split

    def _sections_to_chunks(self, doc: StructuredDocument) -> list[Chunk]:
        """Convert sections into raw chunks with heading context."""
        chunks: list[Chunk] = []
        heading_stack: list[str] = []  # tracks current h1 > h2 > h3 path

        for i, section in enumerate(doc.sections):
            # Update heading stack based on level
            level = section.level
            # Trim stack to parent level
            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(section.title)

            # Build the content: heading path + section content
            content = section.content.strip() if section.content else ""

            # Determine page range
            page_start = section.page
            page_end = section.page
            if i + 1 < len(doc.sections):
                page_end = max(page_start, doc.sections[i + 1].page)

            # Build chunk text with context prefix
            context_prefix = " > ".join(heading_stack)
            if content:
                chunk_text = f"[{context_prefix}]\n\n{content}"
            else:
                # Section with no content (just a heading) — skip
                continue

            chunks.append(Chunk(
                id="",
                text=chunk_text,
                heading_path=list(heading_stack),
                level=level,
                page_start=page_start,
                page_end=page_end,
                source=doc.title,
            ))

        return chunks

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge chunks that are too small with their next sibling."""
        if not chunks:
            return []

        merged: list[Chunk] = []
        buffer: Chunk | None = None

        for chunk in chunks:
            word_count = len(chunk.text.split())

            if buffer is None:
                if word_count < self.min_words:
                    buffer = chunk
                else:
                    merged.append(chunk)
            else:
                # Merge buffer with current chunk if same parent
                buffer_parent = buffer.heading_path[:-1]
                chunk_parent = chunk.heading_path[:-1]

                if buffer_parent == chunk_parent:
                    # Same parent → merge
                    buffer.text += "\n\n" + chunk.text
                    buffer.page_end = chunk.page_end
                    buffer.heading_path = chunk.heading_path  # take latest

                    if len(buffer.text.split()) >= self.min_words:
                        merged.append(buffer)
                        buffer = None
                else:
                    # Different parent → flush buffer, start new
                    merged.append(buffer)
                    if word_count < self.min_words:
                        buffer = chunk
                    else:
                        merged.append(chunk)
                        buffer = None

        if buffer:
            merged.append(buffer)

        return merged

    def _split_large_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Split chunks that exceed max_words at paragraph boundaries."""
        result: list[Chunk] = []

        for chunk in chunks:
            word_count = len(chunk.text.split())
            if word_count <= self.max_words:
                result.append(chunk)
                continue

            # Split at paragraph boundaries (double newline)
            paragraphs = chunk.text.split("\n\n")
            if len(paragraphs) <= 1:
                # Can't split further — keep as is
                result.append(chunk)
                continue

            # Extract context prefix (first line if it starts with [)
            context_prefix = ""
            if paragraphs[0].startswith("["):
                context_prefix = paragraphs[0]
                paragraphs = paragraphs[1:]

            current_parts: list[str] = []
            current_words = 0
            part_idx = 0

            for para in paragraphs:
                para_words = len(para.split())

                if current_words + para_words > self.max_words and current_parts:
                    # Flush current
                    text = context_prefix + "\n\n" + "\n\n".join(current_parts) if context_prefix else "\n\n".join(current_parts)
                    result.append(Chunk(
                        id="",
                        text=text,
                        heading_path=chunk.heading_path,
                        level=chunk.level,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        source=chunk.source,
                    ))
                    # Overlap: keep last N sentences from previous chunk
                    overlap = self._get_overlap(current_parts)
                    current_parts = [overlap] if overlap else []
                    current_words = len(overlap.split()) if overlap else 0
                    part_idx += 1

                current_parts.append(para)
                current_words += para_words

            # Flush remaining
            if current_parts:
                text = context_prefix + "\n\n" + "\n\n".join(current_parts) if context_prefix else "\n\n".join(current_parts)
                result.append(Chunk(
                    id="",
                    text=text,
                    heading_path=chunk.heading_path,
                    level=chunk.level,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    source=chunk.source,
                ))

        return result

    def _get_overlap(self, parts: list[str]) -> str:
        """Get last N sentences from parts for overlap."""
        if not parts or self.overlap_sentences == 0:
            return ""
        last = parts[-1]
        sentences = _split_sentences(last)
        overlap = sentences[-self.overlap_sentences:]
        return " ".join(overlap)

    def _chunk_flat(self, doc: StructuredDocument) -> list[Chunk]:
        """Fallback: chunk by paragraphs when no sections are detected."""
        paragraphs = doc.markdown.split("\n\n")
        chunks: list[Chunk] = []
        current_parts: list[str] = []
        current_words = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            words = len(para.split())

            if current_words + words > self.max_words and current_parts:
                chunks.append(Chunk(
                    id="",
                    text="\n\n".join(current_parts),
                    heading_path=[],
                    level=0,
                    page_start=0,
                    page_end=0,
                    source=doc.title,
                ))
                current_parts = []
                current_words = 0

            current_parts.append(para)
            current_words += words

        if current_parts:
            chunks.append(Chunk(
                id="",
                text="\n\n".join(current_parts),
                heading_path=[],
                level=0,
                page_start=0,
                page_end=0,
                source=doc.title,
            ))

        return chunks


def _slugify(text: str) -> str:
    """Simple slugify for IDs."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:50]


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter for French legal text."""
    import re
    # Split on period/question/exclamation followed by space and uppercase
    # But not on abbreviations like "art.", "al.", "ch.", "cf."
    parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÉÈÀÊÂÔÎÛÜ])', text)
    return [p.strip() for p in parts if p.strip()]
