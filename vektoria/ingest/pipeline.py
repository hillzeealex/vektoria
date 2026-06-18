"""Document ingestion pipeline: extract → chunk → embed → upsert."""

from __future__ import annotations

from .chunk import chunk_text
from .extract import extract_text


class Ingestor:
    """Turns an uploaded document into stored vectors using a given embedder."""

    def __init__(self, embedder):
        self.embedder = embedder

    def ingest(
        self,
        data: bytes,
        filename: str,
        index,
        *,
        max_words: int = 400,
        overlap: int = 40,
    ) -> dict:
        if index.dimension != self.embedder.dimension:
            raise ValueError(
                f"Index dimension {index.dimension} does not match embedder "
                f"dimension {self.embedder.dimension}"
            )

        text = extract_text(data, filename)
        chunks = chunk_text(text, max_words=max_words, overlap=overlap)
        if not chunks:
            return {"source": filename, "chunks": 0, "upserted": 0}

        vectors = self.embedder.embed_documents(chunks)
        items = [
            {
                "id": f"{filename}#{i}",
                "values": vectors[i].tolist(),
                "metadata": {"text": chunks[i], "source": filename, "chunk": i},
            }
            for i in range(len(chunks))
        ]
        upserted = index.upsert(items)
        return {"source": filename, "chunks": len(chunks), "upserted": upserted}
