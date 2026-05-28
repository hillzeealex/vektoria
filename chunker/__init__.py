from .chunker import SemanticChunker, Chunk
from .recursive import RecursiveChunker
from .similarity import SimilarityChunker


def create_chunker(strategy: str = "semantic", **kwargs):
    """
    Factory to create a chunker by strategy name.

    Args:
        strategy: One of "semantic", "recursive", or "similarity".
        **kwargs: Forwarded to the chosen chunker's constructor.
            - semantic: max_words, min_words, overlap_sentences
            - recursive: max_words, overlap_words, separators
            - similarity: embedder (required), similarity_threshold, max_words, min_words

    Returns:
        A chunker instance with a .chunk() method.
    """
    if strategy == "semantic":
        return SemanticChunker(**kwargs)
    elif strategy == "recursive":
        return RecursiveChunker(**kwargs)
    elif strategy == "similarity":
        return SimilarityChunker(**kwargs)
    else:
        raise ValueError(
            f"Unknown chunking strategy: {strategy!r}. "
            f"Choose from: 'semantic', 'recursive', 'similarity'."
        )


__all__ = [
    "Chunk",
    "SemanticChunker",
    "RecursiveChunker",
    "SimilarityChunker",
    "create_chunker",
]
