"""
Local embedding module — zero external API calls.

Supports three backends:
1. sentence-transformers (Python, works on CPU) — for production on VPS
2. Ollama REST API (for when Ollama is running locally)
3. numpy-random (fake embeddings for testing the pipeline without ML deps)

Default model: multilingual-e5-large (best for French legal text on CPU).
"""

import numpy as np
from dataclasses import dataclass
from chunker.chunker import Chunk


@dataclass
class EmbeddedChunk:
    """A chunk with its embedding vector."""
    chunk: Chunk
    vector: np.ndarray  # shape: (dim,)


class LocalEmbedder:
    """
    Embed text chunks using a local model. No external API calls.

    Usage:
        embedder = LocalEmbedder()  # loads model on first call
        vectors = embedder.embed_chunks(chunks)
    """

    def __init__(
        self,
        *,
        backend: str = "sentence-transformers",
        model_name: str = "intfloat/multilingual-e5-large",
        ollama_url: str = "http://localhost:11434",
        batch_size: int = 32,
        prefix: str = "passage: ",
        dimension: int = 384,
    ):
        """
        Args:
            backend: "sentence-transformers", "ollama", or "numpy" (test mode)
            model_name: HuggingFace model name or Ollama model name
            ollama_url: Ollama API base URL (only for ollama backend)
            batch_size: Number of texts to embed at once
            prefix: Prefix for E5 models ("passage: " for docs, "query: " for queries)
            dimension: Embedding dimension (only used for numpy backend)
        """
        self.backend = backend
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.batch_size = batch_size
        self.prefix = prefix
        self._model = None
        self._dim: int = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts. Returns array of shape (n, dim)."""
        self._ensure_loaded()
        prefixed = [self.prefix + t for t in texts]

        if self.backend == "numpy":
            return self._embed_numpy(prefixed)
        elif self.backend == "sentence-transformers":
            return self._embed_st(prefixed)
        elif self.backend == "ollama":
            return self._embed_ollama(prefixed)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query (uses 'query: ' prefix for E5 models)."""
        self._ensure_loaded()
        prefixed = "query: " + query

        if self.backend == "numpy":
            return self._embed_numpy([prefixed])[0]
        elif self.backend == "sentence-transformers":
            return self._embed_st([prefixed])[0]
        elif self.backend == "ollama":
            return self._embed_ollama([prefixed])[0]
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a list of chunks. Returns EmbeddedChunk objects."""
        texts = [c.text for c in chunks]
        all_vectors: list[np.ndarray] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vectors = self.embed_texts(batch)
            all_vectors.append(vectors)

        stacked = np.vstack(all_vectors)
        return [
            EmbeddedChunk(chunk=chunk, vector=stacked[i])
            for i, chunk in enumerate(chunks)
        ]

    # ── Backends ─────────────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._model is not None:
            return

        if self.backend == "numpy":
            self._model = True
            print(f"[embedder] Numpy test backend ready. Dimension: {self._dim}")
        elif self.backend == "sentence-transformers":
            self._load_st()
        elif self.backend == "ollama":
            self._check_ollama()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ── numpy (test/demo) ────────────────────────────────────────────

    def _embed_numpy(self, texts: list[str]) -> np.ndarray:
        """
        Deterministic pseudo-embeddings based on text hash.
        NOT for production — only for testing the pipeline end-to-end.
        Uses a hash-seeded RNG so the same text always gives the same vector.
        """
        vectors = []
        for text in texts:
            seed = hash(text) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self._dim).astype(np.float32)
            vec /= np.linalg.norm(vec)  # L2 normalize
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

    # ── sentence-transformers ────────────────────────────────────────

    def _load_st(self):
        from sentence_transformers import SentenceTransformer

        print(f"[embedder] Loading {self.model_name}...")
        self._model = SentenceTransformer(self.model_name)
        self._dim = self._model.get_sentence_embedding_dimension()
        print(f"[embedder] Loaded. Dimension: {self._dim}")

    def _embed_st(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
            batch_size=self.batch_size,
        )
        return np.array(vectors, dtype=np.float32)

    # ── Ollama ───────────────────────────────────────────────────────

    def _check_ollama(self):
        import urllib.request
        import json

        try:
            req = urllib.request.Request(f"{self.ollama_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                if self.model_name not in models:
                    print(f"[embedder] Model {self.model_name} not found.")
                    print(f"[embedder] Available: {models}")
                    raise RuntimeError(f"Model {self.model_name} not available")
        except urllib.error.URLError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.ollama_url}. "
                "Start with: ollama serve"
            )

        test_vec = self._ollama_embed_single("test")
        self._dim = len(test_vec)
        self._model = True
        print(f"[embedder] Ollama ready. Model: {self.model_name}, Dim: {self._dim}")

    def _embed_ollama(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vec = self._ollama_embed_single(text)
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

    def _ollama_embed_single(self, text: str) -> list[float]:
        import urllib.request
        import json

        payload = json.dumps({
            "model": self.model_name,
            "prompt": text,
        }).encode()

        req = urllib.request.Request(
            f"{self.ollama_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["embedding"]
