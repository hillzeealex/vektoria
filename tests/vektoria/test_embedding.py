"""Embedding backend tests — exercised via the dependency-free HashEmbedder."""

import numpy as np
import pytest

from vektoria.embedding import Embedder, HashEmbedder, make_embedder


def test_hash_embedder_is_deterministic_across_instances():
    a = HashEmbedder(dimension=32)
    b = HashEmbedder(dimension=32)
    # Stable hash → same text yields the same vector even from a fresh instance
    # (this is the fix for Python's per-process-salted hash()).
    assert np.allclose(a.embed_query("clause"), b.embed_query("clause"))
    assert not np.allclose(a.embed_query("clause"), a.embed_query("autre"))


def test_vectors_are_unit_norm():
    emb = HashEmbedder(dimension=64)
    docs = emb.embed_documents(["a", "b", "c"])
    assert docs.shape == (3, 64)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-5)
    q = emb.embed_query("x")
    assert q.shape == (64,)
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-5)


def test_embed_documents_empty_returns_empty_matrix():
    emb = HashEmbedder(dimension=16)
    out = emb.embed_documents([])
    assert out.shape == (0, 16)


def test_dtype_is_float32():
    emb = HashEmbedder(dimension=8)
    assert emb.embed_documents(["a"]).dtype == np.float32
    assert emb.embed_query("a").dtype == np.float32


def test_satisfies_embedder_protocol():
    assert isinstance(HashEmbedder(dimension=8), Embedder)


def test_make_embedder_factory():
    emb = make_embedder("hash", dimension=10)
    assert isinstance(emb, HashEmbedder) and emb.dimension == 10
    with pytest.raises(ValueError):
        make_embedder("nope")
