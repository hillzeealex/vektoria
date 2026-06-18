# tests/vektoria/test_index.py
from vektoria import Index


def test_create_persists_dimension_and_metric(tmp_path):
    idx = Index.create(tmp_path / "myindex", dimension=4, metric="cosine")
    assert idx.dimension == 4
    assert idx.metric == "cosine"
    assert idx.count() == 0
    idx.close()

    reopened = Index(tmp_path / "myindex")
    assert reopened.dimension == 4
    assert reopened.metric == "cosine"
    reopened.close()


def test_create_rejects_unsupported_metric(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        Index.create(tmp_path / "bad", dimension=4, metric="euclidean")


# append to tests/vektoria/test_index.py
import numpy as np
import pytest
from vektoria import Index


def _vec(*xs):
    return [float(x) for x in xs]


def test_upsert_stores_and_counts(tmp_path):
    idx = Index.create(tmp_path / "i", dimension=3)
    n = idx.upsert([
        {"id": "a", "values": _vec(1, 0, 0), "metadata": {"text": "alpha"}},
        {"id": "b", "values": _vec(0, 1, 0), "metadata": {"text": "beta"}},
    ])
    assert n == 2
    assert idx.count() == 2
    idx.close()


def test_upsert_normalizes_vectors(tmp_path):
    idx = Index.create(tmp_path / "i", dimension=3)
    idx.upsert([{"id": "a", "values": _vec(3, 4, 0), "metadata": {}}])  # norm 5
    stored = idx._matrix[0]
    assert np.isclose(np.linalg.norm(stored), 1.0, atol=1e-5)
    assert np.allclose(stored, [0.6, 0.8, 0.0], atol=1e-5)
    idx.close()


def test_upsert_replaces_existing_id(tmp_path):
    idx = Index.create(tmp_path / "i", dimension=3)
    idx.upsert([{"id": "a", "values": _vec(1, 0, 0), "metadata": {"text": "v1"}}])
    idx.upsert([{"id": "a", "values": _vec(0, 1, 0), "metadata": {"text": "v2"}}])
    assert idx.count() == 1  # replaced, not duplicated
    idx.close()


def test_upsert_rejects_wrong_dimension(tmp_path):
    idx = Index.create(tmp_path / "i", dimension=3)
    with pytest.raises(ValueError):
        idx.upsert([{"id": "a", "values": _vec(1, 0), "metadata": {}}])
    idx.close()


# append to tests/vektoria/test_index.py

def test_query_returns_nearest_first(tmp_path):
    idx = Index.create(tmp_path / "i", dimension=3)
    idx.upsert([
        {"id": "x", "values": _vec(1, 0, 0), "metadata": {"text": "x"}},
        {"id": "y", "values": _vec(0, 1, 0), "metadata": {"text": "y"}},
        {"id": "z", "values": _vec(0.9, 0.1, 0), "metadata": {"text": "z"}},
    ])
    matches = idx.query(_vec(1, 0, 0), top_k=2)
    assert [m.id for m in matches] == ["x", "z"]
    assert matches[0].score > matches[1].score
    assert matches[0].metadata["text"] == "x"
    idx.close()


def test_query_empty_index_returns_empty(tmp_path):
    idx = Index.create(tmp_path / "i", dimension=3)
    assert idx.query(_vec(1, 0, 0), top_k=5) == []
    idx.close()


# append to tests/vektoria/test_index.py

def test_query_filters_by_exact_match(tmp_path):
    idx = Index.create(tmp_path / "i", dimension=3)
    idx.upsert([
        {"id": "a", "values": _vec(1, 0, 0), "metadata": {"source": "doc1"}},
        {"id": "b", "values": _vec(0.99, 0.01, 0), "metadata": {"source": "doc2"}},
    ])
    matches = idx.query(_vec(1, 0, 0), top_k=5, filter={"source": "doc2"})
    assert [m.id for m in matches] == ["b"]
    idx.close()


def test_query_filters_by_list_membership(tmp_path):
    idx = Index.create(tmp_path / "i", dimension=3)
    idx.upsert([
        {"id": "a", "values": _vec(1, 0, 0), "metadata": {"source": "doc1"}},
        {"id": "b", "values": _vec(0.9, 0.1, 0), "metadata": {"source": "doc2"}},
        {"id": "c", "values": _vec(0.8, 0.2, 0), "metadata": {"source": "doc3"}},
    ])
    matches = idx.query(_vec(1, 0, 0), top_k=5, filter={"source": ["doc1", "doc3"]})
    assert sorted(m.id for m in matches) == ["a", "c"]
    idx.close()


# append to tests/vektoria/test_index.py

def test_delete_by_ids_removes_rows_and_vectors(tmp_path):
    idx = Index.create(tmp_path / "i", dimension=3)
    idx.upsert([
        {"id": "a", "values": _vec(1, 0, 0), "metadata": {"text": "a"}},
        {"id": "b", "values": _vec(0, 1, 0), "metadata": {"text": "b"}},
    ])
    deleted = idx.delete(ids=["a"])
    assert deleted == 1
    assert idx.count() == 1
    matches = idx.query(_vec(1, 0, 0), top_k=5)
    assert all(m.id != "a" for m in matches)
    assert idx._matrix.shape[0] == 1
    idx.close()


def test_delete_by_filter(tmp_path):
    idx = Index.create(tmp_path / "i", dimension=3)
    idx.upsert([
        {"id": "a", "values": _vec(1, 0, 0), "metadata": {"source": "doc1"}},
        {"id": "b", "values": _vec(0, 1, 0), "metadata": {"source": "doc1"}},
        {"id": "c", "values": _vec(0, 0, 1), "metadata": {"source": "doc2"}},
    ])
    deleted = idx.delete(filter={"source": "doc1"})
    assert deleted == 2
    assert idx.count() == 1
    idx.close()


# append to tests/vektoria/test_index.py

def test_hybrid_query_surfaces_keyword_match(tmp_path):
    idx = Index.create(tmp_path / "i", dimension=3)
    idx.upsert([
        {"id": "a", "values": _vec(1, 0, 0), "metadata": {"text": "contrat de bail"}},
        {"id": "b", "values": _vec(1, 0, 0), "metadata": {"text": "clause de non-concurrence"}},
    ])
    matches = idx.query(
        _vec(1, 0, 0), top_k=2, hybrid=True, alpha=0.5, text="non-concurrence"
    )
    assert matches[0].id == "b"  # keyword breaks the tie
    idx.close()


def test_hybrid_requires_text(tmp_path):
    import pytest
    idx = Index.create(tmp_path / "i", dimension=3)
    idx.upsert([{"id": "a", "values": _vec(1, 0, 0), "metadata": {"text": "x"}}])
    with pytest.raises(ValueError):
        idx.query(_vec(1, 0, 0), hybrid=True)  # no text
    idx.close()
