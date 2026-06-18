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
