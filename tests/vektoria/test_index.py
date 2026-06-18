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
