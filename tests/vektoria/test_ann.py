"""Optional TurboVec (ANN) backend — skipped when turbovec isn't installed."""

import numpy as np
import pytest

pytest.importorskip("turbovec")

from vektoria import Index, IndexManager


def _orthogonalish(n, dim, rng):
    """n well-separated unit vectors so the nearest neighbour is unambiguous."""
    m = rng.standard_normal((n, dim)).astype(np.float32)
    return (m / np.linalg.norm(m, axis=1, keepdims=True)).tolist()


def test_turbovec_create_persists_backend(tmp_path):
    idx = Index.create(tmp_path / "i", dimension=8, backend="turbovec")
    idx.close()
    re = Index(tmp_path / "i")          # reopen reads backend from meta
    assert re._backend == "turbovec" and re._ann is not None
    re.close()


def test_turbovec_query_finds_nearest(tmp_path):
    rng = np.random.default_rng(1)
    idx = Index.create(tmp_path / "i", dimension=16, backend="turbovec")
    vecs = _orthogonalish(40, 16, rng)
    idx.upsert([{"id": f"v{i}", "values": vecs[i], "metadata": {"text": f"doc {i}"}}
                for i in range(40)])
    assert idx.count() == 40
    # querying with an exact stored vector must return that id first
    hit = idx.query(vecs[7], top_k=1)[0]
    assert hit.id == "v7"
    assert hit.metadata["text"] == "doc 7"
    idx.close()


def test_turbovec_delete_and_filter(tmp_path):
    rng = np.random.default_rng(2)
    idx = Index.create(tmp_path / "i", dimension=16, backend="turbovec")
    vecs = _orthogonalish(30, 16, rng)
    idx.upsert([{"id": f"v{i}", "values": vecs[i],
                 "metadata": {"src": "a" if i < 15 else "b"}} for i in range(30)])
    # filter restricts results to a source
    matches = idx.query(vecs[20], top_k=5, filter={"src": "b"})
    assert all(m.metadata["src"] == "b" for m in matches)
    # real delete removes the vector
    assert idx.delete(ids=["v7"]) == 1
    assert idx.count() == 29
    assert all(m.id != "v7" for m in idx.query(vecs[7], top_k=10))
    idx.close()


def test_turbovec_hybrid(tmp_path):
    idx = Index.create(tmp_path / "i", dimension=8, backend="turbovec")
    idx.upsert([
        {"id": "a", "values": [1, 0, 0, 0, 0, 0, 0, 0], "metadata": {"text": "contrat de bail"}},
        {"id": "b", "values": [1, 0, 0, 0, 0, 0, 0, 0], "metadata": {"text": "clause de résiliation"}},
    ])
    matches = idx.query([1, 0, 0, 0, 0, 0, 0, 0], top_k=2, hybrid=True, text="résiliation")
    assert matches[0].id == "b"   # keyword breaks the vector tie
    idx.close()


def test_turbovec_via_manager(tmp_path):
    mgr = IndexManager(tmp_path)
    mgr.create_index("ann", dimension=8, backend="turbovec")
    info = next(i for i in mgr.list_indexes() if i["name"] == "ann")
    assert info["dimension"] == 8
    idx = mgr.get("ann")
    idx.upsert([{"id": "x", "values": [1, 0, 0, 0, 0, 0, 0, 0], "metadata": {}}])
    assert idx.query([1, 0, 0, 0, 0, 0, 0, 0], top_k=1)[0].id == "x"
