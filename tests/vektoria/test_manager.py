# tests/vektoria/test_manager.py
import pytest
from vektoria import IndexManager


def test_create_and_list(tmp_path):
    mgr = IndexManager(tmp_path)
    mgr.create_index("docs", dimension=4)
    mgr.create_index("notes", dimension=8)
    names = {i["name"] for i in mgr.list_indexes()}
    assert names == {"docs", "notes"}
    info = next(i for i in mgr.list_indexes() if i["name"] == "docs")
    assert info["dimension"] == 4
    assert info["count"] == 0


def test_create_duplicate_raises(tmp_path):
    mgr = IndexManager(tmp_path)
    mgr.create_index("docs", dimension=4)
    with pytest.raises(ValueError):
        mgr.create_index("docs", dimension=4)


def test_get_returns_usable_index(tmp_path):
    mgr = IndexManager(tmp_path)
    mgr.create_index("docs", dimension=3)
    idx = mgr.get("docs")
    idx.upsert([{"id": "a", "values": [1.0, 0.0, 0.0], "metadata": {}}])
    assert mgr.get("docs").count() == 1


def test_get_missing_raises(tmp_path):
    mgr = IndexManager(tmp_path)
    with pytest.raises(KeyError):
        mgr.get("nope")


def test_delete_index_removes_data(tmp_path):
    mgr = IndexManager(tmp_path)
    mgr.create_index("docs", dimension=3)
    mgr.delete_index("docs")
    assert mgr.list_indexes() == []
    assert not (tmp_path / "docs").exists()


def test_rejects_unsafe_index_name(tmp_path):
    mgr = IndexManager(tmp_path)
    for bad in ["../escape", "a/b", "", "."]:
        with pytest.raises(ValueError):
            mgr.create_index(bad, dimension=3)


def test_lru_eviction_does_not_lose_data(tmp_path):
    mgr = IndexManager(tmp_path, cache_size=2)
    for n in ["i1", "i2", "i3"]:
        mgr.create_index(n, dimension=3)
        mgr.get(n).upsert([{"id": "a", "values": [1.0, 0.0, 0.0], "metadata": {}}])
    assert len(mgr._open) <= 2
    assert mgr.get("i1").count() == 1  # data still on disk, reopened cleanly
