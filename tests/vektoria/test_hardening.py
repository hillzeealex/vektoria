"""Senior-hardening tests: incremental cache, accurate delete, thread-safety,
and cache-friendly listing."""

import threading

from vektoria import Index, IndexManager


def _v(*xs):
    return [float(x) for x in xs]


def test_replace_updates_vector_and_metadata(tmp_path):
    idx = Index.create(tmp_path / "i", dimension=2)
    idx.upsert([{"id": "a", "values": _v(1, 0), "metadata": {"text": "old", "v": 1}}])
    idx.upsert([{"id": "a", "values": _v(0, 1), "metadata": {"text": "new", "v": 2}}])
    assert idx.count() == 1
    m = idx.query(_v(0, 1), top_k=1)[0]  # vector was replaced (1,0)->(0,1)
    assert m.id == "a"
    assert m.metadata == {"text": "new", "v": 2}
    assert m.score > 0.99
    idx.close()


def test_incremental_upsert_across_batches(tmp_path):
    idx = Index.create(tmp_path / "i", dimension=3)
    idx.upsert([{"id": f"a{i}", "values": _v(1, 0, 0), "metadata": {}} for i in range(5)])
    idx.upsert([{"id": f"b{i}", "values": _v(0, 1, 0), "metadata": {}} for i in range(5)])
    assert idx.count() == 10
    assert idx._matrix.shape == (10, 3)
    assert len(idx.export()["vectors"]) == 10
    idx.close()


def test_delete_count_ignores_nonexistent_ids(tmp_path):
    idx = Index.create(tmp_path / "i", dimension=2)
    idx.upsert([{"id": "a", "values": _v(1, 0), "metadata": {}}])
    assert idx.delete(ids=["a", "ghost"]) == 1  # only 'a' actually existed
    assert idx.delete(ids=["ghost"]) == 0
    assert idx.count() == 0
    idx.close()


def test_query_metadata_served_from_memory(tmp_path):
    idx = Index.create(tmp_path / "i", dimension=2)
    idx.upsert([{"id": "a", "values": _v(1, 0), "metadata": {"text": "x", "k": [1, 2]}}])
    m = idx.query(_v(1, 0), top_k=1)[0]
    assert m.metadata == {"text": "x", "k": [1, 2]}
    idx.close()


def test_concurrent_upsert_and_query_is_safe(tmp_path):
    idx = Index.create(tmp_path / "i", dimension=3)
    errors = []

    def writer(start):
        try:
            for i in range(start, start + 50):
                idx.upsert([{"id": f"v{i}", "values": _v(1, 0, 0), "metadata": {"text": "t"}}])
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def reader():
        try:
            for _ in range(50):
                idx.query(_v(1, 0, 0), top_k=5)
                idx.query(_v(1, 0, 0), top_k=5, hybrid=True, text="t")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [
        threading.Thread(target=writer, args=(0,)),
        threading.Thread(target=writer, args=(50,)),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert idx.count() == 100  # two writers × 50 distinct ids
    idx.close()


def test_list_indexes_does_not_pollute_lru_cache(tmp_path):
    mgr = IndexManager(tmp_path, cache_size=2)
    for n in ["i1", "i2", "i3", "i4"]:
        mgr.create_index(n, dimension=2)
    listed = mgr.list_indexes()
    assert {d["name"] for d in listed} == {"i1", "i2", "i3", "i4"}
    assert all(d["count"] == 0 and d["dimension"] == 2 for d in listed)
    assert len(mgr._open) == 0  # listing opened/cached nothing
