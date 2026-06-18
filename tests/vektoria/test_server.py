# tests/vektoria/test_server.py
from fastapi.testclient import TestClient
from vektoria.server import create_app


def _client(tmp_path, **kw):
    app = create_app(data_dir=str(tmp_path / "data"), **kw)
    return TestClient(app)


def test_health_ok(tmp_path):
    c = _client(tmp_path)
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["indexes"] == 0


def test_create_list_delete_index(tmp_path):
    c = _client(tmp_path)

    r = c.post("/v1/indexes", json={"name": "docs", "dimension": 4})
    assert r.status_code == 201
    assert r.json()["name"] == "docs"

    r = c.get("/v1/indexes")
    assert r.status_code == 200
    items = r.json()["indexes"]
    assert len(items) == 1 and items[0]["name"] == "docs" and items[0]["dimension"] == 4

    r = c.delete("/v1/indexes/docs")
    assert r.status_code == 200
    assert c.get("/v1/indexes").json()["indexes"] == []


def test_create_duplicate_returns_409(tmp_path):
    c = _client(tmp_path)
    c.post("/v1/indexes", json={"name": "docs", "dimension": 4})
    r = c.post("/v1/indexes", json={"name": "docs", "dimension": 4})
    assert r.status_code == 409


def test_create_bad_metric_returns_400(tmp_path):
    c = _client(tmp_path)
    r = c.post("/v1/indexes", json={"name": "docs", "dimension": 4, "metric": "euclidean"})
    assert r.status_code == 400


def test_delete_missing_index_returns_404(tmp_path):
    c = _client(tmp_path)
    assert c.delete("/v1/indexes/nope").status_code == 404


def test_upsert_vectors(tmp_path):
    c = _client(tmp_path)
    c.post("/v1/indexes", json={"name": "docs", "dimension": 3})
    r = c.post("/v1/indexes/docs/upsert", json={"vectors": [
        {"id": "a", "values": [1, 0, 0], "metadata": {"text": "alpha"}},
        {"id": "b", "values": [0, 1, 0], "metadata": {"text": "beta"}},
    ]})
    assert r.status_code == 200
    assert r.json()["upserted"] == 2
    assert c.get("/v1/indexes").json()["indexes"][0]["count"] == 2


def test_upsert_missing_index_404(tmp_path):
    c = _client(tmp_path)
    r = c.post("/v1/indexes/nope/upsert", json={"vectors": [
        {"id": "a", "values": [1, 0, 0], "metadata": {}},
    ]})
    assert r.status_code == 404


def test_upsert_wrong_dimension_400(tmp_path):
    c = _client(tmp_path)
    c.post("/v1/indexes", json={"name": "docs", "dimension": 3})
    r = c.post("/v1/indexes/docs/upsert", json={"vectors": [
        {"id": "a", "values": [1, 0], "metadata": {}},
    ]})
    assert r.status_code == 400


def test_query_returns_matches(tmp_path):
    c = _client(tmp_path)
    c.post("/v1/indexes", json={"name": "docs", "dimension": 3})
    c.post("/v1/indexes/docs/upsert", json={"vectors": [
        {"id": "x", "values": [1, 0, 0], "metadata": {"text": "x"}},
        {"id": "z", "values": [0.9, 0.1, 0], "metadata": {"text": "z"}},
        {"id": "y", "values": [0, 1, 0], "metadata": {"text": "y"}},
    ]})
    r = c.post("/v1/indexes/docs/query", json={"vector": [1, 0, 0], "top_k": 2})
    assert r.status_code == 200
    matches = r.json()["matches"]
    assert [m["id"] for m in matches] == ["x", "z"]
    assert "score" in matches[0] and matches[0]["metadata"]["text"] == "x"


def test_query_hybrid_requires_text_400(tmp_path):
    c = _client(tmp_path)
    c.post("/v1/indexes", json={"name": "docs", "dimension": 3})
    c.post("/v1/indexes/docs/upsert", json={"vectors": [
        {"id": "a", "values": [1, 0, 0], "metadata": {"text": "x"}},
    ]})
    r = c.post("/v1/indexes/docs/query", json={"vector": [1, 0, 0], "hybrid": True})
    assert r.status_code == 400


def test_query_missing_index_404(tmp_path):
    c = _client(tmp_path)
    r = c.post("/v1/indexes/nope/query", json={"vector": [1, 0, 0]})
    assert r.status_code == 404


def test_delete_vectors_by_filter(tmp_path):
    c = _client(tmp_path)
    c.post("/v1/indexes", json={"name": "docs", "dimension": 3})
    c.post("/v1/indexes/docs/upsert", json={"vectors": [
        {"id": "a", "values": [1, 0, 0], "metadata": {"source": "s1"}},
        {"id": "b", "values": [0, 1, 0], "metadata": {"source": "s1"}},
        {"id": "c", "values": [0, 0, 1], "metadata": {"source": "s2"}},
    ]})
    r = c.post("/v1/indexes/docs/delete", json={"filter": {"source": "s1"}})
    assert r.status_code == 200
    assert r.json()["deleted"] == 2
    assert c.get("/v1/indexes").json()["indexes"][0]["count"] == 1


def test_export_returns_dump(tmp_path):
    c = _client(tmp_path)
    c.post("/v1/indexes", json={"name": "docs", "dimension": 3})
    c.post("/v1/indexes/docs/upsert", json={"vectors": [
        {"id": "a", "values": [1, 0, 0], "metadata": {"text": "alpha"}},
    ]})
    r = c.get("/v1/indexes/docs/export")
    assert r.status_code == 200
    body = r.json()
    assert body["dimension"] == 3 and len(body["vectors"]) == 1
    assert body["vectors"][0]["id"] == "a"


def test_export_missing_index_404(tmp_path):
    c = _client(tmp_path)
    assert c.get("/v1/indexes/nope/export").status_code == 404
