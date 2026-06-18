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
