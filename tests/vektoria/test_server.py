# tests/vektoria/test_server.py
from fastapi.testclient import TestClient

from vektoria.embedding import HashEmbedder
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


def test_auth_required_when_key_set(tmp_path):
    c = _client(tmp_path, api_key="secret")
    assert c.get("/health").status_code == 200  # health stays open
    assert c.get("/v1/indexes").status_code == 401  # no key
    assert c.get("/v1/indexes", headers={"Authorization": "Bearer nope"}).status_code == 401
    ok = c.get("/v1/indexes", headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200


def test_no_auth_when_key_unset(tmp_path):
    c = _client(tmp_path)  # no api_key
    assert c.get("/v1/indexes").status_code == 200


def test_cors_origin_allowed_when_configured(tmp_path):
    c = _client(tmp_path, cors_origins=["https://app.example.com"])
    r = c.get("/health", headers={"Origin": "https://app.example.com"})
    assert r.headers.get("access-control-allow-origin") == "https://app.example.com"


# ── ingestion + server-side text query ───────────────────────────────
def test_ingest_then_text_query(tmp_path):
    emb = HashEmbedder(dimension=32)
    c = _client(tmp_path, embedder=emb)
    c.post("/v1/indexes", json={"name": "docs", "dimension": 32})

    text = (" ".join(f"w{i}" for i in range(120))).encode()
    r = c.post(
        "/v1/indexes/docs/ingest?max_words=50&overlap=10",
        files={"file": ("doc.txt", text, "text/plain")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "doc.txt" and body["chunks"] >= 2
    assert body["chunks"] == body["upserted"]

    # text-only query: the server embeds it (no client-side vector)
    r = c.post("/v1/indexes/docs/query", json={"text": "w0", "top_k": 1})
    assert r.status_code == 200
    assert r.json()["matches"][0]["metadata"]["source"] == "doc.txt"


def test_query_without_vector_or_text_400(tmp_path):
    c = _client(tmp_path, embedder=HashEmbedder(8))
    c.post("/v1/indexes", json={"name": "docs", "dimension": 8})
    assert c.post("/v1/indexes/docs/query", json={"top_k": 3}).status_code == 400


def test_ingest_too_large_returns_413(tmp_path):
    c = _client(tmp_path, embedder=HashEmbedder(8), max_upload_mb=0.001)  # ~1 KB cap
    c.post("/v1/indexes", json={"name": "docs", "dimension": 8})
    r = c.post("/v1/indexes/docs/ingest", files={"file": ("big.txt", b"x" * 5000, "text/plain")})
    assert r.status_code == 413


def test_ingest_dimension_mismatch_400(tmp_path):
    c = _client(tmp_path, embedder=HashEmbedder(32))
    c.post("/v1/indexes", json={"name": "docs", "dimension": 8})  # != embedder 32
    r = c.post("/v1/indexes/docs/ingest", files={"file": ("a.txt", b"hello world", "text/plain")})
    assert r.status_code == 400
