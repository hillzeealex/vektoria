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
