"""
Vektoria REST API — a Pinecone-shaped HTTP layer over the core engine.

Build the app with create_app(); configuration comes from explicit arguments
or, when omitted, from environment variables:
  VK_DATA_DIR      data directory (default "./data")
  VK_API_KEY       optional Bearer key; if set, /v1 routes require it
  VK_CORS_ORIGINS  comma-separated allowed origins (default: none)
"""

import os

from fastapi import FastAPI

from vektoria import IndexManager


def create_app(data_dir=None, api_key=None, cors_origins=None) -> FastAPI:
    data_dir = data_dir if data_dir is not None else os.environ.get("VK_DATA_DIR", "./data")
    api_key = api_key if api_key is not None else os.environ.get("VK_API_KEY")
    if cors_origins is None:
        raw = os.environ.get("VK_CORS_ORIGINS", "")
        cors_origins = [o.strip() for o in raw.split(",") if o.strip()]

    manager = IndexManager(data_dir)
    app = FastAPI(title="Vektoria", version="1.0.0")
    app.state.manager = manager
    app.state.api_key = api_key

    @app.get("/health")
    def health():
        return {"status": "ok", "indexes": len(manager.list_indexes())}

    return app
