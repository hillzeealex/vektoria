"""
Vektoria REST API — a Pinecone-shaped HTTP layer over the core engine.

Build the app with create_app(); configuration comes from explicit arguments
or, when omitted, from environment variables:
  VK_DATA_DIR      data directory (default "./data")
  VK_API_KEY       optional Bearer key; if set, /v1 routes require it
  VK_CORS_ORIGINS  comma-separated allowed origins (default: none)
"""

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from vektoria import IndexManager


class CreateIndexRequest(BaseModel):
    name: str
    dimension: int = Field(gt=0)
    metric: str = "cosine"


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

    @app.post("/v1/indexes", status_code=201)
    def create_index(req: CreateIndexRequest):
        try:
            manager.create_index(req.name, dimension=req.dimension, metric=req.metric)
        except ValueError as e:
            msg = str(e)
            if "already exists" in msg:
                raise HTTPException(409, msg)
            raise HTTPException(400, msg)
        return {"name": req.name, "dimension": req.dimension, "metric": req.metric}

    @app.get("/v1/indexes")
    def list_indexes():
        return {"indexes": manager.list_indexes()}

    @app.delete("/v1/indexes/{name}")
    def delete_index(name: str):
        try:
            manager.delete_index(name)
        except KeyError:
            raise HTTPException(404, f"Index {name!r} not found")
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"deleted": name}

    return app
