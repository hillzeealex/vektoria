"""
Vektoria REST API — a Pinecone-shaped HTTP layer over the core engine.

Build the app with create_app(); configuration comes from explicit arguments
or, when omitted, from environment variables:
  VK_DATA_DIR      data directory (default "./data")
  VK_API_KEY       optional Bearer key; if set, /v1 routes require it
  VK_CORS_ORIGINS  comma-separated allowed origins (default: none)
"""

import os

from fastapi import FastAPI, HTTPException, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from vektoria import IndexManager


class CreateIndexRequest(BaseModel):
    name: str
    dimension: int = Field(gt=0)
    metric: str = "cosine"


class VectorItem(BaseModel):
    id: str
    values: list[float]
    metadata: dict = {}


class UpsertRequest(BaseModel):
    vectors: list[VectorItem]


class QueryRequest(BaseModel):
    vector: list[float]
    top_k: int = 5
    filter: dict | None = None
    hybrid: bool = False
    alpha: float = 0.5
    text: str | None = None


class DeleteVectorsRequest(BaseModel):
    ids: list[str] | None = None
    filter: dict | None = None


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

    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    bearer = HTTPBearer(auto_error=False)

    def require_key(creds: HTTPAuthorizationCredentials = Depends(bearer)):
        if not api_key:
            return  # auth disabled
        if creds is None or creds.credentials != api_key:
            raise HTTPException(401, "Missing or invalid API key")

    v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_key)])

    def _get_index(name: str):
        try:
            return manager.get(name)
        except KeyError:
            raise HTTPException(404, f"Index {name!r} not found")

    @app.get("/health")
    def health():
        return {"status": "ok", "indexes": len(manager.list_indexes())}

    @v1.post("/indexes", status_code=201)
    def create_index(req: CreateIndexRequest):
        try:
            manager.create_index(req.name, dimension=req.dimension, metric=req.metric)
        except ValueError as e:
            msg = str(e)
            if "already exists" in msg:
                raise HTTPException(409, msg)
            raise HTTPException(400, msg)
        return {"name": req.name, "dimension": req.dimension, "metric": req.metric}

    @v1.get("/indexes")
    def list_indexes():
        return {"indexes": manager.list_indexes()}

    @v1.delete("/indexes/{name}")
    def delete_index(name: str):
        try:
            manager.delete_index(name)
        except KeyError:
            raise HTTPException(404, f"Index {name!r} not found")
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"deleted": name}

    @v1.post("/indexes/{name}/upsert")
    def upsert(name: str, req: UpsertRequest):
        index = _get_index(name)
        items = [{"id": v.id, "values": v.values, "metadata": v.metadata} for v in req.vectors]
        try:
            n = index.upsert(items)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"upserted": n}

    @v1.post("/indexes/{name}/query")
    def query(name: str, req: QueryRequest):
        index = _get_index(name)
        try:
            matches = index.query(
                req.vector, top_k=req.top_k, filter=req.filter,
                hybrid=req.hybrid, alpha=req.alpha, text=req.text,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"matches": [
            {"id": m.id, "score": m.score, "metadata": m.metadata} for m in matches
        ]}

    @v1.post("/indexes/{name}/delete")
    def delete_vectors(name: str, req: DeleteVectorsRequest):
        index = _get_index(name)
        n = index.delete(ids=req.ids, filter=req.filter)
        return {"deleted": n}

    @v1.get("/indexes/{name}/export")
    def export(name: str):
        index = _get_index(name)
        return index.export()

    app.include_router(v1)
    return app
