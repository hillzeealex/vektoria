"""
SwissVectorStore REST API.

Endpoints:
  POST /ingest     — Upload a PDF, extract, chunk, embed, store
  POST /query      — Search the vector store with a question
  GET  /documents  — List all indexed documents
  DELETE /documents/{source} — Remove a document from the store

Run with: uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pdf_extractor import PdfExtractor
from chunker import SemanticChunker
from embedder import LocalEmbedder
from vector_store import VectorStore


# ── Config ───────────────────────────────────────────────────────

DATA_DIR = os.environ.get("SVS_DATA_DIR", "./data")
EMBED_BACKEND = os.environ.get("SVS_EMBED_BACKEND", "numpy")
EMBED_MODEL = os.environ.get("SVS_EMBED_MODEL", "intfloat/multilingual-e5-large")
EMBED_DIM = int(os.environ.get("SVS_EMBED_DIM", "384"))
OLLAMA_URL = os.environ.get("SVS_OLLAMA_URL", "http://localhost:11434")

# ── Shared instances ─────────────────────────────────────────────

extractor = PdfExtractor()
chunker = SemanticChunker(max_words=800, min_words=50)
embedder: LocalEmbedder | None = None
store: VectorStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, store
    embedder = LocalEmbedder(
        backend=EMBED_BACKEND,
        model_name=EMBED_MODEL,
        ollama_url=OLLAMA_URL,
        dimension=EMBED_DIM,
    )
    store = VectorStore(DATA_DIR)
    print(f"[api] Store loaded: {store.count} chunks in {DATA_DIR}")
    yield
    if store:
        store.close()


app = FastAPI(
    title="SwissVectorStore",
    description="RAG pipeline 100% souverain — zero API cloud",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ──────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    source_filter: str | None = None


class QueryResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    heading_path: list[str]
    page_start: int
    page_end: int
    source: str
    word_count: int


class QueryResponse(BaseModel):
    question: str
    results: list[QueryResult]
    search_time_ms: float


class IngestResponse(BaseModel):
    source: str
    pages: int
    sections: int
    chunks: int
    total_time_ms: float
    timings: dict[str, float]


class DocumentInfo(BaseModel):
    source: str
    chunks: int
    words: int


# ── Endpoints ────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """Upload a PDF, extract, chunk, embed, and store it."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(400, "Empty file")

    timings: dict[str, float] = {}
    t0 = time.time()

    # 1. Extract
    t = time.time()
    doc = extractor.extract(content)
    timings["extract_ms"] = (time.time() - t) * 1000

    # 2. Chunk
    t = time.time()
    chunks = chunker.chunk(doc)
    timings["chunk_ms"] = (time.time() - t) * 1000

    if not chunks:
        raise HTTPException(422, "No content could be extracted from this PDF")

    # 3. Embed
    t = time.time()
    embedded = embedder.embed_chunks(chunks)
    timings["embed_ms"] = (time.time() - t) * 1000

    # 4. Store
    t = time.time()
    store.add(embedded)
    timings["store_ms"] = (time.time() - t) * 1000

    total = (time.time() - t0) * 1000

    return IngestResponse(
        source=doc.title,
        pages=doc.page_count,
        sections=len(doc.sections),
        chunks=len(chunks),
        total_time_ms=round(total, 1),
        timings={k: round(v, 1) for k, v in timings.items()},
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Search the vector store with a natural language question."""
    if store.count == 0:
        raise HTTPException(404, "No documents indexed yet. Use POST /ingest first.")

    t = time.time()
    query_vec = embedder.embed_query(req.question)
    results = store.search(query_vec, top_k=req.top_k, source_filter=req.source_filter)
    search_ms = (time.time() - t) * 1000

    return QueryResponse(
        question=req.question,
        results=[
            QueryResult(
                chunk_id=r.chunk_id,
                text=r.text,
                score=round(r.score, 4),
                heading_path=r.heading_path,
                page_start=r.page_start,
                page_end=r.page_end,
                source=r.source,
                word_count=r.word_count,
            )
            for r in results
        ],
        search_time_ms=round(search_ms, 2),
    )


@app.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    """List all indexed documents."""
    sources = store.list_sources()
    return [
        DocumentInfo(source=s["source"], chunks=s["chunks"], words=s["words"])
        for s in sources
    ]


@app.delete("/documents/{source}")
async def delete_document(source: str):
    """Remove all chunks from a specific document."""
    deleted = store.delete_source(source)
    if deleted == 0:
        raise HTTPException(404, f"Document '{source}' not found")
    return {"deleted": deleted, "source": source}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "documents": len(store.list_sources()),
        "total_chunks": store.count,
        "embed_backend": EMBED_BACKEND,
        "embed_model": EMBED_MODEL,
    }
