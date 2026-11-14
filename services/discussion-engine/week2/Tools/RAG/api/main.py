from __future__ import annotations

import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from helper import close_connection_pool, get_settings  # noqa: E402
from retrieval import _load_index, get_dense, get_reranker, run_search  # noqa: E402

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("api")

def _warm_component(name: str, loader) -> None:
    started_at = time.monotonic()
    logger.info("Warming %s...", name)
    try:
        loader()
    except Exception:
        logger.exception("Failed to warm %s after %.2fs", name, time.monotonic() - started_at)
        raise
    logger.info("Warmed %s in %.2fs", name, time.monotonic() - started_at)

# ── Lifespan ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):

    default_threads = min(os.cpu_count() or 4, 8)
    torch.set_num_threads(int(os.getenv("TORCH_THREADS", default_threads)))

    _warm_component("dense embedding model", get_dense)

    # Warm up reranker if enabled
    settings = get_settings()
    if settings.retrieval_use_reranker:
        _warm_component("cross-encoder reranker", get_reranker)

    # Warm up BM25 index if enabled
    if settings.retrieval_use_bm25:
        _warm_component("BM25 index", _load_index)

    try:
        yield
    finally:
        close_connection_pool()


app = FastAPI(
    title="Knowledge Infrastructure Search",
    description="RAG retrieval over the transformer-architecture corpus.",
    lifespan=lifespan,
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.4f}s")
    return response


# ── Pydantic models ──────────────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language query.")
    top_k: int = Field(5, ge=1, le=100)
    filters: Optional[dict[str, Any]] = None


class ChunkResult(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    section: Optional[str] = None
    section_path: list = Field(default_factory=list)
    title: Optional[str] = None
    source_name: Optional[str] = None
    source_type: Optional[str] = None
    tier: Optional[int] = None
    topics: list = Field(default_factory=list)
    evidence_role: Optional[str] = None
    embedding_text: str
    similarity: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    original_rank: Optional[int] = None


class SearchResponse(BaseModel):
    query: str
    top_k: int
    filters: dict = Field(default_factory=dict)
    results: list[ChunkResult]
    config: dict
    latency_ms: dict
    dense_count: int
    bm25_count: int
    fused_count: int
    rerank_used: bool




# ── Routes ───────────────────────────────────────────────────────────


@app.get("/health")
def health():
    settings = get_settings()
    status: dict[str, Any] = {
        "status": "ok",
        "dense_model_loaded": get_dense() is not None,
    }

    if settings.retrieval_use_reranker:
        
        status["reranker_loaded"] = get_reranker() is not None

    if settings.retrieval_use_bm25:
        
        index, rows = _load_index()
        status["bm25_index_loaded"] = index is not None
        status["bm25_chunk_count"] = len(rows)

    return status


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    try:
        outcome = run_search(
            request.query,
            top_k=request.top_k,
            filters=request.filters or {},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Internal Server Error in /search", exc_info=exc)
        raise HTTPException(
            status_code=500, detail="Internal server error"
        ) from exc

    results = [
        ChunkResult(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            section=row.get("section"),
            section_path=row.get("section_path") or [],
            title=row.get("title"),
            source_name=row.get("source_name"),
            source_type=row.get("source_type"),
            tier=row.get("tier"),
            topics=row.get("topics") or [],
            evidence_role=row.get("evidence_role"),
            embedding_text=row["embedding_text"],
            similarity=row.get("similarity"),
            bm25_score=row.get("bm25_score"),
            rrf_score=row.get("rrf_score"),
            rerank_score=row.get("rerank_score"),
            original_rank=row.get("original_rank"),
        )
        for row in outcome["results"]
    ]
    return SearchResponse(
        query=request.query,
        top_k=request.top_k,
        filters=request.filters or {},
        results=results,
        config=outcome["config"],
        latency_ms=outcome["latency_ms"],
        dense_count=outcome["dense_count"],
        bm25_count=outcome["bm25_count"],
        fused_count=outcome["fused_count"],
        rerank_used=outcome["rerank_used"],
    )


