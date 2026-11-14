"""Master retrieval pipeline.

Composes the modules in this package according to the env-driven settings.
The composition order is:

    dense.top_n
      + bm25.top_n   (if RETRIEVAL_USE_BM25)
      -> RRF fusion  (if BM25 was used; otherwise dense list passes through)
      -> metadata_filters
      -> filters     (min_similarity, dedup_on_text, max_per_document)
      -> rerank      (if RETRIEVAL_USE_RERANKER)
      -> top_k

For single-retriever configurations (A, B, C, D), the BM25 branch is
skipped and the pipeline degenerates to: dense -> filters -> [rerank].

``run_search`` is the single entry point used by both ``/search`` and the
evaluation harness so the two surfaces cannot drift.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from typing import Any

from helper import get_settings

from . import bm25 as bm25_mod
from . import dense as dense_mod
from . import rerank as rerank_mod
from .filters import apply_filters
from .fusion import reciprocal_rank_fusion


def _timed_dense_search(
    query_vector: list[float],
    top_n: int,
    filters: dict | None,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    rows = dense_mod.search_dense(query_vector, top_n=top_n, filters=filters)
    return rows, (time.perf_counter() - started) * 1000


def _timed_bm25_search(
    query: str,
    top_n: int,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    rows = bm25_mod.search_bm25(query, top_n=top_n)
    return rows, (time.perf_counter() - started) * 1000


def _coerce_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """psycopg2 returns JSONB as Python lists already, but be explicit."""
    for key in ("section_path", "topics"):
        v = row.get(key)
        if v is None:
            row[key] = []
        elif isinstance(v, str):
            import json

            try:
                row[key] = json.loads(v)
            except json.JSONDecodeError:
                row[key] = []
    return row


def run_search(
    query: str,
    top_k: int = 5,
    filters: dict | None = None,
) -> dict[str, Any]:
    """Run the full retrieval pipeline and return a diagnostics-rich dict.

    Returned shape::

        {
            "results":        [ChunkResult-shaped dicts, length=top_k],
            "candidates":     [..all post-fusion, pre-rerank, length<=fused_top],
            "dense_count":    int,
            "bm25_count":     int,
            "fused_count":    int,
            "rerank_used":    bool,
            "latency_ms":     {"dense": ..., "bm25": ..., "fusion": ...,
                               "filters": ..., "rerank": ..., "total": ...},
            "config":         {the env-driven settings that shaped this run},
        }
    """
    settings = get_settings()
    cfg = {
        "retrieval_enabled": settings.retrieval_enabled,
        "retrieval_dense_top_n": settings.retrieval_dense_top_n,
        "retrieval_use_bm25": settings.retrieval_use_bm25,
        "retrieval_bm25_top_n": settings.retrieval_bm25_top_n,
        "retrieval_use_reranker": settings.retrieval_use_reranker,
        "retrieval_reranker_top_n": settings.retrieval_reranker_top_n,
        "retrieval_dedup_on_text": settings.retrieval_dedup_on_text,
        "retrieval_max_per_document": settings.retrieval_max_per_document,
        "retrieval_min_similarity": settings.retrieval_min_similarity,
        "retrieval_rrf_k": settings.retrieval_rrf_k,
    }

    timing: dict[str, float] = {}
    started_total = time.perf_counter()

    # If the master switch is off, replay the original /search behavior:
    # dense only, no post-filters, no rerank. The query is encoded but
    # only the SQL path is exercised.
    if not settings.retrieval_enabled:
        t0 = time.perf_counter()
        qv = dense_mod.embed_query(query)
        rows = dense_mod.search_dense(qv, top_n=top_k, filters=filters)
        for r in rows:
            _coerce_metadata(r)
        timing["dense"] = (time.perf_counter() - t0) * 1000
        timing["total"] = (time.perf_counter() - started_total) * 1000
        return {
            "results": rows,
            "candidates": rows,
            "dense_count": len(rows),
            "bm25_count": 0,
            "fused_count": len(rows),
            "rerank_used": False,
            "latency_ms": timing,
            "config": cfg,
        }

    # 1. Encode once, then run the independent dense and lexical branches
    # concurrently. BM25 is warmed at API startup and is an in-memory lookup;
    # dense search waits on PostgreSQL, so overlapping them reduces wall time.
    t0 = time.perf_counter()
    qv = dense_mod.embed_query(query)
    embedding_ms = (time.perf_counter() - t0) * 1000
    bm25_rows: list[dict[str, Any]] = []
    if settings.retrieval_use_bm25:
        with ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="hybrid-retrieval",
        ) as executor:
            dense_future = executor.submit(
                _timed_dense_search,
                qv,
                settings.retrieval_dense_top_n,
                filters,
            )
            bm25_future = executor.submit(
                _timed_bm25_search,
                query,
                settings.retrieval_bm25_top_n,
            )
            dense_rows, dense_search_ms = dense_future.result()
            bm25_rows, bm25_ms = bm25_future.result()
        timing["bm25"] = bm25_ms
    else:
        dense_rows, dense_search_ms = _timed_dense_search(
            qv,
            settings.retrieval_dense_top_n,
            filters,
        )
    for r in dense_rows:
        _coerce_metadata(r)
    timing["dense"] = embedding_ms + dense_search_ms
    dense_count = len(dense_rows)

    # 2. Normalize optional BM25 metadata after the parallel branch joins.
    for r in bm25_rows:
        _coerce_metadata(r)
    bm25_count = len(bm25_rows)

    # 3. Fusion (RRF) - only when both retrievers ran
    t0 = time.perf_counter()
    if bm25_rows:
        fused = reciprocal_rank_fusion(
            [dense_rows, bm25_rows], k=settings.retrieval_rrf_k,
        )
    else:
        # Re-anchor similarity from the dense rows so the rest of the
        # pipeline can sort by a numeric score even in single-retriever mode.
        fused = list(dense_rows)
    timing["fusion"] = (time.perf_counter() - t0) * 1000
    fused_count = len(fused)

    # 4. Post-fusion filters: first metadata filters, then quality filters
    t0 = time.perf_counter()
    candidates = apply_filters(
        fused,
        metadata_filters=filters,                     # <-- new
        min_similarity=settings.retrieval_min_similarity,
        dedup_on_text=settings.retrieval_dedup_on_text,
        max_per_document=settings.retrieval_max_per_document,
    )
    timing["filters"] = (time.perf_counter() - t0) * 1000

    # 5. Optional reranking. We cap the rerank input at
    #    retrieval_reranker_top_n so the reranker cost is bounded even
    #    when retrieval_max_per_document is disabled.
    rerank_used = False
    rerank_input = candidates[: settings.retrieval_reranker_top_n]
    if settings.retrieval_use_reranker and rerank_input:
        t0 = time.perf_counter()
        rerank_input = rerank_mod.rerank(query, rerank_input)
        timing["rerank"] = (time.perf_counter() - t0) * 1000
        rerank_used = True
        results = rerank_input[:top_k]
    else:
        results = candidates[:top_k]

    timing["total"] = (time.perf_counter() - started_total) * 1000

    return {
        "results": results,
        "candidates": candidates,
        "dense_count": dense_count,
        "bm25_count": bm25_count,
        "fused_count": fused_count,
        "rerank_used": rerank_used,
        "latency_ms": timing,
        "config": cfg,
    }
