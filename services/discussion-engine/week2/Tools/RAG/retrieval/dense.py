"""Dense retrieval over pgvector using BGE-M3.

Mirrors the original ``api/main.py::search_chunks`` SQL exactly so that
behavior is reproducible. Optional metadata predicates are translated into a
``WHERE`` clause; the cosine-distance ``<=>`` operator is preserved because
the embeddings were written with ``normalize_embeddings=True`` (cosine on
unit vectors == inner product == 1 - cosine distance).

The query encoder is loaded once and cached at module level. ``embed_query``
also pre-fixes the empty-string guard because BGE-M3 returns a degenerate
vector for an empty input.
"""
from __future__ import annotations

from typing import Any

import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer

from helper import get_settings, pooled_connection

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy + cached. The API lifespan handler also calls this; caching means
    the reranker-friendly ``get_settings()`` (which can change between
    processes via .env) is read once per process."""
    global _model
    if _model is None:
        settings = get_settings()
        device = settings.retrieval_device.strip().lower()
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(settings.model_name, device=device)
    return _model


def reset_model_for_tests() -> None:
    """Drop the cached embedding model for isolated test fixtures."""
    global _model
    _model = None


def embed_query(text: str) -> list[float]:
    """Encode a single query string into a 1024-dim normalized vector."""
    text = (text or "").strip()
    if not text:
        # BGE-M3 returns a 1024-dim zero vector on empty input; we raise
        # because an empty query is almost always a client bug.
        raise ValueError("query is empty")
    model = _get_model()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return vec.tolist()


def _build_filter_clause(filters: dict | None) -> tuple[str, list]:
    """Translate the public filters dict into a parameterized SQL fragment.

    The clause is appended to the ``WHERE`` only when at least one predicate
    is active. Unknown keys are ignored so that the request payload can be
    forward-compatible.

    Supported keys (all optional, all ANDed):
        tier            - exact match (int)
        source_types    - list[str], ``source_type = ANY(%s)``
        topics          - list[str], ``topics ?| %s`` (jsonb overlap on chunks)
        query_groups    - list[str], ``d.topics ?| %s`` (jsonb overlap on documents)
        evidence_roles  - list[str], ``evidence_role = ANY(%s)``
        exclude_reference - bool, default True: not implemented yet
    """
    if not filters:
        return "", []

    clauses: list[str] = []
    params: list = []

    if "tier" in filters and filters["tier"] is not None:
        clauses.append("c.tier = %s")
        params.append(int(filters["tier"]))

    if filters.get("source_types"):
        clauses.append("c.source_type = ANY(%s)")
        params.append(list(filters["source_types"]))

    if filters.get("topics"):
        clauses.append("c.topics ?| %s::text[]")
        params.append(list(filters["topics"]))

    if filters.get("evidence_roles"):
        clauses.append("c.evidence_role = ANY(%s)")
        params.append(list(filters["evidence_roles"]))

    # The joined filters reference documents (d). We always join when at
    # least one of them is requested.
    needs_doc_join = any(
        filters.get(k) for k in ("query_groups",)
    )

    join_sql = ""
    if needs_doc_join:
        join_sql = "JOIN documents d ON d.document_id = c.document_id"
        if filters.get("query_groups"):
            clauses.append("d.topics ?| %s::text[]")
            params.append(list(filters["query_groups"]))

    # Note: relevance_status and exclude_reference are not implemented
    # because the schema does not have these columns. They are ignored
    # to avoid SQL errors.

    where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return f"{join_sql} {where_sql}".strip(), params


def search_dense(
    query_vector: list[float],
    top_n: int,
    filters: dict | None = None,
) -> list[dict[str, Any]]:
    """Return up to ``top_n`` chunks ordered by ascending cosine distance.

    Each result dict contains the full row needed downstream:
        chunk_id, document_id, chunk_index, section, section_path, title,
        source_name, source_type, tier, topics, evidence_role,
        embedding_text, similarity (cosine, 0..1).
    """
    where_sql, params = _build_filter_clause(filters)
    sql = f"""
        SELECT
            c.chunk_id,
            c.document_id,
            c.chunk_index,
            c.section,
            c.section_path,
            c.title,
            c.source_name,
            c.source_type,
            c.tier,
            c.topics,
            c.evidence_role,
            c.embedding_text,
            1 - (c.embedding <=> %s::vector) AS similarity
        FROM chunks c
        {where_sql}
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
    """
    bind = [query_vector, *params, query_vector, top_n]

    with pooled_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, bind)
            return [dict(r) for r in cur.fetchall()]
