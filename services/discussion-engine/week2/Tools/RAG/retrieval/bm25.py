from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import psycopg2
import psycopg2.extras

from helper import pooled_connection

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - handled at runtime
    BM25Okapi = None

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@lru_cache(maxsize=1)
def _load_index() -> tuple[BM25Okapi, list[dict[str, Any]]]:
    """Load every chunk row from the DB and build the BM25 index.

    Cached at module level so the first query pays the build cost and every
    subsequent query is a pure in-memory lookup. After re-running the loader
    with ``--truncate``, call ``reset_index_for_tests()`` in an isolated test
    or restart the service so it picks up the new rows.
    """
    if BM25Okapi is None:
        raise RuntimeError(
            "rank_bm25 is not installed. Run `pip install rank-bm25` to enable "
            "lexical retrieval (RETRIEVAL_USE_BM25=true)."
        )

    with pooled_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT c.chunk_id, c.document_id, c.chunk_index, c.section,
                       c.section_path, c.title, c.source_name, c.source_type,
                       c.tier, c.topics, c.evidence_role, c.embedding_text,
                       d.topics AS document_topics
                FROM chunks c
                JOIN documents d ON d.document_id = c.document_id
                """
            )
            rows = [dict(r) for r in cur.fetchall()]

    tokenized = [_tokenize(r["embedding_text"]) for r in rows]
    return BM25Okapi(tokenized), rows


def reset_index_for_tests() -> None:
    """Drop the cached lexical index for isolated test fixtures."""
    _load_index.cache_clear()


def search_bm25(query: str, top_n: int) -> list[dict[str, Any]]:
    """Return up to ``top_n`` chunks with a ``bm25_score`` field added.

    ``bm25_score`` is the raw BM25 score (higher is better). The pipeline
    fusion step converts it to a rank before RRF, so the raw value's
    magnitude is not directly comparable to dense ``similarity``.
    """
    bm25, rows = _load_index()
    if not rows:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    # Partial selection: only the top_n indices. numpy is not a hard dep
    # so we use a simple heap for clarity.
    import heapq

    top = heapq.nlargest(top_n, enumerate(scores), key=lambda x: x[1])
    out: list[dict[str, Any]] = []
    for idx, score in top:
        if score <= 0:
            # BM25Okapi returns 0 for chunks with no overlapping terms; the
            # pipeline wants a "did anything match?" signal, so we stop on
            # the first non-positive score (the list is sorted descending).
            break
        row = dict(rows[idx])
        row["bm25_score"] = float(score)
        out.append(row)
    return out
