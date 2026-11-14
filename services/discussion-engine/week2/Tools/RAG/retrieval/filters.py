"""Result-level post-filters.

These run on the candidate pool AFTER retrieval and BEFORE reranking so
the reranker never wastes cycles on chunks we already know we don't want.

Filter order is significant:
  1. metadata_filters   - drop chunks that don't match the metadata criteria
  2. min_similarity   - drop low-similarity dense hits (does nothing for BM25)
  3. dedup_on_text    - drop exact-duplicate ``embedding_text``
  4. max_per_document - per-doc cap

Each function returns a new list; the input is not mutated.
"""
from __future__ import annotations

from typing import Any


def apply_metadata_filters(
    candidates: list[dict[str, Any]],
    filters: dict | None,
) -> list[dict[str, Any]]:
    """Drop candidates that do not match the provided metadata filters.

    The filters dict has the same structure as in dense.py. This ensures
    that BM25 results are also filtered.
    """
    if not filters:
        return list(candidates)

    def matches(row: dict[str, Any]) -> bool:
        # tier
        tier = filters.get("tier")
        if tier is not None and row.get("tier") != tier:
            return False
        # source_types
        source_types = filters.get("source_types")
        if source_types and row.get("source_type") not in source_types:
            return False
        # topics (on chunks)
        topics = filters.get("topics")
        if topics:
            row_topics = row.get("topics") or []
            # if any of the requested topics are in row_topics
            if not any(t in row_topics for t in topics):
                return False
        # evidence_roles
        evidence_roles = filters.get("evidence_roles")
        if evidence_roles and row.get("evidence_role") not in evidence_roles:
            return False
        # Dense retrieval enforces query groups in SQL. BM25 rows include the
        # joined document topics so the lexical branch must enforce the same
        # predicate before fusion.
        query_groups = filters.get("query_groups")
        if query_groups:
            document_topics = row.get("document_topics")
            if document_topics is None:
                # A dense hit was already filtered by SQL. A BM25-only row
                # without document metadata is unsafe to retain.
                if row.get("similarity") is None:
                    return False
            elif not any(group in document_topics for group in query_groups):
                return False
        return True

    return [c for c in candidates if matches(c)]


def apply_min_similarity(
    candidates: list[dict[str, Any]],
    min_similarity: float,
) -> list[dict[str, Any]]:
    """Drop chunks whose ``similarity`` is below the threshold.

    Chunks without a ``similarity`` field (e.g. BM25-only candidates) are
    kept - the threshold only applies where it is meaningful.
    """
    if min_similarity <= 0:
        return list(candidates)
    out: list[dict[str, Any]] = []
    for c in candidates:
        sim = c.get("similarity")
        if sim is None or sim >= min_similarity:
            out.append(c)
    return out


def apply_dedup_on_text(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop chunks whose ``embedding_text`` is byte-identical to a prior one.

    Uses the full text for exact deduplication. This is safe and precise.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for c in candidates:
        text = c.get("embedding_text") or ""
        if text in seen:
            continue
        seen.add(text)
        out.append(c)
    return out


def apply_max_per_document(
    candidates: list[dict[str, Any]],
    max_per_document: int,
) -> list[dict[str, Any]]:
    """Cap the number of chunks from any single document.

    ``max_per_document <= 0`` disables the cap. The cap preserves the
    input order, so the upstream ranking (RRF or dense) is the only
    arbiter.
    """
    if max_per_document <= 0:
        return list(candidates)
    counts: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for c in candidates:
        doc_id = c.get("document_id")
        if doc_id is None:
            out.append(c)
            continue
        if counts.get(doc_id, 0) >= max_per_document:
            continue
        out.append(c)
        counts[doc_id] = counts.get(doc_id, 0) + 1
    return out


def apply_filters(
    candidates: list[dict[str, Any]],
    *,
    metadata_filters: dict | None = None,
    min_similarity: float = 0.0,
    dedup_on_text: bool = False,
    max_per_document: int = 0,
) -> list[dict[str, Any]]:
    """One-call wrapper for the post-fusion filter chain."""
    out = apply_metadata_filters(candidates, metadata_filters)
    out = apply_min_similarity(out, min_similarity)
    if dedup_on_text:
        out = apply_dedup_on_text(out)
    out = apply_max_per_document(out, max_per_document)
    return out
