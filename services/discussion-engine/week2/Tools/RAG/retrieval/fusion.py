"""Reciprocal Rank Fusion.

Combines ranked lists from independent retrievers (dense, BM25, ...) into a
single ranked list using the standard RRF formula:

    rrf_score(d) = sum_r  1 / (k + rank_r(d))

``k`` is a smoothing constant (default 60, from Cormack et al. 2009). A
higher ``k`` reduces the influence of the top of one list relative to the
middle of another; the standard value is a safe default for any reasonable
corpus.

Each input list MUST be sorted by relevance descending and MUST contain
``chunk_id`` for every entry. Ties (same chunk in both lists) are summed.
"""
from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Return a single ranked list with ``rrf_score`` attached to each item.

    The "representative" row for a chunk is taken from the first list that
    contains it (in input order). This matters when, e.g., the dense
    retriever fills in the full metadata envelope and the BM25 retriever
    only fills a subset.
    """
    if not ranked_lists:
        return []

    by_id: dict[str, dict[str, Any]] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            cid = item.get("chunk_id")
            if cid is None:
                continue
            contribution = 1.0 / (k + rank)
            if cid in by_id:
                by_id[cid]["rrf_score"] += contribution
                for key, val in item.items():
                    by_id[cid].setdefault(key, val)
            else:
                enriched = dict(item)
                enriched["rrf_score"] = contribution
                by_id[cid] = enriched

    fused = sorted(by_id.values(), key=lambda r: r["rrf_score"], reverse=True)
    return fused
