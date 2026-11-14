"""Shared research for a discussion."""
from __future__ import annotations

from typing import Any, Callable

from .config import DiscussionSettings, get_settings
from .models import SharedResearchContext
from . import week2_adapter

PreparationCallback = Callable[[dict[str, Any]], None]


def _query_progress(result: Any) -> dict[str, Any]:
    context = getattr(result, "context", None)
    return {
        "title": str(getattr(result, "query", "")),
        "status": str(getattr(result, "status", "unknown")),
        "source": str(getattr(context, "source_used", "unknown")),
        "accepted_evidence": len(getattr(context, "chunks", []) or []),
        "rejected_evidence": len(getattr(context, "rejected_chunks", []) or []),
    }

def _rejected_audit(pool: list[Any]) -> list[dict[str, Any]]:
    """Flatten rejected candidates for debugging without feeding them onward."""
    audit: list[dict[str, Any]] = []
    for result in pool:
        context = getattr(result, "context", None)
        for chunk in getattr(context, "rejected_chunks", []) or []:
            item = dict(chunk)
            item["research_query"] = getattr(result, "query", "")
            item["source_used"] = getattr(context, "source_used", "unknown")
            item["accepted"] = False
            audit.append(item)
    return audit


def _bound_evidence(
    pool: list[Any],
    *,
    max_chunks_per_query: int,
    max_chars_per_chunk: int,
    max_total_chunks: int,
) -> None:
    """Bound evidence volume before it reaches consolidation and personas."""
    remaining = max_total_chunks
    for result in pool:
        context = getattr(result, "context", None)
        chunks = list(getattr(context, "chunks", []) or [])
        bounded: list[dict[str, Any]] = []
        for chunk in chunks[:max_chunks_per_query]:
            if remaining <= 0:
                break
            item = dict(chunk)
            for key in ("text", "embedding_text"):
                value = item.get(key)
                if isinstance(value, str) and len(value) > max_chars_per_chunk:
                    item[key] = value[:max_chars_per_chunk]
            bounded.append(item)
            remaining -= 1
        if context is not None and hasattr(context, "chunks"):
            context.chunks[:] = bounded


def dossier_for_prompt(dossier: Any, max_chars: int) -> Any:
    """Return a prompt-only dossier with a bounded summary.

    The full dossier remains on the research artifact for audit and analytics;
    only the repeated persona prompt is compacted.  Cutting at a paragraph
    boundary keeps citation blocks readable and avoids splitting JSON-like
    metadata in the middle of a line.
    """
    summary = str(getattr(dossier, "summary", "") or "")
    if len(summary) <= max_chars:
        return dossier

    notice = (
        "[Prompt evidence truncated at the configured character budget. "
        "The complete dossier remains available in the discussion artifacts.]"
    )
    content_limit = max(1, max_chars - len(notice) - 2)
    cutoff = summary.rfind("\n\n", 0, content_limit)
    if cutoff < content_limit // 2:
        cutoff = summary.rfind("\n", 0, content_limit)
    if cutoff < content_limit // 2:
        cutoff = content_limit
    compact = summary[:cutoff].rstrip() + "\n\n" + notice
    model_copy = getattr(dossier, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"summary": compact})

    class _PromptDossier:
        def __init__(self, value: str):
            self.summary = value

    return _PromptDossier(compact)


def build_shared_research(
    objective: Any,
    *,
    memory_summary: dict[str, Any] | None = None,
    settings: DiscussionSettings | None = None,
    progress_callback: PreparationCallback | None = None,
) -> SharedResearchContext:
    """Run Week 2 planning, retrieval, and consolidation exactly once."""
    settings = settings or get_settings()
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "planning",
                "message": "Creating a bounded research plan",
                "completed_units": 0,
                "total_units": settings.max_initial_queries,
                "queries": [],
            }
        )
    plan = week2_adapter.plan_research(
        objective,
        memory_summary=memory_summary,
        query_limit=settings.max_initial_queries,
    )
    # The planner's fixed-size batch is the complete retrieval budget for a
    # discussion. It is intentionally goal-level and is never expanded by
    # candidate details or by claims raised later in the debate.
    query_history = list(plan.queries)[:settings.max_initial_queries]
    query_states = [
        {"title": str(query), "status": "pending"} for query in query_history
    ]
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "retrieving",
                "message": "Searching and validating research sources",
                "completed_units": 0,
                "total_units": len(query_history),
                "queries": query_states,
            }
        )

    def report_query(result: Any, completed: int, total: int) -> None:
        query_state = _query_progress(result)
        order = int(getattr(result, "order", -1))
        if 0 <= order < len(query_states):
            query_states[order] = query_state
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "retrieving",
                    "message": f"Validated {completed} of {total} research queries",
                    "completed_units": completed,
                    "total_units": total,
                    "queries": [dict(item) for item in query_states],
                }
            )

    evidence_pool = week2_adapter.build_evidence_pool(
        query_history,
        objective=objective,
        **({"progress_callback": report_query} if progress_callback is not None else {}),
    )
    _bound_evidence(
        evidence_pool,
        max_chunks_per_query=settings.max_evidence_chunks_per_query,
        max_chars_per_chunk=settings.max_evidence_chars_per_chunk,
        max_total_chunks=settings.max_total_evidence_chunks,
    )
    for result in evidence_pool:
        order = int(getattr(result, "order", -1))
        if 0 <= order < len(query_states):
            query_states[order] = _query_progress(result)
    accepted_count = sum(
        len(getattr(getattr(result, "context", None), "chunks", []) or [])
        for result in evidence_pool
    )
    rejected_count = sum(
        len(getattr(getattr(result, "context", None), "rejected_chunks", []) or [])
        for result in evidence_pool
    )
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "consolidating",
                "message": "Building the shared evidence dossier",
                "completed_units": len(query_history),
                "total_units": len(query_history),
                "queries": [dict(item) for item in query_states],
                "accepted_evidence": accepted_count,
                "rejected_evidence": rejected_count,
            }
        )
    dossier = week2_adapter.consolidate_evidence(evidence_pool, objective)
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "ready",
                "message": "Evidence dossier ready; starting the discussion",
                "completed_units": len(query_history),
                "total_units": len(query_history),
                "queries": [dict(item) for item in query_states],
                "accepted_evidence": accepted_count,
                "rejected_evidence": rejected_count,
            }
        )
    return SharedResearchContext(
        plan=plan,
        evidence_pool=evidence_pool,
        dossier=dossier,
        query_history=query_history,
        rejected_evidence=_rejected_audit(evidence_pool),
    )
