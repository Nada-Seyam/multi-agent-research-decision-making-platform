"""Evidence pool — run each research query through resolve_context() and collect results.

This is pure orchestration: no grading, expansion, or escalation logic lives here.
``resolve_context()`` handles all of that internally for each query.
"""
from __future__ import annotations

import inspect
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

from src.agents.resolve_context import (
    ContextResolutionResult,
    ModelCallBudget,
    resolve_context,
)
from src.helper.helper import get_settings

_QUERY_PREVIEW_LEN = 80


@dataclass
class EvidenceResult:
    """One research query's full, neutral context-resolution outcome.

    ``status`` describes retrieval sufficiency only. It is not a judgment about
    which candidate or architecture should be preferred.
    """

    query: str
    context: ContextResolutionResult
    order: int
    status: str = "unknown"


def _log(message: str) -> None:
    print(f"[evidence_pool] {message}", file=sys.stderr)


def _preview(text: str, max_len: int = _QUERY_PREVIEW_LEN) -> str:
    return text if len(text) <= max_len else f"{text[:max_len]}..."


def build_evidence_pool(
    queries: list[str],
    objective: Any | None = None,
    *,
    progress_callback: Callable[[EvidenceResult, int, int], None] | None = None,
) -> list[EvidenceResult]:
    """Execute ``resolve_context()`` for each query in order and collect results.

    Each call is independent; the source escalation, grading, and query expansion
    loops inside ``resolve_context()`` run fully for every query.

    Queries run with bounded concurrency while preserving the input order in the
    returned pool. A failed query becomes an empty, non-relevant result so the
    rest of the cycle can still produce a dossier.
    """
    if not queries:
        return []

    total = len(queries)
    try:
        configured_workers = get_settings().EVIDENCE_POOL_MAX_WORKERS
    except Exception:
        # Keep the unit suite independent of local provider credentials.
        configured_workers = 3
    workers = max(1, min(len(queries), configured_workers))
    try:
        model_call_limit = get_settings().RESEARCH_MAX_MODEL_CALLS
    except Exception:
        model_call_limit = 16
    model_call_budget = ModelCallBudget(model_call_limit)
    supports_budget = "model_call_budget" in inspect.signature(resolve_context).parameters

    def resolve_one(index: int, query: str) -> EvidenceResult:
        step = index + 1
        _log(f"Resolving query {step}/{total}: {_preview(query)}")
        try:
            if objective is None:
                ctx = (
                    resolve_context(query, model_call_budget=model_call_budget)
                    if supports_budget
                    else resolve_context(query)
                )
            else:
                ctx = (
                    resolve_context(
                        query,
                        objective=objective,
                        model_call_budget=model_call_budget,
                    )
                    if supports_budget
                    else resolve_context(query, objective=objective)
                )
        except Exception as exc:
            _log(f"Query {step} failed: {exc}")
            ctx = ContextResolutionResult(
                chunks=[],
                resolved_query=query,
                is_relevant=False,
                source_used="unavailable",
                expansion_attempts=0,
            )
            status = "failed"
        else:
            status = (
                "accepted"
                if ctx.is_relevant
                else "partial"
                if ctx.chunks
                else "insufficient"
            )

        _log(
            f"Query {step} {status} "
            f"(source={ctx.source_used}, chunks={len(ctx.chunks)}, "
            f"expansions={ctx.expansion_attempts})"
        )
        return EvidenceResult(query=query, context=ctx, order=index, status=status)

    results: list[EvidenceResult | None] = [None] * total
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="evidence") as executor:
        futures = {
            executor.submit(resolve_one, index, query): index
            for index, query in enumerate(queries)
        }
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results[result.order] = result
            completed += 1
            if progress_callback is not None:
                progress_callback(result, completed, total)

    return [result for result in results if result is not None]
