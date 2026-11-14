"""Source-agnostic context resolution loop.

Walks the ``CONTEXT_SOURCES_ORDER`` list, retrieves chunks from each source,
grades them against the original query, and retries with query expansion when
the grader rejects.

This is the core retrieval loop — called once per research query inside
``evidence_pool.build_evidence_pool()``.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Iterator

from src.agents.query_expander import expand_query
from src.helper.helper import get_settings
from src.agents import retrieval
from Tools import evaluate_retrieval_relevance, search_web


@dataclass(frozen=True)
class ContextResolutionResult:
    """Result of the source-agnostic context resolution loop."""

    chunks: list[dict[str, Any]]
    resolved_query: str
    is_relevant: bool
    source_used: str
    expansion_attempts: int
    rejected_chunks: list[dict[str, Any]] = field(default_factory=list)
    rejection_reason: str = ""

    def __iter__(self) -> Iterator[Any]:
        """Support the five-value interface used by existing retrieval callers."""
        yield self.chunks
        yield self.resolved_query
        yield self.is_relevant
        yield self.source_used
        yield self.expansion_attempts


class ModelCallBudget:
    """Thread-safe budget shared by every research query in one discussion."""

    def __init__(self, limit: int) -> None:
        self.limit = max(1, int(limit))
        self._remaining = self.limit
        self._lock = Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._remaining <= 0:
                return False
            self._remaining -= 1
            return True

    @property
    def used(self) -> int:
        with self._lock:
            return self.limit - self._remaining

_SOURCE_REGISTRY: dict[str, Callable[[str], list[dict[str, Any]]]] = {
    "rag": lambda q: retrieval.retrieve(q),
    "web": lambda q: search_web(q),
}


def _parse_sources_order(raw: str) -> list[str]:
    """Parse a comma-separated ``CONTEXT_SOURCES_ORDER`` into a clean list."""
    out: list[str] = []
    seen: set[str] = set()
    invalid_sources: list[str] = []

    for item in (raw or "").split(","):
        name = item.strip().lower()
        if not name or name in seen:
            continue
        if name not in _SOURCE_REGISTRY:
            invalid_sources.append(name)
            continue
        out.append(name)
        seen.add(name)

    if invalid_sources:
        valid_sources = ", ".join(sorted(_SOURCE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown source(s) in CONTEXT_SOURCES_ORDER: {', '.join(invalid_sources)}. "
            f"Valid sources are: {valid_sources}"
        )

    if not out:
        out = ["rag"]
    return out


def _annotate_chunks(
    chunks: list[dict[str, Any]],
    *,
    source_name: str,
    evaluation: Any | None,
    objective: Any | None,
    accepted: bool,
    rejection_reason: str = "",
    partial_usable: bool = False,
    usable_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Attach provenance and applicability metadata for objective-aware runs."""
    if objective is None:
        return chunks

    annotated: list[dict[str, Any]] = []
    per_chunk = {
        int(getattr(item, "index", 0)): item
        for item in (getattr(evaluation, "chunk_evaluations", []) or [])
        if int(getattr(item, "index", 0)) > 0
    }
    for index, chunk in enumerate(chunks, start=1):
        item = dict(chunk)
        item["source_origin"] = source_name
        chunk_evaluation = per_chunk.get(index)
        if chunk_evaluation is not None:
            item["evidence_quality"] = getattr(
                chunk_evaluation, "evidence_quality", "unknown"
            )
            item["applicability"] = dict(
                getattr(chunk_evaluation, "applicability", {}) or {}
            )
            item["evaluation_reasoning"] = getattr(chunk_evaluation, "reasoning", "")
            item["answered_requirement_ids"] = list(
                getattr(chunk_evaluation, "answered_requirement_ids", []) or []
            )
            item["limitations"] = list(
                getattr(chunk_evaluation, "limitations", []) or []
            )
        elif evaluation is not None:
            item["evidence_quality"] = getattr(evaluation, "evidence_quality", "unknown")
            item["applicability"] = dict(getattr(evaluation, "applicability", {}) or {})
            item["evaluation_reasoning"] = getattr(evaluation, "reasoning", "")
            item["answered_requirement_ids"] = list(
                getattr(evaluation, "answered_requirement_ids", []) or []
            )
            item["limitations"] = list(getattr(evaluation, "limitations", []) or [])
        chunk_is_usable = (
            bool(getattr(chunk_evaluation, "usable", False))
            if chunk_evaluation is not None
            else accepted or (
            partial_usable
            and (not usable_indices or index in usable_indices)
            )
        )
        item["accepted"] = chunk_is_usable
        if partial_usable:
            item["partial_evidence"] = chunk_is_usable
            item["batch_sufficient"] = False
        if rejection_reason:
            item["rejection_reason"] = rejection_reason
        annotated.append(item)
    return annotated


def resolve_context(
    query: str,
    objective: Any | None = None,
    *,
    model_call_budget: ModelCallBudget | None = None,
) -> ContextResolutionResult:
    """Walk the source list, expanding the query, until the grader accepts.

    The function is source-agnostic: it never inspects chunks to decide
    which tool produced them. It only asks the relevance grader whether
    the chunks are useful, and — if not — asks the expander to rewrite
    the query before asking the same source for a new retrieval.

    Returns a ``ContextResolutionResult`` containing the chunks, the
    final (possibly rewritten) query, whether the grader accepted the
    context, which source produced the final chunks, and the number of
    query expansions performed in the final source.

    Behavior summary:
      * For each source in ``CONTEXT_SOURCES_ORDER``:
          * Call the source with the current query.
          * Grade the returned chunks **against the original user query**
            (not the expanded variant) so relevance is always measured
            against the user's actual intent.
          * If relevant → return immediately.
          * Otherwise call ``expand_query`` and retry up to
            ``QUERY_EXPANSION_ATTEMPTS`` times.
          * If all attempts on this source fail, move to the next source
            with the most recent expanded query as the seed.
      * If the loop exhausts every source, return the last chunks
        produced (best-effort fallback) with ``is_relevant=False``.
    """
    settings = get_settings()
    sources = _parse_sources_order(settings.CONTEXT_SOURCES_ORDER)
    max_attempts = max(0, int(settings.QUERY_EXPANSION_ATTEMPTS))

    last_chunks: list[dict[str, Any]] = []
    last_query = query
    last_source = sources[0]
    last_attempts = 0
    last_accepted = False
    rejected_chunks: list[dict[str, Any]] = []
    partial_chunks: list[dict[str, Any]] = []
    rejection_reason = ""
    budget_exhausted = False

    for source_name in sources:
        source_fn = _SOURCE_REGISTRY[source_name]
        current_query = last_query  # carry expansion forward across sources
        chunks: list[dict[str, Any]] = []
        accepted = False
        expansion_count = 0
        source_evaluation: Any | None = None
        rejected_for_source: list[dict[str, Any]] = []

        for attempt_index in range(max_attempts + 1):
            try:
                chunks = source_fn(current_query)
                if (
                    model_call_budget is not None
                    and not model_call_budget.try_acquire()
                ):
                    rejection_reason = "Research model-call budget exhausted."
                    budget_exhausted = True
                    break
                if objective is None:
                    evaluation = evaluate_retrieval_relevance(
                        query=query, chunks=chunks
                    )
                else:
                    evaluation = evaluate_retrieval_relevance(
                        query=query, chunks=chunks, objective=objective
                    )
                source_evaluation = evaluation
            except Exception as exc:
                # A failed source must not prevent trying the next configured
                # source (for example, web fallback when the local RAG is down).
                print(
                    f"[resolver] Source '{source_name}' failed on attempt "
                    f"{attempt_index + 1}: {exc}",
                    file=sys.stderr,
                )
                chunks = []
                break
            print(
                f"[grader] Source '{source_name}' (attempt {attempt_index + 1}/{max_attempts + 1}): "
                f"is_relevant={evaluation.is_relevant} | Rationale: {evaluation.reasoning}",
                file=sys.stderr,
            )
            if evaluation.is_relevant:
                annotated_chunks = _annotate_chunks(
                    chunks,
                    source_name=source_name,
                    evaluation=source_evaluation,
                    objective=objective,
                    accepted=True,
                )
                accepted_chunks = (
                    annotated_chunks
                    if objective is None
                    else [item for item in annotated_chunks if item.get("accepted")]
                )
                rejected_for_source.extend(
                    item
                    for item in annotated_chunks
                    if objective is not None and not item.get("accepted")
                )
                # A batch-level acceptance must still contain at least one
                # individually usable chunk when per-chunk judgments exist.
                if getattr(source_evaluation, "chunk_evaluations", None) and not accepted_chunks:
                    accepted = False
                    rejection_reason = "No individually applicable chunks remained."
                    break
                last_chunks = [*partial_chunks, *accepted_chunks]
                last_query = current_query
                last_source = source_name
                last_attempts = expansion_count
                last_accepted = True
                accepted = True
                break

            reason = getattr(evaluation, "rejection_reason", "") or evaluation.reasoning
            usable_indices = {
                int(index)
                for index in (getattr(evaluation, "usable_chunk_indices", []) or [])
                if isinstance(index, int) and 1 <= index <= len(chunks)
            }
            partial_batch = bool(
                getattr(evaluation, "has_usable_evidence", False)
                and getattr(evaluation, "evidence_quality", "unknown")
                in {"strong", "partial"}
            )
            annotated = _annotate_chunks(
                chunks,
                source_name=source_name,
                evaluation=source_evaluation,
                objective=objective,
                accepted=False,
                rejection_reason=reason,
                partial_usable=partial_batch,
                usable_indices=usable_indices or None,
            )
            partial_chunks.extend(
                item for item in annotated if item.get("partial_evidence")
            )
            rejected_for_source.extend(
                item for item in annotated if not item.get("accepted")
            )
            rejection_reason = reason

            # No more expansion budget on this source.
            if attempt_index >= max_attempts:
                break

            try:
                if (
                    model_call_budget is not None
                    and not model_call_budget.try_acquire()
                ):
                    rejection_reason = "Research model-call budget exhausted."
                    budget_exhausted = True
                    break
                expanded = expand_query(
                    query=current_query,
                    chunks=chunks,
                    grading_reasoning=evaluation.reasoning,
                )
            except Exception as exc:
                print(
                    f"[resolver] Query expansion failed for source "
                    f"'{source_name}': {exc}",
                    file=sys.stderr,
                )
                break
            if not expanded or expanded == current_query:
                # Expander gave up; further attempts would be identical.
                break
            current_query = expanded
            expansion_count += 1

        if accepted:
            rejected_chunks.extend(rejected_for_source)
            break

        if budget_exhausted:
            rejected_chunks.extend(rejected_for_source)
            last_chunks = chunks if objective is None else partial_chunks
            last_query = current_query
            last_source = source_name
            last_attempts = expansion_count
            break

        # Carry best-so-far forward to the next source.
        rejected_chunks.extend(rejected_for_source)
        # The legacy objective-less API returned rejected chunks as a
        # best-effort result. Objective-aware runs must keep them out of the
        # valid evidence stream and expose them only through the audit field.
        if objective is None:
            last_chunks = chunks
        else:
            last_chunks = partial_chunks
        last_query = current_query
        last_source = source_name
        last_attempts = expansion_count

    return ContextResolutionResult(
        chunks=last_chunks,
        resolved_query=last_query,
        is_relevant=last_accepted,
        source_used=last_source,
        expansion_attempts=last_attempts,
        rejected_chunks=rejected_chunks,
        rejection_reason=rejection_reason,
    )
