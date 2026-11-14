"""Unit tests for the evidence pool.

All tests run offline. ``resolve_context`` is monkeypatched so no LLM or
retrieval service is needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

_FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
if str(_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_ROOT))


from src.agents.resolve_context import ContextResolutionResult
from src.agents.evidence_pool import EvidenceResult, build_evidence_pool


def _fake_ctx(
    *,
    chunks=None,
    resolved_query="q",
    is_relevant=True,
    source_used="rag",
    expansion_attempts=0,
) -> ContextResolutionResult:
    return ContextResolutionResult(
        chunks=chunks or [{"text": "chunk"}],
        resolved_query=resolved_query,
        is_relevant=is_relevant,
        source_used=source_used,
        expansion_attempts=expansion_attempts,
    )


def test_pool_length_matches_queries(monkeypatch):
    from src.agents import evidence_pool as ep_mod
    call_order: list[str] = []

    def _resolve(q):
        call_order.append(q)
        return _fake_ctx(resolved_query=q)

    monkeypatch.setattr(ep_mod, "resolve_context", _resolve)

    pool = build_evidence_pool(["Q1", "Q2", "Q3"])
    assert len(pool) == 3
    assert call_order == ["Q1", "Q2", "Q3"]


def test_pool_preserves_order(monkeypatch):
    from src.agents import evidence_pool as ep_mod

    def _resolve(q):
        return _fake_ctx(resolved_query=q, chunks=[{"text": f"for {q}"}])

    monkeypatch.setattr(ep_mod, "resolve_context", _resolve)

    pool = build_evidence_pool(["alpha", "beta", "gamma"])
    assert [e.order for e in pool] == [0, 1, 2]
    assert [e.query for e in pool] == ["alpha", "beta", "gamma"]


def test_pool_wraps_context_result(monkeypatch):
    from src.agents import evidence_pool as ep_mod
    ctx = _fake_ctx(
        chunks=[{"text": "evidence", "source_name": "paper.pdf"}],
        resolved_query="rewritten q",
        is_relevant=True,
        source_used="web",
        expansion_attempts=2,
    )
    monkeypatch.setattr(ep_mod, "resolve_context", lambda q: ctx)

    pool = build_evidence_pool(["Q1"])
    assert len(pool) == 1
    e = pool[0]
    assert isinstance(e, EvidenceResult)
    assert e.query == "Q1"
    assert e.context is ctx
    assert e.context.source_used == "web"
    assert e.context.expansion_attempts == 2
    assert e.status == "accepted"


def test_empty_queries_returns_empty_pool(monkeypatch):
    from src.agents import evidence_pool as ep_mod
    monkeypatch.setattr(ep_mod, "resolve_context", lambda q: _fake_ctx())

    pool = build_evidence_pool([])
    assert pool == []


def test_pool_isolates_a_failed_query(monkeypatch):
    from src.agents import evidence_pool as ep_mod

    def _resolve(query):
        if query == "bad":
            raise RuntimeError("RAG unavailable")
        return _fake_ctx(resolved_query=query)

    monkeypatch.setattr(ep_mod, "resolve_context", _resolve)
    pool = build_evidence_pool(["good", "bad"])

    assert [entry.query for entry in pool] == ["good", "bad"]
    assert pool[0].context.is_relevant is True
    assert pool[1].context.chunks == []
    assert pool[1].context.source_used == "unavailable"
    assert pool[1].status == "failed"
