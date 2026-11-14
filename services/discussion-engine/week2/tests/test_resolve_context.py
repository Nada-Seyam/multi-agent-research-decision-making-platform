"""Unit tests for the source-agnostic context-resolution loop.

The test stubs the ``langgraph`` package so it can run on a Python that
does not have it installed (it is only needed to import ``run.py``).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace



_FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
if str(_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_ROOT))


# ── Stub langgraph so ``src.agents.graph`` can be imported ──────────
def _install_langgraph_stub():
    if "langgraph" in sys.modules and "langgraph.graph" in sys.modules:
        return
    lg = types.ModuleType("langgraph")
    graph = types.ModuleType("langgraph.graph")

    class _StateGraph:
        def __init__(self, _state):
            pass
        def add_node(self, *_a, **_kw): pass
        def add_edge(self, *_a, **_kw): pass
        def add_conditional_edges(self, *_a, **_kw): pass
        def compile(self): return self

    class _END: pass
    class _START: pass

    graph.StateGraph = _StateGraph
    graph.END = _END()
    graph.START = _START()
    sys.modules["langgraph"] = lg
    sys.modules["langgraph.graph"] = graph


_install_langgraph_stub()


# ── Helpers ─────────────────────────────────────────────────────────
def _rel(relevant: bool, reason: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(is_relevant=relevant, reasoning=reason)


class _FakeSettings:
    def __init__(self, *, attempts: int = 3, order: str = "rag,web"):
        self.QUERY_EXPANSION_ATTEMPTS = attempts
        self.CONTEXT_SOURCES_ORDER = order


def _patch_run(monkeypatch, *, expansions, relevance_calls, source_calls):
    """Patch run.py dependencies so resolve_context can be exercised.

    - ``expansions`` is a list of rewritten queries to return, in order.
    - ``relevance_calls`` is a list of (is_relevant, reason) tuples; one is
      popped per grading call.
    - ``source_calls`` is a list of chunk lists; one is popped per source call.
    """
    from src.agents import resolve_context as rc_mod

    def _fake_expand(query, chunks, grading_reasoning):
        if not expansions:
            return query
        return expansions.pop(0)

    def _fake_relevance(query, chunks):
        if not relevance_calls:
            return _rel(True, "default accept")
        return relevance_calls.pop(0)

    monkeypatch.setattr(rc_mod, "evaluate_retrieval_relevance", _fake_relevance)
    monkeypatch.setattr(rc_mod, "expand_query", _fake_expand)
    monkeypatch.setattr(rc_mod, "search_web", lambda q: source_calls.pop(0))
    monkeypatch.setattr(rc_mod, "retrieval", SimpleNamespace(retrieve=lambda q: source_calls.pop(0)))
    monkeypatch.setattr(rc_mod, "get_settings", lambda: _FakeSettings(attempts=3, order="rag,web"))


# ── Tests ───────────────────────────────────────────────────────────
def test_relevant_on_first_try_no_expansion(monkeypatch):
    expansions, relevance_calls, source_calls = [], [_rel(True)], [
        [{"text": "good chunk", "source_name": "p1"}]
    ]
    _patch_run(monkeypatch, expansions=expansions, relevance_calls=relevance_calls, source_calls=source_calls)

    from src.agents.resolve_context import resolve_context

    chunks, final_query, is_relevant, source, attempts = resolve_context("Q")
    assert source == "rag"
    assert attempts == 0
    assert final_query == "Q"
    assert chunks == [{"text": "good chunk", "source_name": "p1"}]
    assert expansions == []  # expander never called


def test_expand_then_accept(monkeypatch):
    """First RAG call irrelevant, expanded RAG call relevant."""
    expansions = ["Q refined"]
    relevance_calls = [_rel(False, "wrong topic"), _rel(True, "now matches")]
    source_calls = [
        [{"text": "bad"}],   # RAG #0
        [{"text": "good"}],  # RAG #1 (expanded)
    ]
    _patch_run(monkeypatch, expansions=expansions, relevance_calls=relevance_calls, source_calls=source_calls)

    from src.agents.resolve_context import resolve_context

    chunks, final_query, is_relevant, source, attempts = resolve_context("Q")
    assert source == "rag"
    assert final_query == "Q refined"
    assert attempts == 1
    assert expansions == []
    assert chunks == [{"text": "good"}]


def test_three_expansions_then_escalate_to_web(monkeypatch):
    """RAG exhausts all 3 expansion attempts; loop escalates to web."""
    expansions = ["Q1", "Q2", "Q3"]
    relevance_calls = [
        _rel(False, "bad0"),
        _rel(False, "bad1"),
        _rel(False, "bad2"),
        _rel(False, "bad3"),
        _rel(True, "web ok"),
    ]
    # Each source_fn call must return a list[dict].  RAG is called 4 times
    # (1 initial + 3 expansions), then web is called once.
    source_calls = [
        [{"text": "rag0"}],
        [{"text": "rag1"}],
        [{"text": "rag2"}],
        [{"text": "rag3"}],
        [{"text": "web0"}],
    ]
    _patch_run(monkeypatch, expansions=expansions, relevance_calls=relevance_calls, source_calls=source_calls)

    from src.agents.resolve_context import resolve_context

    chunks, final_query, is_relevant, source, attempts = resolve_context("Q")
    assert source == "web"
    assert final_query == "Q3"  # last expanded query used for web
    assert attempts == 0  # accepted on first web call
    assert chunks == [{"text": "web0"}]
    assert expansions == []  # all 3 expansions used


def test_source_order_parsed_from_settings(monkeypatch):
    """When CONTEXT_SOURCES_ORDER=web,rag the loop walks web first."""
    from src.agents import resolve_context as rc_mod

    relevance_calls = [_rel(True)]
    source_calls = [[{"text": "from web"}]]

    def _fake_relevance(query, chunks):
        return relevance_calls.pop(0)

    def _fake_expand(query, chunks, grading_reasoning):
        return query

    monkeypatch.setattr(rc_mod, "evaluate_retrieval_relevance", _fake_relevance)
    monkeypatch.setattr(rc_mod, "expand_query", _fake_expand)
    monkeypatch.setattr(rc_mod, "search_web", lambda q: source_calls.pop(0))
    monkeypatch.setattr(rc_mod, "retrieval", SimpleNamespace(retrieve=lambda q: source_calls.pop(0)))

    class _S:
        QUERY_EXPANSION_ATTEMPTS = 3
        CONTEXT_SOURCES_ORDER = "web,rag"
    monkeypatch.setattr(rc_mod, "get_settings", lambda: _S())

    chunks, _, is_relevant, source, _ = rc_mod.resolve_context("Q")
    assert source == "web"
    assert chunks == [{"text": "from web"}]


def test_source_exception_escalates_to_next_source(monkeypatch):
    from src.agents import resolve_context as rc_mod

    monkeypatch.setattr(
        rc_mod,
        "retrieval",
        SimpleNamespace(
            retrieve=lambda _q: (_ for _ in ()).throw(RuntimeError("down"))
        ),
    )
    monkeypatch.setattr(rc_mod, "search_web", lambda _q: [{"text": "web evidence"}])
    monkeypatch.setattr(
        rc_mod,
        "evaluate_retrieval_relevance",
        lambda **_kwargs: _rel(True),
    )
    monkeypatch.setattr(
        rc_mod,
        "get_settings",
        lambda: _FakeSettings(attempts=0, order="rag,web"),
    )

    chunks, _, is_relevant, source, _ = rc_mod.resolve_context("Q")
    assert source == "web"
    assert is_relevant is True
    assert chunks == [{"text": "web evidence"}]


def test_all_sources_fail_returns_last_chunks(monkeypatch):
    """If every source rejects, the loop returns the last chunk list (best-effort)."""
    # 2 sources, each gets 1 initial + 3 expansion attempts = 4 calls per source = 8 total.
    # Each source uses up to 3 expansions = 6 total.
    expansions = ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6"]
    relevance_calls = [_rel(False, "x")] * 8  # all reject
    source_calls = [[{"text": f"chunk{i}"}] for i in range(8)]
    _patch_run(monkeypatch, expansions=expansions, relevance_calls=relevance_calls, source_calls=source_calls)

    from src.agents.resolve_context import resolve_context

    chunks, _, is_relevant, source, _ = resolve_context("Q")
    assert source == "web"
    # The last source call wins; we don't know its exact index but it must
    # be one of the values produced.
    assert chunks[0]["text"].startswith("chunk")
    # All expansions consumed.
    assert expansions == []


def test_expander_stuck_returns_original_query(monkeypatch):
    """If expand_query returns the original query (no progress), the loop stops
    expanding that source and escalates."""
    from src.agents import resolve_context as rc_mod

    expansions_seen = []
    def _fake_expand(query, chunks, grading_reasoning):
        expansions_seen.append(query)
        return query  # simulate expander failure

    relevance_calls = [
        _rel(False, "bad"),  # RAG #0
        _rel(True, "web ok"),  # web #0
    ]
    source_calls = [
        [{"text": "rag0"}],
        [{"text": "web0"}],
    ]
    monkeypatch.setattr(rc_mod, "evaluate_retrieval_relevance", lambda query, chunks: relevance_calls.pop(0))
    monkeypatch.setattr(rc_mod, "expand_query", _fake_expand)
    monkeypatch.setattr(rc_mod, "search_web", lambda q: source_calls.pop(0))
    monkeypatch.setattr(rc_mod, "retrieval", SimpleNamespace(retrieve=lambda q: source_calls.pop(0)))
    monkeypatch.setattr(rc_mod, "get_settings", lambda: _FakeSettings(attempts=3, order="rag,web"))

    chunks, _, is_relevant, source, _ = rc_mod.resolve_context("Q")
    # Expander was called once (for RAG), returned the original query unchanged,
    # so the loop broke out of the inner loop and escalated to web.
    assert source == "web"
    assert chunks == [{"text": "web0"}]
    assert len(expansions_seen) == 1


def test_partial_evidence_is_preserved_while_batch_remains_insufficient(monkeypatch):
    from src.agents import resolve_context as rc_mod

    class Settings:
        QUERY_EXPANSION_ATTEMPTS = 0
        CONTEXT_SOURCES_ORDER = "rag"

    objective = SimpleNamespace(
        goal="Choose an architecture",
        constraints=[],
        core_topics=[],
        decision_criteria=["latency"],
        strict_notes=[],
    )
    chunks = [
        {"text": "Useful architecture evidence"},
        {"text": "Unrelated evidence"},
    ]
    monkeypatch.setattr(rc_mod, "get_settings", lambda: Settings())
    monkeypatch.setattr(
        rc_mod,
        "retrieval",
        SimpleNamespace(retrieve=lambda _query: chunks),
    )
    monkeypatch.setattr(
        rc_mod,
        "evaluate_retrieval_relevance",
        lambda **_kwargs: SimpleNamespace(
            is_relevant=False,
            has_usable_evidence=True,
            usable_chunk_indices=[1],
            evidence_quality="partial",
            answered_requirement_ids=["architecture::r1"],
            reasoning="The first chunk answers one requirement, but the batch is incomplete.",
            rejection_reason="Missing operational evidence",
            applicability={"domain": "medium"},
        ),
    )

    result = rc_mod.resolve_context("Q", objective=objective)

    assert result.is_relevant is False
    assert [chunk["text"] for chunk in result.chunks] == [
        "Useful architecture evidence"
    ]
    assert result.chunks[0]["accepted"] is True
    assert result.chunks[0]["partial_evidence"] is True
    assert result.rejected_chunks[0]["accepted"] is False


def test_batch_acceptance_does_not_copy_relevance_to_rejected_chunks(monkeypatch):
    from src.agents import resolve_context as rc_mod

    class Settings:
        QUERY_EXPANSION_ATTEMPTS = 0
        CONTEXT_SOURCES_ORDER = "rag"

    objective = SimpleNamespace(
        goal="Choose an architecture",
        constraints=["fixed GPU fleet"],
        core_topics=[],
        decision_criteria=["latency"],
        strict_notes=[],
    )
    chunks = [
        {"text": "Target-hardware dense latency measurements"},
        {"text": "A popular but unrelated benchmark"},
    ]
    evaluations = [
        SimpleNamespace(
            index=1,
            usable=True,
            evidence_quality="strong",
            applicability={"hardware": "high"},
            reasoning="Directly applicable.",
            answered_requirement_ids=["latency"],
            limitations=[],
        ),
        SimpleNamespace(
            index=2,
            usable=False,
            evidence_quality="weak",
            applicability={"hardware": "low"},
            reasoning="Different workload and hardware.",
            answered_requirement_ids=[],
            limitations=["not applicable"],
        ),
    ]
    monkeypatch.setattr(rc_mod, "get_settings", lambda: Settings())
    monkeypatch.setattr(rc_mod, "retrieval", SimpleNamespace(retrieve=lambda _query: chunks))
    monkeypatch.setattr(
        rc_mod,
        "evaluate_retrieval_relevance",
        lambda **_kwargs: SimpleNamespace(
            is_relevant=True,
            has_usable_evidence=True,
            usable_chunk_indices=[1],
            evidence_quality="strong",
            answered_requirement_ids=["latency"],
            reasoning="The batch contains one directly applicable chunk.",
            rejection_reason="",
            applicability={"hardware": "mixed"},
            chunk_evaluations=evaluations,
        ),
    )

    result = rc_mod.resolve_context("target hardware p95", objective=objective)

    assert [chunk["text"] for chunk in result.chunks] == [chunks[0]["text"]]
    assert result.chunks[0]["applicability"] == {"hardware": "high"}
    assert [chunk["text"] for chunk in result.rejected_chunks] == [chunks[1]["text"]]
    assert result.rejected_chunks[0]["accepted"] is False


def test_shared_model_budget_stops_expansion_before_extra_llm_call(monkeypatch):
    from src.agents import resolve_context as rc_mod

    class Settings:
        QUERY_EXPANSION_ATTEMPTS = 3
        CONTEXT_SOURCES_ORDER = "rag"

    expansion_calls = []
    monkeypatch.setattr(rc_mod, "get_settings", lambda: Settings())
    monkeypatch.setattr(
        rc_mod,
        "retrieval",
        SimpleNamespace(retrieve=lambda _query: [{"text": "weak evidence"}]),
    )
    monkeypatch.setattr(
        rc_mod,
        "evaluate_retrieval_relevance",
        lambda **_kwargs: _rel(False, "insufficient"),
    )
    monkeypatch.setattr(
        rc_mod,
        "expand_query",
        lambda **_kwargs: expansion_calls.append(True) or "expanded",
    )
    budget = rc_mod.ModelCallBudget(1)

    result = rc_mod.resolve_context("Q", model_call_budget=budget)

    assert budget.used == 1
    assert expansion_calls == []
    assert result.is_relevant is False
    assert result.rejection_reason == "Research model-call budget exhausted."
