"""Unit tests for the research planner.

All tests run offline. The LLM is monkeypatched at the ``_client`` import site.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

_FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
if str(_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_ROOT))

from src.agents.objective import Objective, CoreTopic
from src.agents.planner import (
    _deterministic_fallback,
    _memory_digest,
    _objective_digest,
    plan_research,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _make_objective(
    *,
    goal: str = "Design a small Transformer.",
    topics: list[tuple[str, list[str]]] | None = None,
) -> Objective:
    topics = topics or [
        ("FEED-FORWARD", ["dense vs MoE", "routing"]),
        ("ATTENTION", ["MQA vs GQA", "sliding window"]),
    ]
    return Objective(
        goal=goal,
        constraints=["Limited compute"],
        core_topics=[
            CoreTopic(
                id=t.lower().replace(" ", "_").replace("-", "_"),
                title=t,
                subpoints=sp,
            )
            for t, sp in topics
        ],
        evidence_requirements=["Cite sources"],
        decision_criteria=["Quality", "Compute"],
        strict_notes=["No speculation"],
        final_goal="Produce a working design.",
    )


class _FakeStructured:
    def __init__(self, payload):
        self._payload = payload

    def invoke(self, _messages):
        return self._payload


class _FakeLLM:
    def __init__(self, payload):
        self._payload = payload

    def invoke(self, _messages):
        # Mimic the real interface the planner uses (returns an object with .content).
        return SimpleNamespace(content=self._payload)


def _install_planner_llm(monkeypatch, response_content: str):
    """Patch the LLM inside planner.py to return *response_content* verbatim."""
    fake = _FakeLLM(response_content)
    # planner.py does `from src.agents.llm import _client as get_llm`, so
    # monkeypatching `_client` on the llm module is enough.
    from src.agents import llm as llm_mod
    monkeypatch.setattr(llm_mod, "_client", lambda: fake)


# ── Tests: deterministic fallback ───────────────────────────────────


class TestDeterministicFallback:
    def test_returns_exactly_five(self):
        obj = _make_objective(
            topics=[(f"TOPIC_{i}", [f"subpoint {i}"]) for i in range(8)]
        )
        plan = _deterministic_fallback(obj, memory_summary={})
        assert len(plan.queries) == 5

    def test_skips_topics_in_memory(self):
        obj = _make_objective(
            topics=[
                ("ATTENTION", ["GQA"]),
                ("DEPTH", ["deep vs shallow"]),
                ("WIDTH", ["wide vs narrow"]),
            ]
        )
        # Pre-mark ATTENTION as "resolved" by putting its topic words in shared_facts.
        summary = {"shared_facts": ["The model uses GQA attention."]}
        plan = _deterministic_fallback(obj, memory_summary=summary)
        assert len(plan.queries) == 5
        joined = " ".join(plan.queries).lower()
        assert "attention" not in joined.split("gqa")[0], "ATTENTION topic should have been skipped"

    def test_pads_when_objective_has_fewer_than_five(self):
        obj = _make_objective(
            topics=[("ONLY_TOPIC", ["only one"])]
        )
        plan = _deterministic_fallback(obj, memory_summary={})
        assert len(plan.queries) == 5

    def test_honors_requested_query_limit(self):
        obj = _make_objective(
            topics=[(f"TOPIC_{i}", [f"subpoint {i}"]) for i in range(6)]
        )
        plan = _deterministic_fallback(obj, memory_summary={}, query_limit=3)
        assert len(plan.queries) == 3


# ── Tests: plan_research with mocked LLM ────────────────────────────


class TestPlanResearch:
    def test_exactly_five_queries_from_valid_json(self, monkeypatch):
        payload = json.dumps({
            "queries": ["Q1", "Q2", "Q3", "Q4", "Q5"],
            "rationale": "covers 5 distinct info needs",
        })
        _install_planner_llm(monkeypatch, payload)

        obj = _make_objective()
        plan = plan_research(obj, memory_summary={})
        assert len(plan.queries) == 5

    def test_requested_query_limit_is_forwarded_to_validation(self, monkeypatch):
        payload = json.dumps({
            "queries": ["Q1", "Q2", "Q3"],
            "rationale": "bounded plan",
        })
        _install_planner_llm(monkeypatch, payload)

        plan = plan_research(_make_objective(), memory_summary={}, query_limit=3)

        assert plan.queries == ["Q1", "Q2", "Q3"]

    def test_six_queries_falls_back(self, monkeypatch):
        payload = json.dumps({
            "queries": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"],
            "rationale": "too many",
        })
        _install_planner_llm(monkeypatch, payload)

        obj = _make_objective()
        plan = plan_research(obj, memory_summary={})
        # Falls back to deterministic 5-query plan.
        assert len(plan.queries) == 5

    def test_four_queries_falls_back(self, monkeypatch):
        payload = json.dumps({
            "queries": ["Q1", "Q2", "Q3", "Q4"],
            "rationale": "too few",
        })
        _install_planner_llm(monkeypatch, payload)

        obj = _make_objective()
        plan = plan_research(obj, memory_summary={})
        assert len(plan.queries) == 5

    def test_malformed_json_falls_back(self, monkeypatch):
        _install_planner_llm(monkeypatch, "this is not json {{{")

        obj = _make_objective()
        plan = plan_research(obj, memory_summary={})
        assert len(plan.queries) == 5

    def test_markdown_fences_stripped(self, monkeypatch):
        # Some models wrap JSON in ```json ... ``` even when told not to.
        payload = "```json\n" + json.dumps({
            "queries": ["Q1", "Q2", "Q3", "Q4", "Q5"],
            "rationale": "fenced",
        }) + "\n```"
        _install_planner_llm(monkeypatch, payload)

        obj = _make_objective()
        plan = plan_research(obj, memory_summary={})
        assert len(plan.queries) == 5

    def test_empty_query_in_response_falls_back(self, monkeypatch):
        payload = json.dumps({
            "queries": ["Q1", "Q2", "", "Q4", "Q5"],
            "rationale": "has empty",
        })
        _install_planner_llm(monkeypatch, payload)

        obj = _make_objective()
        plan = plan_research(obj, memory_summary={})
        # Empty query triggered the "empty queries" branch, so fallback runs.
        assert len(plan.queries) == 5

    def test_duplicate_queries_fall_back(self, monkeypatch):
        payload = json.dumps({
            "queries": ["same query", "same query", "Q3", "Q4", "Q5"],
            "rationale": "duplicate",
        })
        _install_planner_llm(monkeypatch, payload)

        plan = plan_research(_make_objective(), memory_summary={})

        assert len(plan.queries) == 5
        assert len({query.lower() for query in plan.queries}) == 5

    def test_second_cycle_deprioritizes_resolved_topic(self, monkeypatch):
        """Cycle 2: shared_facts mention ATTENTION -> planner's chosen 5 must NOT
        include an ATTENTION query (we exercise the deterministic fallback path,
        which is what the planner falls back to on any LLM failure)."""
        obj = _make_objective(
            topics=[
                ("ATTENTION", ["GQA"]),
                ("DEPTH", ["deep vs shallow"]),
                ("WIDTH", ["wide vs narrow"]),
                ("FFN", ["dense vs MoE"]),
                ("NORMALIZATION", ["RMSNorm vs LayerNorm"]),
                ("EMBEDDINGS", ["tied vs untied"]),
            ]
        )
        # LLM raises -> deterministic fallback.
        from src.agents import llm as llm_mod
        def _raise(*_a, **_k):
            raise RuntimeError("LLM down")
        monkeypatch.setattr(llm_mod, "_client", _raise)

        summary = {"shared_facts": ["The model uses GQA attention."]}
        plan = plan_research(obj, memory_summary=summary)
        assert len(plan.queries) == 5
        joined = " ".join(plan.queries).lower()
        # ATTENTION should not have been chosen (it was resolved).
        assert "attention" not in joined.split("gqa")[0]


# ── Tests: helper coverage ──────────────────────────────────────────


def test_memory_digest_empty():
    assert "no prior" in _memory_digest({}).lower()


def test_memory_digest_renders_sections():
    digest = _memory_digest({
        "shared_facts": ["fact a"],
        "decisions_log": ["decision b"],
        "open_questions": ["question c"],
    })
    assert "Established Facts" in digest
    assert "fact a" in digest
    assert "Decisions Log" in digest
    assert "Open Questions" in digest


def test_objective_digest_contains_topics():
    obj = _make_objective(topics=[("FOO", ["x"]), ("BAR", ["y"])])
    digest = _objective_digest(obj)
    assert "FOO" in digest
    assert "BAR" in digest
    assert "Constraints" in digest
    assert "Evidence Requirements" in digest
    assert "Decision Criteria" in digest
    assert "Strict Notes" in digest


def test_fallback_uses_cross_cutting_questions_instead_of_goal_padding():
    obj = _make_objective(topics=[("ONLY_TOPIC", ["one trade-off"])])
    plan = _deterministic_fallback(obj, memory_summary={}, query_limit=5)

    assert len(plan.queries) == 5
    assert len({query.lower() for query in plan.queries}) == 5
    assert all("additional evidence cycle" not in query.lower() for query in plan.queries)
    assert any("counterevidence" in query.lower() for query in plan.queries)
