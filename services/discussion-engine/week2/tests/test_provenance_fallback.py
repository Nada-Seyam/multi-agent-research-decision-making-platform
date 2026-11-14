from __future__ import annotations

from types import SimpleNamespace


def test_objective_aware_resolution_escalates_and_records_web_origin(monkeypatch):
    from src.agents import resolve_context as rc_mod

    class Settings:
        QUERY_EXPANSION_ATTEMPTS = 0
        CONTEXT_SOURCES_ORDER = "rag,web"

    objective = SimpleNamespace(
        goal="Choose a pretraining architecture",
        constraints=["small model"],
        core_topics=[],
        decision_criteria=["quality"],
        strict_notes=[],
    )

    monkeypatch.setattr(rc_mod, "get_settings", lambda: Settings())
    monkeypatch.setattr(
        rc_mod,
        "retrieval",
        SimpleNamespace(retrieve=lambda _query: [{"text": "RAG candidate"}]),
    )
    monkeypatch.setattr(
        rc_mod,
        "search_web",
        lambda _query: [{"text": "Web candidate"}],
    )

    def grade(*, query, chunks, objective):
        assert objective is not None
        return SimpleNamespace(
            is_relevant=chunks[0]["text"] == "Web candidate",
            reasoning="web is applicable" if chunks[0]["text"] == "Web candidate" else "rag is weak",
            evidence_quality="strong" if chunks[0]["text"] == "Web candidate" else "weak",
            applicability={"domain": "high"},
        )

    monkeypatch.setattr(rc_mod, "evaluate_retrieval_relevance", grade)

    chunks, _, relevant, source, _ = rc_mod.resolve_context("Q", objective=objective)

    assert relevant is True
    assert source == "web"
    assert chunks[0]["source_origin"] == "web"
    assert chunks[0]["evidence_quality"] == "strong"


def test_objective_aware_rejected_chunks_are_audit_only(monkeypatch):
    from src.agents import resolve_context as rc_mod

    class Settings:
        QUERY_EXPANSION_ATTEMPTS = 0
        CONTEXT_SOURCES_ORDER = "rag"

    objective = SimpleNamespace(
        goal="Choose a design",
        constraints=[],
        core_topics=[],
        decision_criteria=["quality"],
        strict_notes=[],
    )
    monkeypatch.setattr(rc_mod, "get_settings", lambda: Settings())
    monkeypatch.setattr(
        rc_mod,
        "retrieval",
        SimpleNamespace(retrieve=lambda _query: [{"text": "wrong context"}]),
    )
    monkeypatch.setattr(
        rc_mod,
        "evaluate_retrieval_relevance",
        lambda **_kwargs: SimpleNamespace(
            is_relevant=False,
            reasoning="not applicable",
            rejection_reason="domain mismatch",
            evidence_quality="insufficient",
            applicability={"domain": "low"},
            answered_requirement_ids=[],
            limitations=[],
        ),
    )

    result = rc_mod.resolve_context("Q", objective=objective)

    assert result.chunks == []
    assert result.rejected_chunks[0]["accepted"] is False
    assert result.rejected_chunks[0]["rejection_reason"] == "domain mismatch"
