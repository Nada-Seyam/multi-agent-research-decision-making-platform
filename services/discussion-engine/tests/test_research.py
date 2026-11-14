from __future__ import annotations

from types import SimpleNamespace

from discussion import week2_adapter
from discussion.models import SharedResearchContext
from discussion.research import build_shared_research, dossier_for_prompt


def test_shared_research_runs_each_week2_stage_once(monkeypatch):
    calls: list[tuple[str, object]] = []
    plan = SimpleNamespace(queries=["query one", "query two"])
    pool = ["evidence"]
    dossier = SimpleNamespace(summary="shared dossier")

    def fake_plan(objective, memory_summary=None, query_limit=5):
        calls.append(("plan", objective))
        assert memory_summary == {"prior": "context"}
        assert query_limit == 5
        return plan

    def fake_pool(queries, objective=None):
        calls.append(("pool", queries))
        return pool

    def fake_consolidate(received_pool, objective):
        calls.append(("consolidate", received_pool))
        return dossier

    monkeypatch.setattr(week2_adapter, "plan_research", fake_plan)
    monkeypatch.setattr(week2_adapter, "build_evidence_pool", fake_pool)
    monkeypatch.setattr(week2_adapter, "consolidate_evidence", fake_consolidate)

    result = build_shared_research(
        object(),
        memory_summary={"prior": "context"},
    )

    assert isinstance(result, SharedResearchContext)
    assert result.plan is plan
    assert result.evidence_pool is pool
    assert result.dossier is dossier
    assert [name for name, _ in calls] == ["plan", "pool", "consolidate"]


def test_evidence_is_bounded_before_prompt_construction():
    from discussion.research import _bound_evidence

    context = SimpleNamespace(
        chunks=[
            {"text": "a" * 100, "accepted": True},
            {"text": "b" * 100, "accepted": True},
            {"text": "c" * 100, "accepted": True},
        ]
    )
    pool = [SimpleNamespace(context=context)]

    _bound_evidence(
        pool,
        max_chunks_per_query=2,
        max_chars_per_chunk=10,
        max_total_chunks=2,
    )

    assert len(context.chunks) == 2
    assert all(len(chunk["text"]) == 10 for chunk in context.chunks)


def test_evidence_bound_clears_results_after_global_limit():
    from discussion.research import _bound_evidence

    first = SimpleNamespace(context=SimpleNamespace(chunks=[{"text": "a"}, {"text": "b"}]))
    second = SimpleNamespace(context=SimpleNamespace(chunks=[{"text": "c"}, {"text": "d"}]))

    _bound_evidence(
        [first, second],
        max_chunks_per_query=2,
        max_chars_per_chunk=10,
        max_total_chunks=2,
    )

    assert len(first.context.chunks) == 2
    assert second.context.chunks == []


def test_prompt_dossier_is_bounded_without_mutating_full_artifact():
    full_summary = ("Evidence paragraph.\n\n" * 400).strip()
    dossier = SimpleNamespace(summary=full_summary)

    prompt_dossier = dossier_for_prompt(dossier, 4_000)

    assert len(prompt_dossier.summary) <= 4_000
    assert "Prompt evidence truncated" in prompt_dossier.summary
    assert dossier.summary == full_summary


def test_shared_research_uses_supplied_settings(monkeypatch):
    from discussion import research as research_mod
    from discussion.config import DiscussionSettings

    settings = DiscussionSettings(max_initial_queries=1)
    plan = SimpleNamespace(queries=["q1", "q2"])
    calls = []

    monkeypatch.setattr(
        research_mod.week2_adapter,
        "plan_research",
        lambda objective, memory_summary=None, query_limit=5: plan,
    )
    monkeypatch.setattr(
        research_mod.week2_adapter,
        "build_evidence_pool",
        lambda queries, objective=None: calls.append(list(queries)) or [],
    )
    monkeypatch.setattr(
        research_mod.week2_adapter,
        "consolidate_evidence",
        lambda pool, objective: SimpleNamespace(summary="dossier"),
    )

    research_mod.build_shared_research(object(), settings=settings)

    assert calls == [["q1"]]


def test_initial_query_limit_is_the_total_retrieval_budget(monkeypatch):
    from discussion import research as research_mod
    from discussion.config import DiscussionSettings

    objective = SimpleNamespace(goal="Discover the best solution")
    plan = SimpleNamespace(queries=["goal comparison one", "goal comparison two"])
    retrieved: list[list[str]] = []

    monkeypatch.setattr(
        research_mod.week2_adapter,
        "plan_research",
        lambda objective, memory_summary=None, query_limit=5: plan,
    )
    monkeypatch.setattr(
        research_mod.week2_adapter,
        "build_evidence_pool",
        lambda queries, objective=None: retrieved.append(list(queries)) or [],
    )
    monkeypatch.setattr(
        research_mod.week2_adapter,
        "consolidate_evidence",
        lambda pool, objective: SimpleNamespace(excerpts=[]),
    )

    result = research_mod.build_shared_research(
        objective,
        settings=DiscussionSettings(max_initial_queries=2),
    )

    assert retrieved == [["goal comparison one", "goal comparison two"]]
    assert result.query_history == ["goal comparison one", "goal comparison two"]


def test_shared_research_reports_safe_preparation_progress(monkeypatch):
    from discussion import research as research_mod
    from discussion.config import DiscussionSettings

    plan = SimpleNamespace(queries=["compare latency", "compare memory"])
    events: list[dict] = []

    def fake_pool(queries, objective=None, *, progress_callback=None):
        results = []
        for index, query in enumerate(queries):
            result = SimpleNamespace(
                query=query,
                order=index,
                status="accepted",
                context=SimpleNamespace(
                    source_used="rag",
                    chunks=[{"text": "evidence"}],
                    rejected_chunks=[],
                ),
            )
            results.append(result)
            progress_callback(result, index + 1, len(queries))
        return results

    monkeypatch.setattr(
        research_mod.week2_adapter,
        "plan_research",
        lambda objective, memory_summary=None, query_limit=5: plan,
    )
    monkeypatch.setattr(research_mod.week2_adapter, "build_evidence_pool", fake_pool)
    monkeypatch.setattr(
        research_mod.week2_adapter,
        "consolidate_evidence",
        lambda pool, objective: SimpleNamespace(summary="dossier"),
    )

    research_mod.build_shared_research(
        object(),
        settings=DiscussionSettings(max_initial_queries=2),
        progress_callback=events.append,
    )

    assert [event["stage"] for event in events] == [
        "planning",
        "retrieving",
        "retrieving",
        "retrieving",
        "consolidating",
        "ready",
    ]
    assert events[-1]["accepted_evidence"] == 2
    assert events[-1]["queries"][0] == {
        "title": "compare latency",
        "status": "accepted",
        "source": "rag",
        "accepted_evidence": 1,
        "rejected_evidence": 0,
    }
