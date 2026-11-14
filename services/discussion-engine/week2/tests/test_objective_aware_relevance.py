from __future__ import annotations

from types import SimpleNamespace

from Tools.Web_Serach import relevance


def _objective():
    return SimpleNamespace(
        goal="Choose a small language-model pretraining architecture.",
        constraints=["300M to 1.5B parameters", "pretraining from scratch"],
        core_topics=[],
        decision_criteria=["training feasibility"],
        strict_notes=["Do not use fine-tuning evidence as pretraining evidence."],
    )


def test_grader_receives_objective_constraints_and_rejects_low_applicability(monkeypatch):
    captured = {}

    class Response:
        content = (
            '{"reasoning":"RLHF evidence is the wrong training stage.",'
            '"is_relevant":true,"evidence_quality":"partial",'
            '"domain_match":"high","training_stage_match":"low",'
            '"model_scale_match":"medium","hardware_match":"unknown",'
            '"decision_criterion_match":"medium",'
            '"strict_notes_compliant":false,'
            '"applicability":{"domain":"high","training_stage":"low"}}'
        )

    class FakeLLM:
        def invoke(self, messages):
            captured["prompt"] = messages[-1].content
            return Response()

    monkeypatch.setattr(relevance, "get_llm", lambda: FakeLLM())
    result = relevance.evaluate_retrieval_relevance(
        "pretraining architecture",
        [{"text": "RLHF improves preference alignment."}],
        objective=_objective(),
    )

    assert result.is_relevant is False
    assert result.has_usable_evidence is False
    assert result.usable_chunk_indices == []
    assert "300M to 1.5B" in captured["prompt"]
    assert "pretraining from scratch" in captured["prompt"]
    assert "training stage" in captured["prompt"]


def _run_grader_with(monkeypatch, payload):
    class Response:
        content = payload

    class FakeLLM:
        def invoke(self, _messages):
            return Response()

    monkeypatch.setattr(relevance, "get_llm", lambda: FakeLLM())
    return relevance.evaluate_retrieval_relevance(
        "compare the target design",
        [{"text": "A technically detailed result."}],
        objective=_objective(),
    )


def test_domain_mismatch_cannot_be_accepted(monkeypatch):
    result = _run_grader_with(
        monkeypatch,
        (
            '{"reasoning":"different domain", "is_relevant":true, '
            '"evidence_quality":"strong", "direct_relevance":"high", '
            '"domain_match":"low", "task_match":"high", '
            '"training_stage_match":"high", "model_scale_match":"high", '
            '"hardware_match":"high", "metric_match":"high", '
            '"decision_criterion_match":"high", "strict_notes_compliant":true}'
        ),
    )

    assert result.is_relevant is False
    assert result.rejection_reason


def test_stage_mismatch_cannot_be_accepted(monkeypatch):
    result = _run_grader_with(
        monkeypatch,
        (
            '{"reasoning":"different lifecycle stage", "is_relevant":true, '
            '"evidence_quality":"strong", "direct_relevance":"high", '
            '"domain_match":"high", "task_match":"high", '
            '"training_stage_match":"low", "model_scale_match":"high", '
            '"hardware_match":"high", "metric_match":"high", '
            '"decision_criterion_match":"high", "strict_notes_compliant":true}'
        ),
    )

    assert result.is_relevant is False


def test_scale_mismatch_is_not_directly_accepted(monkeypatch):
    result = _run_grader_with(
        monkeypatch,
        (
            '{"reasoning":"different scale", "is_relevant":true, '
            '"evidence_quality":"strong", "direct_relevance":"high", '
            '"domain_match":"high", "task_match":"high", '
            '"training_stage_match":"high", "model_scale_match":"low", '
            '"hardware_match":"high", "metric_match":"high", '
            '"decision_criterion_match":"high", "strict_notes_compliant":true, '
            '"limitations":["reported scale is not transferable"]}'
        ),
    )

    assert result.is_relevant is False


def test_unknown_context_is_allowed_for_general_architecture_evidence(monkeypatch):
    result = _run_grader_with(
        monkeypatch,
        (
            '{"reasoning":"general architecture evidence with limits", '
            '"is_relevant":true, "evidence_quality":"partial", '
            '"direct_relevance":"high", "domain_match":"high", '
            '"task_match":"medium", "training_stage_match":"unknown", '
            '"model_scale_match":"unknown", "hardware_match":"unknown", '
            '"metric_match":"unknown", "decision_criterion_match":"medium", '
            '"strict_notes_compliant":true, '
            '"limitations":["No target hardware was specified."]}'
        ),
    )

    assert result.is_relevant is True


def test_grader_repairs_unescaped_latex_backslashes(monkeypatch):
    # Some providers repeat LaTeX such as \mathcal{O} in a JSON string without
    # escaping the backslash, which previously caused the evidence to be rejected.
    payload = r'''{"reasoning":"The result compares O(L^2) with \mathcal{O}(L) complexity.",
        "is_relevant":true,"evidence_quality":"partial",
        "direct_relevance":"high","domain_match":"high",
        "task_match":"high","training_stage_match":"unknown",
        "model_scale_match":"unknown","hardware_match":"unknown",
        "metric_match":"medium","decision_criterion_match":"high",
        "strict_notes_compliant":true,"limitations":["Hardware was not specified."]}'''

    result = _run_grader_with(monkeypatch, payload)

    assert result.is_relevant is True
    assert result.evidence_quality == "partial"


def test_partial_batch_reports_usable_evidence_separately(monkeypatch):
    result = _run_grader_with(
        monkeypatch,
        (
            '{"reasoning":"Answers the architecture requirement but not the full query.", '
            '"is_relevant":false, "evidence_quality":"partial", '
            '"answered_requirement_ids":["architecture::r1"], '
            '"usable_chunk_indices":[1], "strict_notes_compliant":true}'
        ),
    )

    assert result.is_relevant is False
    assert result.has_usable_evidence is True
    assert result.usable_chunk_indices == [1]


def test_rejected_batch_cannot_claim_usable_evidence_without_chunk_indices(monkeypatch):
    result = _run_grader_with(
        monkeypatch,
        (
            '{"reasoning":"No applicable evidence.", "is_relevant":false, '
            '"evidence_quality":"partial", "has_usable_evidence":true, '
            '"answered_requirement_ids":["architecture::r1"], '
            '"strict_notes_compliant":true}'
        ),
    )

    assert result.is_relevant is False
    assert result.has_usable_evidence is False
    assert result.usable_chunk_indices == []
