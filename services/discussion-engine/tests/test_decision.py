from __future__ import annotations

from types import SimpleNamespace

from discussion.decision import (
    effective_criterion_weights,
    parse_candidate_assessments,
    parse_switch_assessment,
    select_challenger,
    synthesize_decision,
)
from discussion.models import CandidateAssessment, DiscussionRound, OpinionResult


def objective(**overrides):
    values = {
        "goal": "Choose between a dense Transformer and sparse MoE Transformer",
        "constraints": ["fixed GPU fleet", "predictable latency"],
        "decision_criteria": ["reasoning capability", "serving cost and latency"],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_open_candidate_matrix_accepts_persona_proposed_solutions():
    raw = "\n".join(
        (
            "CANDIDATE_EVAL: novel_recurrent_design | scores=quality:4,cost:5 | evidence=1 | counter=2 | feasibility=conditional | assumptions=scale transfer | gaps=NONE | combination=not_applicable",
            "CANDIDATE_EVAL: memory_augmented_design | scores=quality:5,cost:3 | evidence=3 | counter=4 | feasibility=feasible | assumptions=NONE | gaps=NONE | combination=not_applicable",
        )
    )

    assessments = parse_candidate_assessments(
        raw,
        {"quality": 0.5, "cost": 0.5},
    )

    assert [item.candidate_id for item in assessments] == [
        "novel_recurrent_design",
        "memory_augmented_design",
    ]
    assert assessments[0].weighted_utility == 4.5


def test_persona_priorities_change_operational_criterion_weights():
    capability = effective_criterion_weights(
        {"priority_weights": {"capability": 5, "cost_efficiency": 1, "latency_predictability": 1}},
        objective(),
    )
    cost = effective_criterion_weights(
        {"priority_weights": {"capability": 1, "cost_efficiency": 5, "latency_predictability": 5}},
        objective(),
    )
    assert capability["reasoning_capability"] > cost["reasoning_capability"]
    assert cost["serving_cost_and_latency"] > capability["serving_cost_and_latency"]
    assert round(sum(capability.values()), 5) == 1


def test_candidate_matrix_computes_weighted_utility_for_open_proposals():
    raw = "\n".join(
        (
            "CANDIDATE_EVAL: dense_transformer | scores=quality:3,cost:4 | evidence=1 | counter=2 | feasibility=feasible | assumptions=NONE | gaps=NONE | combination=not_applicable",
            "CANDIDATE_EVAL: hybrid | scores=quality:4,cost:4 | evidence=3 | counter=4 | feasibility=conditional | assumptions=benefits add | gaps=NONE | combination=unsupported",
        )
    )
    assessments = parse_candidate_assessments(raw, {"quality": 0.5, "cost": 0.5})
    by_id = {item.candidate_id: item for item in assessments}
    assert by_id["dense_transformer"].weighted_utility == 3.5
    assert by_id["hybrid"].weighted_utility == 4.0
    assert by_id["hybrid"].combination_evidence == "unsupported"


def test_candidate_matrix_rejects_out_of_range_and_incomplete_scores():
    raw = (
        "CANDIDATE_EVAL: unsafe_parse | scores=quality:15,cost:4 | "
        "evidence=1 | counter=2 | feasibility=feasible | assumptions=NONE | "
        "gaps=NONE | combination=not_applicable"
    )

    (assessment,) = parse_candidate_assessments(
        raw,
        {"quality": 0.5, "cost": 0.5},
    )

    assert assessment.criterion_scores == {"cost": 4.0}
    assert assessment.weighted_utility is None


def test_typed_switch_test_rejects_current_candidate_as_alternative():
    invalid = parse_switch_assessment(
        "current=dense_transformer | alternative=dense_transformer | "
        "condition=target p95 exceeds the SLA | effect=alternative_wins | status=unresolved",
        current_candidate="dense_transformer",
    )
    valid = parse_switch_assessment(
        "current=dense_transformer | alternative=sparse_moe_transformer | "
        "condition=measured target-hardware throughput improves by 30 percent | "
        "effect=alternative_wins | status=satisfied",
        current_candidate="dense_transformer",
    )
    assert invalid is not None and invalid.valid is False
    assert "alternative must differ from current candidate" in invalid.validation_errors
    assert valid is not None and valid.valid is True


def test_strongest_challenger_prefers_architecture_level_disagreement():
    own = OpinionResult("own", 1, "dense", architecture="dense_transformer")
    same = OpinionResult("same", 1, "also dense", architecture="dense_transformer")
    opposed = OpinionResult("opposed", 1, "moe", architecture="sparse_moe_transformer")
    assert select_challenger("own", {item.persona_id: item for item in (own, same, opposed)}) == "opposed"


def test_final_synthesis_preserves_dissent_and_switch_conditions():
    opinions = (
        OpinionResult(
            "a",
            4,
            "dense",
            architecture="dense_transformer",
            candidate_assessments=(CandidateAssessment("dense_transformer", weighted_utility=4.0),),
        ),
        OpinionResult(
            "b",
            4,
            "moe",
            architecture="sparse_moe_transformer",
            switch_test="dense_transformer; p95 regression exceeds 20%; unresolved",
            critical_uncertainty="target p95 remains unresolved",
            decision_critical=True,
            candidate_assessments=(CandidateAssessment("sparse_moe_transformer", weighted_utility=4.2),),
        ),
    )
    summary = synthesize_decision((DiscussionRound(4, opinions, ()),))
    assert summary.preferred_candidate == "sparse_moe_transformer"
    assert summary.dissenting_positions == {"a": "dense_transformer"}
    assert summary.switch_conditions["b"].endswith("unresolved")
    assert summary.unresolved_uncertainties == ("target p95 remains unresolved",)


def test_final_synthesis_preserves_critical_uncertainty_even_when_switch_says_contradicted():
    opinion = OpinionResult(
        "a",
        4,
        "conditional dense",
        architecture="dense_transformer",
        switch_test="sparse_moe_transformer; measured p95 improves; contradicted",
        critical_uncertainty="target-fleet p95 has not been measured",
        decision_critical=True,
    )
    summary = synthesize_decision((DiscussionRound(4, (opinion,), ()),))
    assert summary.unresolved_uncertainties == ("target-fleet p95 has not been measured",)


def test_open_proposal_scored_by_one_advocate_does_not_override_majority():
    opinions = (
        OpinionResult(
            "a",
            4,
            "shared design",
            architecture="shared_design",
            candidate_assessments=(
                CandidateAssessment("shared_design", weighted_utility=3.8),
            ),
        ),
        OpinionResult(
            "b",
            4,
            "shared design",
            architecture="shared_design",
            candidate_assessments=(
                CandidateAssessment("shared_design", weighted_utility=3.7),
            ),
        ),
        OpinionResult(
            "c",
            4,
            "novel design",
            architecture="novel_design",
            candidate_assessments=(
                CandidateAssessment("novel_design", weighted_utility=5.0),
            ),
        ),
    )

    summary = synthesize_decision((DiscussionRound(4, opinions, ()),))

    assert summary.preferred_candidate == "shared_design"
    assert summary.consensus_status == "plurality"


def test_final_synthesis_excludes_unavailable_stale_positions():
    opinions = (
        OpinionResult(
            "healthy",
            4,
            "current recommendation",
            architecture="current_design",
        ),
        OpinionResult(
            "failed",
            4,
            None,
            architecture="stale_design",
            is_relevant=False,
            status="error",
            error="provider unavailable",
        ),
    )

    summary = synthesize_decision((DiscussionRound(4, opinions, ()),))

    assert summary.preferred_candidate == "current_design"
    assert summary.consensus_status == "consensus"
    assert summary.abstaining_personas == {"failed": "provider unavailable"}


def test_final_synthesis_reports_insufficient_evidence_when_everyone_abstains():
    unavailable = OpinionResult(
        "failed",
        4,
        None,
        architecture="stale_design",
        is_relevant=False,
        status="provider_limit",
    )

    summary = synthesize_decision((DiscussionRound(4, (unavailable,), ()),))

    assert summary.preferred_candidate is None
    assert summary.consensus_status == "insufficient_evidence"
    assert summary.abstaining_personas == {"failed": "provider_limit"}


def test_final_synthesis_keeps_current_recommendations_with_validation_warnings():
    warned = OpinionResult(
        "warned",
        4,
        "current but incomplete matrix",
        architecture="current_design",
        status="incomplete_candidate_evaluation",
    )

    summary = synthesize_decision((DiscussionRound(4, (warned,), ()),))

    assert summary.preferred_candidate == "current_design"
    assert summary.abstaining_personas == {}
