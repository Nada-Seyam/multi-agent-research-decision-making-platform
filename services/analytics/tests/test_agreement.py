import pytest

from analytics.agreement import compute_agreement
from analytics.data_loader import load_discussion
from analytics.opinion_change import StanceRecord, compute_opinion_change

def _candidates(round_number, values):
    return {
        f"agent_{index}": [
            StanceRecord(
                agent_id=f"agent_{index}",
                round=round_number,
                candidate=candidate,
                stance=None,
                change=None,
            )
        ]
        for index, candidate in enumerate(values)
    }


def test_unanimous_candidate_choice_is_complete_agreement():
    record = compute_agreement(_candidates(1, ["a", "a", "a", "a"]))[0]

    assert record.score == 1.0
    assert record.disagreement_score == 0.0
    assert record.agreeing_pairs == 6
    assert record.disagreeing_pairs == 0
    assert record.total_pairs == 6


def test_all_distinct_candidates_are_complete_disagreement():
    record = compute_agreement(_candidates(1, ["a", "b", "c", "d"]))[0]

    assert record.score == 0.0
    assert record.disagreement_score == 1.0
    assert record.agreeing_pairs == 0
    assert record.disagreeing_pairs == 6


def test_one_matching_pair_out_of_six_matches_hand_calculation():
    record = compute_agreement(_candidates(1, ["a", "b", "c", "b"]))[0]

    assert record.score == pytest.approx(1 / 6, abs=1e-6)
    assert record.disagreement_score == pytest.approx(5 / 6, abs=1e-6)
    assert record.agreeing_pairs == 1
    assert record.disagreeing_pairs == 5


def test_two_by_two_split_has_two_agreeing_pairs():
    record = compute_agreement(_candidates(1, ["a", "a", "b", "b"]))[0]

    assert record.score == pytest.approx(2 / 6, abs=1e-6)
    assert record.disagreement_score == pytest.approx(4 / 6, abs=1e-6)


def test_single_agent_round_is_insufficient_data():
    record = compute_agreement(_candidates(1, ["a"]))[0]

    assert record.score is None
    assert record.disagreement_score is None
    assert record.num_agents_considered == 1
    assert record.total_pairs == 0
    assert "insufficient_data" in record.reason


def test_missing_candidates_are_excluded():
    record = compute_agreement(_candidates(1, [None, "a", "a"]))[0]

    assert record.num_agents_considered == 2
    assert record.score == 1.0
    assert record.disagreement_score == 0.0


def _discussion_with_anchor(anchor):
    return load_discussion(
        {
            "discussion_id": f"anchor-{anchor}",
            "participants": ["one", "two", "three"],
            "rounds": [{
                "round": 1,
                "opinions": [
                    {"persona_id": "one", "architecture": "a"},
                    {"persona_id": "two", "architecture": "b"},
                    {"persona_id": "three", "architecture": "b"},
                ],
                "messages": [],
            }],
            "decision_summary": {"preferred_candidate": anchor},
        }
    )


def test_agreement_is_independent_of_the_anchor():
    result_a = compute_agreement(
        compute_opinion_change(_discussion_with_anchor("a"))
    )[0]
    result_b = compute_agreement(
        compute_opinion_change(_discussion_with_anchor("b"))
    )[0]

    assert result_a == result_b
    assert result_a.score == pytest.approx(1 / 3, abs=1e-6)


def test_explicit_candidate_aliases_agree_after_loader_normalization():
    discussion = load_discussion(
        {
            "discussion_id": "agreement-aliases",
            "participants": ["one", "two"],
            "configuration": {"candidate_aliases": {"BHGV": "full_name"}},
            "rounds": [{
                "round": 1,
                "opinions": [
                    {"persona_id": "one", "architecture": "BHGV"},
                    {"persona_id": "two", "architecture": "full_name"},
                ],
                "messages": [],
            }],
            "decision_summary": {"preferred_candidate": "full_name"},
        }
    )
    record = compute_agreement(compute_opinion_change(discussion))[0]

    assert record.score == 1.0
    assert record.disagreement_score == 0.0
