from analytics.data_loader import load_discussion
from analytics.opinion_change import compute_opinion_change

def _opinion(agent, candidate, confidence=0.5):
    return {
        "persona_id": agent,
        "position": candidate,
        "architecture": candidate,
        "confidence": confidence,
    }


def _discussion(rounds, anchor="x", participants=None):
    return load_discussion(
        {
            "discussion_id": "synthetic-opinion-change",
            "participants": participants or ["a", "b"],
            "rounds": [
                {"round": index + 1, "opinions": opinions, "messages": []}
                for index, opinions in enumerate(rounds)
            ],
            "decision_summary": {"preferred_candidate": anchor} if anchor else {},
        }
    )


def test_stance_measures_committed_anchor_alignment_not_confidence():
    discussion = _discussion(
        [[_opinion("a", "x", 0.2), _opinion("b", "y", 0.9)]]
    )
    result = compute_opinion_change(discussion)

    assert result["a"][0].stance == 1.0
    assert result["b"][0].stance == -1.0
    assert not hasattr(result["a"][0], "confidence")


def test_switch_to_anchor_has_positive_two_stance_change():
    discussion = _discussion(
        [
            [_opinion("a", "y"), _opinion("b", "x")],
            [_opinion("a", "x"), _opinion("b", "x")],
        ]
    )
    record = compute_opinion_change(discussion)["a"][1]

    assert record.candidate_changed is True
    assert record.stance == 1.0
    assert record.change == 2.0


def test_switch_between_non_anchor_candidates_is_visible_in_candidate_column():
    discussion = _discussion(
        [[_opinion("a", "y")], [_opinion("a", "z")]],
        participants=["a"],
    )
    records = compute_opinion_change(discussion)["a"]

    assert [record.candidate for record in records] == ["y", "z"]
    assert records[1].change == 0.0
    assert records[1].candidate_changed is True


def test_confidence_change_is_not_calculated_as_opinion_change():
    discussion = _discussion(
        [[_opinion("a", "y", 0.8)], [_opinion("a", "y", 0.4)]],
        participants=["a"],
    )
    record = compute_opinion_change(discussion)["a"][1]

    assert record.change == 0.0
    assert not hasattr(record, "confidence_change")


def test_matrix_and_decision_fields_are_not_calculated():
    first = _opinion("a", "x")
    first.update(
        {
            "ranked_candidate": "y",
            "candidate_assessments": [
                {
                    "candidate_id": "x",
                    "criterion_scores": {"quality": 4},
                    "weighted_utility": 4,
                },
                {
                    "candidate_id": "y",
                    "criterion_scores": {"quality": 5},
                    "weighted_utility": 5,
                },
            ],
            "persona_weights": {"quality": 1},
            "update_decision": "retain",
        }
    )
    record = compute_opinion_change(
        _discussion([[first]], participants=["a"])
    )["a"][0]

    for removed_field in (
        "matrix_winner",
        "preference_gap",
        "decision_consistency",
        "change_type",
    ):
        assert not hasattr(record, removed_field)


def test_missing_snapshot_still_produces_one_record_for_that_round():
    discussion = _discussion(
        [
            [_opinion("a", "x"), _opinion("b", "y")],
            [_opinion("a", "x")],
        ]
    )
    records = compute_opinion_change(discussion)["b"]

    assert [record.round for record in records] == [1, 2]
    assert records[1].stance is None
    assert records[1].reason == "missing_opinion_snapshot"


def test_change_is_not_calculated_across_a_missing_round():
    discussion = _discussion(
        [[_opinion("a", "y")], [], [_opinion("a", "x")]],
        participants=["a"],
    )
    records = compute_opinion_change(discussion)["a"]

    assert records[1].stance is None
    assert records[2].stance == 1.0
    assert records[2].change is None
    assert records[2].change_reason == "missing_previous_or_current_round_stance"


def test_missing_confidence_does_not_destroy_known_committed_stance():
    record = compute_opinion_change(
        _discussion([[_opinion("a", "x", None)]], participants=["a"])
    )["a"][0]

    assert record.stance == 1.0


def test_no_anchor_yields_none_stance():
    raw = {
        "discussion_id": "synthetic-no-anchor",
        "participants": ["a"],
        "rounds": [
            {
                "round": 1,
                "opinions": [_opinion("a", None)],
                "messages": [],
            }
        ],
    }
    record = compute_opinion_change(load_discussion(raw))["a"][0]

    assert record.stance is None
    assert record.reason in {
        "no_anchor_candidate_available_for_discussion",
        "unresolvable_candidate_for_this_snapshot",
    }
