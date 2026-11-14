import pytest
from analytics.data_loader import DiscussionValidationError, load_discussion

def test_empty_discussion_raises():
    with pytest.raises(DiscussionValidationError):
        load_discussion({"discussion_id": "empty-run", "rounds": []})


def test_missing_discussion_id_raises():
    with pytest.raises(DiscussionValidationError):
        load_discussion({"rounds": [{"round": 1, "opinions": [], "messages": []}]})


def test_missing_confidence_and_position_are_recorded_not_fabricated():
    raw = {
        "discussion_id": "synthetic-missing-data",
        "participants": ["agent_a", "agent_b"],
        "rounds": [
            {
                "round": 1,
                "opinions": [
                    {"persona_id": "agent_a", "position": "candidate_x", "confidence": 0.7},
                    {"persona_id": "agent_b", "position": None, "confidence": None},
                ],
                "messages": [],
            }
        ],
        "decision_summary": {"preferred_candidate": "candidate_x"},
    }
    discussion = load_discussion(raw)
    agent_b_snap = discussion.snapshots["agent_b"][0]
    assert agent_b_snap.candidate is None
    assert agent_b_snap.confidence is None
    assert any("missing" in issue.lower() or "unresolvable" in issue.lower() for issue in discussion.issues)


def test_candidate_id_normalization_matches_across_formats():
    raw = {
        "discussion_id": "synthetic-normalization",
        "participants": ["agent_a", "agent_b"],
        "rounds": [
            {
                "round": 1,
                "opinions": [
                    {"persona_id": "agent_a", "position": "Attention-SSM Hybrid", "confidence": 0.5},
                    {"persona_id": "agent_b", "position": "attention_ssm_hybrid", "confidence": 0.6},
                ],
                "messages": [],
            }
        ],
        "decision_summary": {"preferred_candidate": "attention_ssm_hybrid"},
    }
    discussion = load_discussion(raw)
    assert discussion.snapshots["agent_a"][0].candidate == "attention_ssm_hybrid"
    assert discussion.snapshots["agent_b"][0].candidate == "attention_ssm_hybrid"


def test_duplicate_opinion_in_same_round_keeps_first_and_logs_issue():
    raw = {
        "discussion_id": "synthetic-duplicate",
        "participants": ["agent_a"],
        "rounds": [
            {
                "round": 1,
                "opinions": [
                    {"persona_id": "agent_a", "position": "x", "confidence": 0.4},
                    {"persona_id": "agent_a", "position": "y", "confidence": 0.9},
                ],
                "messages": [],
            }
        ],
        "decision_summary": {"preferred_candidate": "x"},
    }
    discussion = load_discussion(raw)
    assert len(discussion.snapshots["agent_a"]) == 1
    assert discussion.snapshots["agent_a"][0].candidate == "x"
    assert any("duplicate" in issue.lower() for issue in discussion.issues)


def test_duplicate_round_numbers_are_rejected_as_ambiguous():
    raw = {
        "discussion_id": "duplicate-rounds",
        "participants": ["agent_a"],
        "rounds": [
            {"round": 1, "opinions": [], "messages": []},
            {"round": 1, "opinions": [], "messages": []},
        ],
    }

    with pytest.raises(DiscussionValidationError, match="duplicate round number 1"):
        load_discussion(raw)


def test_missing_decision_summary_falls_back_to_majority_vote():
    raw = {
        "discussion_id": "synthetic-fallback",
        "participants": ["agent_a", "agent_b", "agent_c"],
        "rounds": [
            {
                "round": 1,
                "opinions": [
                    {"persona_id": "agent_a", "position": "x", "confidence": 0.5},
                    {"persona_id": "agent_b", "position": "x", "confidence": 0.5},
                    {"persona_id": "agent_c", "position": "y", "confidence": 0.5},
                ],
                "messages": [],
            }
        ],
    }
    discussion = load_discussion(raw)
    assert discussion.preferred_candidate == "x"
    assert discussion.preferred_candidate_source == "majority_fallback"


def test_architecture_is_authoritative_over_conflicting_position():
    raw = {
        "discussion_id": "architecture-authority",
        "participants": ["a"],
        "rounds": [{
            "round": 1,
            "opinions": [{
                "persona_id": "a", "position": "candidate_x",
                "architecture": "candidate_y", "confidence": 0.5,
            }],
            "messages": [],
        }],
        "decision_summary": {"preferred_candidate": "candidate_y"},
    }
    discussion = load_discussion(raw)

    assert discussion.snapshots["a"][0].candidate == "candidate_y"
    assert any("architecture is authoritative" in issue for issue in discussion.issues)


def test_explicit_candidate_aliases_are_applied_without_fuzzy_matching():
    raw = {
        "discussion_id": "candidate-aliases",
        "participants": ["a"],
        "configuration": {
            "candidate_aliases": {"BHGV": "bitemporal_hybrid_graph_vector_memory"}
        },
        "rounds": [{
            "round": 1,
            "opinions": [{
                "persona_id": "a", "architecture": "BHGV",
                "confidence": 0.5, "ranked_candidate": "BHGV",
            }],
            "messages": [],
        }],
        "decision_summary": {
            "preferred_candidate": "bitemporal_hybrid_graph_vector_memory"
        },
    }
    discussion = load_discussion(raw)
    snapshot = discussion.snapshots["a"][0]

    assert snapshot.candidate == "bitemporal_hybrid_graph_vector_memory"
    assert snapshot.ranked_candidate == "bitemporal_hybrid_graph_vector_memory"


def test_utility_is_recomputed_from_scores_and_normalized_weights():
    raw = {
        "discussion_id": "utility-recompute",
        "participants": ["a"],
        "rounds": [{
            "round": 1,
            "opinions": [{
                "persona_id": "a", "architecture": "x", "confidence": 0.5,
                "persona_weights": {"quality": 2, "cost": 1},
                "candidate_assessments": [{
                    "candidate_id": "x",
                    "criterion_scores": {"quality": 5, "cost": 2},
                    "weighted_utility": 4,
                }],
            }],
            "messages": [],
        }],
        "decision_summary": {"preferred_candidate": "x"},
    }
    discussion = load_discussion(raw)
    assessment = discussion.snapshots["a"][0].candidate_assessments[0]

    assert assessment.weighted_utility == pytest.approx(4.0)
    assert assessment.utility_status == "recomputed"


def test_out_of_range_confidence_is_rejected_not_clipped():
    raw = {
        "discussion_id": "invalid-confidence",
        "participants": ["a"],
        "rounds": [{
            "round": 1,
            "opinions": [{
                "persona_id": "a", "architecture": "x", "confidence": 1.5,
            }],
            "messages": [],
        }],
        "decision_summary": {"preferred_candidate": "x"},
    }
    discussion = load_discussion(raw)

    assert discussion.snapshots["a"][0].confidence is None
    assert any("out-of-range confidence" in issue for issue in discussion.issues)


def test_top_level_opinion_counters_are_validation_only():
    raw = {
        "discussion_id": "counter-validation",
        "participants": ["a"],
        "rounds": [{
            "round": 1,
            "opinions": [{
                "persona_id": "a", "architecture": "x", "confidence": 0.5,
                "changed_mind": False,
            }],
            "messages": [],
        }],
        "opinions": {
            "a": {"mind_change_count": 1, "rounds": [{"round": 1}]}
        },
        "decision_summary": {"preferred_candidate": "x"},
    }
    discussion = load_discussion(raw)

    assert len(discussion.snapshots["a"]) == 1
    assert any("mind_change_count=1" in issue for issue in discussion.issues)


def test_malformed_graph_entries_are_skipped_and_reported():
    raw = {
        "discussion_id": "malformed-graph",
        "participants": ["a", "b"],
        "rounds": [{
            "round": 1,
            "opinions": [
                {"persona_id": "a", "architecture": "x"},
                {"persona_id": "b", "architecture": "x"},
            ],
            "messages": [],
        }],
        "decision_summary": {"preferred_candidate": "x"},
        "graph": {
            "agents": ["a", "b", None],
            "edges": [["a", "b"], ["a"], 42],
            "directed": "yes",
        },
    }

    discussion = load_discussion(raw)

    assert discussion.graph.agents == ["a", "b"]
    assert discussion.graph.edges == [("a", "b")]
    assert discussion.graph.directed is True
    assert any("invalid agent" in issue for issue in discussion.issues)
    assert any("malformed edge" in issue for issue in discussion.issues)
    assert any("must be boolean" in issue for issue in discussion.issues)
