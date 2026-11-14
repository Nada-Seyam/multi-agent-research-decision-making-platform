import copy

import pytest

from analytics.data_loader import load_discussion
from analytics.influence import (
    _candidate_movement,
    _tfidf_relevance_scores,
    compute_influence,
)
from analytics.opinion_change import compute_opinion_change

MESSAGES_POSITIVE = [
    "graph evidence supports adoption",
    "cost latency benchmark",
    "deployment risk",
    "provenance evidence supports architecture",
    "throughput benchmark cost",
    "privacy review",
]

RESPONSES_POSITIVE = [
    "graph evidence supports adoption",
    "privacy controls audit",
    "deployment risk remains",
    "provenance evidence supports architecture",
    "safety governance policy",
    "privacy review remains open",
]

TARGET_CANDIDATES = ["b", "a", "b", "b", "a", "b", "b"]


def _discussion(
    messages=MESSAGES_POSITIVE,
    responses=RESPONSES_POSITIVE,
    target_candidates=TARGET_CANDIDATES,
    anchor="a",
    challenger="sender",
):
    rounds = []
    for index, target_candidate in enumerate(target_candidates):
        round_number = index + 1
        target_opinion = {
            "persona_id": "recipient",
            "architecture": target_candidate,
        }
        if round_number > 1:
            target_opinion.update(
                {
                    "challenger_id": challenger,
                    "challenged_claim": responses[index - 1],
                    "update_reason": responses[index - 1],
                }
            )

        round_messages = []
        if index < len(messages):
            round_messages.append(
                {
                    "round_number": round_number,
                    "sequence": index,
                    "sender": "sender",
                    "recipients": ["recipient"],
                    "content": messages[index],
                }
            )

        rounds.append(
            {
                "round": round_number,
                "opinions": [
                    {"persona_id": "sender", "architecture": "a"},
                    target_opinion,
                ],
                "messages": round_messages,
            }
        )

    return load_discussion(
        {
            "discussion_id": "synthetic-influence",
            "participants": ["sender", "recipient"],
            "rounds": rounds,
            "decision_summary": {"preferred_candidate": anchor},
        }
    )


def _compute(discussion):
    return compute_influence(discussion, compute_opinion_change(discussion))


def test_exact_candidate_movement_does_not_collapse_non_anchor_candidates():
    assert _candidate_movement("b", "a", "c") == 0
    assert _candidate_movement("b", "c", "b") == 1
    assert _candidate_movement("b", "b", "c") == -1


def test_relevance_uses_textual_overlap_not_message_length():
    scores = _tfidf_relevance_scores(
        [
            ("graph provenance evidence", "graph provenance evidence"),
            ("unrelated " * 100, "privacy isolation controls"),
        ]
    )

    assert scores[0] > scores[1]
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == 0.0


def test_semantically_relevant_messages_correlate_with_movement_toward_sender():
    record = _compute(_discussion())["sender"]

    assert record.score is not None
    assert record.score > 0.5
    assert record.num_data_points == 6
    assert record.num_movement_events == 4
    assert record.explicit_attribution_matches == 6


def test_influence_is_independent_of_anchor():
    with_anchor_a = _compute(_discussion(anchor="a"))["sender"]
    with_anchor_z = _compute(_discussion(anchor="z"))["sender"]

    assert with_anchor_a.score == with_anchor_z.score
    assert with_anchor_a.num_movement_events == with_anchor_z.num_movement_events


def test_explicit_challenger_mismatch_is_retained_as_unconfirmed():
    record = _compute(_discussion(challenger="someone_else"))["sender"]

    assert record.score is not None
    assert record.num_data_points == 6
    assert record.explicit_attribution_matches == 0
    assert record.unconfirmed_observations == 6


def test_missing_round_is_not_bridged():
    raw = {
        "discussion_id": "missing-round-influence",
        "participants": ["sender", "recipient"],
        "rounds": [
            {
                "round": 1,
                "opinions": [
                    {"persona_id": "sender", "architecture": "a"},
                    {"persona_id": "recipient", "architecture": "b"},
                ],
                "messages": [{
                    "round_number": 1,
                    "sequence": 1,
                    "sender": "sender",
                    "recipients": ["recipient"],
                    "content": "adopt graph evidence",
                }],
            },
            {
                "round": 3,
                "opinions": [
                    {"persona_id": "sender", "architecture": "a"},
                    {
                        "persona_id": "recipient",
                        "architecture": "a",
                        "challenger_id": "sender",
                        "update_reason": "adopt graph evidence",
                    },
                ],
                "messages": [],
            },
        ],
        "decision_summary": {"preferred_candidate": "a"},
    }
    record = _compute(load_discussion(raw))["sender"]

    assert record.num_data_points == 0
    assert record.score is None


def test_agent_with_no_outgoing_messages_is_insufficient_data():
    record = _compute(_discussion())["recipient"]

    assert record.score is None
    assert "no usable directed messages" in record.reason


def test_one_candidate_movement_gets_low_reliability_when_correlation_exists():
    candidates = ["b", "b", "b", "b", "b", "b", "a"]
    record = _compute(
        _discussion(target_candidates=candidates)
    )["sender"]

    assert record.score is not None
    assert record.num_data_points == 6
    assert record.num_movement_events == 1
    assert record.reliability == "low"


def test_unconfirmed_observations_are_reported_when_challenger_is_missing():
    discussion = _discussion()
    raw = copy.deepcopy(discussion)
    for snapshot in raw.snapshots["recipient"]:
        snapshot.challenger_id = None

    record = _compute(raw)["sender"]

    assert record.num_data_points == 6
    assert record.explicit_attribution_matches == 0
    assert record.unconfirmed_observations == 6


def _indirect_discussion():
    target_candidates = ["b", "b", "a", "b", "b"]
    source_messages = [
        "graph provenance evidence",
        "latency cost benchmark",
        "privacy tenant isolation",
        "reliability circuit breaker",
    ]
    bridge_messages = [
        "initial unrelated proposal",
        "graph provenance evidence",
        "latency cost benchmark",
        "privacy tenant isolation",
    ]
    bridge_responses = [
        "graph provenance evidence",
        "latency concern remains",
        "privacy tenant isolation audit",
        "different operational response",
    ]
    rounds = []
    for index, target_candidate in enumerate(target_candidates):
        round_number = index + 1
        bridge_opinion = {
            "persona_id": "bridge",
            "architecture": "b",
        }
        target_opinion = {
            "persona_id": "target",
            "architecture": target_candidate,
        }
        if round_number > 1:
            bridge_opinion.update({
                "challenger_id": "source",
                "challenged_claim": bridge_responses[index - 1],
                "update_reason": bridge_responses[index - 1],
            })
            target_opinion.update({
                "challenger_id": "bridge",
                "challenged_claim": bridge_messages[index - 1],
                "update_reason": bridge_messages[index - 1],
            })

        messages = []
        if index < len(source_messages):
            messages.extend([
                {
                    "round_number": round_number,
                    "sequence": 1,
                    "sender": "source",
                    "recipients": ["bridge"],
                    "content": source_messages[index],
                },
                {
                    "round_number": round_number,
                    "sequence": 2,
                    "sender": "bridge",
                    "recipients": ["target"],
                    "content": bridge_messages[index],
                },
            ])
        rounds.append({
            "round": round_number,
            "opinions": [
                {"persona_id": "source", "architecture": "a"},
                bridge_opinion,
                target_opinion,
            ],
            "messages": messages,
        })

    return load_discussion({
        "discussion_id": "indirect-influence",
        "participants": ["source", "bridge", "target"],
        "rounds": rounds,
        "decision_summary": {"preferred_candidate": "a"},
    })


def test_two_hop_temporal_path_can_influence_a_non_neighbor():
    record = _compute(_indirect_discussion())["source"]

    assert record.num_direct_observations == 4
    assert record.num_indirect_observations == 3
    assert record.num_data_points == 7
    assert record.num_movement_events == 2
    assert record.num_recipients == 2
    assert record.score is not None


def test_immediate_return_edge_does_not_create_self_influence():
    discussion = _discussion()
    discussion.messages.extend([
        copy.deepcopy(message)
        for message in discussion.messages
    ])
    for message in discussion.messages[len(discussion.messages) // 2:]:
        message.sender = "recipient"
        message.recipients = ["sender"]

    record = _compute(discussion)["sender"]

    assert record.num_indirect_observations == 0


def test_unrelated_intermediary_message_does_not_propagate_influence():
    discussion = _indirect_discussion()
    for message in discussion.messages:
        if message.sender == "bridge" and message.round_number > 1:
            message.content = "unrelated_zulu unrelated_quartz"

    record = _compute(discussion)["source"]

    assert record.num_direct_observations == 4
    assert record.num_indirect_observations == 0
    assert record.num_opportunities == 7
    assert record.opportunity_coverage == pytest.approx(4 / 7)
