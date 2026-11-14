import pytest

from analytics.data_loader import load_discussion
from analytics.sentiment import compute_sentiment

def _discussion_with_message(
    content,
    sender_candidate="alternative_architecture",
    recipients=None,
    recipient_candidates=None,
    sender_extra=None,
):
    recipients = recipients if recipients is not None else ["recipient"]
    recipient_candidates = recipient_candidates or {
        "recipient": "efficient_attention_transformer"
    }
    sender_opinion = {
        "persona_id": "sender",
        "architecture": sender_candidate,
    }
    if sender_extra:
        sender_opinion.update(sender_extra)
    participants = ["sender"] + list(recipient_candidates)
    opinions = [sender_opinion] + [
        {"persona_id": recipient, "architecture": candidate}
        for recipient, candidate in recipient_candidates.items()
    ]
    return load_discussion(
        {
            "discussion_id": "synthetic-sentiment",
            "participants": participants,
            "rounds": [
                {
                    "round": 1,
                    "opinions": opinions,
                    "messages": [
                        {
                            "round_number": 1,
                            "sequence": 1,
                            "sender": "sender",
                            "recipients": recipients,
                            "content": content,
                        }
                    ],
                }
            ],
        }
    )


def _first_evaluation(discussion):
    return compute_sentiment(discussion)[0].evaluations[0]


def test_explicit_support_for_recipient_candidate_is_positive():
    evaluation = _first_evaluation(
        _discussion_with_message(
            "I recommend the efficient attention transformer because it meets the latency constraint."
        )
    )

    assert evaluation.label == "positive"
    assert evaluation.score > 0.15
    assert "candidate_nli" in evaluation.evidence_source


def test_explicit_criticism_of_recipient_candidate_is_negative():
    evaluation = _first_evaluation(
        _discussion_with_message(
            "The efficient attention transformer fails the reliability constraint and creates a critical weakness."
        )
    )

    assert evaluation.label == "negative"
    assert evaluation.score < -0.15


def test_balanced_candidate_evaluation_is_neutral():
    evaluation = _first_evaluation(
        _discussion_with_message(
            "The efficient attention transformer is robust but costly."
        )
    )

    assert evaluation.label == "neutral"
    assert evaluation.score == pytest.approx(0.0)


def test_factual_candidate_mention_is_neutral():
    evaluation = _first_evaluation(
        _discussion_with_message(
            "The efficient attention transformer uses a 512 by 512 attention matrix."
        )
    )

    assert evaluation.label == "neutral"
    assert evaluation.score == 0.0


def test_candidate_not_referenced_is_neutral():
    evaluation = _first_evaluation(
        _discussion_with_message(
            "The deployment matrix has dimensions 512 by 512."
        )
    )

    assert evaluation.score == 0.0
    assert evaluation.label == "neutral"
    assert evaluation.evidence_source == "no_candidate_reference"
    assert evaluation.reason is None


def test_same_committed_candidate_is_positive_structured_evidence():
    evaluation = _first_evaluation(
        _discussion_with_message(
            "The deployment details will be finalized later.",
            sender_candidate="efficient_attention_transformer",
        )
    )

    assert evaluation.label == "positive"
    assert evaluation.score == 1.0
    assert evaluation.evidence_source == "shared_committed_candidate"


def test_candidate_matrix_is_primary_structured_evidence():
    sender_extra = {
        "persona_weights": {"quality": 1.0},
        "candidate_assessments": [
            {
                "candidate_id": "efficient_attention_transformer",
                "criterion_scores": {"quality": 4.5},
                "weighted_utility": 4.5,
            },
            {
                "candidate_id": "alternative_architecture",
                "criterion_scores": {"quality": 2.0},
                "weighted_utility": 2.0,
            },
        ],
    }
    evaluation = _first_evaluation(
        _discussion_with_message(
            "The implementation details will be reviewed.",
            sender_extra=sender_extra,
        )
    )

    assert evaluation.label == "positive"
    assert evaluation.score == pytest.approx(0.5)
    assert evaluation.evidence_source == "candidate_matrix"


def test_different_committed_candidates_without_reference_are_neutral():
    evaluation = _first_evaluation(
        _discussion_with_message("The release schedule is documented separately.")
    )

    assert evaluation.score == 0.0
    assert evaluation.label == "neutral"
    assert evaluation.evidence_source == "no_candidate_reference"


def test_negated_risk_is_not_classified_as_negative():
    evaluation = _first_evaluation(
        _discussion_with_message(
            "The efficient attention transformer does not create a reliability risk."
        )
    )

    assert evaluation.score is not None
    assert evaluation.score > 0
    assert evaluation.label == "positive"


def test_one_message_can_evaluate_two_recipient_candidates_differently():
    discussion = _discussion_with_message(
        "The efficient attention transformer is superior and meets the constraint.\n"
        "The sparse moe transformer fails the reliability requirement.",
        recipients=["recipient_a", "recipient_b"],
        recipient_candidates={
            "recipient_a": "efficient_attention_transformer",
            "recipient_b": "sparse_moe_transformer",
        },
    )
    record = compute_sentiment(discussion)[0]

    assert len(record.evaluations) == 2
    assert record.evaluations[0].label == "positive"
    assert record.evaluations[1].label == "negative"


def test_candidate_acronym_is_learned_from_discussion_text():
    discussion = load_discussion(
        {
            "discussion_id": "synthetic-alias",
            "participants": ["sender", "recipient"],
            "rounds": [
                {
                    "round": 1,
                    "opinions": [
                        {
                            "persona_id": "sender",
                            "architecture": "alternative_architecture",
                        },
                        {
                            "persona_id": "recipient",
                            "architecture": "bitemporal_hybrid_graph_vector_memory",
                        },
                    ],
                    "messages": [
                        {
                            "round_number": 1,
                            "sequence": 1,
                            "sender": "recipient",
                            "recipients": ["sender"],
                            "content": (
                                "Bitemporal Hybrid Graph-Vector Memory (BHGV) "
                                "is the full candidate name."
                            ),
                        },
                        {
                            "round_number": 1,
                            "sequence": 2,
                            "sender": "sender",
                            "recipients": ["recipient"],
                            "content": "BHGV should not be chosen.",
                        },
                    ],
                }
            ],
        }
    )

    evaluation = compute_sentiment(discussion)[1].evaluations[0]

    assert evaluation.label == "negative"
    assert evaluation.score < -0.15
    assert evaluation.evidence_source == "candidate_nli"


def test_empty_message_gets_no_evaluations_with_reason():
    record = compute_sentiment(_discussion_with_message(""))[0]

    assert record.evaluations == []
    assert record.reason == "empty_or_missing_message_content"


def test_message_without_recipients_gets_clear_reason():
    record = compute_sentiment(
        _discussion_with_message("A technical message", recipients=[])
    )[0]

    assert record.evaluations == []
    assert record.reason == "message_has_no_recipients"


def test_round_filter_skips_inference_for_previous_messages(monkeypatch):
    class CountingNLI:
        pair_count = 0

        def entailment_probabilities(self, pairs):
            self.pair_count += len(pairs)
            return [0.9 if index % 2 == 0 else 0.05 for index, _ in enumerate(pairs)]

    classifier = CountingNLI()
    monkeypatch.setattr(
        "analytics.sentiment._get_nli_classifier",
        lambda: classifier,
    )
    discussion = load_discussion(
        {
            "discussion_id": "round-filter",
            "participants": ["sender", "recipient"],
            "rounds": [
                {
                    "round": round_number,
                    "opinions": [
                        {"persona_id": "sender", "architecture": "alternative"},
                        {"persona_id": "recipient", "architecture": "candidate_x"},
                    ],
                    "messages": [
                        {
                            "round_number": round_number,
                            "sequence": round_number,
                            "sender": "sender",
                            "recipients": ["recipient"],
                            "content": "I recommend candidate x.",
                        }
                    ],
                }
                for round_number in (1, 2)
            ],
        }
    )

    records = compute_sentiment(discussion, round_number=2)

    assert [record.round for record in records] == [2]
    assert classifier.pair_count == 2


def test_nli_pairs_are_batched_once_and_reused_from_cache(monkeypatch):
    from analytics import sentiment

    class CountingNLI:
        def __init__(self):
            self.calls = 0
            self.pair_count = 0

        def entailment_probabilities(self, pairs):
            self.calls += 1
            self.pair_count += len(pairs)
            return [
                0.9 if "supports choosing" in hypothesis else 0.05
                for _, hypothesis in pairs
            ]

    classifier = CountingNLI()
    monkeypatch.setattr(sentiment, "_get_nli_classifier", lambda: classifier)
    with sentiment._NLI_SCORE_CACHE_LOCK:
        sentiment._NLI_SCORE_CACHE.clear()
    discussion = load_discussion(
        {
            "discussion_id": "batched-nli",
            "participants": ["sender", "recipient"],
            "rounds": [
                {
                    "round": 1,
                    "opinions": [
                        {"persona_id": "sender", "architecture": "alternative"},
                        {"persona_id": "recipient", "architecture": "candidate_x"},
                    ],
                    "messages": [
                        {
                            "round_number": 1,
                            "sequence": sequence,
                            "sender": "sender",
                            "recipients": ["recipient"],
                            "content": content,
                        }
                        for sequence, content in enumerate(
                            [
                                "I recommend candidate x for latency.",
                                "Candidate x meets the reliability target.",
                            ],
                            start=1,
                        )
                    ],
                }
            ],
        }
    )

    first = compute_sentiment(discussion)
    second = compute_sentiment(discussion)

    assert len(first) == len(second) == 2
    assert classifier.calls == 1
    assert classifier.pair_count == 4
