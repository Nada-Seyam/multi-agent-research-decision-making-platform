"""Unified analytics engine — the single entry point for Week 5+.

Combines Tasks 1-4 (opinion change, agreement, influence, sentiment) into
one structured result, so downstream consumers don't need to know how to
call four separate modules or in what order.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from .agreement import AgreementRecord, compute_agreement
from .influence import InfluenceRecord, compute_influence
from .models import Discussion
from .opinion_change import StanceRecord, compute_opinion_change
from .sentiment import SentimentRecord, compute_sentiment


@dataclass
class AnalyticsResult:
    discussion_id: str
    opinion_change: Dict[str, List[StanceRecord]]
    agreement: List[AgreementRecord]
    influence: Dict[str, InfluenceRecord]
    sentiment: List[SentimentRecord]
    issues: List[str] = field(default_factory=list)


@dataclass
class RoundAnalyticsResult:
    """Metrics that are valid immediately after a completed round."""

    discussion_id: str
    round: int
    anchor_candidate: Optional[str]
    anchor_source: str
    stances: Dict[str, StanceRecord]
    agreement: Optional[AgreementRecord]
    sentiment: List[SentimentRecord]
    provisional_anchor: bool = False
    issues: List[str] = field(default_factory=list)
    deferred_metrics: List[str] = field(
        default_factory=lambda: ["influence", "discussion_level_trends"]
    )


def analyze_discussion(discussion: Discussion) -> AnalyticsResult:
    """Run all four analytics categories on a loaded discussion.

    This is the single call site Week 5 (or anything else) needs:
    `analyze_discussion(load_discussion(path))` returns opinion-change,
    agreement, influence, and sentiment data together, computed in the
    correct dependency order (agreement and influence consume committed
    candidates from the opinion-change result).
    """
    opinion_change_result = compute_opinion_change(discussion)
    agreement_result = compute_agreement(opinion_change_result)
    influence_result = compute_influence(discussion, opinion_change_result)
    sentiment_result = compute_sentiment(discussion)

    return AnalyticsResult(
        discussion_id=discussion.discussion_id,
        opinion_change=opinion_change_result,
        agreement=agreement_result,
        influence=influence_result,
        sentiment=sentiment_result,
        issues=list(discussion.issues),
    )


def analyze_round(
    discussion: Discussion,
    round_number: Optional[int] = None,
) -> RoundAnalyticsResult:
    """Analyze one completed round without calculating final-only influence."""
    if not discussion.rounds_present:
        raise ValueError("Cannot analyze a discussion with no completed rounds.")

    selected_round = (
        max(discussion.rounds_present) if round_number is None else round_number
    )
    if selected_round not in discussion.rounds_present:
        raise ValueError(
            f"Round {selected_round} is not present in discussion "
            f"'{discussion.discussion_id}'."
        )

    anchor_is_final = discussion.preferred_candidate_source == "decision_summary"
    analysis_discussion = (
        discussion
        if anchor_is_final
        else replace(
            discussion,
            preferred_candidate=None,
            preferred_candidate_source="pending_final_decision",
        )
    )
    opinion_change_result = compute_opinion_change(analysis_discussion)
    stances = {
        agent_id: record
        for agent_id, records in opinion_change_result.items()
        for record in records
        if record.round == selected_round
    }
    agreement = next(
        (
            record
            for record in compute_agreement(opinion_change_result)
            if record.round == selected_round
        ),
        None,
    )
    sentiment = compute_sentiment(discussion, round_number=selected_round)

    return RoundAnalyticsResult(
        discussion_id=discussion.discussion_id,
        round=selected_round,
        anchor_candidate=analysis_discussion.preferred_candidate,
        anchor_source=analysis_discussion.preferred_candidate_source,
        stances=stances,
        agreement=agreement,
        sentiment=sentiment,
        provisional_anchor=not anchor_is_final,
        issues=list(discussion.issues),
    )
