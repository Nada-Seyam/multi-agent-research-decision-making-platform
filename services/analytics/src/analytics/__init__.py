"""
Week 4 — Analytics & Intelligence Layer.

This package implements all four required analytics categories plus a
unified engine:

    - opinion_change   (Task 1)
    - agreement        (Task 2)  — consumes committed candidates from opinion_change
    - sentiment        (Task 3)  — independent of Tasks 1/2/4
    - influence        (Task 4)  — consumes exact candidates from opinion_change
    - engine           unified entry point combining all four

Each metric module exposes a single pure function that takes a
`Discussion` (loaded via `analytics.data_loader.load_discussion`) and
returns plain dataclasses describing its results.
"""

from .models import (
    AgentRoundSnapshot,
    CandidateAssessment,
    Discussion,
    DiscussionGraph,
    Message,
    SwitchAssessment,
)
from .data_loader import load_discussion, DiscussionValidationError
from .opinion_change import compute_opinion_change, StanceRecord
from .agreement import compute_agreement, AgreementRecord
from .sentiment import CandidateSentiment, compute_sentiment, SentimentRecord
from .influence import compute_influence, InfluenceRecord
from .engine import analyze_discussion, AnalyticsResult

__all__ = [
    "Discussion",
    "DiscussionGraph",
    "Message",
    "AgentRoundSnapshot",
    "CandidateAssessment",
    "SwitchAssessment",
    "load_discussion",
    "DiscussionValidationError",
    "compute_opinion_change",
    "StanceRecord",
    "compute_agreement",
    "AgreementRecord",
    "compute_sentiment",
    "CandidateSentiment",
    "SentimentRecord",
    "compute_influence",
    "InfluenceRecord",
    "analyze_discussion",
    "AnalyticsResult",
]
