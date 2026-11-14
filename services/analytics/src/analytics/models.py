"""Plain data structures representing a validated Week 3 discussion.

These structures retain the Week 3 fields needed to distinguish committed
position, evaluated matrix preference, confidence, and reasoning updates,
plus the messages used by the other analytics modules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CandidateAssessment:
    """One candidate's structured evaluation in an opinion snapshot.

    ``weighted_utility`` is the usable value after validation.  It is
    recomputed from criterion scores and persona weights when possible;
    ``declared_weighted_utility`` preserves the Week 3 value for audit.
    """

    raw_candidate_id: Optional[str]
    candidate_id: Optional[str]
    criterion_scores: Dict[str, float] = field(default_factory=dict)
    weighted_utility: Optional[float] = None
    declared_weighted_utility: Optional[float] = None
    utility_status: str = "missing"


@dataclass
class SwitchAssessment:
    """The Week 3 contract governing whether a mind change is allowed."""

    current_candidate: Optional[str] = None
    alternative_candidate: Optional[str] = None
    condition_effect: Optional[str] = None
    status: Optional[str] = None
    valid: Optional[bool] = None
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class AgentRoundSnapshot:
    """One agent's opinion snapshot at one round, already normalized."""

    agent_id: str
    round_number: int
    raw_position: Optional[str]
    candidate: Optional[str]
    confidence: Optional[float]
    raw_architecture: Optional[str] = None
    ranked_candidate: Optional[str] = None
    candidate_assessments: List[CandidateAssessment] = field(default_factory=list)
    ranking_margin: Optional[float] = None
    ranking_changed: Optional[bool] = None
    update_decision: Optional[str] = None
    update_reason: Optional[str] = None
    changed_mind: Optional[bool] = None
    material_update: Optional[bool] = None
    assumptions_changed: Optional[bool] = None
    confidence_changed: Optional[bool] = None
    challenger_id: Optional[str] = None
    challenged_claim: Optional[str] = None
    switch_assessment: Optional[SwitchAssessment] = None


@dataclass
class Message:
    """One discussion message, with a synthesized stable id.

    Week 3 message records do not include a unique id field, so we
    synthesize one deterministically from (round, sequence, sender).
    This id is purely a reference key for downstream results (e.g.
    sentiment output) — it is not part of the original Week 3 data.
    """

    message_id: str
    round_number: int
    sequence: int
    sender: Optional[str]
    recipients: List[str]
    content: Optional[str]


@dataclass
class DiscussionGraph:
    """Validated communication topology declared by the Week 3 runner."""

    agents: List[str] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    directed: bool = True


@dataclass
class Discussion:
    """A validated, normalized view of a single Week 3 discussion run."""

    discussion_id: str
    agents: List[str]
    rounds_present: List[int]
    preferred_candidate: Optional[str]
    preferred_candidate_source: str  # "decision_summary" | "majority_fallback" | "unavailable"
    snapshots: Dict[str, List[AgentRoundSnapshot]]  # agent_id -> snapshots sorted by round
    messages: List[Message]
    issues: List[str] = field(default_factory=list)  # human-readable validation notes
    topic: Optional[str] = None  # from data['objective']['goal'], report context only
    week3_consensus_status: Optional[str] = None  # from decision_summary.consensus_status
    utility_consistency_tolerance: float = 0.25
    graph: Optional[DiscussionGraph] = None
