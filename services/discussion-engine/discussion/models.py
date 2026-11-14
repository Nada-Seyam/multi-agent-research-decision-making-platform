"""Compact discussion results and runtime records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

SCHEMA_VERSION = "1.0"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


@dataclass(frozen=True)
class SharedResearchContext:
    plan: Any
    evidence_pool: list[Any]
    dossier: Any
    query_history: list[str] = field(default_factory=list)
    rejected_evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class CandidateAssessment:
    """One persona's evidence-linked evaluation of one candidate."""

    candidate_id: str
    criterion_scores: dict[str, float] = field(default_factory=dict)
    weighted_utility: float | None = None
    evidence_refs: tuple[int, ...] = ()
    counter_evidence_refs: tuple[int, ...] = ()
    assumptions: tuple[str, ...] = ()
    evidence_gaps: tuple[str, ...] = ()
    feasibility: str = "unknown"
    combination_evidence: str = "not_applicable"


@dataclass(frozen=True)
class SwitchAssessment:
    """Machine-checkable counterfactual for changing architecture."""

    current_candidate: str | None = None
    alternative_candidate: str | None = None
    condition: str | None = None
    condition_effect: str | None = None
    status: str | None = None
    valid: bool = False
    validation_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionSummary:
    """Deterministic, minority-preserving synthesis of final persona state."""

    preferred_candidate: str | None
    consensus_status: str
    support_personas: tuple[str, ...] = ()
    dissenting_positions: dict[str, str] = field(default_factory=dict)
    strongest_alternative: str | None = None
    aggregate_utilities: dict[str, float] = field(default_factory=dict)
    unresolved_uncertainties: tuple[str, ...] = ()
    switch_conditions: dict[str, str] = field(default_factory=dict)
    abstaining_personas: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OpinionResult:
    persona_id: str
    round_number: int
    opinion: str | None
    position: str | None = None
    architecture: str | None = None
    components: str | None = None
    considered_alternatives: str | None = None
    switch_test: str | None = None
    switch_assessment: SwitchAssessment | None = None
    strongest_alternative: str | None = None
    criteria_scores: str | None = None
    decisive_criterion: str | None = None
    update_decision: str | None = None
    update_reason: str | None = None
    confidence: float = 0.5
    changed_mind: bool = False
    change_reason: str | None = None
    is_relevant: bool = True
    status: str = "ok"
    error: str | None = None
    raw_opinion: str | None = None
    candidate_assessments: tuple[CandidateAssessment, ...] = ()
    persona_weights: dict[str, float] = field(default_factory=dict)
    challenger_id: str | None = None
    challenged_claim: str | None = None
    decision_critical: bool = False
    critical_uncertainty: str | None = None
    ranked_candidate: str | None = None
    previous_ranked_candidate: str | None = None
    ranking_margin: float | None = None
    ranking_changed: bool = False
    near_tie: bool = False
    assumptions_changed: bool = False
    confidence_changed: bool = False
    material_update: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class DiscussionMessage:
    round_number: int
    sequence: int
    sender: str
    recipients: tuple[str, ...]
    content: str

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class DiscussionRound:
    round_number: int
    opinions: tuple[OpinionResult, ...]
    messages: tuple[DiscussionMessage, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical round representation used everywhere."""
        return {
            "round": self.round_number,
            "opinions": [item.to_dict() for item in self.opinions],
            "messages": [item.to_dict() for item in self.messages],
        }


@dataclass(frozen=True)
class DiscussionResult:
    discussion_id: str
    objective: Any
    participants: tuple[str, ...]
    graph: dict[str, Any]
    research: SharedResearchContext | None
    rounds: tuple[DiscussionRound, ...]
    preparation: dict[str, Any] | None = None
    configuration: dict[str, Any] = field(default_factory=dict)
    logs: dict[str, str] = field(default_factory=dict)
    decision_summary: DecisionSummary | None = None
    conclusion: str | None = None
    status: str = "completed"
    started_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None
    failure: dict[str, Any] | None = None
    warnings: tuple[dict[str, Any], ...] = ()

    @property
    def opinions(self) -> tuple[OpinionResult, ...]:
        """Flatten rounds for callers of the original result shape."""
        return tuple(
            opinion
            for discussion_round in self.rounds
            for opinion in discussion_round.opinions
        )

    def _topics(self) -> list[dict[str, Any]]:
        return [
            {
                "id": str(getattr(topic, "id", "topic")),
                "title": str(getattr(topic, "title", "")),
                "subpoints": list(getattr(topic, "subpoints", []) or []),
            }
            for topic in (getattr(self.objective, "core_topics", []) or [])
        ]

    def _opinion_output(self) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for discussion_round in self.rounds:
            for opinion in discussion_round.opinions:
                entry = output.setdefault(
                    opinion.persona_id,
                    {
                        "mind_change_count": 0,
                        "material_update_count": 0,
                        "ranking_change_count": 0,
                        "assumption_change_count": 0,
                        "confidence_change_count": 0,
                        "rounds": [],
                    },
                )
                entry["mind_change_count"] += int(opinion.changed_mind)
                entry["material_update_count"] += int(opinion.material_update)
                entry["ranking_change_count"] += int(opinion.ranking_changed)
                entry["assumption_change_count"] += int(opinion.assumptions_changed)
                entry["confidence_change_count"] += int(opinion.confidence_changed)
                entry["rounds"].append(
                    {
                        "round": opinion.round_number,
                        "position": opinion.position,
                        "architecture": opinion.architecture,
                        **(
                            {"components": opinion.components}
                            if opinion.components
                            else {}
                        ),
                        **(
                            {"considered_alternatives": opinion.considered_alternatives}
                            if opinion.considered_alternatives
                            else {}
                        ),
                        **(
                            {"switch_test": opinion.switch_test}
                            if opinion.switch_test
                            else {}
                        ),
                        **(
                            {"switch_assessment": _jsonable(opinion.switch_assessment)}
                            if opinion.switch_assessment
                            else {}
                        ),
                        **(
                            {"strongest_alternative": opinion.strongest_alternative}
                            if opinion.strongest_alternative
                            else {}
                        ),
                        **(
                            {"criteria_scores": opinion.criteria_scores}
                            if opinion.criteria_scores
                            else {}
                        ),
                        **(
                            {"decisive_criterion": opinion.decisive_criterion}
                            if opinion.decisive_criterion
                            else {}
                        ),
                        **(
                            {"update_decision": opinion.update_decision}
                            if opinion.update_decision
                            else {}
                        ),
                        **(
                            {"update_reason": opinion.update_reason}
                            if opinion.update_reason
                            else {}
                        ),
                        "opinion": opinion.opinion,
                        "candidate_assessments": _jsonable(
                            opinion.candidate_assessments
                        ),
                        "persona_weights": dict(opinion.persona_weights),
                        **(
                            {"challenger_id": opinion.challenger_id}
                            if opinion.challenger_id
                            else {}
                        ),
                        **(
                            {"challenged_claim": opinion.challenged_claim}
                            if opinion.challenged_claim
                            else {}
                        ),
                        "decision_critical": opinion.decision_critical,
                        **(
                            {"critical_uncertainty": opinion.critical_uncertainty}
                            if opinion.critical_uncertainty
                            else {}
                        ),
                        "ranked_candidate": opinion.ranked_candidate,
                        "previous_ranked_candidate": opinion.previous_ranked_candidate,
                        "ranking_margin": opinion.ranking_margin,
                        "ranking_changed": opinion.ranking_changed,
                        "near_tie": opinion.near_tie,
                        "assumptions_changed": opinion.assumptions_changed,
                        "confidence_changed": opinion.confidence_changed,
                        "material_update": opinion.material_update,
                        "confidence": opinion.confidence,
                        "changed_mind": opinion.changed_mind,
                        **(
                            {"change_reason": opinion.change_reason}
                            if opinion.changed_mind and opinion.change_reason
                            else {}
                        ),
                        **(
                            {"status": opinion.status} if opinion.status != "ok" else {}
                        ),
                    }
                )
        return output

    def _round_output(self) -> list[dict[str, Any]]:
        return [discussion_round.to_dict() for discussion_round in self.rounds]

    def _progress_output(self) -> dict[str, int]:
        total_rounds = self.configuration.get("num_rounds", len(self.rounds))
        try:
            total_rounds = int(total_rounds)
        except (TypeError, ValueError):
            total_rounds = len(self.rounds)
        return {
            "current_round": len(self.rounds),
            "completed_rounds": len(self.rounds),
            "total_rounds": total_rounds,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "discussion_id": self.discussion_id,
            "status": self.status,
            "progress": self._progress_output(),
            "preparation": _jsonable(self.preparation),
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "failure": _jsonable(self.failure),
            "warnings": _jsonable(self.warnings),
            "objective": _jsonable(self.objective),
            "core_topics": self._topics(),
            "participants": list(self.participants),
            "graph": _jsonable(self.graph),
            "configuration": _jsonable(self.configuration),
            "rounds": self._round_output(),
            "opinions": self._opinion_output(),
            "decision_summary": _jsonable(self.decision_summary),
            "logs": dict(self.logs),
            "conclusion": self.conclusion,
        }

    def logs_payload(self) -> dict[str, Any]:
        research = self.research
        return {
            "schema_version": SCHEMA_VERSION,
            "discussion_id": self.discussion_id,
            "status": self.status,
            "progress": self._progress_output(),
            "preparation": _jsonable(self.preparation),
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "failure": _jsonable(self.failure),
            "warnings": _jsonable(self.warnings),
            "objective": _jsonable(self.objective),
            "participants": list(self.participants),
            "graph": _jsonable(self.graph),
            "configuration": _jsonable(self.configuration),
            "decision_summary": _jsonable(self.decision_summary),
            "conclusion": self.conclusion,
            "research": {
                "plan": _jsonable(research.plan) if research else None,
                "evidence_pool": _jsonable(research.evidence_pool) if research else [],
                "dossier": _jsonable(research.dossier) if research else None,
                "query_history": list(research.query_history) if research else [],
                "rejected_evidence": _jsonable(research.rejected_evidence)
                if research
                else [],
            },
            "rounds": self._round_output(),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, indent=indent, default=str
        )
