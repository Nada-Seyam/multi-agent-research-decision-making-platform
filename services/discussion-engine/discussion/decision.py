from __future__ import annotations

from collections import Counter
import re
from typing import Any, Iterable, Mapping

from .candidate_identity import build_candidate_aliases, canonical_candidate
from .models import CandidateAssessment, DecisionSummary, OpinionResult, SwitchAssessment
from .structured_output import parse_marker_line


_DIMENSION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "capability": ("reasoning", "quality", "coding", "capability", "benchmark"),
    "latency_predictability": ("latency", "interactive", "predictable", "response time"),
    "cost_efficiency": ("cost", "compute", "flop", "throughput", "efficiency", "memory"),
    "long_context": ("long-context", "long context", "retrieval", "context"),
    "reliability": ("reliability", "robust", "failure", "recover", "stability"),
    "operational_complexity": ("complexity", "debug", "serving", "deployment", "operate"),
    "safety_auditability": ("safety", "audit", "governance", "control", "observable"),
    "user_value": ("user", "product", "access", "adoption", "workflow"),
    "flexibility": ("flexibility", "adaptive", "dynamic", "future"),
    "training_stability": ("training", "reward", "rl", "optimization", "stability"),
}

_ABSTENTION_STATUSES = {
    "error",
    "provider_limit",
    "no_opinion",
    "missing_recommendation",
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "item"


def criteria_registry(objective: Any) -> tuple[tuple[str, str], ...]:
    values = list(getattr(objective, "decision_criteria", []) or [])
    if not values:
        values = [
            "quality and capability",
            "latency and resource efficiency",
            "reliability and operational complexity",
        ]
    used: Counter[str] = Counter()
    result: list[tuple[str, str]] = []
    for value in values:
        text = str(value).strip()
        base = _slug(text)[:64]
        used[base] += 1
        criterion_id = base if used[base] == 1 else f"{base}_{used[base]}"
        result.append((criterion_id, text))
    return tuple(result)


def effective_criterion_weights(
    persona: Mapping[str, Any], objective: Any
) -> dict[str, float]:
    """Map a persona's semantic weights onto this objective's criteria."""
    dimension_weights = persona.get("priority_weights") or {}
    if not isinstance(dimension_weights, Mapping):
        dimension_weights = {}
    raw: dict[str, float] = {}
    for criterion_id, criterion_text in criteria_registry(objective):
        lowered = criterion_text.lower()
        matched = [
            float(dimension_weights.get(dimension, 1.0))
            for dimension, keywords in _DIMENSION_KEYWORDS.items()
            if any(keyword in lowered for keyword in keywords)
        ]
        raw[criterion_id] = max(matched, default=1.0)
    total = sum(raw.values()) or 1.0
    return {key: round(value / total, 6) for key, value in raw.items()}


def _parse_refs(value: str) -> tuple[int, ...]:
    return tuple(sorted({int(item) for item in re.findall(r"\d+", value)}))


def _split_list_field(value: str) -> tuple[str, ...]:
    if value.strip().lower() in {"", "none", "n/a", "not_applicable"}:
        return ()
    return tuple(item.strip() for item in re.split(r"\s*;\s*", value) if item.strip())


def parse_candidate_assessments(
    raw: str,
    criterion_weights: Mapping[str, float],
) -> tuple[CandidateAssessment, ...]:
    """Parse persona-proposed alternatives and compute weighted utilities."""
    assessments: dict[str, CandidateAssessment] = {}
    for line in raw.splitlines():
        marker = parse_marker_line(line, {"CANDIDATE_EVAL"})
        if marker is None:
            continue
        _, marker_value = marker
        parts = [part.strip() for part in marker_value.split("|")]
        candidate_id = _slug(parts[0]) if parts else ""
        if not candidate_id:
            continue
        fields: dict[str, str] = {}
        for part in parts[1:]:
            if "=" in part:
                key, value = part.split("=", 1)
                fields[_slug(key)] = value.strip()
        score_text = fields.get("scores", "")
        scores: dict[str, float] = {}
        for score_item in re.split(r"[,;]", score_text):
            score_match = re.fullmatch(
                r"\s*([a-zA-Z0-9_\-]+)\s*[:=]\s*"
                r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*",
                score_item,
            )
            if score_match is None:
                continue
            criterion_id = _slug(score_match.group(1))
            score = float(score_match.group(2))
            if criterion_id in criterion_weights and 0.0 <= score <= 5.0:
                scores[criterion_id] = score
        expected_criteria = set(criterion_weights)
        utility = None
        if expected_criteria and expected_criteria.issubset(scores):
            utility = sum(
                scores[key] * weight for key, weight in criterion_weights.items()
            )
        combination = _slug(fields.get("combination", "not_applicable"))
        gaps = _split_list_field(fields.get("gaps", ""))
        assessments[candidate_id] = CandidateAssessment(
            candidate_id=candidate_id,
            criterion_scores=scores,
            weighted_utility=(round(max(0.0, utility), 3) if utility is not None else None),
            evidence_refs=_parse_refs(fields.get("evidence", "")),
            counter_evidence_refs=_parse_refs(fields.get("counter", "")),
            assumptions=_split_list_field(fields.get("assumptions", "")),
            evidence_gaps=tuple(dict.fromkeys(gaps)),
            feasibility=_slug(fields.get("feasibility", "unknown")),
            combination_evidence=combination,
        )
    return tuple(assessments.values())


def parse_switch_assessment(
    value: str | None,
    *,
    current_candidate: str | None,
    known_candidates: Iterable[str] = (),
    candidate_aliases: Mapping[str, str] | None = None,
) -> SwitchAssessment | None:
    """Parse and validate a typed architecture counterfactual.

    The preferred format is ``current=... | alternative=... | condition=... |
    effect=alternative_wins | status=...``. The explicit effect makes the
    intended direction machine-checkable instead of implicit. The old
    semicolon format remains readable so existing saved prompts and callers do
    not break.
    """
    raw = (value or "").strip()
    if not raw or raw.upper() == "NONE":
        return None

    fields: dict[str, str] = {}
    legacy = not ("|" in raw and "=" in raw)
    if legacy:
        parts = [item.strip() for item in raw.split(";", 2)]
        fields = {
            "current": current_candidate or "",
            "alternative": parts[0] if parts else "",
            "condition": parts[1] if len(parts) > 1 else "",
            "effect": "alternative_wins",
            "status": parts[2] if len(parts) > 2 else "",
        }
    else:
        for part in raw.split("|"):
            if "=" in part:
                key, item = part.split("=", 1)
                fields[_slug(key)] = item.strip()

    candidate_values = (
        *known_candidates,
        current_candidate,
        fields.get("current"),
        fields.get("alternative"),
    )
    aliases = build_candidate_aliases(
        candidate_values,
        preferred=(*known_candidates, current_candidate),
    )
    aliases.update(candidate_aliases or {})
    current = canonical_candidate(fields.get("current"), aliases)
    alternative = canonical_candidate(fields.get("alternative"), aliases)
    expected_current = canonical_candidate(current_candidate, aliases)
    condition = fields.get("condition", "").strip() or None
    condition_effect = _slug(fields.get("effect", "")) if fields.get("effect") else None
    status_text = re.sub(r"[^a-z]+", "_", fields.get("status", "").lower()).strip("_")
    status = next(
        (item for item in ("satisfied", "contradicted", "unresolved") if item in status_text),
        None,
    )
    errors: list[str] = []
    if not current:
        errors.append("current candidate is missing")
    elif expected_current and current != expected_current:
        errors.append("current candidate does not match the validated position")
    if not alternative:
        errors.append("alternative candidate is missing")
    if current and alternative and current == alternative:
        errors.append("alternative must differ from current candidate")
    if not condition:
        errors.append("switch condition is missing")
    elif len(condition.split()) < 2:
        errors.append("switch condition is not concrete")
    if condition_effect != "alternative_wins":
        errors.append("condition effect must be alternative_wins")
    if status is None:
        errors.append("status must be satisfied, contradicted, or unresolved")

    return SwitchAssessment(
        current_candidate=current,
        alternative_candidate=alternative,
        condition=condition,
        condition_effect=condition_effect,
        status=status,
        valid=not errors,
        validation_errors=tuple(errors),
    )


def select_challenger(
    persona_id: str, previous: Mapping[str, OpinionResult]
) -> str | None:
    """Select the previous-round opinion with the greatest decision distance."""
    own = previous.get(persona_id)
    if own is None:
        return None
    others = [item for key, item in previous.items() if key != persona_id and item.opinion]
    if not others:
        return None

    own_utilities = {
        item.candidate_id: item.weighted_utility
        for item in own.candidate_assessments
        if item.weighted_utility is not None
    }

    def distance(other: OpinionResult) -> tuple[float, float, float]:
        architecture_distance = float(bool(other.architecture and other.architecture != own.architecture))
        other_utilities = {
            item.candidate_id: item.weighted_utility
            for item in other.candidate_assessments
            if item.weighted_utility is not None
        }
        shared = set(own_utilities) & set(other_utilities)
        utility_distance = sum(
            abs(float(own_utilities[key]) - float(other_utilities[key])) for key in shared
        )
        criterion_distance = float(
            bool(other.decisive_criterion and other.decisive_criterion != own.decisive_criterion)
        )
        return architecture_distance, utility_distance, criterion_distance

    selected = max(others, key=distance)
    if distance(selected) == (0.0, 0.0, 0.0):
        return None
    return selected.persona_id


def synthesize_decision(rounds: Iterable[Any]) -> DecisionSummary:
    round_list = list(rounds)
    final = list(round_list[-1].opinions) if round_list else []
    usable = [
        item
        for item in final
        if item.architecture
        and item.is_relevant
        and item.status not in _ABSTENTION_STATUSES
    ]
    abstentions = {
        item.persona_id: item.error or item.status or "unavailable"
        for item in final
        if item not in usable
    }
    if not usable:
        return DecisionSummary(
            None,
            "insufficient_evidence",
            abstaining_personas=abstentions,
        )

    candidate_values = [
        candidate
        for item in usable
        for candidate in (
            item.architecture,
            item.ranked_candidate,
            *(assessment.candidate_id for assessment in item.candidate_assessments),
        )
        if candidate
    ]
    aliases = build_candidate_aliases(candidate_values)
    architecture_by_persona = {
        item.persona_id: canonical_candidate(item.architecture, aliases)
        for item in usable
    }
    votes = Counter(
        candidate
        for candidate in architecture_by_persona.values()
        if candidate is not None
    )
    utility_values: dict[str, list[float]] = {}
    for opinion in usable:
        for assessment in opinion.candidate_assessments:
            if assessment.weighted_utility is not None:
                candidate_id = canonical_candidate(assessment.candidate_id, aliases)
                if candidate_id is None:
                    continue
                utility_values.setdefault(candidate_id, []).append(
                    assessment.weighted_utility
                )
    aggregates = {
        candidate: round(sum(values) / len(values), 3)
        for candidate, values in utility_values.items()
        if values
    }
    max_votes = max(votes.values())
    vote_leaders = [candidate for candidate, count in votes.items() if count == max_votes]
    preferred = max(
        vote_leaders,
        key=lambda candidate: aggregates.get(candidate, float("-inf")),
    )
    ranked_votes = sorted(
        votes,
        key=lambda candidate: (
            votes[candidate],
            aggregates.get(candidate, float("-inf")),
        ),
        reverse=True,
    )
    alternative = next(
        (candidate for candidate in ranked_votes if candidate != preferred),
        None,
    )

    supporters = tuple(
        item.persona_id
        for item in usable
        if architecture_by_persona[item.persona_id] == preferred
    )
    dissent = {
        item.persona_id: str(architecture_by_persona[item.persona_id])
        for item in usable
        if architecture_by_persona[item.persona_id] != preferred
    }
    consensus_status = (
        "consensus"
        if len(votes) == 1
        else "plurality"
        if votes[preferred] > len(usable) / 2
        else "no_consensus"
    )
    unresolved = tuple(
        dict.fromkeys(
            item.critical_uncertainty or item.switch_test or ""
            for item in usable
            if item.decision_critical
            and (
                bool(item.critical_uncertainty)
                or (
                    item.switch_assessment is not None
                    and item.switch_assessment.status == "unresolved"
                )
                or re.search(
                    r"\b(unresolved|unknown|unclear|open)\b",
                    (item.switch_test or "").lower(),
                )
            )
        )
    )
    unresolved = tuple(value for value in unresolved if value)
    switch_conditions = {
        item.persona_id: item.switch_test
        for item in usable
        if item.switch_test
    }
    return DecisionSummary(
        preferred_candidate=preferred,
        consensus_status=consensus_status,
        support_personas=supporters,
        dissenting_positions=dissent,
        strongest_alternative=alternative,
        aggregate_utilities=aggregates,
        unresolved_uncertainties=unresolved,
        switch_conditions=switch_conditions,
        abstaining_personas=abstentions,
    )
