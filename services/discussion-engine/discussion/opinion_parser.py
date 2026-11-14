"""Parse and validate one persona response into an opinion record."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from .candidate_identity import (
    build_candidate_aliases,
    canonical_candidate,
    candidate_key,
    explicit_recommendation,
    recommendation_refinement,
)
from .decision import parse_candidate_assessments, parse_switch_assessment
from .models import OpinionResult
from .structured_output import parse_marker_line

_MARKER_NAMES = {
    "ARCHITECTURE",
    "COMPONENTS",
    "CONSIDERED",
    "SWITCH_TEST",
    "STRONGEST_ALTERNATIVE",
    "CRITERIA_SCORES",
    "DECISIVE_CRITERION",
    "UPDATE_DECISION",
    "UPDATE_REASON",
    "POSITION",
    "CONFIDENCE",
    "CHANGE_OF_MIND",
    "CHANGE_REASON",
    "CANDIDATE_EVAL",
    "DECISION_CRITICAL",
    "CRITICAL_UNCERTAINTY",
    "CHALLENGED_CLAIM",
}
def _markers(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in raw.splitlines():
        marker = parse_marker_line(line, _MARKER_NAMES)
        if marker:
            name, value = marker
            values[name] = value
    return values


def _opinion_body(raw: str) -> str:
    lines: list[str] = []
    for line in raw.splitlines():
        if parse_marker_line(line, _MARKER_NAMES) is None:
            lines.append(line)
    return "\n".join(lines).strip()


def architecture_key(position: str | None) -> str:
    """Create a stable id from a persona-proposed solution name."""
    return candidate_key(position) or ""


def _confidence(
    value: str | None,
    previous: OpinionResult | None,
    *,
    coverage_ratio: float = 1.0,
    decision_critical: bool = False,
    switch_status: str | None = None,
    matrix_complete: bool = True,
) -> float:
    if value is None:
        result = previous.confidence if previous else 0.5
    else:
        try:
            result = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            result = previous.confidence if previous else 0.5
    if previous is not None:
        result = max(previous.confidence - 0.2, min(previous.confidence + 0.2, result))
        result = min(result, previous.confidence)
    evidence_cap = 0.45 + 0.45 * max(0.0, min(1.0, coverage_ratio))
    result = min(result, evidence_cap)
    if not matrix_complete:
        result = min(result, 0.55)
    if decision_critical and switch_status == "unresolved":
        unresolved_cap = 0.65
        if previous is not None:
            unresolved_cap = min(
                unresolved_cap,
                max(0.0, previous.confidence - 0.05),
            )
        result = min(result, unresolved_cap)
    return round(result, 3)


def _update_decision(
    value: str | None,
    previous: OpinionResult | None,
) -> str | None:
    text = re.sub(r"[^a-z]+", " ", (value or "").lower()).strip()
    aliases = {
        "switch": "switch",
        "changed": "switch",
        "change": "switch",
        "retain": "retain",
        "keep": "retain",
        "unchanged": "retain",
        "narrow": "narrow",
        "lower confidence": "lower_confidence",
        "lowerconfidence": "lower_confidence",
        "unresolved": "lower_confidence",
        "uncertain": "lower_confidence",
        "unclear": "lower_confidence",
    }
    if text in {"", "none"}:
        return previous.update_decision if previous else None
    return aliases.get(text)


def _switch_test_status(value: str | None) -> str | None:
    text = re.sub(r"[^a-z]+", " ", (value or "").lower()).strip()
    if not text:
        return None
    if re.search(r"\b(unresolved|unknown|unclear|open)\b", text):
        return "unresolved"
    if re.search(
        r"\b(unsatisfied|not satisfied|not met|not confirmed|contradicted|failed|false)\b",
        text,
    ):
        return "contradicted"
    if re.search(r"\b(satisfied|met|confirmed|true)\b", text):
        return "satisfied"
    return None


def _optional_marker(
    markers: dict[str, str],
    name: str,
    previous_value: str | None = None,
) -> str | None:
    value = markers.get(name, "").strip() or previous_value
    return None if value and value.upper() == "NONE" else value


def parse_opinion_response(
    persona_id: str,
    round_number: int,
    response: dict[str, Any],
    previous: OpinionResult | None = None,
    *,
    persona_weights: dict[str, float] | None = None,
    challenger_id: str | None = None,
    utility_tolerance: float = 0.25,
    known_candidates: tuple[str, ...] = (),
) -> OpinionResult:
    raw_value = response.get("opinion")
    raw = "" if raw_value is None else str(raw_value).strip()
    markers = _markers(raw)
    body = _opinion_body(raw) or markers.get("POSITION") or None

    structured_position = (
        markers.get("ARCHITECTURE", "").strip()
        or markers.get("POSITION", "").strip()
        or None
    )
    raw_explicit_architecture = architecture_key(structured_position) or None
    prose_recommendation = explicit_recommendation(body)
    if raw_explicit_architecture is None:
        raw_explicit_architecture = prose_recommendation
    refined_architecture = recommendation_refinement(
        raw_explicit_architecture, body
    )
    explicit_architecture = refined_architecture or raw_explicit_architecture
    explicit_position = structured_position or explicit_architecture
    weights = dict(persona_weights or {})
    assessments = parse_candidate_assessments(raw, weights)
    established_candidates = (
        *known_candidates,
        *((previous.architecture, previous.ranked_candidate) if previous else ()),
    )
    aliases = build_candidate_aliases(
        (
            *established_candidates,
            explicit_architecture,
            *(item.candidate_id for item in assessments),
        ),
        preferred=established_candidates,
    )
    if raw_explicit_architecture and refined_architecture:
        aliases[raw_explicit_architecture] = (
            canonical_candidate(refined_architecture, aliases)
            or refined_architecture
        )
    explicit_architecture = canonical_candidate(explicit_architecture, aliases)
    assessments = tuple(
        replace(
            item,
            candidate_id=canonical_candidate(item.candidate_id, aliases)
            or item.candidate_id,
        )
        for item in assessments
    )
    expected_criterion_ids = set(weights)
    matrix_complete = bool(len(assessments) >= 2) and all(
        not expected_criterion_ids
        or expected_criterion_ids <= set(assessment.criterion_scores)
        for assessment in assessments
    )
    status = str(response.get("status") or ("ok" if raw else "no_opinion"))

    ranked_candidate: str | None = None
    ranking_margin: float | None = None
    if matrix_complete and assessments:
        ranked = sorted(
            assessments,
            key=lambda item: (
                item.weighted_utility if item.weighted_utility is not None else -1.0
            ),
            reverse=True,
        )
        top = ranked[0]
        ranked_candidate = top.candidate_id
        if len(ranked) > 1:
            ranking_margin = round(
                float(top.weighted_utility or 0.0)
                - float(ranked[1].weighted_utility or 0.0),
                3,
            )
        chosen = next(
            (
                item
                for item in assessments
                if item.candidate_id == explicit_architecture
            ),
            None,
        )
        if explicit_architecture is None:
            # A score winner is not an agent recommendation. Leave the
            # recommendation missing (or retain the prior round below) rather
            # than manufacturing an analytics stance from advisory scores.
            status = "missing_recommendation"
        elif (
            chosen is not None
            and top.candidate_id != explicit_architecture
            and top.weighted_utility is not None
            and chosen.weighted_utility is not None
            and top.weighted_utility - chosen.weighted_utility > utility_tolerance
            and status == "ok"
        ):
            # The matrix is advisory. Preserve the agent's declared
            # recommendation for analytics and expose the score disagreement
            # only as consistency metadata.
            status = "recommendation_utility_conflict"
    elif status == "ok":
        status = "incomplete_candidate_evaluation"

    position = explicit_position or (previous.position if previous else None)
    architecture = (
        explicit_architecture
        or (previous.architecture if previous else None)
        or None
    )
    components = markers.get("COMPONENTS", "").strip() or (
        previous.components if previous else None
    )
    considered_alternatives = _optional_marker(
        markers,
        "CONSIDERED",
        previous.considered_alternatives if previous else None,
    )
    switch_test = _optional_marker(
        markers,
        "SWITCH_TEST",
        previous.switch_test if previous else None,
    )
    strongest_alternative = _optional_marker(
        markers,
        "STRONGEST_ALTERNATIVE",
        previous.strongest_alternative if previous else None,
    )
    criteria_scores = _optional_marker(
        markers,
        "CRITERIA_SCORES",
        previous.criteria_scores if previous else None,
    )
    decisive_criterion = _optional_marker(
        markers,
        "DECISIVE_CRITERION",
        previous.decisive_criterion if previous else None,
    )
    update_decision = _update_decision(markers.get("UPDATE_DECISION"), previous)
    update_reason = _optional_marker(
        markers,
        "UPDATE_REASON",
        previous.update_reason if previous else None,
    )
    decision_critical = re.sub(
        r"[^a-z]+",
        "",
        markers.get("DECISION_CRITICAL", "").lower(),
    ) in {"yes", "true", "critical"}
    critical_uncertainty = _optional_marker(markers, "CRITICAL_UNCERTAINTY")
    challenged_claim = _optional_marker(markers, "CHALLENGED_CLAIM")
    changed = bool(
        previous
        and architecture
        and previous.architecture
        and architecture != previous.architecture
    )
    reason = markers.get("CHANGE_REASON", "").strip() or None
    baseline_architecture = previous.architecture if previous else architecture
    switch_assessment = parse_switch_assessment(
        switch_test,
        current_candidate=baseline_architecture,
        known_candidates=(
            *known_candidates,
            architecture,
            *(item.candidate_id for item in assessments),
        ),
        candidate_aliases=aliases,
    )
    switch_status = (
        switch_assessment.status
        if switch_assessment is not None
        else _switch_test_status(switch_test)
    )
    if switch_assessment is not None:
        strongest_alternative = (
            switch_assessment.alternative_candidate or strongest_alternative
        )
        if not switch_assessment.valid:
            status = "invalid_switch_test"
            decision_critical = True
    if strongest_alternative:
        alternative_aliases = build_candidate_aliases(
            (
                *known_candidates,
                architecture,
                *(item.candidate_id for item in assessments),
                strongest_alternative,
            ),
            preferred=(*known_candidates, architecture),
        )
        strongest_alternative = canonical_candidate(
            strongest_alternative, alternative_aliases
        )
    decision_critical = decision_critical or bool(
        switch_status == "unresolved" and strongest_alternative and decisive_criterion
    )
    if decision_critical and not critical_uncertainty:
        critical_uncertainty = switch_test
    if decision_critical and critical_uncertainty and switch_status == "contradicted":
        status = "inconsistent_uncertainty_state"

    if assessments:
        covered_count = sum(
            1
            for assessment in assessments
            if assessment.evidence_refs
            and assessment.counter_evidence_refs
            and not assessment.evidence_gaps
        )
        coverage_ratio = covered_count / len(assessments)
    else:
        coverage_ratio = 0.0
    confidence = _confidence(
        markers.get("CONFIDENCE"),
        previous,
        coverage_ratio=coverage_ratio,
        decision_critical=decision_critical,
        switch_status=switch_status,
        matrix_complete=matrix_complete,
    )
    if previous and update_decision == "lower_confidence":
        confidence = min(confidence, round(max(0.0, previous.confidence - 0.05), 3))

    valid_switch = bool(
        switch_assessment
        and switch_assessment.valid
        and switch_assessment.status == "satisfied"
        and switch_assessment.alternative_candidate == architecture
    )
    if previous and changed and (update_decision != "switch" or not valid_switch):
        # Never rewrite an agent's declared recommendation. Switch metadata is
        # validation/audit context; analytics must observe the recommendation
        # the agent actually made in this round.
        status = (
            "unresolved_switch_condition"
            if switch_status != "satisfied"
            else "conflicting_update_metadata"
        )
    elif (
        previous
        and switch_assessment is not None
        and switch_assessment.valid
        and switch_status == "satisfied"
        and (
            update_decision != "switch"
            or architecture != switch_assessment.alternative_candidate
        )
    ):
        status = "satisfied_switch_not_applied"
        decision_critical = True
    previous_ranked_candidate = (
        previous.ranked_candidate or previous.architecture if previous else None
    )
    ranking_changed = bool(
        previous
        and ranked_candidate
        and previous_ranked_candidate
        and ranked_candidate != previous_ranked_candidate
    )
    near_tie = bool(ranking_margin is not None and ranking_margin <= utility_tolerance)
    if ranking_changed and near_tie and update_decision == "retain":
        update_decision = "narrow"
        if status == "ok":
            status = "near_tie_ranking_update"

    def assumptions_for(items: tuple[Any, ...]) -> dict[str, tuple[str, ...]]:
        return {item.candidate_id: tuple(item.assumptions) for item in items}

    assumptions_changed = bool(
        previous
        and assumptions_for(assessments)
        != assumptions_for(previous.candidate_assessments)
    )
    confidence_changed = bool(previous and confidence != previous.confidence)
    material_update = bool(
        changed or ranking_changed or assumptions_changed or confidence_changed
    )

    return OpinionResult(
        persona_id=persona_id,
        round_number=round_number,
        opinion=body,
        position=position,
        architecture=architecture,
        components=components,
        considered_alternatives=considered_alternatives,
        switch_test=switch_test,
        switch_assessment=switch_assessment,
        strongest_alternative=strongest_alternative,
        criteria_scores=criteria_scores,
        decisive_criterion=decisive_criterion,
        update_decision=update_decision,
        update_reason=update_reason,
        confidence=confidence,
        changed_mind=changed,
        change_reason=reason if changed else None,
        is_relevant=bool(response.get("is_relevant", True)),
        status=status,
        error=response.get("error"),
        raw_opinion=raw,
        candidate_assessments=assessments,
        persona_weights=weights,
        challenger_id=challenger_id,
        challenged_claim=challenged_claim,
        decision_critical=decision_critical,
        critical_uncertainty=critical_uncertainty,
        ranked_candidate=ranked_candidate,
        previous_ranked_candidate=previous_ranked_candidate,
        ranking_margin=ranking_margin,
        ranking_changed=ranking_changed,
        near_tie=near_tie,
        assumptions_changed=assumptions_changed,
        confidence_changed=confidence_changed,
        material_update=material_update,
    )
