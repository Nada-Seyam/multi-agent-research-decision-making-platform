"""Execute and commit one synchronous discussion round."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

from . import week2_adapter
from .candidate_identity import (
    build_candidate_aliases,
    canonical_candidate,
    candidate_key,
    recommendation_refinement,
)
from .config import DiscussionSettings
from .decision import effective_criterion_weights, select_challenger
from .models import CandidateAssessment, DiscussionMessage, DiscussionRound, OpinionResult
from .opinion_parser import architecture_key, parse_opinion_response
from .prompting import build_discussion_context


@dataclass(frozen=True)
class RoundExecution:
    """The completed round and the next global message sequence."""

    discussion_round: DiscussionRound
    next_message_sequence: int


@dataclass(frozen=True)
class _PersonaRoundTask:
    """Immutable inputs for one independent persona call in a round."""

    persona_id: str
    previous_opinion: OpinionResult | None
    persona_weights: dict[str, float]
    challenger_id: str | None
    memory_context: str
    recipients: tuple[str, ...]


def _opinion_candidates(opinion: OpinionResult) -> tuple[str, ...]:
    switch = opinion.switch_assessment
    values = (
        opinion.architecture,
        opinion.ranked_candidate,
        opinion.previous_ranked_candidate,
        opinion.strongest_alternative,
        switch.current_candidate if switch else None,
        switch.alternative_candidate if switch else None,
        *(item.candidate_id for item in opinion.candidate_assessments),
    )
    return tuple(value for value in values if value)


def _canonicalize_opinion(
    opinion: OpinionResult,
    aliases: dict[str, str],
) -> OpinionResult:
    architecture = canonical_candidate(opinion.architecture, aliases)
    position_key = candidate_key(opinion.position)
    position = (
        architecture
        if position_key and canonical_candidate(position_key, aliases) == architecture
        else opinion.position
    )
    assessments: dict[str, CandidateAssessment] = {}
    for item in opinion.candidate_assessments:
        candidate_id = canonical_candidate(item.candidate_id, aliases) or item.candidate_id
        normalized = replace(item, candidate_id=candidate_id)
        current = assessments.get(candidate_id)
        if current is None or (
            len(normalized.criterion_scores),
            normalized.weighted_utility is not None,
        ) > (
            len(current.criterion_scores),
            current.weighted_utility is not None,
        ):
            assessments[candidate_id] = normalized

    switch = opinion.switch_assessment
    if switch is not None:
        switch = replace(
            switch,
            current_candidate=canonical_candidate(switch.current_candidate, aliases),
            alternative_candidate=canonical_candidate(
                switch.alternative_candidate, aliases
            ),
        )
    return replace(
        opinion,
        position=position,
        architecture=architecture,
        candidate_assessments=tuple(assessments.values()),
        ranked_candidate=canonical_candidate(opinion.ranked_candidate, aliases),
        previous_ranked_candidate=canonical_candidate(
            opinion.previous_ranked_candidate, aliases
        ),
        strongest_alternative=canonical_candidate(
            opinion.strongest_alternative, aliases
        ),
        switch_assessment=switch,
    )


def canonicalize_round_opinions(
    opinions: list[OpinionResult],
    proposed_alternatives: dict[str, str],
) -> tuple[OpinionResult, ...]:
    """Resolve harmless naming variants before a round is published."""
    established = tuple(dict.fromkeys(proposed_alternatives.values()))
    aliases = build_candidate_aliases(
        (
            *established,
            *(candidate for opinion in opinions for candidate in _opinion_candidates(opinion)),
        ),
        preferred=established,
    )
    for opinion in opinions:
        position_key = candidate_key(opinion.position)
        refinement = recommendation_refinement(position_key, opinion.opinion)
        if not position_key or not refinement:
            continue
        refined_canonical = canonical_candidate(refinement, aliases)
        architecture_canonical = canonical_candidate(opinion.architecture, aliases)
        if refined_canonical and refined_canonical == architecture_canonical:
            aliases[position_key] = architecture_canonical
    for alias, canonical in aliases.items():
        proposed_alternatives[alias] = canonical
    return tuple(_canonicalize_opinion(opinion, aliases) for opinion in opinions)


def _error_opinion(
    *,
    persona_id: str,
    round_number: int,
    previous: OpinionResult | None,
    persona_weights: dict[str, float],
    challenger_id: str | None,
    error: Exception,
) -> OpinionResult:
    return OpinionResult(
        persona_id=persona_id,
        round_number=round_number,
        opinion=None,
        position=previous.position if previous else None,
        architecture=previous.architecture if previous else None,
        components=previous.components if previous else None,
        considered_alternatives=(
            previous.considered_alternatives if previous else None
        ),
        switch_test=previous.switch_test if previous else None,
        strongest_alternative=(previous.strongest_alternative if previous else None),
        criteria_scores=previous.criteria_scores if previous else None,
        decisive_criterion=previous.decisive_criterion if previous else None,
        update_decision=previous.update_decision if previous else None,
        update_reason=previous.update_reason if previous else None,
        confidence=previous.confidence if previous else 0.5,
        is_relevant=False,
        status="error",
        error=f"{type(error).__name__}: {error}",
        persona_weights=persona_weights,
        challenger_id=challenger_id,
    )


def _run_persona_task(
    task: _PersonaRoundTask,
    *,
    round_number: int,
    objective: Any,
    dossier: Any,
    utility_tolerance: float,
    known_candidates: tuple[str, ...],
) -> OpinionResult:
    """Generate and parse one response without mutating shared round state."""
    try:
        response = week2_adapter.run_for_persona(
            task.persona_id,
            objective,
            dossier,
            memory_context=task.memory_context,
        )
        if not isinstance(response, dict):
            response = {"opinion": response}
        return parse_opinion_response(
            task.persona_id,
            round_number,
            response,
            task.previous_opinion,
            persona_weights=task.persona_weights,
            challenger_id=task.challenger_id,
            utility_tolerance=utility_tolerance,
            known_candidates=known_candidates,
        )
    except Exception as exc:
        return _error_opinion(
            persona_id=task.persona_id,
            round_number=round_number,
            previous=task.previous_opinion,
            persona_weights=task.persona_weights,
            challenger_id=task.challenger_id,
            error=exc,
        )


def execute_round(
    *,
    round_number: int,
    objective: Any,
    settings: DiscussionSettings,
    participants: list[str],
    personas: dict[str, dict[str, Any]],
    graph: Any,
    dossier: Any,
    history: dict[str, list[OpinionResult]],
    proposed_alternatives: dict[str, str],
    message_sequence: int,
) -> RoundExecution:
    """Collect every persona response against the previous round barrier."""
    previous = {
        persona_id: opinions[-1] for persona_id, opinions in history.items() if opinions
    }
    known_candidates = tuple(dict.fromkeys(proposed_alternatives.values()))
    tasks: list[_PersonaRoundTask] = []
    for persona_id in participants:
        challenger_id = (
            select_challenger(persona_id, previous)
            if settings.dynamic_challenger_routing
            else None
        )
        neighbor_opinions = {
            neighbor_id: previous[neighbor_id]
            for neighbor_id in graph.get_neighbors(persona_id)
            if neighbor_id in previous and previous[neighbor_id].opinion
        }
        if challenger_id and challenger_id in previous:
            neighbor_opinions[challenger_id] = previous[challenger_id]

        persona = personas.get(persona_id, {})
        persona_weights = effective_criterion_weights(persona, objective)
        recipients = tuple(graph.get_neighbors(persona_id))
        context = build_discussion_context(
            round_number,
            persona,
            previous.get(persona_id),
            neighbor_opinions,
            objective=objective,
            challenger_id=challenger_id,
            proposed_alternatives=known_candidates,
        )
        tasks.append(
            _PersonaRoundTask(
                persona_id=persona_id,
                previous_opinion=previous.get(persona_id),
                persona_weights=persona_weights,
                challenger_id=challenger_id,
                memory_context=context,
                recipients=recipients,
            )
        )

    def run_task(task: _PersonaRoundTask) -> OpinionResult:
        return _run_persona_task(
            task,
            round_number=round_number,
            objective=objective,
            dossier=dossier,
            utility_tolerance=settings.utility_consistency_tolerance,
            known_candidates=known_candidates,
        )

    worker_count = min(settings.max_parallel_agents, len(tasks))
    if worker_count == 1:
        opinions = [run_task(task) for task in tasks]
    else:
        # Every prompt observes only the previous-round snapshot, so these
        # network-bound model calls are independent. executor.map retains the
        # participant order even when responses finish out of order.
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix=f"discussion-round-{round_number}",
        ) as executor:
            opinions = list(executor.map(run_task, tasks))

    messages: list[DiscussionMessage] = []
    for task, opinion in zip(tasks, opinions, strict=True):
        if opinion.opinion:
            message_sequence += 1
            messages.append(
                DiscussionMessage(
                    round_number=round_number,
                    sequence=message_sequence,
                    sender=task.persona_id,
                    recipients=task.recipients,
                    content=opinion.opinion,
                )
            )

    canonical_opinions = canonicalize_round_opinions(
        opinions, proposed_alternatives
    )
    return RoundExecution(
        discussion_round=DiscussionRound(
            round_number=round_number,
            opinions=canonical_opinions,
            messages=tuple(messages),
        ),
        next_message_sequence=message_sequence,
    )


def commit_round(
    discussion_round: DiscussionRound,
    history: dict[str, list[OpinionResult]],
    proposed_alternatives: dict[str, str],
) -> None:
    """Commit a completed round to state used by subsequent rounds."""
    for opinion in discussion_round.opinions:
        history[opinion.persona_id].append(opinion)
        for assessment in opinion.candidate_assessments:
            proposed_alternatives.setdefault(
                assessment.candidate_id,
                assessment.candidate_id,
            )
        if opinion.architecture:
            proposed_alternatives.setdefault(
                opinion.architecture,
                opinion.architecture,
            )
        if opinion.strongest_alternative:
            alternative_id = architecture_key(opinion.strongest_alternative)
            if alternative_id:
                proposed_alternatives.setdefault(alternative_id, alternative_id)
