import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .models import (
    AgentRoundSnapshot,
    CandidateAssessment,
    Discussion,
    DiscussionGraph,
    Message,
    SwitchAssessment,
)
from .utils import clip_confidence, normalize_candidate_id


DEFAULT_UTILITY_CONSISTENCY_TOLERANCE = 0.25
UTILITY_RECOMPUTE_TOLERANCE = 0.02
VALID_UPDATE_DECISIONS = {"retain", "narrow", "lower_confidence", "switch"}


class DiscussionValidationError(ValueError):
    """Raised when a discussion file is missing data required to proceed at all."""


def load_discussion(source: Union[str, Path, Dict[str, Any]]) -> Discussion:
    """Load and validate a Week 3 discussion.

    Parameters
    ----------
    source: path to a discussion JSON file, or an already-parsed dict.

    Raises
    ------
    DiscussionValidationError
        If the discussion is missing structurally-required data (no
        rounds at all, or not a recognizable discussion object). Data
        problems that are recoverable (a missing confidence value, an
        unresolvable candidate, an empty message) are NOT raised —
        they are recorded in `Discussion.issues` and surfaced as
        `None` / documented sentinel values in the downstream metrics,
        per the spec's "fail clearly, don't silently mislead" rule.
    """
    data = _load_raw(source)
    issues: List[str] = []

    if not isinstance(data, dict):
        raise DiscussionValidationError("Discussion source is not a JSON object.")

    discussion_id = data.get("discussion_id")
    if not discussion_id:
        raise DiscussionValidationError("Discussion is missing required field 'discussion_id'.")

    rounds = data.get("rounds")
    if not isinstance(rounds, list) or len(rounds) == 0:
        raise DiscussionValidationError(
            f"Discussion '{discussion_id}' has no rounds — cannot compute any metrics."
        )

    candidate_aliases = _load_candidate_aliases(data, issues)
    utility_tolerance = _load_utility_tolerance(data, issues)

    participants = data.get("participants")
    declared_agents = [a for a in participants if isinstance(a, str) and a.strip()] if isinstance(participants, list) else []
    if not declared_agents:
        issues.append("No valid 'participants' list found; agent set will be inferred from opinion snapshots.")

    snapshots: Dict[str, List[AgentRoundSnapshot]] = {}
    messages: List[Message] = []
    rounds_present: List[int] = []
    seen_message_keys = set()

    seen_round_numbers = set()
    for round_entry in rounds:
        if not isinstance(round_entry, dict):
            issues.append("Skipped a malformed round entry (not an object).")
            continue

        round_number = round_entry.get("round")
        if not isinstance(round_number, int) or isinstance(round_number, bool):
            issues.append("Skipped a round with a missing or non-integer 'round' number.")
            continue
        if round_number in seen_round_numbers:
            raise DiscussionValidationError(
                f"Discussion '{discussion_id}' contains duplicate round number "
                f"{round_number}; refusing to merge ambiguous round histories."
            )
        seen_round_numbers.add(round_number)
        rounds_present.append(round_number)

        # --- opinions for this round ---
        opinions = round_entry.get("opinions")
        if not isinstance(opinions, list):
            issues.append(f"Round {round_number}: 'opinions' missing or malformed; no snapshots recorded for this round.")
            opinions = []

        seen_agents_this_round = set()
        for op in opinions:
            if not isinstance(op, dict):
                issues.append(f"Round {round_number}: skipped a malformed opinion entry.")
                continue

            agent_id = op.get("persona_id")
            if not agent_id or not isinstance(agent_id, str):
                issues.append(f"Round {round_number}: skipped an opinion with missing/invalid agent id.")
                continue

            if agent_id in seen_agents_this_round:
                issues.append(f"Round {round_number}: duplicate opinion for agent '{agent_id}'; kept the first occurrence.")
                continue
            seen_agents_this_round.add(agent_id)

            raw_position = op.get("position")
            raw_architecture = op.get("architecture")
            position_candidate = _canonical_candidate(raw_position, candidate_aliases)
            architecture_candidate = _canonical_candidate(raw_architecture, candidate_aliases)
            if (
                position_candidate is not None
                and architecture_candidate is not None
                and position_candidate != architecture_candidate
            ):
                issues.append(
                    f"Round {round_number}, agent '{agent_id}': 'position' and "
                    "'architecture' identify different candidates; architecture is authoritative."
                )
            candidate = architecture_candidate or position_candidate
            if candidate is None:
                issues.append(f"Round {round_number}, agent '{agent_id}': no usable candidate/position value.")

            raw_confidence = op.get("confidence")
            confidence = clip_confidence(raw_confidence)
            if confidence is None:
                issues.append(
                    f"Round {round_number}, agent '{agent_id}': missing, non-numeric, "
                    "or out-of-range confidence value (expected [0, 1])."
                )

            update_decision = _clean_optional_string(op.get("update_decision"))
            if update_decision and update_decision not in VALID_UPDATE_DECISIONS:
                issues.append(
                    f"Round {round_number}, agent '{agent_id}': unknown update_decision "
                    f"'{update_decision}'."
                )

            candidate_assessments = _parse_candidate_assessments(
                op,
                candidate_aliases,
                round_number,
                agent_id,
                issues,
            )

            snapshots.setdefault(agent_id, []).append(
                AgentRoundSnapshot(
                    agent_id=agent_id,
                    round_number=round_number,
                    raw_position=raw_position,
                    candidate=candidate,
                    confidence=confidence,
                    raw_architecture=raw_architecture,
                    ranked_candidate=_canonical_candidate(op.get("ranked_candidate"), candidate_aliases),
                    candidate_assessments=candidate_assessments,
                    ranking_margin=_bounded_float(op.get("ranking_margin"), 0.0, 5.0),
                    ranking_changed=_optional_bool(op.get("ranking_changed")),
                    update_decision=update_decision,
                    update_reason=_clean_optional_string(op.get("update_reason")),
                    changed_mind=_optional_bool(op.get("changed_mind")),
                    material_update=_optional_bool(op.get("material_update")),
                    assumptions_changed=_optional_bool(op.get("assumptions_changed")),
                    confidence_changed=_optional_bool(op.get("confidence_changed")),
                    challenger_id=_clean_optional_string(op.get("challenger_id")),
                    challenged_claim=_clean_optional_string(op.get("challenged_claim")),
                    switch_assessment=_parse_switch_assessment(op.get("switch_assessment"), candidate_aliases),
                )
            )

        # --- messages for this round ---
        round_messages = round_entry.get("messages")
        if not isinstance(round_messages, list):
            issues.append(f"Round {round_number}: 'messages' missing or malformed; no messages recorded for this round.")
            round_messages = []

        for msg in round_messages:
            if not isinstance(msg, dict):
                issues.append(f"Round {round_number}: skipped a malformed message entry.")
                continue

            sequence = msg.get("sequence")
            sender = msg.get("sender")
            content = msg.get("content")
            recipients = msg.get("recipients") if isinstance(msg.get("recipients"), list) else []

            dedup_key = (round_number, sequence, sender, content)
            if dedup_key in seen_message_keys:
                issues.append(f"Round {round_number}: skipped a duplicate message from '{sender}' (sequence {sequence}).")
                continue
            seen_message_keys.add(dedup_key)

            if declared_agents and sender not in declared_agents:
                issues.append(f"Round {round_number}: message sender '{sender}' is not in the declared participants list.")

            if not content or not str(content).strip():
                issues.append(f"Round {round_number}: message from '{sender}' (sequence {sequence}) has empty content.")

            message_id = f"r{round_number}_s{sequence}_{sender or 'unknown'}"
            messages.append(
                Message(
                    message_id=message_id,
                    round_number=round_number,
                    sequence=sequence if isinstance(sequence, int) else -1,
                    sender=sender,
                    recipients=recipients,
                    content=content,
                )
            )

    for agent_id in snapshots:
        snapshots[agent_id].sort(key=lambda s: s.round_number)
    messages.sort(key=lambda m: (m.round_number, m.sequence))

    agents = declared_agents if declared_agents else sorted(snapshots.keys())
    for agent_id in snapshots:
        if agent_id not in agents:
            issues.append(f"Agent '{agent_id}' appears in opinions but not in the declared participants list; included anyway.")
            agents.append(agent_id)

    unique_rounds = sorted(set(rounds_present))
    for agent_id in agents:
        agent_rounds = {snap.round_number for snap in snapshots.get(agent_id, [])}
        for round_number in unique_rounds:
            if round_number not in agent_rounds:
                issues.append(
                    f"Round {round_number}, agent '{agent_id}': missing opinion snapshot."
                )

    preferred_candidate, preferred_source = _resolve_anchor_candidate(
        data, snapshots, rounds_present, issues, candidate_aliases
    )

    _validate_top_level_opinion_history(data, snapshots, issues)

    objective = data.get("objective")
    topic = objective.get("goal") if isinstance(objective, dict) else None

    decision_summary = data.get("decision_summary")
    consensus_status = decision_summary.get("consensus_status") if isinstance(decision_summary, dict) else None
    graph = _parse_graph(data.get("graph"), agents, issues)

    return Discussion(
        discussion_id=discussion_id,
        agents=agents,
        rounds_present=unique_rounds,
        preferred_candidate=preferred_candidate,
        preferred_candidate_source=preferred_source,
        snapshots=snapshots,
        messages=messages,
        issues=issues,
        topic=topic if isinstance(topic, str) and topic.strip() else None,
        week3_consensus_status=consensus_status if isinstance(consensus_status, str) and consensus_status.strip() else None,
        utility_consistency_tolerance=utility_tolerance,
        graph=graph,
    )


def _load_raw(source: Union[str, Path, Dict[str, Any]]) -> Any:
    if isinstance(source, dict):
        return source
    path = Path(source)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_graph(
    raw_graph: Any,
    discussion_agents: List[str],
    issues: List[str],
) -> Optional[DiscussionGraph]:
    """Validate and retain the runner-declared communication topology."""
    if raw_graph is None:
        return None
    if not isinstance(raw_graph, dict):
        issues.append("Top-level 'graph' is malformed; expected an object.")
        return None

    raw_agents = raw_graph.get("agents")
    graph_agents: List[str] = []
    if isinstance(raw_agents, list):
        for raw_agent in raw_agents:
            if not isinstance(raw_agent, str) or not raw_agent.strip():
                issues.append("Graph contains an invalid agent identifier; skipped it.")
                continue
            agent = raw_agent.strip()
            if agent not in graph_agents:
                graph_agents.append(agent)
    else:
        issues.append("Graph 'agents' is missing or malformed; inferred nodes from edges.")

    raw_edges = raw_graph.get("edges")
    edges: List[Tuple[str, str]] = []
    if isinstance(raw_edges, list):
        for raw_edge in raw_edges:
            if (
                not isinstance(raw_edge, (list, tuple))
                or len(raw_edge) != 2
                or not all(
                    isinstance(endpoint, str) and endpoint.strip()
                    for endpoint in raw_edge
                )
            ):
                issues.append("Graph contains a malformed edge; skipped it.")
                continue
            edge = (raw_edge[0].strip(), raw_edge[1].strip())
            if edge not in edges:
                edges.append(edge)
            for endpoint in edge:
                if endpoint not in graph_agents:
                    graph_agents.append(endpoint)
                    issues.append(
                        f"Graph edge references undeclared node '{endpoint}'; included it."
                    )
    else:
        issues.append("Graph 'edges' is missing or malformed; no edges retained.")

    directed = raw_graph.get("directed", True)
    if not isinstance(directed, bool):
        issues.append("Graph 'directed' must be boolean; defaulted to true.")
        directed = True

    for agent in graph_agents:
        if agent not in discussion_agents:
            issues.append(
                f"Graph node '{agent}' is not in the discussion participants list."
            )

    return DiscussionGraph(agents=graph_agents, edges=edges, directed=directed)


def _resolve_anchor_candidate(
    data: Dict[str, Any],
    snapshots: Dict[str, List[AgentRoundSnapshot]],
    rounds_present: List[int],
    issues: List[str],
    candidate_aliases: Dict[str, str],
) -> Tuple[Optional[str], str]:
    """Resolve the reference candidate used as the stance anchor (Task 1).

    Preferred source: `decision_summary.preferred_candidate`, which is the
    Week 3 discussion's own recorded outcome. If that is absent, fall
    back to a majority vote of agents' candidates at the final available
    round. If neither is available, the anchor is None and stance values
    cannot be computed (documented, not fabricated).
    """
    decision_summary = data.get("decision_summary")
    if isinstance(decision_summary, dict):
        anchor = _canonical_candidate(decision_summary.get("preferred_candidate"), candidate_aliases)
        if anchor:
            return anchor, "decision_summary"

    issues.append("No usable 'decision_summary.preferred_candidate'; falling back to a majority vote at the final round.")

    if not rounds_present:
        return None, "unavailable"

    final_round = max(rounds_present)
    final_round_candidates = [
        snap.candidate
        for agent_snaps in snapshots.values()
        for snap in agent_snaps
        if snap.round_number == final_round and snap.candidate is not None
    ]
    if not final_round_candidates:
        issues.append("No valid candidates found at the final round; stance anchor is unavailable.")
        return None, "unavailable"

    counts = Counter(final_round_candidates)
    top_count = max(counts.values())
    winners = [c for c, n in counts.items() if n == top_count]
    if len(winners) > 1:
        issues.append(f"Majority-vote fallback found a tie among {winners}; picking the first alphabetically for determinism.")
        winners.sort()
    return winners[0], "majority_fallback"


def _clean_optional_string(raw: Any) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    return cleaned or None


def _optional_bool(raw: Any) -> Optional[bool]:
    return raw if isinstance(raw, bool) else None


def _bounded_float(raw: Any, minimum: float, maximum: float) -> Optional[float]:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value or not minimum <= value <= maximum:
        return None
    return value


def _nonnegative_float(raw: Any) -> Optional[float]:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")) or value < 0:
        return None
    return value


def _load_utility_tolerance(data: Dict[str, Any], issues: List[str]) -> float:
    configuration = data.get("configuration")
    raw = configuration.get("utility_consistency_tolerance") if isinstance(configuration, dict) else None
    if raw is None:
        return DEFAULT_UTILITY_CONSISTENCY_TOLERANCE
    value = _bounded_float(raw, 0.0, 5.0)
    if value is None:
        issues.append(
            "Invalid configuration.utility_consistency_tolerance; using default 0.25."
        )
        return DEFAULT_UTILITY_CONSISTENCY_TOLERANCE
    return value


def _load_candidate_aliases(data: Dict[str, Any], issues: List[str]) -> Dict[str, str]:
    """Load only explicit aliases; semantic/fuzzy candidate merging is unsafe."""
    aliases: Dict[str, str] = {}
    configuration = data.get("configuration")
    raw_aliases = configuration.get("candidate_aliases") if isinstance(configuration, dict) else None
    if raw_aliases is None:
        raw_aliases = data.get("candidate_aliases")
    if raw_aliases is None:
        return aliases
    if not isinstance(raw_aliases, dict):
        issues.append("Ignored malformed candidate_aliases mapping.")
        return aliases
    for raw_alias, raw_canonical in raw_aliases.items():
        alias = normalize_candidate_id(raw_alias)
        canonical = normalize_candidate_id(raw_canonical)
        if alias is None or canonical is None:
            issues.append("Ignored an invalid candidate alias entry.")
            continue
        aliases[alias] = canonical
    return aliases


def _canonical_candidate(raw: Any, aliases: Dict[str, str]) -> Optional[str]:
    candidate = normalize_candidate_id(raw)
    if candidate is None:
        return None
    seen = set()
    while candidate in aliases and candidate not in seen:
        seen.add(candidate)
        candidate = aliases[candidate]
    return candidate


def _parse_candidate_assessments(
    opinion: Dict[str, Any],
    aliases: Dict[str, str],
    round_number: int,
    agent_id: str,
    issues: List[str],
) -> List[CandidateAssessment]:
    raw_assessments = opinion.get("candidate_assessments")
    if not isinstance(raw_assessments, list):
        return []

    raw_weights = opinion.get("persona_weights")
    weights: Dict[str, float] = {}
    if isinstance(raw_weights, dict):
        for criterion, raw_weight in raw_weights.items():
            weight = _nonnegative_float(raw_weight)
            if weight is not None:
                weights[str(criterion)] = weight

    assessments: List[CandidateAssessment] = []
    for raw_assessment in raw_assessments:
        if not isinstance(raw_assessment, dict):
            issues.append(
                f"Round {round_number}, agent '{agent_id}': skipped malformed candidate assessment."
            )
            continue

        raw_candidate_id = raw_assessment.get("candidate_id")
        candidate_id = _canonical_candidate(raw_candidate_id, aliases)
        raw_scores = raw_assessment.get("criterion_scores")
        scores: Dict[str, float] = {}
        scores_valid = isinstance(raw_scores, dict) and bool(raw_scores)
        if isinstance(raw_scores, dict):
            for criterion, raw_score in raw_scores.items():
                score = _bounded_float(raw_score, 0.0, 5.0)
                if score is None:
                    scores_valid = False
                    issues.append(
                        f"Round {round_number}, agent '{agent_id}', candidate "
                        f"'{raw_candidate_id}': invalid score for criterion '{criterion}'."
                    )
                else:
                    scores[str(criterion)] = score

        declared_utility = _bounded_float(raw_assessment.get("weighted_utility"), 0.0, 5.0)
        usable_utility: Optional[float] = None
        utility_status = "missing"

        expected_criteria = {criterion for criterion, weight in weights.items() if weight > 0}
        can_recompute = (
            scores_valid
            and bool(expected_criteria)
            and expected_criteria.issubset(scores.keys())
        )
        if can_recompute:
            weight_total = sum(weights.values())
            if weight_total > 0:
                recomputed = sum(
                    scores[criterion] * (weight / weight_total)
                    for criterion, weight in weights.items()
                    if weight > 0
                )
                usable_utility = round(recomputed, 6)
                utility_status = "recomputed"
                if (
                    declared_utility is not None
                    and abs(declared_utility - recomputed) > UTILITY_RECOMPUTE_TOLERANCE
                ):
                    utility_status = "inconsistent"
                    issues.append(
                        f"Round {round_number}, agent '{agent_id}', candidate "
                        f"'{raw_candidate_id}': declared weighted utility {declared_utility:.3f} "
                        f"differs from recomputed utility {recomputed:.3f}."
                    )
        elif scores and declared_utility is not None:
            usable_utility = declared_utility
            utility_status = "declared_incomplete"
        elif declared_utility is not None and declared_utility > 0:
            usable_utility = declared_utility
            utility_status = "declared_only"

        assessments.append(
            CandidateAssessment(
                raw_candidate_id=raw_candidate_id if isinstance(raw_candidate_id, str) else None,
                candidate_id=candidate_id,
                criterion_scores=scores,
                weighted_utility=usable_utility,
                declared_weighted_utility=declared_utility,
                utility_status=utility_status,
            )
        )

    return assessments


def _parse_switch_assessment(raw: Any, aliases: Dict[str, str]) -> Optional[SwitchAssessment]:
    if not isinstance(raw, dict):
        return None
    raw_errors = raw.get("validation_errors")
    errors = [str(error) for error in raw_errors] if isinstance(raw_errors, list) else []
    return SwitchAssessment(
        current_candidate=_canonical_candidate(raw.get("current_candidate"), aliases),
        alternative_candidate=_canonical_candidate(raw.get("alternative_candidate"), aliases),
        condition_effect=_clean_optional_string(raw.get("condition_effect")),
        status=_clean_optional_string(raw.get("status")),
        valid=_optional_bool(raw.get("valid")),
        validation_errors=errors,
    )


def _validate_top_level_opinion_history(
    data: Dict[str, Any],
    snapshots: Dict[str, List[AgentRoundSnapshot]],
    issues: List[str],
) -> None:
    """Use the duplicated Week 3 opinion history only as a consistency check."""
    history = data.get("opinions")
    if not isinstance(history, dict):
        return

    counter_fields = {
        "mind_change_count": "changed_mind",
        "material_update_count": "material_update",
        "ranking_change_count": "ranking_changed",
        "assumption_change_count": "assumptions_changed",
        "confidence_change_count": "confidence_changed",
    }
    for agent_id, agent_history in history.items():
        if not isinstance(agent_history, dict):
            continue
        agent_snapshots = snapshots.get(agent_id, [])
        history_rounds = agent_history.get("rounds")
        if isinstance(history_rounds, list):
            recorded_rounds = {
                entry.get("round")
                for entry in history_rounds
                if isinstance(entry, dict) and isinstance(entry.get("round"), int)
            }
            parsed_rounds = {snapshot.round_number for snapshot in agent_snapshots}
            if recorded_rounds != parsed_rounds:
                issues.append(
                    f"Agent '{agent_id}': top-level opinion history rounds do not match rounds[].opinions."
                )

        for count_field, snapshot_field in counter_fields.items():
            declared_count = agent_history.get(count_field)
            if not isinstance(declared_count, int):
                continue
            actual_count = sum(
                1 for snapshot in agent_snapshots if getattr(snapshot, snapshot_field) is True
            )
            if declared_count != actual_count:
                issues.append(
                    f"Agent '{agent_id}': {count_field}={declared_count}, "
                    f"but rounds[].opinions implies {actual_count}."
                )
