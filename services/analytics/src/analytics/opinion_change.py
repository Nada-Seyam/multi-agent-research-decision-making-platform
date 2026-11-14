"""Opinion-change analysis for the structured Week 3 discussion format.

The metric deliberately measures only committed alignment with the discussion
anchor. Matrix ranking, confidence, and update metadata are not folded into
the stance because they are different signals and are not required by any
downstream metric.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .models import AgentRoundSnapshot, Discussion


@dataclass
class StanceRecord:
    """One agent's committed stance for one discussion round."""

    agent_id: str
    round: int
    candidate: Optional[str]
    stance: Optional[float]
    change: Optional[float]
    reason: Optional[str] = None
    candidate_changed: Optional[bool] = None
    change_reason: Optional[str] = None


def compute_opinion_change(discussion: Discussion) -> Dict[str, List[StanceRecord]]:
    """Return one opinion record per declared agent per discussion round.

    ``change`` is calculated only against the immediately preceding
    discussion round. Candidate identity is retained so a switch between two
    non-anchor candidates remains observable even though both have stance -1.
    """
    anchor = discussion.preferred_candidate
    results: Dict[str, List[StanceRecord]] = {}

    for agent_id in discussion.agents:
        snapshots_by_round: Dict[int, AgentRoundSnapshot] = {}
        for snapshot in discussion.snapshots.get(agent_id, []):
            snapshots_by_round.setdefault(snapshot.round_number, snapshot)

        records: List[StanceRecord] = []
        previous_record: Optional[StanceRecord] = None

        for round_number in discussion.rounds_present:
            snapshot = snapshots_by_round.get(round_number)
            if snapshot is None:
                record = StanceRecord(
                    agent_id=agent_id,
                    round=round_number,
                    candidate=None,
                    stance=None,
                    change=None,
                    reason="missing_opinion_snapshot",
                    change_reason="missing_current_round_snapshot",
                )
                records.append(record)
                previous_record = record
                continue

            stance, reason = _committed_stance(snapshot.candidate, anchor)
            candidate_changed = (
                snapshot.candidate != previous_record.candidate
                if previous_record is not None
                and snapshot.candidate is not None
                and previous_record.candidate is not None
                else None
            )

            change: Optional[float] = None
            change_reason: Optional[str] = None
            if previous_record is not None:
                if stance is not None and previous_record.stance is not None:
                    change = round(stance - previous_record.stance, 6)
                else:
                    change_reason = "missing_previous_or_current_round_stance"

            record = StanceRecord(
                agent_id=agent_id,
                round=round_number,
                candidate=snapshot.candidate,
                stance=stance,
                change=change,
                reason=reason,
                candidate_changed=candidate_changed,
                change_reason=change_reason,
            )
            records.append(record)
            previous_record = record

        results[agent_id] = records

    return results


def _committed_stance(
    candidate: Optional[str], anchor: Optional[str]
) -> Tuple[Optional[float], Optional[str]]:
    if anchor is None:
        return None, "no_anchor_candidate_available_for_discussion"
    if candidate is None:
        return None, "unresolvable_candidate_for_this_snapshot"
    return (1.0 if candidate == anchor else -1.0), None
