import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional

from .opinion_change import StanceRecord


@dataclass
class AgreementRecord:
    round: int
    score: Optional[float]
    num_agents_considered: int
    reason: Optional[str] = None
    disagreement_score: Optional[float] = None
    agreeing_pairs: int = 0
    disagreeing_pairs: int = 0
    total_pairs: int = 0


def compute_agreement(
    opinion_change_result: Dict[str, List[StanceRecord]],
) -> List[AgreementRecord]:
    """Compute candidate agreement and disagreement for every observed round.

    ``score`` is the proportion of usable agent pairs whose committed
    candidates match. ``disagreement_score`` is the complementary proportion
    whose candidates differ. At least two usable candidates are required.
    """
    candidates_by_round: Dict[int, List[str]] = {}
    all_rounds = set()

    for records in opinion_change_result.values():
        for record in records:
            all_rounds.add(record.round)
            if record.candidate is not None:
                candidates_by_round.setdefault(record.round, []).append(
                    record.candidate
                )

    results: List[AgreementRecord] = []
    for round_number in sorted(all_rounds):
        candidates = candidates_by_round.get(round_number, [])

        if len(candidates) < 2:
            results.append(
                AgreementRecord(
                    round=round_number,
                    score=None,
                    disagreement_score=None,
                    num_agents_considered=len(candidates),
                    reason=(
                        "insufficient_data: fewer than 2 agents have a "
                        f"resolvable committed candidate in round {round_number} "
                        f"(found {len(candidates)})"
                    ),
                )
            )
            continue

        candidate_pairs = list(itertools.combinations(candidates, 2))
        agreeing_pairs = sum(
            1 for left, right in candidate_pairs if left == right
        )
        total_pairs = len(candidate_pairs)
        disagreeing_pairs = total_pairs - agreeing_pairs

        agreement_score = agreeing_pairs / total_pairs
        disagreement_score = disagreeing_pairs / total_pairs

        results.append(
            AgreementRecord(
                round=round_number,
                score=round(agreement_score, 6),
                disagreement_score=round(disagreement_score, 6),
                num_agents_considered=len(candidates),
                agreeing_pairs=agreeing_pairs,
                disagreeing_pairs=disagreeing_pairs,
                total_pairs=total_pairs,
            )
        )

    return results
