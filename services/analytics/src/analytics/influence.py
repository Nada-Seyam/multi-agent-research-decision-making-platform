import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .models import AgentRoundSnapshot, Discussion
from .opinion_change import StanceRecord


MIN_DATA_POINTS = 3
MODERATE_DATA_POINTS = 6
HIGH_DATA_POINTS = 12
MAX_PATH_HOPS = 2
INDIRECT_DECAY = 0.5
MIN_PROPAGATION_RELEVANCE = 0.05
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]+")


@dataclass
class InfluenceRecord:
    agent_id: str
    score: Optional[float]
    num_data_points: int
    reason: Optional[str] = None
    num_recipients: int = 0
    num_movement_events: int = 0
    explicit_attribution_matches: int = 0
    unconfirmed_observations: int = 0
    num_direct_observations: int = 0
    num_indirect_observations: int = 0
    num_opportunities: int = 0
    opportunity_coverage: Optional[float] = None
    reliability: str = "insufficient"


@dataclass
class _Observation:
    sender: str
    recipient: str
    round_number: int
    end_round: int
    path: Tuple[str, ...]
    message_texts: Tuple[str, ...]
    response_text: str
    movement: int
    explicit_attribution: bool
    relevance: float = 0.0

    @property
    def hop_count(self) -> int:
        return len(self.path) - 1

    @property
    def key(self) -> Tuple[str, str, int, int]:
        return self.sender, self.recipient, self.round_number, self.end_round


def compute_influence(
    discussion: Discussion,
    opinion_change_result: Dict[str, List[StanceRecord]],
) -> Dict[str, InfluenceRecord]:
    """Return graph-aware exact-candidate influence associations.

    Direct observations cover ``sender -> recipient`` in consecutive rounds.
    Indirect observations cover ``sender -> intermediary -> recipient`` over
    two consecutive transitions. Indirect paths are retained only when the
    first argument is textually present in the intermediary's later message.
    """
    candidates = {
        (agent_id, record.round): record.candidate
        for agent_id, records in opinion_change_result.items()
        for record in records
    }
    snapshots = {
        (agent_id, snapshot.round_number): snapshot
        for agent_id, agent_snapshots in discussion.snapshots.items()
        for snapshot in agent_snapshots
    }

    directed_messages, senders_with_messages = _group_directed_messages(
        discussion
    )
    path_candidates, opportunities = _build_path_candidates(
        discussion.rounds_present,
        directed_messages,
        candidates,
        snapshots,
    )
    observations = _score_and_select_observations(path_candidates)

    observations_by_sender: Dict[str, List[_Observation]] = defaultdict(list)
    for observation in observations:
        observations_by_sender[observation.sender].append(observation)

    results: Dict[str, InfluenceRecord] = {}
    for agent_id in discussion.agents:
        agent_observations = observations_by_sender.get(agent_id, [])
        recipients = {obs.recipient for obs in agent_observations}
        movement_events = sum(
            1 for obs in agent_observations if obs.movement != 0
        )
        explicit_matches = sum(
            1 for obs in agent_observations if obs.explicit_attribution
        )
        unconfirmed = len(agent_observations) - explicit_matches
        direct = sum(1 for obs in agent_observations if obs.hop_count == 1)
        indirect = sum(1 for obs in agent_observations if obs.hop_count == 2)
        num_opportunities = len(opportunities.get(agent_id, set()))
        coverage = (
            len(agent_observations) / num_opportunities
            if num_opportunities
            else None
        )

        common = dict(
            agent_id=agent_id,
            num_data_points=len(agent_observations),
            num_recipients=len(recipients),
            num_movement_events=movement_events,
            explicit_attribution_matches=explicit_matches,
            unconfirmed_observations=unconfirmed,
            num_direct_observations=direct,
            num_indirect_observations=indirect,
            num_opportunities=num_opportunities,
            opportunity_coverage=(round(coverage, 6) if coverage is not None else None),
        )

        if agent_id not in senders_with_messages:
            results[agent_id] = InfluenceRecord(
                score=None,
                reason="insufficient_data: agent sent no usable directed messages",
                **common,
            )
            continue

        if len(agent_observations) < MIN_DATA_POINTS:
            results[agent_id] = InfluenceRecord(
                score=None,
                reason=(
                    "insufficient_data: only "
                    f"{len(agent_observations)} usable graph-aware observations "
                    f"available (need >= {MIN_DATA_POINTS})"
                ),
                **common,
            )
            continue

        relevance = [obs.relevance for obs in agent_observations]
        movement = [float(obs.movement) for obs in agent_observations]
        if not _has_variance(relevance):
            results[agent_id] = InfluenceRecord(
                score=None,
                reason="insufficient_data: no variation in argument relevance",
                **common,
            )
            continue
        if not _has_variance(movement):
            results[agent_id] = InfluenceRecord(
                score=None,
                reason="insufficient_data: no variation in candidate movement",
                **common,
            )
            continue

        correlation = _spearman(relevance, movement)
        if correlation is None:
            results[agent_id] = InfluenceRecord(
                score=None,
                reason="insufficient_data: correlation is undefined",
                **common,
            )
            continue

        path_support = sum(
            INDIRECT_DECAY ** (obs.hop_count - 1)
            for obs in agent_observations
        ) / len(agent_observations)
        results[agent_id] = InfluenceRecord(
            score=round(correlation * path_support, 6),
            reliability=_reliability(
                len(agent_observations), movement_events, coverage
            ),
            **common,
        )

    return results


def _group_directed_messages(
    discussion: Discussion,
) -> Tuple[Dict[Tuple[str, str, int], str], Set[str]]:
    grouped: Dict[Tuple[str, str, int], List[Tuple[int, str]]] = defaultdict(list)
    senders: Set[str] = set()
    for message in discussion.messages:
        if not message.sender or not message.content or not message.content.strip():
            continue
        for recipient in message.recipients:
            if not recipient or recipient == message.sender:
                continue
            grouped[(message.sender, recipient, message.round_number)].append(
                (message.sequence, message.content.strip())
            )
            senders.add(message.sender)

    combined = {
        key: "\n".join(text for _, text in sorted(parts))
        for key, parts in grouped.items()
    }
    return combined, senders


def _build_path_candidates(
    rounds_present: Sequence[int],
    directed_messages: Dict[Tuple[str, str, int], str],
    candidates: Dict[Tuple[str, int], Optional[str]],
    snapshots: Dict[Tuple[str, int], AgentRoundSnapshot],
) -> Tuple[List[_Observation], Dict[str, Set[Tuple[str, str, int, int]]]]:
    """Build direct and two-hop candidates without crossing missing rounds."""
    rounds = set(rounds_present)
    outgoing: Dict[Tuple[str, int], List[Tuple[str, str]]] = defaultdict(list)
    for (sender, recipient, round_number), text in directed_messages.items():
        outgoing[(sender, round_number)].append((recipient, text))
    for messages in outgoing.values():
        messages.sort(key=lambda item: item[0])

    path_candidates: List[_Observation] = []
    opportunities: Dict[str, Set[Tuple[str, str, int, int]]] = defaultdict(set)

    for (source, recipient, start_round), first_text in sorted(
        directed_messages.items()
    ):
        next_round = start_round + 1
        if next_round not in rounds:
            continue

        direct_key = (source, recipient, start_round, next_round)
        opportunities[source].add(direct_key)
        direct = _make_observation(
            source=source,
            recipient=recipient,
            start_round=start_round,
            end_round=next_round,
            path=(source, recipient),
            message_texts=(first_text,),
            candidates=candidates,
            snapshots=snapshots,
        )
        if direct is not None:
            path_candidates.append(direct)

        if MAX_PATH_HOPS < 2 or start_round + 2 not in rounds:
            continue

        for target, second_text in outgoing.get((recipient, next_round), []):
            # Simple paths only: cycles must not create self-influence or
            # duplicate credit through immediate return edges.
            if target in {source, recipient}:
                continue
            end_round = start_round + 2
            indirect_key = (source, target, start_round, end_round)
            opportunities[source].add(indirect_key)
            indirect = _make_observation(
                source=source,
                recipient=target,
                start_round=start_round,
                end_round=end_round,
                path=(source, recipient, target),
                message_texts=(first_text, second_text),
                candidates=candidates,
                snapshots=snapshots,
            )
            if indirect is not None:
                path_candidates.append(indirect)

    return path_candidates, opportunities


def _make_observation(
    source: str,
    recipient: str,
    start_round: int,
    end_round: int,
    path: Tuple[str, ...],
    message_texts: Tuple[str, ...],
    candidates: Dict[Tuple[str, int], Optional[str]],
    snapshots: Dict[Tuple[str, int], AgentRoundSnapshot],
) -> Optional[_Observation]:
    source_candidate = candidates.get((source, start_round))
    recipient_before_candidate = candidates.get((recipient, end_round - 1))
    recipient_after_candidate = candidates.get((recipient, end_round))
    if (
        source_candidate is None
        or recipient_before_candidate is None
        or recipient_after_candidate is None
    ):
        return None

    final_snapshot = snapshots.get((recipient, end_round))
    if final_snapshot is None:
        return None
    response_text = _response_text(final_snapshot)
    if response_text is None:
        return None

    if len(path) == 2:
        explicit_attribution = final_snapshot.challenger_id == source
    else:
        intermediary = path[1]
        intermediary_snapshot = snapshots.get((intermediary, start_round + 1))
        intermediary_match = (
            intermediary_snapshot is not None
            and intermediary_snapshot.challenger_id == source
        )
        final_match = final_snapshot.challenger_id in {source, intermediary}
        explicit_attribution = intermediary_match and final_match

    return _Observation(
        sender=source,
        recipient=recipient,
        round_number=start_round,
        end_round=end_round,
        path=path,
        message_texts=message_texts,
        response_text=response_text,
        movement=_candidate_movement(
            source_candidate,
            recipient_before_candidate,
            recipient_after_candidate,
        ),
        explicit_attribution=explicit_attribution,
    )


def _response_text(snapshot: AgentRoundSnapshot) -> Optional[str]:
    parts = [
        part
        for part in (snapshot.challenged_claim, snapshot.update_reason)
        if part
    ]
    return "\n".join(parts) if parts else None


def _score_and_select_observations(
    path_candidates: Sequence[_Observation],
) -> List[_Observation]:
    """Score paths and keep one strongest path per source/target/window."""
    if not path_candidates:
        return []

    texts: List[str] = []
    for candidate in path_candidates:
        texts.extend(candidate.message_texts)
        texts.append(candidate.response_text)
    vectors = _tfidf_vectors(list(dict.fromkeys(texts)))

    selected: Dict[Tuple[str, str, int, int], _Observation] = {}
    for candidate in sorted(
        path_candidates,
        key=lambda item: (item.key, item.path),
    ):
        message_links = [
            _cosine(vectors[left], vectors[right])
            for left, right in zip(
                candidate.message_texts,
                candidate.message_texts[1:],
            )
        ]
        if (
            candidate.hop_count > 1
            and any(
                similarity < MIN_PROPAGATION_RELEVANCE
                for similarity in message_links
            )
        ):
            continue

        final_link = _cosine(
            vectors[candidate.message_texts[-1]],
            vectors[candidate.response_text],
        )
        similarities = message_links + [final_link]
        candidate.relevance = round(_geometric_mean(similarities), 12)

        previous = selected.get(candidate.key)
        if previous is None or candidate.relevance > previous.relevance:
            selected[candidate.key] = candidate

    return sorted(
        selected.values(),
        key=lambda item: (item.sender, item.round_number, item.end_round, item.recipient),
    )


def _candidate_movement(
    sender_candidate: str,
    recipient_candidate: str,
    recipient_next_candidate: str,
) -> int:
    before = 0 if recipient_candidate == sender_candidate else 1
    after = 0 if recipient_next_candidate == sender_candidate else 1
    return before - after


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def _tfidf_vectors(texts: Sequence[str]) -> Dict[str, Dict[str, float]]:
    tokenized = [_tokenize(text) for text in texts]
    document_frequency: Counter = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))

    document_count = len(tokenized)
    idf = {
        token: math.log((1 + document_count) / (1 + frequency)) + 1.0
        for token, frequency in document_frequency.items()
    }
    return {
        text: _tfidf_vector(tokens, idf)
        for text, tokens in zip(texts, tokenized)
    }




def _tfidf_vector(
    tokens: Sequence[str], idf: Dict[str, float]
) -> Dict[str, float]:
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {
        token: (frequency / total) * idf[token]
        for token, frequency in counts.items()
    }


def _cosine(left: Dict[str, float], right: Dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    shared = left.keys() & right.keys()
    dot = sum(left[token] * right[token] for token in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _geometric_mean(values: Sequence[float]) -> float:
    if not values or any(value <= 0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _reliability(
    num_data_points: int,
    num_movement_events: int,
    coverage: Optional[float],
) -> str:
    if (
        num_data_points >= HIGH_DATA_POINTS
        and num_movement_events >= 4
        and coverage is not None
        and coverage >= 0.75
    ):
        return "high"
    if (
        num_data_points >= MODERATE_DATA_POINTS
        and num_movement_events >= 2
        and coverage is not None
        and coverage >= 0.5
    ):
        return "moderate"
    return "low"


def _rank(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[indexed[position][0]] = average_rank
        cursor = end
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or not xs:
        return None
    return _pearson(_rank(xs), _rank(ys))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0:
        return None
    correlation = sum(
        left * right for left, right in zip(centered_x, centered_y)
    ) / denominator
    return max(-1.0, min(1.0, correlation))


def _has_variance(values: Iterable[float]) -> bool:
    iterator = iter(values)
    try:
        first = next(iterator)
    except StopIteration:
        return False
    return any(value != first for value in iterator)
