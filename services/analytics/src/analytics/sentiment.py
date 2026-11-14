"""Candidate-directed sentiment for technical multi-agent discussions.

Structured Week 3 evidence remains authoritative. When only prose is
available, a Natural Language Inference (NLI) model decides whether the
message supports or opposes the recipient's current candidate. This avoids
treating technical words such as "risk" or "complex" as emotional sentiment.
"""

import os
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .models import AgentRoundSnapshot, Discussion


POSITIVE_THRESHOLD = 0.15
NEGATIVE_THRESHOLD = -0.15
NLI_MODEL_NAME = os.getenv(
    "ANALYTICS_NLI_MODEL", "cross-encoder/nli-deberta-v3-base"
)
NLI_BATCH_SIZE = max(1, int(os.getenv("ANALYTICS_NLI_BATCH_SIZE", "32")))
NLI_MAX_LENGTH = 512

WORD_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")
ALIAS_DEFINITION_RE = re.compile(
    r"(?P<long>[A-Za-z][A-Za-z0-9/-]*"
    r"(?:[\s-]+[A-Za-z][A-Za-z0-9/-]*){1,12})"
    r"\s*\(\s*(?P<short>[A-Z][A-Z0-9/+_-]{1,11})\s*\)"
)


@dataclass
class CandidateSentiment:
    recipient: str
    target_candidate: Optional[str]
    score: Optional[float]
    label: Optional[str]
    evidence_source: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class SentimentRecord:
    message_id: str
    round: int
    sender: Optional[str]
    evaluations: List[CandidateSentiment] = field(default_factory=list)
    reason: Optional[str] = None


class _TransformersNLIClassifier:
    """Lazy Transformers wrapper with a small, testable interface."""

    def __init__(self, model_name: str = NLI_MODEL_NAME) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - deployment-only failure
            raise RuntimeError(
                "Candidate sentiment requires torch, transformers, and "
                "sentencepiece. Install the pinned requirements.txt dependencies."
            ) from exc

        self._torch = torch
        try:
            # Avoid a Hugging Face metadata request on every analytics run once
            # the model has already been downloaded.
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=True, use_fast=False
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_name, local_files_only=True
            )
        except OSError:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=False
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            )
        configured_device = os.getenv("ANALYTICS_NLI_DEVICE", "auto").strip().lower()
        device_name = (
            "cuda" if configured_device == "auto" and torch.cuda.is_available()
            else "cpu" if configured_device == "auto"
            else configured_device
        )
        self._device = torch.device(device_name)
        self._model.to(self._device)
        self._model.eval()

        id2label = {
            int(index): str(label).strip().lower()
            for index, label in self._model.config.id2label.items()
        }
        self._entailment_index = next(
            (index for index, label in id2label.items() if "entail" in label),
            None,
        )
        if self._entailment_index is None:
            raise RuntimeError(
                f"{model_name} does not expose an entailment output label"
            )

    def entailment_probabilities(
        self, pairs: Sequence[Tuple[str, str]]
    ) -> List[float]:
        probabilities: List[float] = []
        for start in range(0, len(pairs), NLI_BATCH_SIZE):
            batch = pairs[start : start + NLI_BATCH_SIZE]
            premises = [premise for premise, _ in batch]
            hypotheses = [hypothesis for _, hypothesis in batch]
            encoded = self._tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=NLI_MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                logits = self._model(**encoded).logits
                batch_probabilities = self._torch.softmax(logits, dim=-1)[
                    :, self._entailment_index
                ]
            probabilities.extend(
                float(value) for value in batch_probabilities.detach().cpu().tolist()
            )
        return probabilities


_NLI_CLASSIFIER: Optional[_TransformersNLIClassifier] = None
_NLI_CLASSIFIER_LOCK = Lock()
_NLI_SCORE_CACHE: "OrderedDict[Tuple[object, str, str], float]" = OrderedDict()
_NLI_SCORE_CACHE_LOCK = Lock()
_NLI_INFERENCE_LOCK = Lock()
_NLI_SCORE_CACHE_SIZE = max(128, int(os.getenv("ANALYTICS_NLI_CACHE_SIZE", "8192")))


def _get_nli_classifier() -> _TransformersNLIClassifier:
    global _NLI_CLASSIFIER
    if _NLI_CLASSIFIER is None:
        with _NLI_CLASSIFIER_LOCK:
            if _NLI_CLASSIFIER is None:
                _NLI_CLASSIFIER = _TransformersNLIClassifier()
    return _NLI_CLASSIFIER


@dataclass
class _PendingCandidateEvaluation:
    result: CandidateSentiment
    evidence: List[Tuple[float, float, str]]
    pairs: List[Tuple[str, str]]


def _classify(score: float) -> str:
    if score > POSITIVE_THRESHOLD:
        return "positive"
    if score < NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def compute_sentiment(
    discussion: Discussion,
    *,
    round_number: Optional[int] = None,
) -> List[SentimentRecord]:
    """Evaluate messages against each recipient's current candidate.

    ``round_number`` limits expensive NLI inference to one completed round.
    Candidate aliases are still learned from the complete available snapshot,
    so this produces the same records as filtering a full analysis afterward.
    """
    snapshots = {
        (agent_id, snapshot.round_number): snapshot
        for agent_id, agent_snapshots in discussion.snapshots.items()
        for snapshot in agent_snapshots
    }
    aliases = _candidate_aliases(discussion)
    _learn_message_aliases(discussion, aliases)
    results: List[SentimentRecord] = []
    pending: List[_PendingCandidateEvaluation] = []

    messages = (
        message
        for message in discussion.messages
        if round_number is None or message.round_number == round_number
    )
    for message in messages:
        if not message.content or not str(message.content).strip():
            results.append(
                SentimentRecord(
                    message_id=message.message_id,
                    round=message.round_number,
                    sender=message.sender,
                    reason="empty_or_missing_message_content",
                )
            )
            continue

        if not message.recipients:
            results.append(
                SentimentRecord(
                    message_id=message.message_id,
                    round=message.round_number,
                    sender=message.sender,
                    reason="message_has_no_recipients",
                )
            )
            continue

        sender_snapshot = snapshots.get((message.sender, message.round_number))
        evaluations: List[CandidateSentiment] = []
        for recipient in message.recipients:
            recipient_snapshot = snapshots.get((recipient, message.round_number))
            target_candidate = (
                recipient_snapshot.candidate if recipient_snapshot else None
            )
            evaluation, deferred = _prepare_candidate_evaluation(
                recipient=recipient,
                target_candidate=target_candidate,
                message_text=str(message.content),
                sender_snapshot=sender_snapshot,
                sender_history=discussion.snapshots.get(message.sender or "", []),
                aliases=aliases,
                round_number=message.round_number,
            )
            evaluations.append(evaluation)
            if deferred is not None:
                pending.append(deferred)

        results.append(
            SentimentRecord(
                message_id=message.message_id,
                round=message.round_number,
                sender=message.sender,
                evaluations=evaluations,
            )
        )

    if pending:
        all_pairs = [pair for item in pending for pair in item.pairs]
        probabilities = _cached_entailment_probabilities(all_pairs)
        offset = 0
        for item in pending:
            item_probabilities = probabilities[offset : offset + len(item.pairs)]
            offset += len(item.pairs)
            unit_scores = [
                item_probabilities[index] - item_probabilities[index + 1]
                for index in range(0, len(item_probabilities), 2)
            ]
            if unit_scores:
                item.evidence.append(
                    (
                        max(-1.0, min(1.0, sum(unit_scores) / len(unit_scores))),
                        1.0,
                        "candidate_nli",
                    )
                )
            _finalize_candidate_evaluation(item.result, item.evidence)

    return results


def _prepare_candidate_evaluation(
    recipient: str,
    target_candidate: Optional[str],
    message_text: str,
    sender_snapshot: Optional[AgentRoundSnapshot],
    sender_history: Sequence[AgentRoundSnapshot],
    aliases: Dict[str, Set[str]],
    round_number: int,
) -> Tuple[CandidateSentiment, Optional[_PendingCandidateEvaluation]]:
    if target_candidate is None:
        return CandidateSentiment(
            recipient=recipient,
            target_candidate=None,
            score=None,
            label=None,
            reason="recipient_candidate_unavailable",
        ), None
    if sender_snapshot is None:
        return CandidateSentiment(
            recipient=recipient,
            target_candidate=target_candidate,
            score=None,
            label=None,
            reason="sender_opinion_snapshot_unavailable",
        ), None

    evidence: List[Tuple[float, float, str]] = []

    matrix_score = _matrix_evaluation(sender_snapshot, target_candidate)
    if matrix_score is not None:
        evidence.append((matrix_score, 2.0, "candidate_matrix"))

    if sender_snapshot.candidate == target_candidate:
        evidence.append((1.0, 1.5, "shared_committed_candidate"))
    elif _switched_away_from(
        sender_history, round_number, target_candidate, sender_snapshot
    ):
        evidence.append((-1.0, 1.5, "switched_away_from_candidate"))

    pairs = _targeted_nli_pairs(
        message_text,
        target_candidate,
        aliases.get(target_candidate, {target_candidate}),
    )
    result = CandidateSentiment(
        recipient=recipient,
        target_candidate=target_candidate,
        score=None,
        label=None,
    )
    if pairs:
        return result, _PendingCandidateEvaluation(result, evidence, pairs)

    _finalize_candidate_evaluation(result, evidence)
    return result, None


def _finalize_candidate_evaluation(
    result: CandidateSentiment,
    evidence: List[Tuple[float, float, str]],
) -> None:
    if not evidence:
        result.score = 0.0
        result.label = "neutral"
        result.evidence_source = "no_candidate_reference"
        return

    weighted_sum = sum(score * weight for score, weight, _ in evidence)
    total_weight = sum(weight for _, weight, _ in evidence)
    score = max(-1.0, min(1.0, weighted_sum / total_weight))
    rounded_score = round(score, 6)
    result.score = rounded_score
    result.label = _classify(rounded_score)
    result.evidence_source = " + ".join(source for _, _, source in evidence)


def _evaluate_candidate(
    recipient: str,
    target_candidate: Optional[str],
    message_text: str,
    sender_snapshot: Optional[AgentRoundSnapshot],
    sender_history: Sequence[AgentRoundSnapshot],
    aliases: Dict[str, Set[str]],
    round_number: int,
) -> CandidateSentiment:
    """Evaluate one target while retaining compatibility with internal callers."""
    result, pending = _prepare_candidate_evaluation(
        recipient,
        target_candidate,
        message_text,
        sender_snapshot,
        sender_history,
        aliases,
        round_number,
    )
    if pending is None:
        return result
    probabilities = _cached_entailment_probabilities(pending.pairs)
    unit_scores = [
        probabilities[index] - probabilities[index + 1]
        for index in range(0, len(probabilities), 2)
    ]
    if unit_scores:
        pending.evidence.append(
            (
                max(-1.0, min(1.0, sum(unit_scores) / len(unit_scores))),
                1.0,
                "candidate_nli",
            )
        )
    _finalize_candidate_evaluation(result, pending.evidence)
    return result


def _matrix_evaluation(
    snapshot: AgentRoundSnapshot,
    target_candidate: str,
) -> Optional[float]:
    """Return target utility relative to the sender's assessed baseline."""
    utilities: Dict[str, float] = {}
    for assessment in snapshot.candidate_assessments:
        if (
            assessment.candidate_id is not None
            and assessment.weighted_utility is not None
        ):
            utilities[assessment.candidate_id] = assessment.weighted_utility

    target_utility = utilities.get(target_candidate)
    if target_utility is None:
        return None

    if len(utilities) == 1:
        return max(-1.0, min(1.0, (target_utility - 2.5) / 2.5))

    baseline = sum(utilities.values()) / len(utilities)
    return max(-1.0, min(1.0, (target_utility - baseline) / 2.5))


def _switched_away_from(
    history: Sequence[AgentRoundSnapshot],
    round_number: int,
    target_candidate: str,
    current: AgentRoundSnapshot,
) -> bool:
    previous = next(
        (
            snapshot
            for snapshot in history
            if snapshot.round_number == round_number - 1
        ),
        None,
    )
    return (
        previous is not None
        and previous.candidate == target_candidate
        and current.candidate != target_candidate
    )


def _candidate_aliases(discussion: Discussion) -> Dict[str, Set[str]]:
    aliases: Dict[str, Set[str]] = defaultdict(set)
    for agent_snapshots in discussion.snapshots.values():
        for snapshot in agent_snapshots:
            if snapshot.candidate:
                aliases[snapshot.candidate].update(_alias_forms(snapshot.candidate))
                for raw in (snapshot.raw_architecture, snapshot.raw_position):
                    if raw:
                        aliases[snapshot.candidate].update(_alias_forms(raw))
            for assessment in snapshot.candidate_assessments:
                if assessment.candidate_id:
                    aliases[assessment.candidate_id].update(
                        _alias_forms(assessment.candidate_id)
                    )
                    if assessment.raw_candidate_id:
                        aliases[assessment.candidate_id].update(
                            _alias_forms(assessment.raw_candidate_id)
                        )
    return aliases


def _learn_message_aliases(
    discussion: Discussion, aliases: Dict[str, Set[str]]
) -> None:
    """Learn definitions such as `Bitemporal ... Memory (BHGV)` globally."""
    for message in discussion.messages:
        if not message.content:
            continue
        for match in ALIAS_DEFINITION_RE.finditer(str(message.content)):
            long_form = _normalize_phrase(match.group("long"))
            short_form = _normalize_phrase(match.group("short"))
            if not long_form or not short_form:
                continue

            candidates: List[Tuple[int, str]] = []
            for candidate, forms in aliases.items():
                for form in forms:
                    if " " not in form:
                        continue
                    if _contains_alias(long_form, form):
                        candidates.append((len(form), candidate))

            if not candidates:
                continue
            longest = max(length for length, _ in candidates)
            matched_candidates = {
                candidate for length, candidate in candidates if length == longest
            }
            if len(matched_candidates) == 1:
                aliases[next(iter(matched_candidates))].add(short_form)


def _alias_forms(raw: str) -> Set[str]:
    cleaned = _normalize_phrase(raw)
    if not cleaned:
        return set()
    forms = {cleaned}
    words = cleaned.split()
    if len(words) >= 3:
        acronym = "".join(word[0] for word in words if word)
        if len(acronym) >= 3:
            forms.add(acronym)
    return forms


def _normalize_phrase(raw: str) -> str:
    separated = raw.lower().replace("_", " ").replace("-", " ").replace("/", " ")
    return " ".join(WORD_RE.findall(separated))


def _targeted_nli_pairs(
    text: str,
    target_candidate: str,
    candidate_aliases: Set[str],
) -> List[Tuple[str, str]]:
    units = _candidate_units(text, candidate_aliases)
    if not units:
        return []

    display_candidate = target_candidate.replace("_", " ")
    support_hypothesis = f"The author supports choosing {display_candidate}."
    oppose_hypothesis = f"The author opposes choosing {display_candidate}."
    glossary = _candidate_glossary(display_candidate, candidate_aliases)

    pairs: List[Tuple[str, str]] = []
    for unit in units:
        premise = f"{glossary}Message excerpt: {unit}"
        pairs.append((premise, support_hypothesis))
        pairs.append((premise, oppose_hypothesis))
    return pairs


def _cached_entailment_probabilities(
    pairs: Sequence[Tuple[str, str]],
) -> List[float]:
    """Score unique pairs in full batches and retain a bounded cross-round cache."""
    if not pairs:
        return []

    classifier = _get_nli_classifier()
    keys = [(classifier, premise, hypothesis) for premise, hypothesis in pairs]
    values: Dict[Tuple[object, str, str], float] = {}
    missing_keys: List[Tuple[object, str, str]] = []
    missing_pairs: List[Tuple[str, str]] = []
    seen_missing: Set[Tuple[object, str, str]] = set()

    with _NLI_SCORE_CACHE_LOCK:
        for key, pair in zip(keys, pairs):
            cached = _NLI_SCORE_CACHE.get(key)
            if cached is not None:
                values[key] = cached
                _NLI_SCORE_CACHE.move_to_end(key)
            elif key not in seen_missing:
                seen_missing.add(key)
                missing_keys.append(key)
                missing_pairs.append(pair)

    if missing_pairs:
        # The singleton model may be shared by several discussion workers.
        # Serializing only the expensive forward pass avoids unsafe concurrent
        # access to one GPU model while leaving preparation and aggregation free.
        with _NLI_INFERENCE_LOCK:
            # Another worker may have filled these entries while this worker
            # waited for the model. Recheck to avoid duplicate forward passes.
            still_missing_keys: List[Tuple[object, str, str]] = []
            still_missing_pairs: List[Tuple[str, str]] = []
            with _NLI_SCORE_CACHE_LOCK:
                for key, pair in zip(missing_keys, missing_pairs):
                    cached = _NLI_SCORE_CACHE.get(key)
                    if cached is None:
                        still_missing_keys.append(key)
                        still_missing_pairs.append(pair)
                    else:
                        values[key] = cached
                        _NLI_SCORE_CACHE.move_to_end(key)
            inferred = (
                classifier.entailment_probabilities(still_missing_pairs)
                if still_missing_pairs
                else []
            )
            if len(inferred) != len(still_missing_pairs):
                raise RuntimeError(
                    "NLI classifier returned an unexpected number of scores"
                )
            with _NLI_SCORE_CACHE_LOCK:
                for key, value in zip(still_missing_keys, inferred):
                    score = float(value)
                    values[key] = score
                    _NLI_SCORE_CACHE[key] = score
                    _NLI_SCORE_CACHE.move_to_end(key)
                while len(_NLI_SCORE_CACHE) > _NLI_SCORE_CACHE_SIZE:
                    _NLI_SCORE_CACHE.popitem(last=False)

    return [values[key] for key in keys]


def _targeted_nli_evaluation(
    text: str,
    target_candidate: str,
    candidate_aliases: Set[str],
) -> Optional[float]:
    pairs = _targeted_nli_pairs(text, target_candidate, candidate_aliases)
    if not pairs:
        return None

    probabilities = _cached_entailment_probabilities(pairs)

    unit_scores = [
        probabilities[index] - probabilities[index + 1]
        for index in range(0, len(probabilities), 2)
    ]
    return max(-1.0, min(1.0, sum(unit_scores) / len(unit_scores)))


def _candidate_glossary(display_candidate: str, aliases: Set[str]) -> str:
    compact_aliases = sorted(
        alias.upper()
        for alias in aliases
        if " " not in alias and alias != display_candidate.replace(" ", "")
    )
    if not compact_aliases:
        return ""
    joined = ", ".join(compact_aliases)
    return f"Candidate glossary: {joined} means {display_candidate}. "


def _candidate_units(text: str, candidate_aliases: Set[str]) -> List[str]:
    normalized_aliases = {
        alias.lower().strip()
        for alias in candidate_aliases
        if alias and len(alias.strip()) >= 3
    }
    units: List[str] = []
    seen = set()
    for raw_unit in re.split(r"\n+|(?<=[.!?])\s+", text):
        original = " ".join(raw_unit.strip().split())
        normalized = _normalize_phrase(raw_unit)
        if not normalized:
            continue
        if any(_contains_alias(normalized, alias) for alias in normalized_aliases):
            if original not in seen:
                seen.add(original)
                units.append(original)
    return units


def _contains_alias(text: str, alias: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text) is not None
