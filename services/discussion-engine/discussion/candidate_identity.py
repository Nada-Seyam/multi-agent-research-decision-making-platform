from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Iterable, Mapping


_NON_DISTINGUISHING_TOKENS = {
    "approach",
    "architecture",
    "base",
    "baseline",
    "canonical",
    "classic",
    "configuration",
    "conventional",
    "default",
    "design",
    "family",
    "framework",
    "mechanism",
    "method",
    "model",
    "reference",
    "solution",
    "standard",
    "system",
    "traditional",
    "vanilla",
}

_EQUIVALENT_PHRASES = {
    ("mixture", "of", "experts"): "moe",
}

_RECOMMENDATION_PATTERNS = (
    re.compile(
        r"\bi\s+recommend\s+(?:\\?\*\*|__)?"
        r"(?P<candidate>[A-Za-z][A-Za-z0-9 _/\-]{1,80}?)"
        r"(?=(?:\\?\*\*|__)?[.,;:\n])",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:my\s+)?(?:current\s+|final\s+)?recommendation\s+"
        r"(?:is|remains)\s+(?:\\?\*\*|__)?"
        r"(?P<candidate>[A-Za-z][A-Za-z0-9 _/\-]{1,80}?)"
        r"(?=(?:\\?\*\*|__)?[.,;:\n])",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:primary|final)\s+recommendation\s*:\s*"
        r"(?:\\?\*\*|__)?"
        r"(?P<candidate>[A-Za-z][A-Za-z0-9 _/\-]{1,80}?)"
        r"(?=(?:\\?\*\*|__)?[.,;:\n]|(?:\\?\*\*|__)\s*$|\s*$)",
        re.IGNORECASE | re.MULTILINE,
    ),
)


def candidate_key(value: object) -> str | None:
    """Return the stable lowercase identifier used by the discussion schema."""
    if not isinstance(value, str) or not value.strip():
        return None
    key = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return key or None


def _signature(candidate: str) -> tuple[str, ...]:
    raw_tokens = tuple(token for token in candidate.split("_") if token)
    tokens: list[str] = []
    index = 0
    while index < len(raw_tokens):
        match = next(
            (
                (phrase, replacement)
                for phrase, replacement in _EQUIVALENT_PHRASES.items()
                if raw_tokens[index : index + len(phrase)] == phrase
            ),
            None,
        )
        if match is None:
            tokens.append(raw_tokens[index])
            index += 1
            continue
        phrase, replacement = match
        tokens.append(replacement)
        index += len(phrase)
    meaningful = tuple(
        token for token in tokens if token not in _NON_DISTINGUISHING_TOKENS
    )
    # Never collapse a name made entirely from qualifiers into an empty group.
    # Agents often emit the same composite architecture with its components in
    # a different order. The order of an identifier is presentation variance,
    # so compare the complete token multiset in a stable order.
    return tuple(sorted(meaningful or tuple(tokens)))


def build_candidate_aliases(
    candidates: Iterable[object],
    *,
    preferred: Iterable[object] = (),
) -> dict[str, str]:
    """Build alias -> canonical mappings without fuzzy semantic matching.

    Existing registry values are preferred so a canonical id remains stable
    across rounds.  For a new group, the most frequently used spelling wins;
    ties prefer the more descriptive identifier.
    """
    normalized = [key for value in candidates if (key := candidate_key(value))]
    if not normalized:
        return {}

    counts = Counter(normalized)
    preferred_keys = [
        key for value in preferred if (key := candidate_key(value))
    ]
    preferred_order = {key: index for index, key in enumerate(preferred_keys)}
    groups: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for candidate in normalized:
        groups[_signature(candidate)].add(candidate)

    aliases: dict[str, str] = {}
    for members in groups.values():
        established = [member for member in members if member in preferred_order]
        if established:
            canonical = min(established, key=lambda item: preferred_order[item])
        else:
            canonical = max(
                members,
                key=lambda item: (
                    counts[item],
                    len(item.split("_")),
                    len(item),
                    item,
                ),
            )
        aliases.update({member: canonical for member in members})
    return aliases


def canonical_candidate(
    value: object,
    aliases: Mapping[str, str] | None = None,
) -> str | None:
    """Normalize one candidate and follow a supplied alias chain."""
    candidate = candidate_key(value)
    if candidate is None:
        return None
    mapping = aliases or {}
    seen: set[str] = set()
    while candidate in mapping and candidate not in seen:
        seen.add(candidate)
        candidate = mapping[candidate]
    return candidate


def recommendation_refinement(
    current_candidate: object,
    message: object,
) -> str | None:
    """Return a more specific candidate explicitly recommended in prose.

    Refinement is allowed only when every token in the structured candidate is
    present in the recommendation.  Thus ``attention`` may be refined to
    ``standard_softmax_attention`` when the author explicitly recommends it,
    while unrelated candidates mentioned elsewhere in the response cannot
    replace the structured vote.
    """
    current = candidate_key(current_candidate)
    if current is None or not isinstance(message, str):
        return None
    current_tokens = set(current.split("_"))
    # Refinement exists for genuinely generic structured labels such as
    # ``attention``.  A detailed structured candidate is authoritative; prose
    # may append words such as "architecture" without defining a new option.
    if len(_signature(current)) != 1:
        return None
    for pattern in _RECOMMENDATION_PATTERNS:
        for match in pattern.finditer(message):
            recommended = candidate_key(match.group("candidate"))
            if recommended is None or recommended == current:
                continue
            recommended_tokens = set(recommended.split("_"))
            added_meaningful_tokens = {
                token
                for token in recommended_tokens - current_tokens
                if token not in _NON_DISTINGUISHING_TOKENS
            }
            if current_tokens < recommended_tokens and added_meaningful_tokens:
                return recommended
    return None


def explicit_recommendation(message: object) -> str | None:
    """Return a candidate only when the prose explicitly recommends it."""
    if not isinstance(message, str):
        return None
    matches: list[tuple[int, str]] = []
    for pattern in _RECOMMENDATION_PATTERNS:
        for match in pattern.finditer(message):
            candidate = candidate_key(match.group("candidate"))
            if candidate:
                matches.append((match.start(), candidate))
    return min(matches, default=(0, None), key=lambda item: item[0])[1]
