"""Small shared helpers used across the analytics modules."""

import re
from typing import Optional

_WHITESPACE_OR_HYPHEN = re.compile(r"[\s\-]+")
_NON_ID_CHARS = re.compile(r"[^a-z0-9_]")
_MULTI_UNDERSCORE = re.compile(r"_+")


def normalize_candidate_id(raw: Optional[str]) -> Optional[str]:
    """Normalize a candidate/position string into a canonical id.

    Week 3 opinion snapshots represent the same candidate in inconsistent
    forms across rounds and personas, e.g.:

        "Attention-SSM Hybrid"  ->  attention_ssm_hybrid
        "attention_ssm_hybrid"  ->  attention_ssm_hybrid
        "Bitemporal Hybrid Graph-Vector Memory" -> bitemporal_hybrid_graph_vector_memory

    Normalizing (lowercase, collapse whitespace/hyphens to underscores,
    strip anything else) lets us compare a persona's chosen candidate
    against `decision_summary.preferred_candidate` reliably, without
    relying on the free-text `criteria_scores` field or the frequently
    inconsistent `ranked_candidate` field (see README "Design Decisions").

    Returns None for missing/blank/non-string input.
    """
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip().lower()
    if not s:
        return None
    s = _WHITESPACE_OR_HYPHEN.sub("_", s)
    s = _NON_ID_CHARS.sub("", s)
    s = _MULTI_UNDERSCORE.sub("_", s).strip("_")
    return s or None


def clip_confidence(raw) -> Optional[float]:
    """Coerce a confidence value into a float in [0, 1], or None if invalid.

    Returns None (rather than a fabricated default) when the value is
    missing or not numeric, so callers can explicitly treat it as
    "insufficient data" instead of silently guessing a confidence.
    """
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if val != val:  # NaN check
        return None
    if not 0.0 <= val <= 1.0:
        return None
    return val
