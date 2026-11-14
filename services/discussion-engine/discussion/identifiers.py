"""Stable and unique identifiers for discussion runs."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from .config import DiscussionSettings


def objective_id(objective: Any) -> str:
    short_id = getattr(objective, "short_id", None)
    value = short_id() if callable(short_id) else short_id
    return str(value or "discussion")


def discussion_id(
    objective: Any,
    participants: list[str],
    settings: DiscussionSettings,
) -> str:
    identity = {
        "goal": getattr(objective, "goal", str(objective)),
        "participants": participants,
        "seed": settings.random_seed,
        "rounds": settings.num_rounds,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    return f"{objective_id(objective)}-{digest}-{uuid.uuid4().hex[:12]}"
