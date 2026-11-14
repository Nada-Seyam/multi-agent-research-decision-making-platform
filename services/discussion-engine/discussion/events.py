"""Versioned lifecycle events for incremental discussion consumers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .models import SCHEMA_VERSION


class DiscussionEventType(str, Enum):
    STARTED = "discussion.started"
    ROUND_COMPLETED = "round.completed"
    COMPLETED = "discussion.completed"
    FAILED = "discussion.failed"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def snapshot_sha256(snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(
        snapshot,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DiscussionEvent:
    schema_version: str
    event_id: str
    event_type: DiscussionEventType
    discussion_id: str
    sequence: int
    occurred_at: str
    snapshot_sha256: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "discussion_id": self.discussion_id,
            "sequence": self.sequence,
            "occurred_at": self.occurred_at,
            "snapshot_sha256": self.snapshot_sha256,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DiscussionEvent:
        return cls(
            schema_version=str(value["schema_version"]),
            event_id=str(value["event_id"]),
            event_type=DiscussionEventType(value["event_type"]),
            discussion_id=str(value["discussion_id"]),
            sequence=int(value["sequence"]),
            occurred_at=str(value["occurred_at"]),
            snapshot_sha256=str(value["snapshot_sha256"]),
            data=dict(value["data"]),
        )


def make_discussion_event(
    event_type: DiscussionEventType,
    snapshot: dict[str, Any],
    *,
    sequence: int,
    round_data: dict[str, Any] | None = None,
) -> DiscussionEvent:
    discussion_id = str(snapshot["discussion_id"])
    if event_type is DiscussionEventType.ROUND_COMPLETED:
        if round_data is None:
            raise ValueError("A round.completed event requires round_data.")
        event_id = f"{discussion_id}:round:{round_data['round']}"
    else:
        suffix = event_type.value.split(".", 1)[1]
        event_id = f"{discussion_id}:{suffix}"

    data: dict[str, Any] = {"discussion": snapshot}
    if round_data is not None:
        data["completed_round"] = round_data["round"]
        data["round"] = round_data

    return DiscussionEvent(
        schema_version=SCHEMA_VERSION,
        event_id=event_id,
        event_type=event_type,
        discussion_id=discussion_id,
        sequence=sequence,
        occurred_at=utc_now(),
        snapshot_sha256=snapshot_sha256(snapshot),
        data=data,
    )
