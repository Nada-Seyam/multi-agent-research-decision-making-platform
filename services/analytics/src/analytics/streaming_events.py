"""Validation and construction of producer and analytics events."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


SUPPORTED_EVENT_TYPES = {
    "discussion.started",
    "round.completed",
    "discussion.completed",
    "discussion.failed",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ProducerEvent:
    schema_version: str
    event_id: str
    event_type: str
    discussion_id: str
    sequence: int
    occurred_at: str
    snapshot_sha256: str
    data: dict[str, Any]

    @property
    def discussion(self) -> dict[str, Any]:
        return self.data["discussion"]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProducerEvent":
        if not isinstance(payload, dict):
            raise ValueError("event payload must be a JSON object")
        try:
            event = cls(
                schema_version=str(payload["schema_version"]),
                event_id=str(payload["event_id"]),
                event_type=str(payload["event_type"]),
                discussion_id=str(payload["discussion_id"]),
                sequence=int(payload["sequence"]),
                occurred_at=str(payload["occurred_at"]),
                snapshot_sha256=str(payload["snapshot_sha256"]),
                data=dict(payload["data"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed producer event: {exc}") from exc

        if event.schema_version != "1.0":
            raise ValueError(
                f"unsupported schema_version '{event.schema_version}'"
            )
        if event.event_type not in SUPPORTED_EVENT_TYPES:
            raise ValueError(f"unsupported event_type '{event.event_type}'")
        if not event.event_id or not event.discussion_id:
            raise ValueError("event_id and discussion_id must not be empty")
        if event.sequence < 0:
            raise ValueError("sequence must be non-negative")

        discussion = event.data.get("discussion")
        if not isinstance(discussion, dict):
            raise ValueError("data.discussion must be a JSON object")
        if str(discussion.get("discussion_id")) != event.discussion_id:
            raise ValueError(
                "event discussion_id does not match data.discussion.discussion_id"
            )
        if canonical_sha256(discussion) != event.snapshot_sha256:
            raise ValueError("snapshot_sha256 does not match data.discussion")

        if event.event_type == "round.completed":
            round_data = event.data.get("round")
            completed_round = event.data.get("completed_round")
            if not isinstance(round_data, dict):
                raise ValueError("round.completed requires data.round")
            if round_data.get("round") != completed_round:
                raise ValueError(
                    "data.completed_round must match data.round.round"
                )
        return event


def make_analysis_event(
    source: ProducerEvent,
    event_type: str,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "event_id": f"{source.event_id}:analysis",
        "event_type": event_type,
        "discussion_id": source.discussion_id,
        "source_event_id": source.event_id,
        "source_sequence": source.sequence,
        "occurred_at": utc_now(),
        "data": {"analysis": analysis},
    }
