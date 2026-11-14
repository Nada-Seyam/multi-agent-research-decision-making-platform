"""Replay durable discussion events without rerunning the discussion."""

from __future__ import annotations

from dataclasses import dataclass

from .artifacts import FileDiscussionArtifactStore
from .event_sinks import DiscussionEventSink


@dataclass(frozen=True)
class ReplaySummary:
    attempted: int
    delivered: int
    pending: int


def replay_pending_events(
    discussion_id: str,
    store: FileDiscussionArtifactStore,
    sink: DiscussionEventSink,
) -> ReplaySummary:
    attempted = 0
    delivered = 0
    pending = 0
    for event in store.pending_events(discussion_id):
        attempted += 1
        result = sink.publish(event)
        store.record_delivery(
            event.event_id,
            status=result.status,
            attempts=result.attempts,
            last_error=result.last_error,
            status_code=result.status_code,
            attempted_at=result.attempted_at,
        )
        if result.delivered:
            delivered += 1
        else:
            pending += 1
    return ReplaySummary(attempted=attempted, delivered=delivered, pending=pending)
