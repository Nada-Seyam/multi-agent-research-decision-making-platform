"""Event-driven orchestration for incremental and final analytics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any

from .data_loader import DiscussionValidationError, load_discussion
from .engine import analyze_discussion, analyze_round
from .streaming_events import ProducerEvent, make_analysis_event, utc_now
from .streaming_store import AnalyticsArtifactStore


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    return value


@dataclass(frozen=True)
class EventProcessingResult:
    status: str
    event: ProducerEvent
    artifact_path: Path
    analysis_event: dict[str, Any] | None = None
    should_process: bool = False


class StreamingAnalyticsService:
    """Consume producer events idempotently and update one durable artifact."""

    def __init__(self, store: AnalyticsArtifactStore):
        self.store = store
        self._locks_guard = Lock()
        self._ingest_locks: dict[str, Lock] = {}
        self._processing_locks: dict[str, Lock] = {}

    def _lock_for(self, locks: dict[str, Lock], discussion_id: str) -> Lock:
        with self._locks_guard:
            return locks.setdefault(discussion_id, Lock())

    @staticmethod
    def _matching_analysis_event(
        artifact: dict[str, Any], event_id: str
    ) -> dict[str, Any] | None:
        return next(
            (
                item
                for item in artifact.get("analysis_events", [])
                if item.get("source_event_id") == event_id
            ),
            None,
        )

    @staticmethod
    def _new_artifact(discussion_id: str) -> dict[str, Any]:
        now = utc_now()
        return {
            "schema_version": "1.0",
            "discussion_id": discussion_id,
            "status": "pending",
            "created_at": now,
            "updated_at": now,
            "last_source_sequence": -1,
            "last_source_event": None,
            "round_analyses": [],
            "final_analysis": None,
            "failure": None,
            "processing_failures": {},
            "analysis_events": [],
            "last_analysis_event": None,
            "processed_source_events": [],
        }

    def ingest(
        self,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> EventProcessingResult:
        """Durably accept an event without running expensive analytics."""
        event = ProducerEvent.from_dict(payload)
        if idempotency_key != event.event_id:
            raise ValueError("Idempotency-Key must match event_id")
        if event.event_type in {"round.completed", "discussion.completed"}:
            # Reject malformed snapshots before acknowledging them. This parse
            # is cheap compared with NLI inference and preserves the API's 422
            # contract while the expensive analysis stays off the request path.
            load_discussion(event.discussion)

        with self._lock_for(self._ingest_locks, event.discussion_id):
            existing = self.store.read_event(event.discussion_id, event.event_id)
            artifact = self.store.read_artifact(event.discussion_id) or {}
            processed = set(artifact.get("processed_source_events", []))
            if existing is not None:
                if existing != payload:
                    raise EventConflictError(
                        "event_id was already used for different content"
                    )
                return EventProcessingResult(
                    status="duplicate",
                    event=event,
                    artifact_path=self.store.artifact_path(event.discussion_id),
                    analysis_event=self._matching_analysis_event(
                        artifact, event.event_id
                    ),
                    should_process=event.event_id not in processed,
                )

            self.store.save_event(payload)
            return EventProcessingResult(
                status="accepted",
                event=event,
                artifact_path=self.store.artifact_path(event.discussion_id),
                should_process=True,
            )

    def process_ingested(
        self,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> EventProcessingResult:
        """Process one previously persisted event, idempotently."""
        event = ProducerEvent.from_dict(payload)
        if idempotency_key != event.event_id:
            raise ValueError("Idempotency-Key must match event_id")

        with self._lock_for(self._processing_locks, event.discussion_id):
            artifact = self.store.read_artifact(event.discussion_id) or {}
            processed = list(artifact.get("processed_source_events", []))
            if event.event_id in processed:
                return EventProcessingResult(
                    status="duplicate",
                    event=event,
                    artifact_path=self.store.artifact_path(event.discussion_id),
                    analysis_event=self._matching_analysis_event(
                        artifact, event.event_id
                    ),
                )

            artifact, analysis_event = self._apply_event(event)
            processing_failures = dict(
                artifact.get("processing_failures", {})
            )
            processing_failures.pop(event.event_id, None)
            artifact["processing_failures"] = processing_failures
            if processing_failures:
                artifact["status"] = (
                    "completed_with_errors"
                    if artifact.get("final_analysis") is not None
                    else "analysis_failed"
                )
            elif artifact.get("status") == "completed_with_errors":
                artifact["status"] = "completed"
            processed = list(artifact.get("processed_source_events", []))
            if event.event_id not in processed:
                processed.append(event.event_id)
            artifact["processed_source_events"] = processed
            artifact_path = self.store.save_artifact(artifact)
            return EventProcessingResult(
                status="accepted",
                event=event,
                artifact_path=artifact_path,
                analysis_event=analysis_event,
            )

    def process_ingested_safely(
        self,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> EventProcessingResult:
        """Process a background event and persist failures for later retry."""
        try:
            return self.process_ingested(payload, idempotency_key)
        except Exception as exc:
            event = ProducerEvent.from_dict(payload)
            with self._lock_for(self._processing_locks, event.discussion_id):
                artifact = self.store.read_artifact(
                    event.discussion_id
                ) or self._new_artifact(event.discussion_id)
                failures = dict(artifact.get("processing_failures", {}))
                previous = failures.get(event.event_id, {})
                failures[event.event_id] = {
                    "event_type": event.event_type,
                    "source_sequence": event.sequence,
                    "attempts": int(previous.get("attempts", 0)) + 1,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "last_failed_at": utc_now(),
                }
                artifact["processing_failures"] = failures
                artifact["status"] = (
                    "completed_with_errors"
                    if artifact.get("final_analysis") is not None
                    else "analysis_failed"
                )
                artifact["updated_at"] = utc_now()
                artifact_path = self.store.save_artifact(artifact)
            return EventProcessingResult(
                status="processing_failed",
                event=event,
                artifact_path=artifact_path,
            )

    def process(
        self,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> EventProcessingResult:
        accepted = self.ingest(payload, idempotency_key)
        if not accepted.should_process:
            return accepted
        return self.process_ingested(payload, idempotency_key)

    def recover_pending(self) -> int:
        """Replay accepted-but-unprocessed events after a service restart."""
        recovered = 0
        for payload in self.store.pending_event_payloads():
            event_id = str(payload.get("event_id", ""))
            result = self.process_ingested_safely(payload, event_id)
            if result.status != "processing_failed":
                recovered += 1
        return recovered

    def _apply_event(
        self, event: ProducerEvent
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        artifact = self.store.read_artifact(
            event.discussion_id
        ) or self._new_artifact(event.discussion_id)
        previous_status = str(artifact.get("status", "pending"))
        previous_sequence = int(artifact.get("last_source_sequence", -1))
        is_latest_event = event.sequence >= previous_sequence

        analysis_event: dict[str, Any] | None = None
        if event.event_type == "round.completed":
            discussion = load_discussion(event.discussion)
            completed_round = int(event.data["completed_round"])
            analysis = jsonable(analyze_round(discussion, completed_round))
            analysis.update(
                {
                    "scope": "round",
                    "generated_at": utc_now(),
                    "source_event_id": event.event_id,
                    "source_sequence": event.sequence,
                }
            )
            round_analyses = [
                item
                for item in artifact.get("round_analyses", [])
                if item.get("round") != completed_round
            ]
            round_analyses.append(analysis)
            artifact["round_analyses"] = sorted(
                round_analyses, key=lambda item: item["round"]
            )
            # Background consumers for consecutive HTTP requests may acquire
            # the per-discussion lock out of sequence. If the final analysis
            # has already landed, immediately re-anchor this late round too.
            if artifact.get("final_analysis") is not None:
                self._finalize_round_analyses(
                    artifact,
                    artifact["final_analysis"],
                )
            if previous_status not in {"completed", "failed"}:
                artifact["status"] = "running"
            analysis_event = make_analysis_event(
                event, "analysis.round.completed", analysis
            )

        elif event.event_type == "discussion.completed":
            discussion = load_discussion(event.discussion)
            analysis = jsonable(analyze_discussion(discussion))
            analysis.update(
                {
                    "scope": "discussion",
                    "generated_at": utc_now(),
                    "source_event_id": event.event_id,
                    "source_sequence": event.sequence,
                    "final_only_metrics": ["influence"],
                    "anchor_candidate": discussion.preferred_candidate,
                    "anchor_source": discussion.preferred_candidate_source,
                }
            )
            artifact["final_analysis"] = analysis
            self._finalize_round_analyses(artifact, analysis)
            if is_latest_event:
                artifact["status"] = "completed"
            analysis_event = make_analysis_event(
                event, "analysis.discussion.completed", analysis
            )

        elif event.event_type == "discussion.failed":
            if is_latest_event:
                artifact["status"] = "failed"
                artifact["failure"] = event.discussion.get("failure")
        elif previous_status not in {"completed", "failed"}:
            artifact["status"] = "running"

        if is_latest_event:
            artifact["last_source_sequence"] = event.sequence
            artifact["last_source_event"] = {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "occurred_at": event.occurred_at,
                "snapshot_sha256": event.snapshot_sha256,
            }
        artifact["updated_at"] = utc_now()
        if analysis_event is not None:
            analysis_events = [
                item
                for item in artifact.get("analysis_events", [])
                if item.get("source_event_id") != event.event_id
            ]
            analysis_events.append(analysis_event)
            artifact["analysis_events"] = sorted(
                analysis_events,
                key=lambda item: item.get("source_sequence", -1),
            )
            artifact["last_analysis_event"] = analysis_event
        return artifact, analysis_event

    @staticmethod
    def _finalize_round_analyses(
        artifact: dict[str, Any],
        final_analysis: dict[str, Any],
    ) -> None:
        """Re-anchor every stored round to the final decision consistently."""
        stances_by_round: dict[int, dict[str, Any]] = {}
        for agent_id, records in final_analysis.get("opinion_change", {}).items():
            for record in records:
                stances_by_round.setdefault(int(record["round"]), {})[agent_id] = record
        agreement_by_round = {
            int(record["round"]): record
            for record in final_analysis.get("agreement", [])
        }
        sentiment_by_round: dict[int, list[dict[str, Any]]] = {}
        for record in final_analysis.get("sentiment", []):
            sentiment_by_round.setdefault(int(record["round"]), []).append(record)

        for round_analysis in artifact.get("round_analyses", []):
            round_number = int(round_analysis["round"])
            round_analysis["anchor_candidate"] = final_analysis.get(
                "anchor_candidate"
            )
            round_analysis["anchor_source"] = final_analysis.get(
                "anchor_source", "decision_summary"
            )
            round_analysis["provisional_anchor"] = False
            round_analysis["stances"] = stances_by_round.get(round_number, {})
            round_analysis["agreement"] = agreement_by_round.get(round_number)
            round_analysis["sentiment"] = sentiment_by_round.get(round_number, [])


class EventConflictError(ValueError):
    """Raised when an event ID is reused for different content."""


__all__ = [
    "DiscussionValidationError",
    "EventConflictError",
    "EventProcessingResult",
    "StreamingAnalyticsService",
]
