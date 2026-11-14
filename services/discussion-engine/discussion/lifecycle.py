"""Create, persist, and publish incremental discussion snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import (
    DiscussionArtifactStore,
    FileDiscussionArtifactStore,
    NullDiscussionArtifactStore,
    atomic_write_text,
)
from .config import DiscussionSettings
from .event_sinks import DiscussionEventSink, EventDeliveryError
from .events import DiscussionEventType, make_discussion_event, utc_now
from .models import DiscussionResult, DiscussionRound


def _log_root(settings: DiscussionSettings, discussion_id: str) -> Path:
    return settings.resolve_logs_dir() / discussion_id


def _persist_logs(result: DiscussionResult, settings: DiscussionSettings) -> None:
    if result.research is None:
        return
    log_dir = _log_root(settings, result.discussion_id)
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = result.logs_payload()
    atomic_write_text(
        log_dir / "details.json",
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
    )

    dossier_path = log_dir / "dossier.json"
    if not dossier_path.exists():
        atomic_write_text(
            dossier_path,
            json.dumps(
                payload["research"]["dossier"],
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
        )
    rejected_path = log_dir / "rejected_evidence.jsonl"
    if not rejected_path.exists():
        atomic_write_text(
            rejected_path,
            "".join(
                json.dumps(item, ensure_ascii=False, default=str) + "\n"
                for item in result.research.rejected_evidence
            ),
        )
    atomic_write_text(
        log_dir / "messages.jsonl",
        "".join(
            json.dumps(message.to_dict(), ensure_ascii=False, default=str) + "\n"
            for discussion_round in result.rounds
            for message in discussion_round.messages
        ),
    )


def _runtime_configuration(
    settings: DiscussionSettings,
    proposed_alternatives: dict[str, str],
) -> dict[str, Any]:
    configuration = settings.model_dump(mode="json", exclude={"event_token"})
    configuration["solution_space"] = "open"
    configuration["proposed_alternatives"] = list(
        dict.fromkeys(proposed_alternatives.values())
    )
    configuration["candidate_aliases"] = {
        alias: canonical
        for alias, canonical in proposed_alternatives.items()
        if alias != canonical
    }
    return configuration


def _result_log_paths(
    settings: DiscussionSettings,
    discussion_id: str,
) -> dict[str, str]:
    log_root = _log_root(settings, discussion_id)
    return {
        "dossier": (log_root / "dossier.json").as_posix(),
        "rejected_evidence": (log_root / "rejected_evidence.jsonl").as_posix(),
        "messages": (log_root / "messages.jsonl").as_posix(),
        "details": (log_root / "details.json").as_posix(),
        "events": (log_root / "events.jsonl").as_posix(),
        "delivery_state": (log_root / "delivery-state.json").as_posix(),
    }


def build_result_snapshot(
    *,
    discussion_id: str,
    objective: Any,
    participants: list[str],
    graph: dict[str, Any],
    research: Any,
    rounds: list[DiscussionRound],
    settings: DiscussionSettings,
    proposed_alternatives: dict[str, str],
    status: str,
    started_at: str,
    decision_summary: Any = None,
    conclusion: str | None = None,
    completed_at: str | None = None,
    failure: dict[str, Any] | None = None,
    warnings: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    preparation: dict[str, Any] | None = None,
) -> DiscussionResult:
    return DiscussionResult(
        discussion_id=discussion_id,
        objective=objective,
        participants=tuple(participants),
        graph=graph,
        research=research,
        rounds=tuple(rounds),
        preparation=preparation,
        configuration=_runtime_configuration(settings, proposed_alternatives),
        logs=_result_log_paths(settings, discussion_id),
        decision_summary=decision_summary,
        conclusion=conclusion,
        status=status,
        started_at=started_at,
        updated_at=utc_now(),
        completed_at=completed_at,
        failure=failure,
        warnings=tuple(warnings),
    )


def default_artifact_store(settings: DiscussionSettings) -> DiscussionArtifactStore:
    if not settings.persist_artifacts:
        return NullDiscussionArtifactStore()
    return FileDiscussionArtifactStore(
        settings.resolve_output_dir(),
        settings.resolve_logs_dir(),
    )


class DiscussionLifecycle:
    """Own durable snapshot persistence and at-least-once event delivery."""

    def __init__(
        self,
        settings: DiscussionSettings,
        event_sink: DiscussionEventSink,
        artifact_store: DiscussionArtifactStore,
    ):
        self.settings = settings
        self.event_sink = event_sink
        self.artifact_store = artifact_store

    def checkpoint(self, result: DiscussionResult) -> None:
        """Persist a live snapshot without adding a durable replay event.

        Preparation can update several times before the first discussion round.
        These checkpoints make that work visible to polling clients while the
        lifecycle journal remains focused on replayable discussion boundaries.
        """
        self.artifact_store.save_snapshot(result.to_dict())

    def publish(
        self,
        result: DiscussionResult,
        event_type: DiscussionEventType,
        *,
        sequence: int,
        round_data: dict[str, Any] | None = None,
        require_delivery: bool | None = None,
    ) -> None:
        snapshot = result.to_dict()
        event = make_discussion_event(
            event_type,
            snapshot,
            sequence=sequence,
            round_data=round_data,
        )
        # Journal the complete snapshot first. If a later snapshot write is
        # interrupted, replay still has the state needed to recover it.
        self.artifact_store.append_event(event)
        self.artifact_store.save_snapshot(snapshot)
        if self.settings.persist_artifacts and result.research is not None:
            _persist_logs(result, self.settings)
        delivery = self.event_sink.publish(event)
        self.artifact_store.record_delivery(
            event.event_id,
            status=delivery.status,
            attempts=delivery.attempts,
            last_error=delivery.last_error,
            status_code=delivery.status_code,
            attempted_at=delivery.attempted_at,
        )
        must_deliver = (
            self.settings.event_delivery_required
            if require_delivery is None
            else require_delivery
        )
        if must_deliver and not delivery.delivered:
            raise EventDeliveryError(
                f"Required delivery failed for event '{event.event_id}': "
                f"{delivery.last_error or delivery.status}."
            )
