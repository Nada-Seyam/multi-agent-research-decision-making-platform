"""Durable, atomic storage for discussion snapshots and delivery state."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol

from .events import DiscussionEvent


def atomic_write_text(path: Path, content: str) -> None:
    """Replace a text file without exposing a partially written document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _safe_id(discussion_id: str) -> str:
    if not discussion_id or Path(discussion_id).name != discussion_id:
        raise ValueError("discussion_id must be a non-empty file-safe name.")
    return discussion_id


class DiscussionArtifactStore(Protocol):
    def save_snapshot(self, snapshot: dict[str, Any]) -> Path | None: ...

    def append_event(self, event: DiscussionEvent) -> None: ...

    def record_delivery(
        self,
        event_id: str,
        *,
        status: str,
        attempts: int,
        last_error: str | None,
        status_code: int | None,
        attempted_at: str,
    ) -> None: ...

    def pending_events(self, discussion_id: str) -> list[DiscussionEvent]: ...

    def snapshot_path(self, discussion_id: str) -> Path | None: ...

    def list_snapshots(
        self,
        *,
        limit: int,
        offset: int,
        status: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]: ...

    def replay_timeline(self, discussion_id: str) -> dict[str, Any]: ...


class NullDiscussionArtifactStore:
    """No-op store for tests and explicitly non-persistent runs."""

    def save_snapshot(self, snapshot: dict[str, Any]) -> None:
        return None

    def append_event(self, event: DiscussionEvent) -> None:
        return None

    def record_delivery(
        self,
        event_id: str,
        *,
        status: str,
        attempts: int,
        last_error: str | None,
        status_code: int | None,
        attempted_at: str,
    ) -> None:
        return None

    def pending_events(self, discussion_id: str) -> list[DiscussionEvent]:
        return []

    def snapshot_path(self, discussion_id: str) -> None:
        return None

    def list_snapshots(
        self,
        *,
        limit: int,
        offset: int,
        status: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        return [], 0

    def replay_timeline(self, discussion_id: str) -> dict[str, Any]:
        return {"discussion_id": _safe_id(discussion_id), "frames": [], "warnings": []}


class FileDiscussionArtifactStore:
    def __init__(self, output_dir: Path, logs_dir: Path):
        self.output_dir = output_dir
        self.logs_dir = logs_dir

    def snapshot_path(self, discussion_id: str) -> Path:
        return self.output_dir / f"{_safe_id(discussion_id)}.json"

    def _log_dir(self, discussion_id: str) -> Path:
        return self.logs_dir / _safe_id(discussion_id)

    def _events_path(self, discussion_id: str) -> Path:
        return self._log_dir(discussion_id) / "events.jsonl"

    def _delivery_path(self, discussion_id: str) -> Path:
        return self._log_dir(discussion_id) / "delivery-state.json"

    def save_snapshot(self, snapshot: dict[str, Any]) -> Path:
        path = self.snapshot_path(str(snapshot["discussion_id"]))
        atomic_write_text(
            path,
            json.dumps(snapshot, ensure_ascii=False, indent=2, default=str),
        )
        return path

    def append_event(self, event: DiscussionEvent) -> None:
        path = self._events_path(event.discussion_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False, default=str))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def record_delivery(
        self,
        event_id: str,
        *,
        status: str,
        attempts: int,
        last_error: str | None,
        status_code: int | None,
        attempted_at: str,
    ) -> None:
        discussion_id = event_id.split(":", 1)[0]
        path = self._delivery_path(discussion_id)
        state: dict[str, Any] = {"events": {}}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("events"), dict):
                    state = loaded
            except (OSError, json.JSONDecodeError):
                state = {"events": {}}
        state["events"][event_id] = {
            "status": status,
            "attempts": attempts,
            "last_attempt_at": attempted_at,
            "last_error": last_error,
            "status_code": status_code,
        }
        atomic_write_text(path, json.dumps(state, ensure_ascii=False, indent=2))

    def pending_events(self, discussion_id: str) -> list[DiscussionEvent]:
        events_path = self._events_path(discussion_id)
        if not events_path.is_file():
            return []
        delivery_path = self._delivery_path(discussion_id)
        delivered: set[str] = set()
        if delivery_path.is_file():
            try:
                state = json.loads(delivery_path.read_text(encoding="utf-8"))
                delivered = {
                    event_id
                    for event_id, record in state.get("events", {}).items()
                    if record.get("status") == "delivered"
                }
            except (OSError, json.JSONDecodeError, AttributeError):
                delivered = set()

        events: list[DiscussionEvent] = []
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = DiscussionEvent.from_dict(json.loads(line))
            if event.event_id not in delivered:
                events.append(event)
        return sorted(events, key=lambda event: event.sequence)

    @staticmethod
    def _objective_title(snapshot: dict[str, Any]) -> str:
        objective = snapshot.get("objective")
        if isinstance(objective, dict):
            for key in ("goal", "final_goal", "title"):
                value = objective.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        if isinstance(objective, str) and objective.strip():
            return objective.strip().splitlines()[0]
        return "Untitled discussion"

    @classmethod
    def _history_item(cls, snapshot: dict[str, Any]) -> dict[str, Any]:
        progress = snapshot.get("progress")
        if not isinstance(progress, dict):
            progress = {}
        rounds = snapshot.get("rounds")
        round_count = len(rounds) if isinstance(rounds, list) else 0
        participants = snapshot.get("participants")
        if not isinstance(participants, list):
            participants = []
        decision = snapshot.get("decision_summary")
        if not isinstance(decision, dict):
            decision = {}
        conclusion = snapshot.get("conclusion")
        conclusion_preview = None
        if isinstance(conclusion, str) and conclusion.strip():
            compact = " ".join(conclusion.split())
            conclusion_preview = compact[:240]
        return {
            "discussion_id": str(snapshot.get("discussion_id", "")),
            "title": cls._objective_title(snapshot),
            "status": str(snapshot.get("status", "unknown")),
            "participants": [str(value) for value in participants],
            "completed_rounds": int(progress.get("completed_rounds", round_count)),
            "total_rounds": int(progress.get("total_rounds", round_count)),
            "started_at": snapshot.get("started_at"),
            "updated_at": snapshot.get("updated_at"),
            "completed_at": snapshot.get("completed_at"),
            "preferred_candidate": decision.get("preferred_candidate"),
            "consensus_status": decision.get("consensus_status"),
            "conclusion_preview": conclusion_preview,
        }

    def list_snapshots(
        self,
        *,
        limit: int,
        offset: int,
        status: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return newest persisted discussions without exposing heavy snapshots."""
        if not self.output_dir.is_dir():
            return [], 0
        entries: list[tuple[int, dict[str, Any]]] = []
        for path in self.output_dir.glob("*.json"):
            try:
                snapshot = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(snapshot, dict):
                    continue
                item = self._history_item(snapshot)
                if not item["discussion_id"]:
                    continue
                if status is not None and item["status"] != status:
                    continue
                entries.append((path.stat().st_mtime_ns, item))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                # A damaged or concurrently replaced legacy artifact should not
                # make the complete history endpoint unavailable.
                continue
        entries.sort(key=lambda entry: entry[0], reverse=True)
        total = len(entries)
        return [item for _, item in entries[offset : offset + limit]], total

    def replay_timeline(self, discussion_id: str) -> dict[str, Any]:
        """Return lightweight lifecycle frames used to replay one saved run."""
        safe_id = _safe_id(discussion_id)
        frames: list[dict[str, Any]] = []
        warnings: list[str] = []
        events_path = self._events_path(safe_id)
        if events_path.is_file():
            try:
                lines = events_path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                lines = []
                warnings.append(f"The event journal could not be read: {exc}")
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    event = DiscussionEvent.from_dict(json.loads(line))
                    snapshot = event.data.get("discussion", {})
                    progress = snapshot.get("progress", {})
                    frames.append(
                        {
                            "sequence": event.sequence,
                            "event_type": event.event_type.value,
                            "occurred_at": event.occurred_at,
                            "status": snapshot.get("status"),
                            "completed_round": event.data.get(
                                "completed_round",
                                progress.get("completed_rounds", 0),
                            ),
                            "snapshot_sha256": event.snapshot_sha256,
                        }
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    warnings.append(
                        f"Skipped invalid event journal line {line_number}: {exc}"
                    )
        frames.sort(key=lambda frame: frame["sequence"])
        return {
            "discussion_id": safe_id,
            "frames": frames,
            "warnings": warnings,
        }
