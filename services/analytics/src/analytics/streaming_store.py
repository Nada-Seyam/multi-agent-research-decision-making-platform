"""Durable local storage for incoming events and incremental analytics."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .streaming_events import canonical_sha256


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def atomic_write_json(path: Path, value: Any) -> None:
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
            json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class AnalyticsArtifactStore:
    def __init__(self, output_dir: Path, logs_dir: Path):
        self.output_dir = output_dir
        self.logs_dir = logs_dir

    @staticmethod
    def validate_discussion_id(discussion_id: str) -> None:
        if not SAFE_ID.fullmatch(discussion_id):
            raise ValueError(
                "discussion_id may contain only letters, digits, '.', '_' and '-'"
            )

    def artifact_path(self, discussion_id: str) -> Path:
        self.validate_discussion_id(discussion_id)
        return self.output_dir / f"{discussion_id}.json"

    def _event_path(self, discussion_id: str, event_id: str) -> Path:
        self.validate_discussion_id(discussion_id)
        prefix = f"{discussion_id}:"
        suffix = event_id[len(prefix) :] if event_id.startswith(prefix) else event_id
        if suffix == "started":
            filename = "discussion-started-source-event.json"
        elif suffix == "completed":
            filename = "discussion-completed-source-event.json"
        elif suffix == "failed":
            filename = "discussion-failed-source-event.json"
        elif suffix.startswith("round:") and suffix[6:].isdigit():
            filename = f"round-{int(suffix[6:]):02d}-completed-source-event.json"
        else:
            readable = re.sub(r"[^A-Za-z0-9._-]+", "-", suffix).strip("-")
            readable = readable[:60] or "unknown"
            event_key = canonical_sha256(event_id)[:8]
            filename = f"{readable}-{event_key}-source-event.json"
        return self.logs_dir / discussion_id / "events" / filename

    def read_artifact(self, discussion_id: str) -> dict[str, Any] | None:
        path = self.artifact_path(discussion_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None

    def save_artifact(self, artifact: dict[str, Any]) -> Path:
        path = self.artifact_path(str(artifact["discussion_id"]))
        atomic_write_json(path, artifact)
        return path

    def read_event(
        self, discussion_id: str, event_id: str
    ) -> dict[str, Any] | None:
        path = self._event_path(discussion_id, event_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None

    def save_event(self, payload: dict[str, Any]) -> Path:
        path = self._event_path(
            str(payload["discussion_id"]), str(payload["event_id"])
        )
        atomic_write_json(path, payload)
        return path

    def pending_event_payloads(self) -> list[dict[str, Any]]:
        """Return durably ingested events not yet reflected in an artifact."""
        pending: list[dict[str, Any]] = []
        if not self.logs_dir.exists():
            return pending
        for discussion_dir in self.logs_dir.iterdir():
            if not discussion_dir.is_dir():
                continue
            try:
                artifact = self.read_artifact(discussion_dir.name) or {}
            except (OSError, ValueError, json.JSONDecodeError):
                artifact = {}
            processed = set(artifact.get("processed_source_events", []))
            events_dir = discussion_dir / "events"
            if not events_dir.is_dir():
                continue
            for path in events_dir.glob("*.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if (
                    isinstance(payload, dict)
                    and payload.get("event_id") not in processed
                ):
                    pending.append(payload)
        return sorted(
            pending,
            key=lambda item: (
                str(item.get("discussion_id", "")),
                int(item.get("sequence", -1)),
            ),
        )
