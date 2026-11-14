"""Public entry points for running a discussion."""
from __future__ import annotations

from typing import Any

from . import orchestrator
from .artifacts import DiscussionArtifactStore
from .config import DiscussionSettings
from .event_sinks import DiscussionEventSink
from .models import DiscussionResult


def run_discussion(
    objective: Any,
    *,
    settings: DiscussionSettings | None = None,
    event_sink: DiscussionEventSink | None = None,
    artifact_store: DiscussionArtifactStore | None = None,
    run_id: str | None = None,
) -> DiscussionResult:
    """Run the configured multi-round discussion."""
    return orchestrator.run_discussion(
        objective,
        settings=settings,
        event_sink=event_sink,
        artifact_store=artifact_store,
        run_id=run_id,
    )
