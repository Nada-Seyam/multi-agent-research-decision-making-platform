"""FastAPI application that consumes producer lifecycle events."""

from __future__ import annotations

import asyncio
import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse

from .data_loader import DiscussionValidationError
from .settings import AnalyticsSettings
from .streaming_service import EventConflictError, StreamingAnalyticsService
from .streaming_store import AnalyticsArtifactStore


settings = AnalyticsSettings.from_environment()
store = AnalyticsArtifactStore(
    settings.resolve_output_dir(), settings.resolve_logs_dir()
)
service = StreamingAnalyticsService(store)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    recovery_task = asyncio.create_task(asyncio.to_thread(service.recover_pending))
    try:
        yield
    finally:
        await recovery_task

app = FastAPI(
    title="Discussion Analytics Consumer",
    description=(
        "Consumes producer lifecycle events, emits round analytics after each "
        "round, and emits influence plus full-discussion analytics only after "
        "the discussion completes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _authorize(authorization: str | None) -> None:
    if settings.event_token is None:
        return
    expected = f"Bearer {settings.event_token}"
    if authorization is None or not hmac.compare_digest(authorization, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid bearer token",
        )


@app.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "consumer": "ready",
        "event_endpoint": "/api/v1/discussion-events",
        "round_metrics": ["stances", "opinion_change", "agreement", "sentiment"],
        "final_only_metrics": ["influence", "discussion_level_trends"],
    }


@app.post("/api/v1/discussion-events", tags=["events"])
def receive_discussion_event(
    event: dict[str, Any],
    background_tasks: BackgroundTasks,
    idempotency_key: str = Header(alias="Idempotency-Key"),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    _authorize(authorization)
    try:
        result = service.ingest(event, idempotency_key)
    except EventConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except (DiscussionValidationError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    if result.should_process:
        background_tasks.add_task(
            service.process_ingested_safely,
            event,
            idempotency_key,
        )

    return JSONResponse(
        status_code=(
            status.HTTP_201_CREATED
            if result.status == "accepted"
            else status.HTTP_200_OK
        ),
        content={
            "status": result.status,
            "event_id": result.event.event_id,
            "event_type": result.event.event_type,
            "discussion_id": result.event.discussion_id,
            "artifact_path": result.artifact_path.as_posix(),
            "analysis_event": result.analysis_event,
            "processing_status": (
                "queued" if result.should_process else "completed"
            ),
        },
    )


@app.get("/api/v1/discussions/{discussion_id}/analysis", tags=["analysis"])
def get_discussion_analysis(discussion_id: str) -> dict[str, Any]:
    try:
        artifact = store.read_artifact(discussion_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    if artifact is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="analysis artifact was not found",
        )
    return artifact


@app.get(
    "/api/v1/discussions/{discussion_id}/analysis/rounds/{round_number}",
    tags=["analysis"],
)
def get_round_analysis(
    discussion_id: str, round_number: int
) -> dict[str, Any]:
    artifact = get_discussion_analysis(discussion_id)
    for analysis in artifact.get("round_analyses", []):
        if analysis.get("round") == round_number:
            return analysis
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"analysis for round {round_number} was not found",
    )
