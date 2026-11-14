"""Single-process local FastAPI entrypoint for discussion production."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from fastapi.staticfiles import StaticFiles
import uvicorn
import httpx
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, model_validator

from . import week2_adapter
from .artifacts import FileDiscussionArtifactStore
from .config import DiscussionSettings, get_settings
from .event_sinks import HttpEventSink
from .events import DiscussionEvent, utc_now
from .graph import build_week2_agent_graph
from .identifiers import discussion_id
from .lifecycle import build_result_snapshot
from .participants import ParticipantSelectionError, select_participant_ids
from .runner import run_discussion

API_HOST = os.getenv("DISCUSSION_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("DISCUSSION_API_PORT", "8001"))
EVENT_PATH = "/api/v1/discussion-events"
MAX_OBJECTIVE_BYTES = 1_000_000
OBJECTIVE_FILE_EXTENSIONS = {".txt", ".md"}
DISCUSSION_STATUSES = {"queued", "preparing", "running", "completed", "failed"}


class ParticipantSelectionInput(BaseModel):
    """Participant choices shared by JSON and multipart start requests."""

    participant_selection_mode: Literal["random", "custom"] = "random"
    num_participants: int = Field(default=4, ge=3, le=6)
    selected_participant_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_selection(self) -> "ParticipantSelectionInput":
        if self.participant_selection_mode == "random":
            if self.selected_participant_ids:
                raise ValueError("Random selection cannot include explicit participant ids.")
            return self
        if len(self.selected_participant_ids) < 3:
            raise ValueError("Choose at least three participants.")
        if len(self.selected_participant_ids) != len(set(self.selected_participant_ids)):
            raise ValueError("Participant ids must be unique.")
        if self.num_participants != len(self.selected_participant_ids):
            raise ValueError("Participant count must match the selected roster.")
        return self


class DiscussionCreateRequest(ParticipantSelectionInput):
    """The small, browser-friendly input contract for a new discussion."""

    topic: str = Field(min_length=3, max_length=160)
    objective: str = Field(min_length=10, max_length=4_000)
    constraints: str = Field(default="", max_length=4_000)
    rounds: int = Field(default=4, ge=3, le=6)


class MockEventConsumer:
    """Thread-safe, in-memory idempotent receiver used for local development."""

    def __init__(self) -> None:
        self._events: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def accept(
        self,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> tuple[str, DiscussionEvent]:
        try:
            event = DiscussionEvent.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"invalid discussion event: {exc}",
            ) from exc

        if idempotency_key != event.event_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Idempotency-Key must match event_id",
            )

        with self._lock:
            previous = self._events.get(event.event_id)
            if previous is not None:
                if previous != payload:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="event_id was already used for different content",
                    )
                return "duplicate", event
            self._events[event.event_id] = payload
        return "accepted", event

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._events)


mock_consumer = MockEventConsumer()

_executor_lock = Lock()
_discussion_executor: ThreadPoolExecutor | None = None
_reserved_discussions = 0


def _reserve_discussion_capacity(settings: DiscussionSettings) -> bool:
    """Reserve one bounded running-or-queued discussion slot."""
    global _reserved_discussions
    limit = settings.max_concurrent_discussions + settings.max_queued_discussions
    with _executor_lock:
        if _reserved_discussions >= limit:
            return False
        _reserved_discussions += 1
        return True


def _release_discussion_capacity() -> None:
    global _reserved_discussions
    with _executor_lock:
        _reserved_discussions = max(0, _reserved_discussions - 1)


def _get_discussion_executor(settings: DiscussionSettings) -> ThreadPoolExecutor:
    global _discussion_executor
    with _executor_lock:
        if _discussion_executor is None:
            _discussion_executor = ThreadPoolExecutor(
                max_workers=settings.max_concurrent_discussions,
                thread_name_prefix="discussion-job",
            )
        return _discussion_executor


def _shutdown_discussion_executor() -> None:
    global _discussion_executor, _reserved_discussions
    with _executor_lock:
        executor, _discussion_executor = _discussion_executor, None
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=False)
    with _executor_lock:
        _reserved_discussions = 0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    started_rag = False
    startup_error: str | None = None
    try:
        started_rag = await asyncio.to_thread(week2_adapter.start_local_rag)
    except Exception as exc:
        # RAG readiness is reported through /health. A slow or unavailable RAG
        # must not take down the producer and analytics APIs with it.
        startup_error = f"{type(exc).__name__}: {exc}"
    app.state.rag_started_by_api = started_rag
    app.state.rag_startup_error = startup_error
    try:
        yield
    finally:
        await asyncio.to_thread(_shutdown_discussion_executor)
        if started_rag:
            await asyncio.to_thread(week2_adapter.stop_local_rag)


app = FastAPI(
    title="Multi-Agent Discussion Producer",
    description=(
        "Local producer API with automatic RAG startup. It delivers events "
        "to DISCUSSION_EVENT_ENDPOINT when configured and otherwise uses an "
        "in-process mock consumer."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

cors_origins = [
    origin.strip()
    for origin in "http://127.0.0.1:5173,http://localhost:5173".split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _local_event_endpoint(request: Request) -> str:
    return str(request.base_url).rstrip("/") + EVENT_PATH


def _settings_for_local_api(request: Request) -> DiscussionSettings:
    configured = get_settings()
    external_endpoint = (configured.event_endpoint or "").strip()
    return configured.model_copy(
        update={
            "event_endpoint": external_endpoint or _local_event_endpoint(request),
            "event_delivery_required": True,
            "persist_artifacts": True,
        }
    )


def _artifact_store(settings: DiscussionSettings) -> FileDiscussionArtifactStore:
    return FileDiscussionArtifactStore(
        settings.resolve_output_dir(),
        settings.resolve_logs_dir(),
    )


def _objective_text(payload: DiscussionCreateRequest) -> str:
    constraint_lines = [
        line.strip().lstrip("- ")
        for line in payload.constraints.splitlines()
        if line.strip()
    ] or ["The final recommendation must directly satisfy the stated objective."]
    constraints = "\n".join(f"- {line}" for line in constraint_lines)
    return f"""OBJECTIVE
{payload.objective.strip()}

PRIMARY CONSTRAINTS
{constraints}

CORE TOPICS TO DISCUSS

1. {payload.topic.strip().upper()}
- Compare viable approaches and their trade-offs.
- Identify the strongest evidence, risks, and unresolved questions.

EVIDENCE REQUIREMENTS
- Ground claims in retrieved evidence where available.
- Clearly identify assumptions and missing evidence.

DECISION CRITERIA
1. Fit with the stated objective
2. Feasibility
3. Evidence strength
4. Risk and implementation complexity

STRICT NOTES
- Preserve dissent and uncertainty in the final decision.
- Do not invent evidence.

FINAL GOAL
Reach a clear, evidence-aware recommendation for {payload.topic.strip()}.
"""


def _load_json_objective(
    payload: DiscussionCreateRequest,
    settings: DiscussionSettings,
) -> Any:
    return _load_uploaded_objective(
        _objective_text(payload).encode("utf-8"),
        "objective.txt",
        settings,
    )


def _build_sink(settings: DiscussionSettings, request: Request) -> HttpEventSink:
    return HttpEventSink(
        settings.event_endpoint or _local_event_endpoint(request),
        token=(
            settings.event_token.get_secret_value()
            if settings.event_token is not None
            else None
        ),
        timeout_seconds=settings.event_timeout_seconds,
        max_attempts=settings.event_max_attempts,
        retry_initial_seconds=settings.event_retry_initial_seconds,
        retry_max_seconds=settings.event_retry_max_seconds,
    )


def _run_background_discussion(
    objective: Any,
    settings: DiscussionSettings,
    run_id: str,
    sink: HttpEventSink,
    store: FileDiscussionArtifactStore,
) -> None:
    try:
        run_discussion(
            objective,
            settings=settings,
            event_sink=sink,
            artifact_store=store,
            run_id=run_id,
        )
    except Exception:
        # The orchestrator persists a structured failed snapshot. Keeping this
        # exception inside the background task prevents an unhandled task error
        # while still making the failure visible to polling clients.
        return
    finally:
        sink.close()


def _run_reserved_discussion(
    objective: Any,
    settings: DiscussionSettings,
    run_id: str,
    sink: HttpEventSink,
    store: FileDiscussionArtifactStore,
) -> None:
    try:
        _run_background_discussion(objective, settings, run_id, sink, store)
    finally:
        _release_discussion_capacity()


def _submit_background_discussion(
    objective: Any,
    settings: DiscussionSettings,
    run_id: str,
    sink: HttpEventSink,
    store: FileDiscussionArtifactStore,
) -> None:
    """Submit quickly so FastAPI worker threads never wait behind long jobs."""
    try:
        _get_discussion_executor(settings).submit(
            _run_reserved_discussion,
            objective,
            settings,
            run_id,
            sink,
            store,
        )
    except BaseException:
        sink.close()
        _release_discussion_capacity()
        raise


def _participant_settings_update(
    selection: ParticipantSelectionInput,
) -> dict[str, Any]:
    """Translate the public participant contract into run settings."""
    return {
        "participant_selection_mode": selection.participant_selection_mode,
        "num_participants": selection.num_participants,
        "selected_participant_ids": selection.selected_participant_ids,
    }


def _parse_uploaded_participant_selection(
    participant_selection_mode: str,
    num_participants: int,
    selected_participant_ids: str,
) -> ParticipantSelectionInput:
    try:
        decoded_ids = json.loads(selected_participant_ids)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="selected_participant_ids must be a JSON array",
        ) from exc
    try:
        return ParticipantSelectionInput(
            participant_selection_mode=participant_selection_mode,
            num_participants=num_participants,
            selected_participant_ids=decoded_ids,
        )
    except ValidationError as exc:
        first_error = exc.errors(include_url=False)[0]
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=first_error["msg"],
        ) from exc


def _queue_discussion(
    objective: Any,
    settings: DiscussionSettings,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Persist the initial snapshot and schedule one discussion run."""
    settings = settings.with_selected_random_seed()
    assert settings.random_seed is not None
    personas = week2_adapter.load_week2_personas(settings.resolve_week2_root())
    try:
        participants = select_participant_ids(
            personas,
            num_participants=settings.num_participants,
            random_seed=settings.random_seed,
            selected_participant_ids=(
                settings.selected_participant_ids
                if settings.participant_selection_mode == "custom"
                else None
            ),
        )
    except ParticipantSelectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    graph = build_week2_agent_graph(
        participants,
        random_seed=settings.random_seed,
    ).to_dict()
    graph["directed"] = True
    run_id = discussion_id(objective, participants, settings)
    store = _artifact_store(settings)
    if not _reserve_discussion_capacity(settings):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "discussion_capacity_reached",
                "message": "The discussion queue is full. Try again shortly.",
            },
            headers={"Retry-After": "10"},
        )
    sink: HttpEventSink | None = None
    try:
        queued = build_result_snapshot(
            discussion_id=run_id,
            objective=objective,
            participants=participants,
            graph=graph,
            research=None,
            rounds=[],
            settings=settings,
            proposed_alternatives={},
            status="queued",
            started_at=utc_now(),
            preparation={
                "stage": "queued",
                "message": "Waiting for a discussion worker",
                "completed_units": 0,
                "total_units": settings.max_initial_queries,
                "queries": [],
            },
        )
        sink = _build_sink(settings, request)
        store.save_snapshot(queued.to_dict())
        background_tasks.add_task(
            _submit_background_discussion,
            objective,
            settings,
            run_id,
            sink,
            store,
        )
    except BaseException:
        if sink is not None:
            sink.close()
        _release_discussion_capacity()
        raise
    return {
        "discussion_id": run_id,
        "status": "queued",
        "discussion_url": f"/api/v1/discussions/{run_id}",
        "analysis_url": f"/api/v1/discussions/{run_id}/analysis",
    }


def _analytics_base_url(settings: DiscussionSettings) -> str:
    explicit = os.getenv("ANALYTICS_API_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    endpoint = (settings.event_endpoint or "").strip()
    suffix = "/api/v1/discussion-events"
    if endpoint.endswith(suffix):
        return endpoint[: -len(suffix)].rstrip("/")
    return "http://127.0.0.1:8002"


def _load_uploaded_objective(
    content: bytes, filename: str, settings: DiscussionSettings
) -> Any:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="objective file must be UTF-8 text",
        ) from exc
    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="objective file is empty",
        )

    suffix = Path(filename).suffix or ".txt"
    with tempfile.TemporaryDirectory(prefix="discussion-objective-") as directory:
        objective_path = Path(directory) / f"objective{suffix}"
        objective_path.write_text(text, encoding="utf-8")
        try:
            return week2_adapter.load_objective(
                objective_path,
                settings.resolve_week2_root(),
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc


def _rag_health_snapshot() -> tuple[bool, str, Any]:
    rag_url = week2_adapter.retrieval_api_base_url()
    try:
        response = httpx.get(f"{rag_url}/health", timeout=5.0)
        rag_ready = response.is_success
        rag_details: Any = response.json() if response.is_success else response.text
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        rag_ready = False
        rag_details = f"{type(exc).__name__}: {exc}"
    return rag_ready, rag_url, rag_details


def _require_rag_ready() -> None:
    rag_ready, rag_url, rag_details = _rag_health_snapshot()
    if rag_ready:
        return
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "code": "rag_warming_up",
            "message": "Retrieval is still warming up or unavailable. Try again shortly.",
            "rag_url": rag_url,
            "details": rag_details,
        },
        headers={"Retry-After": "10"},
    )


@app.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    rag_ready, rag_url, rag_details = _rag_health_snapshot()

    return {
        "status": "ok" if rag_ready else "degraded",
        "producer": "ready",
        "rag": {
            "ready": rag_ready,
            "url": rag_url,
            "details": rag_details,
        },
        "mock_consumer": {
            "ready": True,
            "received_events": mock_consumer.count,
        },
    }


@app.post(
    "/api/v1/discussions",
    tags=["discussions"],
    status_code=status.HTTP_202_ACCEPTED,
)
def create_discussion(
    request: Request,
    payload: DiscussionCreateRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    _require_rag_ready()
    settings = _settings_for_local_api(request).model_copy(
        update={
            "num_rounds": payload.rounds,
            **_participant_settings_update(payload),
        }
    )
    objective = _load_json_objective(payload, settings)
    return _queue_discussion(objective, settings, request, background_tasks)


@app.post(
    "/api/v1/discussions/upload",
    tags=["discussions"],
    status_code=status.HTTP_202_ACCEPTED,
)
def create_discussion_from_upload(
    request: Request,
    background_tasks: BackgroundTasks,
    objective_file: UploadFile = File(...),
    rounds: int = Form(default=4, ge=3, le=6),
    participant_selection_mode: Literal["random", "custom"] = Form(default="random"),
    num_participants: int = Form(default=4, ge=3, le=6),
    selected_participant_ids: str = Form(default="[]"),
) -> dict[str, Any]:
    _require_rag_ready()
    filename = objective_file.filename or "objective.txt"
    if Path(filename).suffix.lower() not in OBJECTIVE_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="objective_file must be a .txt or .md file",
        )

    content = objective_file.file.read(MAX_OBJECTIVE_BYTES + 1)
    if len(content) > MAX_OBJECTIVE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"objective file must not exceed {MAX_OBJECTIVE_BYTES} bytes",
        )

    selection = _parse_uploaded_participant_selection(
        participant_selection_mode,
        num_participants,
        selected_participant_ids,
    )
    settings = _settings_for_local_api(request).model_copy(
        update={
            "num_rounds": rounds,
            **_participant_settings_update(selection),
        }
    )
    objective = _load_uploaded_objective(content, filename, settings)
    return _queue_discussion(objective, settings, request, background_tasks)


@app.get("/api/v1/discussions", tags=["discussions"])
def list_discussions(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    discussion_status: str | None = Query(default=None, alias="status"),
) -> dict[str, Any]:
    """List lightweight summaries for saved discussions, newest first."""
    if discussion_status is not None and discussion_status not in DISCUSSION_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "status must be one of: "
                + ", ".join(sorted(DISCUSSION_STATUSES))
            ),
        )
    settings = get_settings().model_copy(update={"persist_artifacts": True})
    items, total = _artifact_store(settings).list_snapshots(
        limit=limit,
        offset=offset,
        status=discussion_status,
    )
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + len(items) < total,
    }


@app.get("/api/v1/discussions/{discussion_id}", tags=["discussions"])
def get_discussion(discussion_id: str) -> dict[str, Any]:
    settings = get_settings().model_copy(update={"persist_artifacts": True})
    try:
        path = _artifact_store(settings).snapshot_path(discussion_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="discussion was not found",
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="discussion snapshot could not be read",
        ) from exc


@app.get("/api/v1/discussions/{discussion_id}/replay", tags=["discussions"])
def get_discussion_replay(discussion_id: str) -> dict[str, Any]:
    """Return lifecycle timing metadata for round-by-round replay."""
    settings = get_settings().model_copy(update={"persist_artifacts": True})
    store = _artifact_store(settings)
    try:
        snapshot_path = store.snapshot_path(discussion_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    if not snapshot_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="discussion was not found",
        )
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        timeline = store.replay_timeline(discussion_id)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="discussion replay could not be read",
        ) from exc

    # Older snapshots may predate the lifecycle journal. They remain fully
    # replayable because every public snapshot contains all completed rounds.
    if not timeline["frames"]:
        rounds = snapshot.get("rounds", [])
        started_at = snapshot.get("started_at") or snapshot.get("updated_at")
        timeline["frames"] = [
            {
                "sequence": 0,
                "event_type": "discussion.started",
                "occurred_at": started_at,
                "status": "preparing",
                "completed_round": 0,
                "snapshot_sha256": None,
            },
            *[
                {
                    "sequence": index,
                    "event_type": "round.completed",
                    "occurred_at": snapshot.get("updated_at"),
                    "status": "running",
                    "completed_round": index,
                    "snapshot_sha256": None,
                }
                for index, _ in enumerate(rounds, start=1)
            ],
        ]
        if snapshot.get("status") in {"completed", "failed"}:
            timeline["frames"].append(
                {
                    "sequence": len(rounds) + 1,
                    "event_type": f"discussion.{snapshot['status']}",
                    "occurred_at": snapshot.get("completed_at")
                    or snapshot.get("updated_at"),
                    "status": snapshot["status"],
                    "completed_round": len(rounds),
                    "snapshot_sha256": None,
                }
            )
        timeline["warnings"].append(
            "This discussion predates lifecycle journals; replay timing is approximate."
        )
    return timeline


@app.get("/api/v1/discussions/{discussion_id}/analysis", tags=["analysis"])
def get_discussion_analysis(discussion_id: str) -> JSONResponse:
    settings = get_settings()
    url = f"{_analytics_base_url(settings)}/api/v1/discussions/{discussion_id}/analysis"
    try:
        response = httpx.get(url, timeout=10.0)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="analytics service is unavailable",
        ) from exc
    try:
        content = response.json()
    except json.JSONDecodeError:
        content = {"detail": "analytics service returned an invalid response"}
    return JSONResponse(status_code=response.status_code, content=content)


@app.post(EVENT_PATH, include_in_schema=False)
def receive_discussion_event(
    event: dict[str, Any],
    idempotency_key: str = Header(alias="Idempotency-Key"),
) -> JSONResponse:
    outcome, parsed_event = mock_consumer.accept(event, idempotency_key)
    return JSONResponse(
        status_code=(
            status.HTTP_201_CREATED if outcome == "accepted" else status.HTTP_200_OK
        ),
        content={
            "status": outcome,
            "event_id": parsed_event.event_id,
        },
    )


static_dir_value = os.getenv("DISCUSSION_STATIC_DIR", "").strip()
if static_dir_value:
    static_dir = Path(static_dir_value).resolve()
    if not static_dir.is_dir():
        raise RuntimeError(f"DISCUSSION_STATIC_DIR does not exist: {static_dir}")
    # Keep this mount last: the API, health, OpenAPI, and docs routes declared
    # above must take precedence over the frontend's root path.
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")



def main() -> None:
    uvicorn.run(
        "discussion.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
