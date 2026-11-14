"""CLI entrypoint for the objective-driven multi-query research pipeline.

Lifecycle:
    load_objective → load_rolling_summary → plan_research → build_evidence_pool
    → consolidate_evidence → run both personas → persist output → update_memory

CLI:
    python -m src.agents --objective <path>

``resolve_context()`` is invoked once per research query (5 times per cycle)
inside ``evidence_pool.build_evidence_pool()``. The persona pipelines only
receive the final consolidated dossier — they never call ``resolve_context``
directly.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from urllib.parse import urlparse

import httpx

from src.agents.config import get_personas
from src.agents.graph import run_for_persona
from src.agents.llm import chat as llm_chat
from src.agents.memory import (
    build_memory_context,
    load_session,
    load_working_buffer,
    record_cycle,
    update_memory_from_cycle,
)
from src.agents.objective import Objective, load_objective
from src.agents.planner import plan_research
from src.agents.evidence_pool import build_evidence_pool
from src.agents.consolidator import consolidate_evidence
from src.helper.helper import get_settings

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"
FRAMEWORK_ROOT = Path(__file__).resolve().parents[2]

# Slot for the RAG subprocess we may spawn in this process.
_RAG_CHILD: list[subprocess.Popen] = []
_RAG_CHILD_LOCK = threading.Lock()

_RAG_HEALTH_TIMEOUT_S = 2.0
_RAG_POLL_INTERVAL_S = 2.0
_RAG_TEARDOWN_TIMEOUT_S = 5.0
_RAG_ERROR_TAIL_CHARS = 2000

_RAG_APP_DIR = "Tools/RAG"
_RAG_APP_MODULE = "api.main:app"


# ── Output helpers ──────────────────────────────────────────────────

def _slugify(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    if not s:
        s = "objective"
    return s[:max_len]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT-%M-%SZ")


def _objective_hash(objective: Objective) -> str:
    content = objective.goal.strip().encode("utf-8")
    return f"sha256:{hashlib.sha256(content).hexdigest()[:16]}"


def _write_outputs(payload: dict) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _utc_stamp()
    slug = _slugify(payload.get("objective_id", "run"))
    timestamped = OUTPUT_DIR / f"{stamp}_{slug}.json"
    latest = OUTPUT_DIR / "latest.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    timestamped.write_text(text, encoding="utf-8")
    fd, tmp_path = tempfile.mkstemp(
        dir=str(OUTPUT_DIR), suffix=".tmp", prefix=".latest_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, str(latest))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return timestamped, latest


# ── RAG auto-start ──────────────────────────────────────────────────

def _is_localhost(base_url: str) -> bool:
    try:
        host = (urlparse(base_url).hostname or "").lower()
    except ValueError:
        return False
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")


def _rag_health_ok(base_url: str) -> bool:
    try:
        r = httpx.get(f"{base_url}/health", timeout=_RAG_HEALTH_TIMEOUT_S)
    except httpx.HTTPError:
        return False
    return r.status_code == 200


def _rag_ready_timeout_seconds() -> float:
    """Return the CLI readiness timeout, keeping the web API non-blocking."""
    raw = os.getenv("RAG_READY_TIMEOUT_SECONDS", "180")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("RAG_READY_TIMEOUT_SECONDS must be a number") from exc
    if value <= 0:
        raise ValueError("RAG_READY_TIMEOUT_SECONDS must be greater than zero")
    return value


def _drain_subprocess_output(proc: subprocess.Popen) -> threading.Thread:
    def _pump():
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stderr.write(f"[rag] {line}")
        sys.stderr.flush()

    t = threading.Thread(target=_pump, name="rag-stdout-drain", daemon=True)
    t.start()
    return t


def _start_rag_subprocess() -> subprocess.Popen:
    env = {**os.environ}
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", _RAG_APP_MODULE, "--app-dir", _RAG_APP_DIR],
        cwd=str(FRAMEWORK_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        env=env,
    )


def _wait_for_rag_health(base_url: str, proc: subprocess.Popen) -> None:
    timeout_seconds = _rag_ready_timeout_seconds()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        rc = proc.poll()
        if rc is not None:
            tail = ""
            if proc.stdout is not None:
                tail = proc.stdout.read() or ""
            raise RuntimeError(
                f"RAG subprocess exited with code {rc} before /health was ready. "
                f"Output tail:\n{tail[-_RAG_ERROR_TAIL_CHARS:]}"
            )
        if _rag_health_ok(base_url):
            return
        time.sleep(_RAG_POLL_INTERVAL_S)
    raise RuntimeError(
        f"RAG /health did not become ready within {timeout_seconds:g}s "
        f"at {base_url}"
    )


def _configured_rag_base_url() -> str:
    base_url = get_settings().RETRIEVAL_API_BASE_URL
    if not base_url:
        raise RuntimeError(
            "RETRIEVAL_API_BASE_URL is not set. "
            "Set it in src/.env or export it in the shell."
        )
    return str(base_url).rstrip("/")


def _live_rag_child() -> subprocess.Popen | None:
    with _RAG_CHILD_LOCK:
        if not _RAG_CHILD:
            return None
        proc = _RAG_CHILD[0]
        if proc.poll() is None:
            return proc
        _RAG_CHILD.clear()
        return None


def _start_rag_if_needed() -> bool:
    """Start a local RAG child without waiting for model warm-up.

    Returns ``True`` only when this call created the child process. Remote RAG
    endpoints are never managed by this process.
    """
    base_url = _configured_rag_base_url()

    if not _is_localhost(base_url):
        return False

    if _rag_health_ok(base_url):
        return False

    with _RAG_CHILD_LOCK:
        if _RAG_CHILD:
            proc = _RAG_CHILD[0]
            if proc.poll() is None:
                return False
            _RAG_CHILD.clear()

        print(
            f"[rag] /health not responding at {base_url}; spawning uvicorn...",
            file=sys.stderr,
        )
        proc = _start_rag_subprocess()
        _RAG_CHILD.append(proc)

    _drain_subprocess_output(proc)
    return True


def _ensure_rag_up() -> bool:
    """Make sure local RAG is ready. Returns True if we spawned it."""
    base_url = _configured_rag_base_url()

    if not _is_localhost(base_url) or _rag_health_ok(base_url):
        return False

    started = _start_rag_if_needed()
    proc = _live_rag_child()
    if proc is None:
        raise RuntimeError(f"RAG failed to start at {base_url}")
    _wait_for_rag_health(base_url, proc)
    print(f"[rag] /health OK at {base_url}", file=sys.stderr)
    return started


def _teardown_rag_subprocess() -> None:
    with _RAG_CHILD_LOCK:
        if not _RAG_CHILD:
            return
        proc = _RAG_CHILD.pop(0)
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=_RAG_TEARDOWN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=_RAG_TEARDOWN_TIMEOUT_S)


# ── Core pipeline ───────────────────────────────────────────────────

def run_cycle(objective: Objective) -> dict:
    """Execute one full research cycle for the given objective.

    Returns the complete payload dict (written to output/latest.json).
    """
    settings = get_settings()
    mem_session = load_session() if settings.MEMORY_ENABLED else {}
    working_buffer = load_working_buffer() if settings.MEMORY_ENABLED else []

    # 1. Plan research (5 queries)
    print("[cycle] Planning research...", file=sys.stderr)
    plan = plan_research(objective, memory_summary=mem_session)

    # 2. Build evidence pool (resolve_context × 5)
    print("[cycle] Building evidence pool...", file=sys.stderr)
    pool = build_evidence_pool(plan.queries, objective=objective)

    # 3. Consolidate into dossier
    print("[cycle] Consolidating evidence...", file=sys.stderr)
    dossier = consolidate_evidence(pool, objective)

    # 4. Run each persona through the graph
    personas = {
        persona_id: persona
        for persona_id, persona in get_personas().get("personas", {}).items()
        if persona_id != "shared_rules" and isinstance(persona, dict)
    }
    results: dict[str, Any] = {}
    persona_opinions: dict[str, str] = {}

    for persona_id in personas.keys():
        print(f"[llm] Generating opinion for persona: {persona_id}...", file=sys.stderr)
        mem_ctx = None
        if settings.MEMORY_ENABLED:
            mem_ctx = build_memory_context(mem_session, working_buffer, objective.short_id(), persona_id)
        result = run_for_persona(persona_id, objective, dossier, memory_context=mem_ctx)
        opinion_text = result.get("opinion", "")
        results[persona_id] = {
            "opinion": opinion_text,
            "is_relevant": result.get("is_relevant", True),
        }
        if "error" in result:
            results[persona_id]["error"] = result["error"]
        persona_opinions[persona_id] = opinion_text
        print(f"[llm] Finished {persona_id}.", file=sys.stderr)

    # 5. Memory write-back
    print("[cycle] Writing memory...", file=sys.stderr)
    plan_text = json.dumps(
        {"queries": plan.queries, "rationale": plan.rationale},
        ensure_ascii=False,
    ) if plan else None

    # Summarize persona outputs for memory
    mem_summaries: dict[str, str] = {}
    for pid, text in persona_opinions.items():
        if isinstance(text, str) and text.strip():
            summary = llm_chat(
                "You are a summarization assistant.",
                f"Summarize the core architectural recommendation and rationale in 1 to 2 sentences max:\n\n{text}"
            )
            mem_summaries[pid] = summary

    record_cycle(objective.short_id(), objective.goal, plan_text, mem_summaries)
    update_memory_from_cycle(objective, dossier)

    # 6. Assemble output payload
    research_queries_payload = []
    for i, q in enumerate(plan.queries):
        entry: dict[str, Any] = {"id": i + 1, "query": q}
        # Find matching pool entry
        if i < len(pool):
            entry["source_used"] = pool[i].context.source_used
            entry["is_relevant"] = pool[i].context.is_relevant
            entry["num_chunks"] = len(pool[i].context.chunks)
        research_queries_payload.append(entry)

    payload: dict[str, Any] = {
        "objective_id": objective.short_id(),
        "objective": {
            "goal": objective.goal,
            "final_goal": objective.final_goal,
            "constraints": objective.constraints,
            "decision_criteria": objective.decision_criteria,
            "core_topics": [{"id": t.id, "title": t.title} for t in objective.core_topics],
        },
        "research_queries": research_queries_payload,
        "research_dossier": {
            "summary": dossier.summary if dossier else "",
            "excerpt_count": len(dossier.excerpts) if dossier else 0,
            "contradictions": [
                {"description": c.description, "indices": c.excerpt_indices}
                for c in (dossier.contradictions if dossier else [])
            ],
            "per_topic_summaries": dossier.per_topic_summaries if dossier else {},
        },
        "results": results,
    }

    # 7. Write resource usage log (per-cycle, not per-persona)
    log_file = Path(__file__).resolve().parents[2] / "logs" / "query_resource_usage.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "objective_id": objective.short_id(),
        "objective_hash": _objective_hash(objective),
        "research_queries": plan.queries if plan else [],
        "per_query": [
            {
                "id": i + 1,
                "source_used": pool[i].context.source_used,
                "expansion_attempts": pool[i].context.expansion_attempts,
            }
            for i in range(len(pool))
        ],
        "dossier_size_chars": len(dossier.summary) if dossier else 0,
    }
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[Logging] Error writing log file: {e}")

    return payload


# ── CLI ─────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in ("-h", "--help"):
        print(
            "Usage:\n"
            "  python -m src.agents --objective <path>   Run one research cycle\n",
            file=sys.stderr,
        )
        return 0 if args and args[0] in ("-h", "--help") else 2

    if args[0] == "--objective":
        if len(args) < 2:
            print("Error: --objective requires a file path.", file=sys.stderr)
            return 2
        objective_path = args[1]
    else:
        # Accept positional path for backwards compatibility.
        objective_path = args[0]

    obj_path = Path(objective_path)
    if not obj_path.exists():
        print(f"Error: Objective file not found: {objective_path}", file=sys.stderr)
        return 2

    try:
        objective = load_objective(obj_path)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        started_rag = _ensure_rag_up()
    except Exception as exc:
        print(f"error: failed to start RAG: {exc}", file=sys.stderr)
        return 1

    try:
        payload = run_cycle(objective)
        ts_path, latest_path = _write_outputs(payload)
        print(f"wrote {ts_path}", file=sys.stderr)
        print(f"wrote {latest_path}", file=sys.stderr)
        return 0
    finally:
        if started_rag:
            _teardown_rag_subprocess()

if __name__ == "__main__":
    sys.exit(main())
