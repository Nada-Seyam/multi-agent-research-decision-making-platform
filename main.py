"""Start the producer and analytics consumer APIs with one command."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from threading import Thread
from typing import Any


ROOT = Path(__file__).resolve().parent
CONSUMER_ROOT = ROOT / "services" / "analytics"
CONSUMER_SRC = CONSUMER_ROOT / "src"
PRODUCER_ROOT = ROOT / "services" / "discussion-engine"


def _add_project_paths() -> None:
    """Make both independent projects importable without installing them."""
    for path in (CONSUMER_SRC, PRODUCER_ROOT):
        resolved = str(path)
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def _environment_port(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if not 1 <= value <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535")
    return value


def _configure_local_connection(consumer_port: int) -> str:
    """Route producer lifecycle events to this launcher's consumer."""
    endpoint = os.getenv(
        "ANALYTICS_CONSUMER_URL",
        f"http://127.0.0.1:{consumer_port}/api/v1/discussion-events",
    )
    os.environ["DISCUSSION_EVENT_ENDPOINT"] = endpoint
    os.environ["DISCUSSION_EVENT_DELIVERY_REQUIRED"] = "true"

    consumer_token = os.getenv("ANALYTICS_EVENT_TOKEN")
    if consumer_token:
        os.environ["DISCUSSION_EVENT_TOKEN"] = consumer_token
    return endpoint


def _wait_until_started(
    server: Any,
    thread: Thread,
    *,
    service_name: str,
    timeout_seconds: float = 15.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if server.started:
            return
        if not thread.is_alive():
            raise RuntimeError(f"{service_name} stopped during startup")
        time.sleep(0.05)
    raise TimeoutError(
        f"{service_name} did not start within {timeout_seconds:g} seconds"
    )


def _stop_background_server(server: Any, thread: Thread) -> None:
    server.should_exit = True
    thread.join(timeout=10.0)
    if thread.is_alive():
        server.force_exit = True
        thread.join(timeout=2.0)


def main() -> None:
    consumer_host = os.getenv("ANALYTICS_API_HOST", "127.0.0.1")
    consumer_port = _environment_port("ANALYTICS_API_PORT", 8002)
    producer_host = os.getenv("DISCUSSION_API_HOST", "127.0.0.1")
    producer_port = _environment_port("DISCUSSION_API_PORT", 8001)
    event_endpoint = _configure_local_connection(consumer_port)
    _add_project_paths()

    import uvicorn
    from analytics.api import app as consumer_app
    from discussion.api import app as producer_app

    consumer_server = uvicorn.Server(
        uvicorn.Config(
            consumer_app,
            host=consumer_host,
            port=consumer_port,
            log_level="info",
        )
    )
    consumer_thread = Thread(
        target=consumer_server.run,
        name="analytics-consumer-api",
        daemon=True,
    )
    consumer_thread.start()

    try:
        _wait_until_started(
            consumer_server,
            consumer_thread,
            service_name="analytics consumer",
        )
        print(
            f"Analytics consumer ready: http://127.0.0.1:{consumer_port}/docs",
            flush=True,
        )
        print(f"Producer event delivery: {event_endpoint}", flush=True)
        print(
            f"Discussion producer: http://127.0.0.1:{producer_port}/docs",
            flush=True,
        )
        print("Press Ctrl+C to stop both APIs.", flush=True)

        producer_server = uvicorn.Server(
            uvicorn.Config(
                producer_app,
                host=producer_host,
                port=producer_port,
                log_level="info",
            )
        )
        producer_server.run()
    finally:
        _stop_background_server(consumer_server, consumer_thread)


if __name__ == "__main__":
    main()
