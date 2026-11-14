"""Standalone entrypoint for the analytics consumer API."""

from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analytics.api import app, settings  # noqa: E402


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
