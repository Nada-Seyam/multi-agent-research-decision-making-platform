"""Environment-backed settings for the streaming analytics API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AnalyticsSettings:
    host: str = "127.0.0.1"
    port: int = 8002
    output_dir: Path = Path("outputs/analytics")
    logs_dir: Path = Path("logs/analytics")
    event_token: str | None = None

    @classmethod
    def from_environment(cls) -> "AnalyticsSettings":
        return cls(
            host=os.getenv("ANALYTICS_API_HOST", "127.0.0.1"),
            port=int(os.getenv("ANALYTICS_API_PORT", "8002")),
            output_dir=Path(
                os.getenv("ANALYTICS_OUTPUT_DIR", "outputs/analytics")
            ),
            logs_dir=Path(os.getenv("ANALYTICS_LOGS_DIR", "logs/analytics")),
            event_token=os.getenv("ANALYTICS_EVENT_TOKEN") or None,
        )

    def resolve_output_dir(self) -> Path:
        return self._resolve(self.output_dir)

    def resolve_logs_dir(self) -> Path:
        return self._resolve(self.logs_dir)

    @staticmethod
    def _resolve(path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path
