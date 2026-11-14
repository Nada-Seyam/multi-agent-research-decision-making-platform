from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict



class Settings(BaseSettings):
    """Project-wide configuration.

    Field names map 1-to-1 to env-var names (uppercased).
    e.g.  ``request_delay_seconds``  ↔  ``REQUEST_DELAY_SECONDS``

    ``API_KEY`` is passed to the LLM provider. For local providers
    like Ollama it can be any non-empty string (defaults to "ollama").
    """

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    API_KEY: str
    RETRIEVAL_API_BASE_URL: str = "http://localhost:8000"
    RETRIEVAL_TOP_K: int = 5
    LLM_BASE_URL: str = "https://ollama.com/v1"
    LLM_MODEL: str = "gemma4:31b"
    LLM_MAX_CONCURRENT_REQUESTS: int = 8
    LLM_REQUEST_TIMEOUT_SECONDS: float = 120.0
    LLM_MAX_RETRIES: int = 2

    # ── Query expansion loop (per source) ────────────────────────
    # Number of additional retrieval attempts after the initial one.
    # Set to 0 to disable expansion entirely (legacy single-shot behavior).
    QUERY_EXPANSION_ATTEMPTS: int = 3
    # When the loop escalates, the order it walks sources in.
    # Comma-separated. Recognized values: "rag", "web".
    CONTEXT_SOURCES_ORDER: str = "rag,web"
    # Maximum number of independent planned queries to resolve concurrently.
    # Keep this bounded to avoid overwhelming a local RAG service or LLM endpoint.
    EVIDENCE_POOL_MAX_WORKERS: int = 3
    RESEARCH_MAX_MODEL_CALLS: int = 16

    # ── Session memory (Cross-Run Memory) ───────────────────────────────
    # Master switch for the rolling-summary cross-run memory.
    MEMORY_ENABLED: bool = True
    # Max execution turns kept in RAM before older ones are compacted.
    MEMORY_WINDOW_SIZE: int = 4
    # Soft token budget safety net; compaction fires when the working buffer exceeds this.
    MEMORY_TOKEN_BUDGET: int = 2000
@lru_cache(maxsize=1)
def get_settings() -> Settings:
     """Return a cached singleton so the .env file is read only once."""
     return Settings()


def clean_json(text: str) -> str:
    """Remove optional Markdown JSON fences from an LLM response."""
    import re
    text = text.strip()
    # Some OpenAI-compatible providers emit a fenced response despite a JSON
    # schema request. Accept one to three backticks because a few providers
    # collapse Markdown fences to a single backtick.
    text = re.sub(r"^`{1,3}(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*`{1,3}$", "", text)
    return text.strip()
