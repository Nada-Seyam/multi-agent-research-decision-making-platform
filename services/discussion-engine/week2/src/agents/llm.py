from __future__ import annotations

from functools import lru_cache
from threading import BoundedSemaphore
from typing import Any

from langchain_openai import ChatOpenAI

from src.helper.helper import get_settings


class _BoundedRunnable:
    """Wrap a LangChain runnable while sharing the provider concurrency cap."""

    def __init__(self, runnable: Any, slots: BoundedSemaphore) -> None:
        self._runnable = runnable
        self._slots = slots

    def invoke(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        with self._slots:
            return self._runnable.invoke(messages, *args, **kwargs)


class _BoundedChatClient(_BoundedRunnable):
    """Small proxy that enforces one process-wide provider concurrency cap."""

    def __init__(self, client: ChatOpenAI, max_concurrent: int) -> None:
        self._client = client
        super().__init__(client, BoundedSemaphore(max(1, max_concurrent)))

    def with_structured_output(self, *args: Any, **kwargs: Any) -> _BoundedRunnable:
        """Preserve structured-output behavior without bypassing the cap."""
        return _BoundedRunnable(
            self._client.with_structured_output(*args, **kwargs),
            self._slots,
        )


@lru_cache(maxsize=1)
def _client() -> _BoundedChatClient:
    s = get_settings()
    return _BoundedChatClient(
        ChatOpenAI(
            model=s.LLM_MODEL,
            base_url=s.LLM_BASE_URL,
            api_key=s.API_KEY,
            max_tokens=1024,
            temperature=1,
            timeout=s.LLM_REQUEST_TIMEOUT_SECONDS,
            max_retries=s.LLM_MAX_RETRIES,
        ),
        s.LLM_MAX_CONCURRENT_REQUESTS,
    )


def chat(system: str, user: str) -> str:
    """Call the chat model with a system + user message and return the text."""
    from langchain_core.messages import HumanMessage, SystemMessage

    response = _client().invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    content = response.content
    if isinstance(content, list):
        return "".join(
            (block.get("text", "") if isinstance(block, dict) else str(block))
            for block in content
        )
    return str(content)
