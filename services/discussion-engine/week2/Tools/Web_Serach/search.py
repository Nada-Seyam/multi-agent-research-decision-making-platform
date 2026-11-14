from __future__ import annotations

import ipaddress
import json
import re
import socket
from typing import Any
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from src.agents.llm import _client as get_llm


MAX_WEBPAGE_BYTES = 2_000_000
MAX_REDIRECTS = 5
WEB_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)


def _validate_public_web_url(url: str) -> None:
    """Reject non-web and private-network targets before fetching."""
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("only absolute HTTP(S) URLs are allowed")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URLs containing credentials are not allowed")
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(
                parsed.hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        }
    except socket.gaierror as exc:
        raise ValueError("URL host could not be resolved") from exc
    if not addresses:
        raise ValueError("URL host did not resolve to an address")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("private, loopback, link-local, and reserved hosts are blocked")


class OptimizedSearch(BaseModel):
    """Structured output for search query optimization."""

    search_query: str = Field(
        description=(
            "A highly targeted, keyword-dense search query for DuckDuckGo. "
            "Strip all conversational filler."
        )
    )


QUERY_OPTIMIZER_PROMPT = (
    "You are an expert AI research assistant. Your task is to convert a verbose "
    "user question into a highly targeted, keyword-dense search engine query.\n\n"
    "CRITICAL RULES:\n"
    "1. Extract ONLY the core technical entities, architectures, and comparison "
    "metrics (e.g., 'INT4 AWQ', 'FP8 quantization', 'memory reduction').\n"
    "2. Remove all conversational filler (e.g., 'We are preparing our 7B model...', "
    "'Should we use...').\n"
    "3. Add terms like 'comparison', 'benchmark', or 'paper' if they help retrieve "
    "technical deep-dives rather than generic blogs.\n"
    "4. VERY IMPORTANT: The final query MUST be concise. Limit the query to a "
    "MAXIMUM of 8-10 keywords. Do NOT exceed 100 characters."
)


def optimize_query_for_search(original_query: str) -> str:
    """Convert a verbose user query into a concise search-engine query."""

    from src.helper.helper import clean_json

    try:
        llm = get_llm()

        system_prompt = (
            QUERY_OPTIMIZER_PROMPT
            + '\n\nOutput valid JSON only. '
            'Schema: {"search_query": "..."}'
        )

        raw = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"Optimize this query: {original_query}"
                ),
            ]
        )

        content = raw.content

        # Some chat models return structured/multimodal content
        # instead of a plain string.
        if isinstance(content, list):
            content = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in content
            )

        cleaned = clean_json(str(content))
        data = json.loads(cleaned)

        response = OptimizedSearch.model_validate(data)

        if response.search_query:
            return response.search_query.strip()

        return original_query

    except Exception as e:
        print(
            f"[WebSearch] Query optimization failed: {e}. "
            "Using original query."
        )
        return original_query


def clean_webpage_text(text: str) -> str:
    """
    Clean extracted webpage text.

    Removes excessive whitespace and repeated blank lines while
    preserving the actual textual content of the page.
    """

    if not text:
        return ""

    # Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive spaces/tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove whitespace around lines.
    lines = [line.strip() for line in text.splitlines()]

    # Remove empty lines at the beginning/end.
    text = "\n".join(lines).strip()

    return text


def fetch_webpage_text(url: str) -> str:
    """
    Fetch a webpage and return its extracted textual content.

    Redirects are followed explicitly so every target is checked against
    the public-network policy before a request is made.
    """

    try:
        current_url = url
        for _ in range(MAX_REDIRECTS + 1):
            _validate_public_web_url(current_url)
            with requests.get(
                current_url,
                timeout=15,
                headers={"User-Agent": WEB_USER_AGENT},
                allow_redirects=False,
                stream=True,
            ) as response:
                if response.is_redirect or response.is_permanent_redirect:
                    location = response.headers.get("location")
                    if not location:
                        return ""
                    current_url = urljoin(current_url, location)
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if not any(
                    allowed in content_type
                    for allowed in ("text/html", "application/xhtml+xml", "text/plain")
                ):
                    return ""
                content = bytearray()
                for block in response.iter_content(chunk_size=64 * 1024):
                    content.extend(block)
                    if len(content) > MAX_WEBPAGE_BYTES:
                        raise ValueError("webpage exceeds the download size limit")
                encoding = response.encoding or "utf-8"
                decoded = content.decode(encoding, errors="replace")
                if "text/plain" in content_type:
                    return clean_webpage_text(decoded)
                soup = BeautifulSoup(decoded, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                return clean_webpage_text(soup.get_text("\n"))
        raise ValueError("webpage exceeded the redirect limit")

    except Exception as e:
        print(f"[WebSearch] Failed to fetch webpage {url}: {e}")
        return ""


def search_web(
    query: str,
    max_results: int = 5,
    max_chars_per_page: int = 30000,
) -> list[dict[str, Any]]:
    """
    Search the web and return extracted webpage text.
    Args:
        query:
            Original user query.

        max_results:
            Maximum number of search results/webpages to fetch.

        max_chars_per_page:
            Maximum amount of text retained from each webpage.

    Returns:
        A list of dictionaries containing webpage text and metadata.
    """

    optimized_query = optimize_query_for_search(query)[:200]

    print(
        f"[WebSearch] Optimized query: '{optimized_query}'"
    )

    try:
        search = DuckDuckGoSearchResults(
            max_results=max_results,
            output_format="list",
        )
        raw_results = search.invoke(optimized_query)

    except Exception as e:
        print(f"[WebSearch] LangChain search failed: {e}")
        raw_results = _search_with_legacy_client(optimized_query, max_results)

    # DuckDuckGo should return a list when output_format="list".
    if not isinstance(raw_results, list):
        print("[WebSearch] Unexpected search result format.")
        return []

    chunks: list[dict[str, Any]] = []

    for index, result in enumerate(raw_results, start=1):

        if not isinstance(result, dict):
            continue

        url = result.get("link")

        if not url:
            print(
                f"[WebSearch] Result {index} has no URL. Skipping."
            )
            continue

        print(
            f"[WebSearch] Fetching page {index}/{len(raw_results)}: {url}"
        )

        page_text = fetch_webpage_text(url)

        if not page_text:
            print(
                f"[WebSearch] No text extracted from: {url}"
            )
            continue

        # Prevent extremely large webpages from overwhelming
        # the downstream agent/RAG context.
        if len(page_text) > max_chars_per_page:
            page_text = page_text[:max_chars_per_page]

        chunks.append(
            {
                "source_name": url,
                "section": "Web Page",
                "text": page_text,
                "rerank_score": 1.0,
            }
        )

    print(
        f"[WebSearch] Successfully extracted "
        f"{len(chunks)}/{len(raw_results)} webpages."
    )

    return chunks


def _search_with_legacy_client(query: str, max_results: int) -> list[dict[str, Any]]:
    """Use ``duckduckgo_search`` when the newer ``ddgs`` package is absent."""
    try:
        from duckduckgo_search import DDGS
    except ImportError as exc:
        print(f"[WebSearch] No DuckDuckGo client available: {exc}")
        return []

    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as exc:
        print(f"[WebSearch] Legacy DuckDuckGo client failed: {exc}")
        return []

    normalized: list[dict[str, Any]] = []
    for result in results or []:
        if not isinstance(result, dict):
            continue
        normalized.append(
            {
                "link": result.get("href") or result.get("link"),
                "title": result.get("title", ""),
                "snippet": result.get("body") or result.get("snippet", ""),
            }
        )
    return normalized
