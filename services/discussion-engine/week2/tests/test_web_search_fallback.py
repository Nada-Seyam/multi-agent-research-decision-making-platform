from __future__ import annotations

import pytest


def test_search_web_falls_back_when_langchain_constructor_fails(monkeypatch):
    from Tools.Web_Serach import search as search_mod

    monkeypatch.setattr(search_mod, "optimize_query_for_search", lambda query: query)

    class BrokenSearch:
        def __init__(self, **_kwargs):
            raise ImportError("ddgs is not installed")

    monkeypatch.setattr(search_mod, "DuckDuckGoSearchResults", BrokenSearch)
    monkeypatch.setattr(
        search_mod,
        "_search_with_legacy_client",
        lambda query, max_results: [
            {"link": "https://example.com/paper", "title": "Paper"}
        ],
    )
    monkeypatch.setattr(
        search_mod,
        "fetch_webpage_text",
        lambda url: f"Extracted text from {url}",
    )

    chunks = search_mod.search_web("transformer architecture", max_results=1)

    assert chunks == [
        {
            "source_name": "https://example.com/paper",
            "section": "Web Page",
            "text": "Extracted text from https://example.com/paper",
            "rerank_score": 1.0,
        }
    ]


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "10.1.2.3", "169.254.169.254", "::1"],
)
def test_web_fetch_url_validation_blocks_non_public_addresses(
    monkeypatch, address
):
    from Tools.Web_Serach import search as search_mod

    monkeypatch.setattr(
        search_mod.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (None, None, None, None, (address, 443))
        ],
    )

    with pytest.raises(ValueError, match="blocked"):
        search_mod._validate_public_web_url("https://example.test/paper")


def test_web_fetch_url_validation_allows_public_addresses(monkeypatch):
    from Tools.Web_Serach import search as search_mod

    monkeypatch.setattr(
        search_mod.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (None, None, None, None, ("93.184.216.34", 443))
        ],
    )

    search_mod._validate_public_web_url("https://example.test/paper")
