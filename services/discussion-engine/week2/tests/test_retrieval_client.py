from __future__ import annotations

import httpx

from src.agents import retrieval


def test_retrieve_reuses_one_http_client(monkeypatch):
    created_clients = []
    requests = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.closed = False
            created_clients.append((self, kwargs))

        def post(self, url, *, json):
            requests.append((url, json))
            return httpx.Response(
                200,
                json={"results": [{"chunk_id": json["query"]}]},
                request=httpx.Request("POST", url),
            )

        def close(self):
            self.closed = True

    retrieval.close_client()
    monkeypatch.setattr(retrieval.httpx, "Client", FakeClient)
    monkeypatch.setattr(retrieval, "_get_base_url", lambda: "http://rag.test")

    first = retrieval.retrieve("first", top_k=2)
    second = retrieval.retrieve("second", top_k=3)
    retrieval.close_client()

    assert first == [{"chunk_id": "first"}]
    assert second == [{"chunk_id": "second"}]
    assert len(created_clients) == 1
    assert [request[1]["top_k"] for request in requests] == [2, 3]
    assert created_clients[0][0].closed is True
