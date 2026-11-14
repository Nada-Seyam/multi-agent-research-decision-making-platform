import main


def test_configure_local_connection_wires_producer_to_consumer(monkeypatch):
    monkeypatch.delenv("ANALYTICS_CONSUMER_URL", raising=False)
    monkeypatch.delenv("ANALYTICS_EVENT_TOKEN", raising=False)
    monkeypatch.delenv("DISCUSSION_EVENT_ENDPOINT", raising=False)

    endpoint = main._configure_local_connection(8123)

    assert endpoint == "http://127.0.0.1:8123/api/v1/discussion-events"
    assert main.os.environ["DISCUSSION_EVENT_ENDPOINT"] == endpoint
    assert main.os.environ["DISCUSSION_EVENT_DELIVERY_REQUIRED"] == "true"


def test_configure_local_connection_shares_consumer_token(monkeypatch):
    monkeypatch.setenv("ANALYTICS_EVENT_TOKEN", "shared-secret")
    monkeypatch.delenv("DISCUSSION_EVENT_TOKEN", raising=False)

    main._configure_local_connection(8002)

    assert main.os.environ["DISCUSSION_EVENT_TOKEN"] == "shared-secret"


def test_environment_port_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("TEST_API_PORT", "70000")

    try:
        main._environment_port("TEST_API_PORT", 8000)
    except ValueError as exc:
        assert "between 1 and 65535" in str(exc)
    else:
        raise AssertionError("invalid ports must be rejected")
