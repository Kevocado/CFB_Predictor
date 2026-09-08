"""main.py's global exception handler. Confirmed live (security review):
this used to return a raw traceback to any caller on any unhandled
exception -- a real information-disclosure risk on the public, internet-
reachable deployment, not just a local debugging nicety."""

from fastapi.testclient import TestClient

from cfb_predictor.api import main


def _client_with_broken_route():
    @main.app.get("/api/__test_boom")
    def boom():
        raise RuntimeError("something broke")

    return TestClient(main.app, raise_server_exceptions=False)


def test_public_mode_returns_a_generic_error_not_a_traceback(monkeypatch):
    monkeypatch.setattr(main, "PUBLIC_MODE", True)
    client = _client_with_broken_route()

    response = client.get("/api/__test_boom")

    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert "RuntimeError" not in response.text
    assert "Traceback" not in response.text


def test_private_mode_still_returns_the_traceback_for_local_debugging(monkeypatch):
    monkeypatch.setattr(main, "PUBLIC_MODE", False)
    client = _client_with_broken_route()

    response = client.get("/api/__test_boom")

    assert response.status_code == 500
    assert "DEBUG_TRACEBACK" in response.text
    assert "RuntimeError" in response.text
