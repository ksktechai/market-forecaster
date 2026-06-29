"""HTTP layer via FastAPI TestClient, wired to fake sources (offline)."""

from fastapi.testclient import TestClient

from market_forecaster.app import create_app


def _client(config, fake_pipeline):
    app = create_app(config=config, pipeline=fake_pipeline)
    return TestClient(app)


def test_health(config, fake_pipeline):
    client = _client(config, fake_pipeline)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "NASDAQ" in body["known_symbols"]


def test_post_forecast(config, fake_pipeline):
    client = _client(config, fake_pipeline)
    resp = client.post("/forecast", json={"symbols": ["NASDAQ"], "horizon": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["symbol"] == "NASDAQ"
    # baseline must always be present
    assert body["results"][0]["baseline"] is not None
    assert body["telegram_text"]


def test_get_single_symbol(config, fake_pipeline):
    client = _client(config, fake_pipeline)
    resp = client.get("/forecast/NZX50")
    assert resp.status_code == 200
    assert resp.json()["symbol"] == "NZX50"


def test_unknown_symbol_rejected(config, fake_pipeline):
    client = _client(config, fake_pipeline)
    assert client.get("/forecast/DOGE").status_code == 404
    assert client.post("/forecast", json={"symbols": ["DOGE"]}).status_code == 400
