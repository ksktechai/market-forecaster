"""Ollama narrator: JSON parsing, <think> stripping, graceful failure -- all mocked."""

from market_forecaster.data.base import Headline
from market_forecaster.narrative.ollama import OllamaNarrator, _extract_json, _strip_think


class FakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = {"content-type": "application/json"}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, resp=None, raise_exc=None):
        self._resp = resp
        self._raise = raise_exc

    def post(self, url, json=None, timeout=None):
        if self._raise:
            raise self._raise
        return self._resp

    def get(self, url, timeout=None):
        return FakeResp(200)


def _narrator(session):
    return OllamaNarrator("http://x", "qwen3", session=session)


def test_strip_think():
    assert _strip_think("<think>secret reasoning</think>answer") == "answer"


def test_extract_json_from_noisy_output():
    text = '<think>...</think> Sure: {"sentiment": "bearish", "confidence": 0.6, "narrative": "down"}'
    obj = _extract_json(text)
    assert obj["sentiment"] == "bearish"


def test_narrate_parses_json():
    payload = {"response": '{"sentiment":"bullish","confidence":0.8,"narrative":"Up within band."}'}
    narrator = _narrator(FakeSession(resp=FakeResp(200, payload)))
    out = narrator.narrate(
        display="NASDAQ", current_price=100.0, horizon=5,
        point_forecast=[101, 102], band=([99], [104]),
        headlines=[Headline(headline="x")],
    )
    assert out.sentiment == "bullish"
    assert out.confidence == 0.8
    assert "band" in out.narrative.lower()


def test_narrate_handles_network_error():
    import requests

    narrator = _narrator(FakeSession(raise_exc=requests.RequestException("boom")))
    out = narrator.narrate(
        display="NASDAQ", current_price=100.0, horizon=5,
        point_forecast=[101], band=None, headlines=[],
    )
    assert out.sentiment == "unknown"
    assert out.error is not None
