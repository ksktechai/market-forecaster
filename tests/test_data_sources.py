"""Data-source tests with the network mocked (no Finnhub / Yahoo calls)."""

from datetime import datetime, timezone

import pytest

from market_forecaster.config import load_config
from market_forecaster.data.base import DataSourceUnavailable, SymbolNotAvailable
from market_forecaster.data.chained import ChainedPriceSource
from market_forecaster.data.finnhub import FinnhubNewsSource, FinnhubPriceSource
from market_forecaster.data.yahoo import YahooPriceSource


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text="", content_type="application/json"):
        self.status_code = status_code
        self._json = json_data
        self.text = text
        self.headers = {"content-type": content_type}

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class FakeSession:
    def __init__(self, handler):
        self._handler = handler

    def get(self, url, params=None, timeout=None):
        return self._handler(url, params or {})


def _fake_candles(n=3):
    base = datetime(2024, 1, 2, tzinfo=timezone.utc)
    from datetime import timedelta

    return [(base + timedelta(days=i), 100.0 + i) for i in range(n)]


@pytest.fixture
def config():
    return load_config(env={"FINNHUB_API_KEY": "fake-key"})


# --- Finnhub ----------------------------------------------------------------


def test_finnhub_candle_403_raises_unavailable(config):
    session = FakeSession(lambda url, params: FakeResponse(status_code=403, text="forbidden"))
    src = FinnhubPriceSource(config, session=session)
    with pytest.raises(DataSourceUnavailable):
        src.fetch_prices(config.spec("NASDAQ"), lookback_days=30)


def test_finnhub_candle_ok_parses(config):
    body = {"s": "ok", "t": [1704153600, 1704240000], "c": [100.0, 101.0]}
    session = FakeSession(lambda url, params: FakeResponse(json_data=body))
    src = FinnhubPriceSource(config, session=session)
    series = src.fetch_prices(config.spec("NASDAQ"), lookback_days=30)
    assert series.closes == [100.0, 101.0]
    assert series.source == "finnhub"


def test_finnhub_news_parses(config):
    body = [
        {"headline": "Markets rally", "source": "Reuters", "url": "http://x", "datetime": 1704153600},
        {"headline": "", "source": "Empty"},  # dropped
    ]
    session = FakeSession(lambda url, params: FakeResponse(json_data=body))
    src = FinnhubNewsSource(config, session=session)
    headlines = src.fetch_news(config.spec("NASDAQ"))
    assert len(headlines) == 1
    assert headlines[0].headline == "Markets rally"


# --- Yahoo (downloader injected, no network) --------------------------------


def test_yahoo_parses_downloader_output(config):
    src = YahooPriceSource(downloader=lambda sym, days: _fake_candles(3))
    series = src.fetch_prices(config.spec("NZX50"), lookback_days=30)
    assert series.source == "yahoo"
    assert series.closes == [100.0, 101.0, 102.0]


def test_yahoo_empty_raises_not_available(config):
    src = YahooPriceSource(downloader=lambda sym, days: [])
    with pytest.raises(SymbolNotAvailable):
        src.fetch_prices(config.spec("NZX50"), lookback_days=30)


def test_yahoo_network_error_raises_unavailable(config):
    def boom(sym, days):
        raise RuntimeError("yahoo down")

    src = YahooPriceSource(downloader=boom)
    with pytest.raises(DataSourceUnavailable):
        src.fetch_prices(config.spec("NASDAQ"), lookback_days=30)


# --- Chain: Finnhub 403 -> Yahoo serves -------------------------------------


def test_chain_falls_back_from_finnhub_to_yahoo(config):
    finnhub_session = FakeSession(lambda url, params: FakeResponse(status_code=403, text="forbidden"))
    chain = ChainedPriceSource([
        FinnhubPriceSource(config, session=finnhub_session),
        YahooPriceSource(downloader=lambda sym, days: _fake_candles(3)),
    ])
    series = chain.fetch_prices(config.spec("NASDAQ"), lookback_days=30)
    assert series.source == "yahoo"
    assert len(series) == 3
