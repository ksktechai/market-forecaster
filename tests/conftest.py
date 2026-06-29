"""Shared fakes so the whole suite runs offline (no Finnhub / Ollama / TimesFM)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from market_forecaster.config import SymbolSpec, load_config
from market_forecaster.data.base import Headline, NewsSource, PriceDataSource, PriceSeries
from market_forecaster.forecast.base import Forecaster, ForecastOutput


def make_closes(n: int, start: float = 100.0) -> list[tuple[datetime, float]]:
    """Deterministic upward-drifting series with a little curvature."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    out = []
    for i in range(n):
        value = start + i * 0.5 + 3.0 * math.sin(i / 7.0)
        out.append((base + timedelta(days=i), round(value, 2)))
    return out


class FakePriceSource(PriceDataSource):
    name = "fake"

    def __init__(self, points: int = 200):
        self._points = points

    def fetch_prices(self, spec: SymbolSpec, lookback_days: int) -> PriceSeries:
        return PriceSeries(
            canonical=spec.canonical, source=self.name, candles=make_closes(self._points)
        )


class FakeNewsSource(NewsSource):
    name = "fake"

    def fetch_news(self, spec: SymbolSpec, limit: int = 10) -> list[Headline]:
        return [
            Headline(headline="Tech stocks rally on strong earnings", source="Test"),
            Headline(headline="Central bank holds rates steady", source="Test"),
        ][:limit]


class FakeForecaster(Forecaster):
    name = "fake"

    def __init__(self):
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def forecast(self, history: list[float], horizon: int) -> ForecastOutput:
        last = history[-1]
        point = [round(last * (1 + 0.001 * (i + 1)), 4) for i in range(horizon)]
        quantiles = {
            "0.1": [round(p * 0.98, 4) for p in point],
            "0.5": list(point),
            "0.9": [round(p * 1.02, 4) for p in point],
        }
        return ForecastOutput(point=point, quantiles=quantiles)


class FakeNarrator:
    def __init__(self, sentiment="bullish"):
        self._sentiment = sentiment

    def is_ready(self) -> bool:
        return True

    def narrate(self, **kwargs):
        from market_forecaster.models import NewsSentiment

        return NewsSentiment(
            sentiment=self._sentiment,
            confidence=0.7,
            narrative="Forecast drifts up within a moderate uncertainty band.",
            headlines_considered=len(kwargs.get("headlines", [])),
            model="fake",
        )


@pytest.fixture
def config():
    return load_config(env={"FINNHUB_API_KEY": "fake-key", "OLLAMA_BASE_URL": "http://x"})


@pytest.fixture
def fake_pipeline(config):
    from market_forecaster.pipeline import Pipeline

    return Pipeline(
        config=config,
        price_source=FakePriceSource(points=200),
        news_source=FakeNewsSource(),
        forecaster=FakeForecaster(),
        narrator=FakeNarrator(),
    )
