"""Pydantic models for the structured forecast result (also the HTTP response schema)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Candle(BaseModel):
    """A single daily close."""

    date: datetime
    close: float


class QuantileForecast(BaseModel):
    """Quantile bands for the horizon. Keys are quantile levels (e.g. '0.1')."""

    levels: dict[str, list[float]] = Field(default_factory=dict)

    def band(self, lo: str = "0.1", hi: str = "0.9") -> tuple[list[float], list[float]] | None:
        if lo in self.levels and hi in self.levels:
            return self.levels[lo], self.levels[hi]
        return None


class BaselineComparison(BaseModel):
    """Always-present comparison of the model forecast vs a naive baseline.

    Computed on a held-out backtest window when the history is long enough.
    ``mase`` is model MAE divided by the in-sample naive one-step MAE (the scale).
    ``skill_vs_naive`` is 1 - (model MAE / naive MAE): positive means the model
    beat the naive baseline on the backtest.
    """

    method: str
    backtest_horizon: int | None = None
    model_mae: float | None = None
    naive_mae: float | None = None
    mase: float | None = None
    skill_vs_naive: float | None = None
    beats_naive: bool | None = None
    note: str = ""


class NewsHeadline(BaseModel):
    headline: str
    source: str | None = None
    url: str | None = None
    published: datetime | None = None


class NewsSentiment(BaseModel):
    sentiment: str = "unknown"  # bullish / bearish / neutral / mixed / unknown
    confidence: float | None = None
    narrative: str = ""
    headlines_considered: int = 0
    model: str | None = None
    error: str | None = None


class SymbolForecast(BaseModel):
    symbol: str
    display: str
    market: str = "US"
    source: str | None = None            # which data source actually served the data
    is_proxy: bool = False
    proxy_note: str = ""
    as_of: datetime | None = None
    horizon: int = 0
    current_price: float | None = None
    history_points: int = 0

    point_forecast: list[float] = Field(default_factory=list)
    quantiles: QuantileForecast = Field(default_factory=QuantileForecast)
    baseline: BaselineComparison | None = None
    news: NewsSentiment | None = None
    headlines: list[NewsHeadline] = Field(default_factory=list)

    telegram_text: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class RunResult(BaseModel):
    correlation_id: str
    generated_at: datetime
    horizon: int
    symbols_requested: list[str]
    results: list[SymbolForecast] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    # A single combined Telegram message for the whole run (all symbols).
    telegram_text: str = ""
