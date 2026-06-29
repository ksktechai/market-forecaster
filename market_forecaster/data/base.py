"""Interfaces and shared types for price + news data sources.

The price-data source is abstracted so the default (Finnhub) can be swapped for a
fallback (Yahoo) or anything else, and so tests can inject a fake. Sources work in
terms of *canonical* symbols (NASDAQ, NZX50); each source maps those onto its own
ticker via the SymbolSpec in the config.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime

from market_forecaster.config import SymbolSpec


@dataclass
class PriceSeries:
    """A daily close series for one canonical symbol, plus provenance."""

    canonical: str
    source: str
    candles: list[tuple[datetime, float]] = field(default_factory=list)

    @property
    def closes(self) -> list[float]:
        return [c for _, c in self.candles]

    @property
    def dates(self) -> list[datetime]:
        return [d for d, _ in self.candles]

    @property
    def last_close(self) -> float | None:
        return self.candles[-1][1] if self.candles else None

    def __len__(self) -> int:
        return len(self.candles)


@dataclass
class Headline:
    headline: str
    source: str | None = None
    url: str | None = None
    when: datetime | None = None


# --- exceptions -------------------------------------------------------------


class DataSourceError(Exception):
    """Base for data-source problems."""


class SymbolNotAvailable(DataSourceError):
    """The source has no data for this symbol (and the app should try a fallback)."""


class DataSourceUnavailable(DataSourceError):
    """The source itself is unreachable / unauthorized (e.g. Finnhub 403 on free tier)."""


# --- interfaces -------------------------------------------------------------


class PriceDataSource(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def fetch_prices(self, spec: SymbolSpec, lookback_days: int) -> PriceSeries:
        """Return a daily close series for ``spec``.

        Raises ``SymbolNotAvailable`` if this source can't serve the symbol, or
        ``DataSourceUnavailable`` if the whole source is down/forbidden.
        """

    def ping(self) -> bool:
        """Best-effort reachability check for /health. Default: assume reachable."""
        return True


class NewsSource(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def fetch_news(self, spec: SymbolSpec, limit: int = 10) -> list[Headline]:
        """Return recent headlines relevant to ``spec``. Should not raise on empty."""
