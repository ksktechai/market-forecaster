"""Chained price source: try each source in order, fall through on failure.

This is what makes the app resilient to Finnhub's free-tier 403 on candles: the
default chain is just [Yahoo] (or [Finnhub, Yahoo] with a premium Finnhub plan).
If one source can't serve a symbol, we transparently fall through and record which
source actually served the data, so the output is honest about provenance.
"""

from __future__ import annotations

from market_forecaster.config import SymbolSpec
from market_forecaster.data.base import (
    DataSourceError,
    PriceDataSource,
    PriceSeries,
    SymbolNotAvailable,
)
from market_forecaster.logging_setup import get_logger

log = get_logger(__name__)


class ChainedPriceSource(PriceDataSource):
    name = "chained"

    def __init__(self, sources: list[PriceDataSource]):
        if not sources:
            raise ValueError("ChainedPriceSource needs at least one source")
        self._sources = sources

    def fetch_prices(self, spec: SymbolSpec, lookback_days: int) -> PriceSeries:
        errors: list[str] = []
        for source in self._sources:
            try:
                series = source.fetch_prices(spec, lookback_days)
                if len(series) == 0:
                    raise SymbolNotAvailable(f"{source.name} returned empty series")
                if errors:
                    log.info(
                        "data.fallback_used",
                        symbol=spec.canonical,
                        served_by=source.name,
                        skipped=errors,
                    )
                return series
            except DataSourceError as exc:
                msg = f"{source.name}: {exc}"
                log.warning("data.source_failed", symbol=spec.canonical, detail=msg)
                errors.append(msg)
        raise SymbolNotAvailable(
            f"No source could serve {spec.canonical}. Tried: " + " | ".join(errors)
        )

    def ping(self) -> bool:
        return any(s.ping() for s in self._sources)
