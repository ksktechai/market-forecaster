"""Yahoo Finance price source (via the ``yfinance`` library) -- keyless daily history.

This is the working default price source. yfinance handles Yahoo's crumb/cookie/
retry handshake internally (raw requests to the chart API get 429'd), and Yahoo
covers what we need keylessly:
  * ^IXIC  -> NASDAQ Composite
  * ^NZ50  -> the real S&P/NZX 50 index (no ETF proxy needed)
  * ENZL   -> iShares MSCI NZ ETF (available as a documented override)

The actual download is behind an injectable callable so the test suite can run
offline without yfinance touching the network.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from market_forecaster.config import SymbolSpec
from market_forecaster.data.base import (
    DataSourceUnavailable,
    PriceDataSource,
    PriceSeries,
    SymbolNotAvailable,
)
from market_forecaster.logging_setup import get_logger

log = get_logger(__name__)

# A downloader takes (yahoo_symbol, lookback_days) and returns [(datetime, close)].
Downloader = "Callable[[str, int], list[tuple[datetime, float]]]"


def _browser_session():
    """A curl_cffi session impersonating Chrome.

    Yahoo anti-bot/429s plain clients; impersonating a real browser is the
    documented way to make yfinance reliable. Falls back to None (yfinance's own
    client) if curl_cffi isn't installed.
    """
    try:
        from curl_cffi import requests as cffi_requests

        return cffi_requests.Session(impersonate="chrome")
    except Exception:  # pragma: no cover - only when curl_cffi missing
        return None


def _yfinance_download(symbol: str, lookback_days: int) -> list[tuple[datetime, float]]:
    """Default downloader: pull daily closes from Yahoo via yfinance.

    Retries a few times because Yahoo is intermittently flaky (429 / empty body).
    """
    import yfinance as yf  # lazy: keeps the core/test path import-light

    period = f"{max(lookback_days, 7)}d"
    session = _browser_session()
    last_exc: Exception | None = None

    for attempt in range(3):
        try:
            ticker = yf.Ticker(symbol, session=session) if session else yf.Ticker(symbol)
            df = ticker.history(period=period, interval="1d", auto_adjust=False)
            if df is not None and not df.empty and "Close" in df:
                out: list[tuple[datetime, float]] = []
                for ts, close in df["Close"].dropna().items():
                    when = ts.to_pydatetime()
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=timezone.utc)
                    out.append((when.astimezone(timezone.utc), float(close)))
                if out:
                    out.sort(key=lambda x: x[0])
                    return out
        except Exception as exc:  # network / parse hiccup -> retry
            last_exc = exc
            log.warning("yahoo.retry", symbol=symbol, attempt=attempt + 1, error=str(exc))
        time.sleep(0.6 * (attempt + 1))

    if last_exc is not None:
        raise last_exc
    return []  # genuinely no rows (e.g. delisted) -> caller raises SymbolNotAvailable


class YahooPriceSource(PriceDataSource):
    name = "yahoo"

    def __init__(self, downloader=None):
        self._download = downloader or _yfinance_download

    def fetch_prices(self, spec: SymbolSpec, lookback_days: int) -> PriceSeries:
        if not spec.yahoo:
            raise SymbolNotAvailable(f"No Yahoo ticker mapped for {spec.canonical}")
        log.info(">>> yahoo.request", symbol=spec.yahoo, lookback_days=lookback_days)
        t0 = time.monotonic()
        try:
            candles = self._download(spec.yahoo, lookback_days)
        except Exception as exc:  # network / library errors -> let the chain fall through
            log.error("<<< yahoo.error", symbol=spec.yahoo, error=str(exc))
            raise DataSourceUnavailable(f"Yahoo request failed: {exc}") from exc
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        log.info("<<< yahoo.response", symbol=spec.yahoo, latency_ms=latency_ms, rows=len(candles))
        if not candles:
            raise SymbolNotAvailable(f"Yahoo returned no data for {spec.yahoo}")
        return PriceSeries(canonical=spec.canonical, source=self.name, candles=candles)

    def ping(self) -> bool:
        try:
            return bool(self._download("^GSPC", 5))
        except Exception:
            return False
