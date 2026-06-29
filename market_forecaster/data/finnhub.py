"""Finnhub data source: price candles + market news.

IMPORTANT: Finnhub's free tier no longer serves historical stock *candles* -- the
``/stock/candle`` endpoint returns HTTP 403 on free keys (moved to premium). We
therefore:
  * raise ``DataSourceUnavailable`` on 403 so the app can fall back to Yahoo, and
  * keep using Finnhub for *news*, which still works on the free tier.

Every request and a response *summary* are logged with the run's correlation id.
The API key is sent as a query param but is registered as a secret and scrubbed
from logs (we also never log the raw URL with the token).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import requests

from market_forecaster.config import Config, SymbolSpec
from market_forecaster.data.base import (
    DataSourceUnavailable,
    Headline,
    NewsSource,
    PriceDataSource,
    PriceSeries,
    SymbolNotAvailable,
)
from market_forecaster.logging_setup import get_logger

log = get_logger(__name__)


class _FinnhubClient:
    """Shared HTTP plumbing + correlation-aware logging for Finnhub calls."""

    def __init__(self, config: Config, session: requests.Session | None = None):
        self._base = config.finnhub_base_url.rstrip("/")
        self._key = config.finnhub_api_key
        self._session = session or requests.Session()
        self._timeout = 15

    def get(self, endpoint: str, params: dict) -> tuple[int, object]:
        if not self._key:
            raise DataSourceUnavailable("FINNHUB_API_KEY is not set")
        url = f"{self._base}{endpoint}"
        # Log params WITHOUT the token.
        safe_params = {k: v for k, v in params.items() if k != "token"}
        log.info(
            ">>> finnhub.request",
            endpoint=endpoint,
            params=safe_params,
        )
        start = time.monotonic()
        call_params = dict(params)
        call_params["token"] = self._key
        try:
            resp = self._session.get(url, params=call_params, timeout=self._timeout)
        except requests.RequestException as exc:
            log.error("<<< finnhub.error", endpoint=endpoint, error=str(exc))
            raise DataSourceUnavailable(f"Finnhub request failed: {exc}") from exc
        latency_ms = round((time.monotonic() - start) * 1000, 1)

        if resp.status_code in (401, 403):
            log.warning(
                "<<< finnhub.forbidden",
                endpoint=endpoint,
                status=resp.status_code,
                latency_ms=latency_ms,
                hint="free tier likely lacks access to this endpoint",
            )
            raise DataSourceUnavailable(
                f"Finnhub {endpoint} returned {resp.status_code} "
                "(endpoint likely requires a premium plan)"
            )
        if resp.status_code == 429:
            raise DataSourceUnavailable("Finnhub rate limit (429)")
        if resp.status_code >= 400:
            log.warning(
                "<<< finnhub.http_error",
                endpoint=endpoint,
                status=resp.status_code,
                latency_ms=latency_ms,
            )
            raise DataSourceUnavailable(f"Finnhub {endpoint} HTTP {resp.status_code}")

        try:
            body = resp.json()
        except ValueError as exc:
            raise DataSourceUnavailable(f"Finnhub {endpoint} returned non-JSON") from exc

        row_count = len(body) if isinstance(body, list) else (
            len(body.get("t", [])) if isinstance(body, dict) else 0
        )
        log.info(
            "<<< finnhub.response",
            endpoint=endpoint,
            status=resp.status_code,
            latency_ms=latency_ms,
            rows=row_count,
        )
        return resp.status_code, body

    def ping(self) -> bool:
        try:
            self.get("/quote", {"symbol": "AAPL"})
            return True
        except Exception:
            return False


class FinnhubPriceSource(PriceDataSource):
    name = "finnhub"

    def __init__(self, config: Config, session: requests.Session | None = None):
        self._client = _FinnhubClient(config, session)

    def fetch_prices(self, spec: SymbolSpec, lookback_days: int) -> PriceSeries:
        if not spec.finnhub:
            raise SymbolNotAvailable(f"No Finnhub ticker mapped for {spec.canonical}")
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=lookback_days)
        _status, body = self._client.get(
            "/stock/candle",
            {
                "symbol": spec.finnhub,
                "resolution": "D",
                "from": int(start.timestamp()),
                "to": int(now.timestamp()),
            },
        )
        if not isinstance(body, dict) or body.get("s") != "ok":
            raise SymbolNotAvailable(
                f"Finnhub has no candle data for {spec.finnhub} "
                f"(status={body.get('s') if isinstance(body, dict) else 'n/a'})"
            )
        times = body.get("t", [])
        closes = body.get("c", [])
        candles = [
            (datetime.fromtimestamp(t, tz=timezone.utc), float(c))
            for t, c in zip(times, closes)
        ]
        candles.sort(key=lambda x: x[0])
        return PriceSeries(canonical=spec.canonical, source=self.name, candles=candles)

    def ping(self) -> bool:
        return self._client.ping()


class FinnhubNewsSource(NewsSource):
    name = "finnhub"

    def __init__(self, config: Config, session: requests.Session | None = None):
        self._client = _FinnhubClient(config, session)

    def fetch_news(self, spec: SymbolSpec, limit: int = 10) -> list[Headline]:
        # Indices have no useful /company-news, so we use general market news.
        # This still works on the free tier.
        try:
            _status, body = self._client.get("/news", {"category": "general"})
        except DataSourceUnavailable as exc:
            log.warning("finnhub.news_unavailable", error=str(exc))
            return []
        if not isinstance(body, list):
            return []
        headlines: list[Headline] = []
        for item in body[:limit]:
            if not isinstance(item, dict):
                continue
            ts = item.get("datetime")
            when = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
            headlines.append(
                Headline(
                    headline=str(item.get("headline", "")).strip(),
                    source=item.get("source"),
                    url=item.get("url"),
                    when=when,
                )
            )
        return [h for h in headlines if h.headline]
