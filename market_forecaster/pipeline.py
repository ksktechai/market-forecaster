"""The single-run forecast pipeline: fetch -> forecast -> backtest -> news -> narrate.

One ``run()`` call == one correlation id, logged with ``>>>`` / ``<<<`` markers so a
whole run is traceable end-to-end across Finnhub, TimesFM and Ollama. Per-symbol
failures are isolated: a symbol we can't serve is reported in its own result with
an ``error`` set, and the rest of the run continues.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

from market_forecaster.config import Config
from market_forecaster.data.base import (
    DataSourceError,
    NewsSource,
    PriceDataSource,
)
from market_forecaster.data.chained import ChainedPriceSource
from market_forecaster.data.finnhub import FinnhubNewsSource, FinnhubPriceSource
from market_forecaster.data.yahoo import YahooPriceSource
from market_forecaster.forecast import baseline
from market_forecaster.forecast.base import Forecaster
from market_forecaster.forecast.timesfm_forecaster import TimesFMForecaster
from market_forecaster.formatting import format_run, format_symbol
from market_forecaster.logging_setup import correlation_scope, get_logger
from market_forecaster.models import (
    NewsHeadline,
    NewsSentiment,
    QuantileForecast,
    RunResult,
    SymbolForecast,
)
from market_forecaster.narrative.ollama import OllamaNarrator

log = get_logger(__name__)

# Minimum context we insist on before asking the model to forecast at all.
_MIN_CONTEXT = 12


@dataclass
class Pipeline:
    config: Config
    price_source: PriceDataSource
    news_source: NewsSource
    forecaster: Forecaster
    narrator: OllamaNarrator | None = None

    # --- public API --------------------------------------------------------

    def run(self, symbols: list[str] | None = None, horizon: int | None = None) -> RunResult:
        cid_symbols = symbols or list(self.config.symbols)
        h = horizon or self.config.forecast_horizon
        with correlation_scope() as cid:
            t0 = time.monotonic()
            log.info(">>> forecast.run", symbols=cid_symbols, horizon=h)
            results: list[SymbolForecast] = []
            errors: list[str] = []
            for canonical in cid_symbols:
                try:
                    results.append(self.forecast_symbol(canonical, h))
                except Exception as exc:  # never let one symbol kill the run
                    log.exception("forecast.symbol_crashed", symbol=canonical, error=str(exc))
                    errors.append(f"{canonical}: {exc}")
                    results.append(
                        SymbolForecast(symbol=canonical, display=canonical, horizon=h, error=str(exc))
                    )
            run = RunResult(
                correlation_id=cid,
                generated_at=datetime.now(timezone.utc),
                horizon=h,
                symbols_requested=cid_symbols,
                results=results,
                errors=errors,
            )
            run.telegram_text = format_run(run)
            log.info(
                "<<< forecast.run",
                horizon=h,
                ok=[r.symbol for r in results if r.ok],
                failed=[r.symbol for r in results if not r.ok],
                latency_ms=round((time.monotonic() - t0) * 1000, 1),
            )
            return run

    def forecast_symbol(self, canonical: str, horizon: int) -> SymbolForecast:
        spec = self.config.spec(canonical)
        log.info("forecast.symbol_start", symbol=spec.canonical, horizon=horizon)

        # 1) Prices (chained source; Yahoo by default, Finnhub first if premium).
        try:
            series = self.price_source.fetch_prices(spec, self.config.lookback_days)
        except DataSourceError as exc:
            log.warning("forecast.no_prices", symbol=spec.canonical, error=str(exc))
            return SymbolForecast(
                symbol=spec.canonical,
                display=spec.display,
                market=spec.market,
                horizon=horizon,
                is_proxy=spec.proxy,
                proxy_note=spec.proxy_note,
                error=f"no price data: {exc}",
            )

        closes = series.closes
        if len(closes) < _MIN_CONTEXT:
            return SymbolForecast(
                symbol=spec.canonical,
                display=spec.display,
                market=spec.market,
                source=series.source,
                horizon=horizon,
                history_points=len(closes),
                is_proxy=spec.proxy,
                proxy_note=spec.proxy_note,
                error=f"insufficient history ({len(closes)} points, need >= {_MIN_CONTEXT})",
            )

        # 2) Forecast the actual future. Isolate model failures (e.g. TimesFM/HF
        #    cache or weight-load problems) so they keep the symbol's context.
        try:
            output = self.forecaster.forecast(closes, horizon)
            comparison = self._backtest(closes, horizon)  # 3) baseline (ALWAYS present)
        except Exception as exc:
            log.exception("forecast.model_failed", symbol=spec.canonical, error=str(exc))
            return SymbolForecast(
                symbol=spec.canonical,
                display=spec.display,
                market=spec.market,
                source=series.source,
                is_proxy=spec.proxy,
                proxy_note=spec.proxy_note,
                as_of=series.dates[-1] if series.dates else None,
                horizon=horizon,
                current_price=series.last_close,
                history_points=len(closes),
                error=f"forecast model failed: {exc}",
            )

        # 4) News + 5) narrative.
        headlines = self._fetch_news(spec)
        news = self._narrate(spec, series.last_close, horizon, output, headlines)

        sf = SymbolForecast(
            symbol=spec.canonical,
            display=spec.display,
            market=spec.market,
            source=series.source,
            is_proxy=spec.proxy,
            proxy_note=spec.proxy_note,
            as_of=series.dates[-1] if series.dates else None,
            horizon=horizon,
            current_price=series.last_close,
            history_points=len(closes),
            point_forecast=output.point,
            quantiles=QuantileForecast(levels=output.quantiles),
            baseline=comparison,
            news=news,
            headlines=[
                NewsHeadline(headline=h.headline, source=h.source, url=h.url, published=h.when)
                for h in headlines
            ],
        )
        sf.telegram_text = format_symbol(sf)
        return sf

    # --- steps -------------------------------------------------------------

    def _backtest(self, closes: list[float], horizon: int):
        if not self.config.enable_backtest:
            return baseline.no_backtest("backtest disabled (ENABLE_BACKTEST=false)")
        if len(closes) < 2 * horizon + _MIN_CONTEXT:
            return baseline.no_backtest(
                f"history too short for a holdout backtest (have {len(closes)})"
            )
        train = closes[:-horizon]
        test = closes[-horizon:]
        bt = self.forecaster.forecast(train, horizon)
        naive_pred = baseline.naive_last_value(train, horizon)
        comparison = baseline.compare(
            test_actual=test, model_pred=bt.point, naive_pred=naive_pred, train=train
        )
        log.info(
            "forecast.baseline",
            model_mae=comparison.model_mae,
            naive_mae=comparison.naive_mae,
            mase=comparison.mase,
            beats_naive=comparison.beats_naive,
        )
        return comparison

    def _fetch_news(self, spec):
        try:
            return self.news_source.fetch_news(spec, limit=10)
        except Exception as exc:
            log.warning("forecast.news_failed", symbol=spec.canonical, error=str(exc))
            return []

    def _narrate(self, spec, current_price, horizon, output, headlines) -> NewsSentiment:
        if self.narrator is None:
            return NewsSentiment(sentiment="unknown", narrative="", headlines_considered=len(headlines))
        band = QuantileForecast(levels=output.quantiles).band()
        try:
            return self.narrator.narrate(
                display=spec.display,
                current_price=current_price,
                horizon=horizon,
                point_forecast=output.point,
                band=band,
                headlines=headlines,
            )
        except Exception as exc:
            log.warning("forecast.narrate_failed", symbol=spec.canonical, error=str(exc))
            return NewsSentiment(
                sentiment="unknown",
                narrative="",
                headlines_considered=len(headlines),
                error=str(exc),
            )


# --- delivery ---------------------------------------------------------------


def post_to_hermes(config: Config, run: RunResult, session: requests.Session | None = None) -> bool:
    """POST a completed run to the HermesTechBot webhook, if configured."""
    if not config.hermes_webhook_url:
        log.info("hermes.skip", reason="HERMES_WEBHOOK_URL not set")
        return False
    session = session or requests.Session()
    payload = {
        "correlation_id": run.correlation_id,
        "text": run.telegram_text,
        "result": run.model_dump(mode="json"),
    }
    log.info(">>> hermes.post", url=config.hermes_webhook_url, correlation_id=run.correlation_id)
    try:
        resp = session.post(config.hermes_webhook_url, json=payload, timeout=20)
        log.info("<<< hermes.post_done", status=resp.status_code)
        return resp.status_code < 400
    except requests.RequestException as exc:
        log.error("<<< hermes.post_failed", error=str(exc))
        return False


# --- factory ----------------------------------------------------------------


def build_pipeline(config: Config, session: requests.Session | None = None) -> Pipeline:
    """Wire the default production pipeline: (Finnhub if premium) -> Yahoo, TimesFM, Ollama.

    Finnhub candles are only chained in when ENABLE_FINNHUB_CANDLES is set (they
    need a premium plan); otherwise Yahoo is the working primary price source.
    """
    sources: list[PriceDataSource] = []
    if config.enable_finnhub_candles and config.finnhub_api_key:
        sources.append(FinnhubPriceSource(config, session))
    if config.enable_yahoo_source:
        sources.append(YahooPriceSource())
    if not sources:
        # Everything disabled: still give Yahoo so the app actually works.
        sources.append(YahooPriceSource())

    price_source = ChainedPriceSource(sources)
    news_source = FinnhubNewsSource(config, session)
    forecaster = TimesFMForecaster()
    narrator = OllamaNarrator(config.ollama_base_url, config.ollama_model, session)
    return Pipeline(
        config=config,
        price_source=price_source,
        news_source=news_source,
        forecaster=forecaster,
        narrator=narrator,
    )
