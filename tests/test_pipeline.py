from tests.conftest import FakeForecaster, FakeNarrator, FakeNewsSource, FakePriceSource

from market_forecaster.pipeline import Pipeline


def test_run_produces_baseline_for_every_symbol(fake_pipeline):
    run = fake_pipeline.run(symbols=["NASDAQ", "NZX50"], horizon=5)
    assert len(run.results) == 2
    # The invariant: a baseline comparison is ALWAYS present.
    for sf in run.results:
        assert sf.baseline is not None, f"{sf.symbol} missing baseline"
    assert run.telegram_text
    assert run.correlation_id


def test_backtest_metrics_present_with_long_history(fake_pipeline):
    sf = fake_pipeline.forecast_symbol("NASDAQ", 5)
    assert sf.baseline.method == "holdout-backtest"
    assert sf.baseline.mase is not None
    assert sf.baseline.beats_naive in (True, False)
    assert sf.point_forecast and len(sf.point_forecast) == 5
    assert sf.quantiles.band() is not None
    assert sf.news.sentiment == "bullish"
    assert sf.telegram_text


def test_short_history_still_has_baseline(config):
    pipe = Pipeline(
        config=config,
        price_source=FakePriceSource(points=20),  # too short for a 5-step holdout
        news_source=FakeNewsSource(),
        forecaster=FakeForecaster(),
        narrator=FakeNarrator(),
    )
    sf = pipe.forecast_symbol("NASDAQ", 5)
    assert sf.baseline is not None
    assert sf.baseline.method == "none"


def test_unservable_symbol_reported_not_crashing(config):
    from market_forecaster.data.base import PriceDataSource, SymbolNotAvailable

    class DeadSource(PriceDataSource):
        name = "dead"

        def fetch_prices(self, spec, lookback_days):
            raise SymbolNotAvailable("nope")

    pipe = Pipeline(
        config=config,
        price_source=DeadSource(),
        news_source=FakeNewsSource(),
        forecaster=FakeForecaster(),
        narrator=FakeNarrator(),
    )
    run = pipe.run(symbols=["NASDAQ"], horizon=5)
    assert run.results[0].error is not None
    assert run.results[0].ok is False
    # Run as a whole still succeeds.
    assert run.telegram_text
