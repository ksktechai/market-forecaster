"""Scheduler builds one job per configured market, in that market's timezone."""

from market_forecaster.config import load_config
from market_forecaster.scheduler import _symbols_for_market, build_scheduler


def _pipeline(config):
    from market_forecaster.pipeline import Pipeline
    from tests.conftest import FakeForecaster, FakeNarrator, FakeNewsSource, FakePriceSource

    return Pipeline(config, FakePriceSource(), FakeNewsSource(), FakeForecaster(), FakeNarrator())


def test_symbols_grouped_by_market():
    cfg = load_config(env={"SYMBOLS": "NASDAQ,UK,INDIA"})
    assert _symbols_for_market(cfg, "US") == ["NASDAQ"]
    assert _symbols_for_market(cfg, "UK") == ["UK"]
    assert _symbols_for_market(cfg, "IN") == ["INDIA"]
    assert _symbols_for_market(cfg, "NZ") == []


def test_one_job_per_market_with_symbols():
    cfg = load_config(env={"SYMBOLS": "NASDAQ,UK,KOREA"})
    scheduler = build_scheduler(_pipeline(cfg), cfg)
    job_ids = {j.id for j in scheduler.get_jobs()}
    # US (NASDAQ), UK, KR (KOREA) -> jobs; nothing else.
    assert job_ids == {"us_close", "uk_close", "kr_close"}


def test_job_uses_market_local_timezone():
    cfg = load_config(env={"SYMBOLS": "UK"})
    scheduler = build_scheduler(_pipeline(cfg), cfg)
    uk = scheduler.get_job("uk_close")
    assert "Europe/London" in str(uk.trigger.timezone)


def test_per_market_override():
    cfg = load_config(env={"SYMBOLS": "INDIA", "IN_CLOSE_CRON": "0 16 * * 1-5", "IN_TZ": "Asia/Kolkata"})
    scheduler = build_scheduler(_pipeline(cfg), cfg)
    assert scheduler.get_job("in_close") is not None
    assert cfg.market_schedules["IN"].cron == "0 16 * * 1-5"


def test_bad_timezone_skips_job_without_crashing():
    cfg = load_config(env={"SYMBOLS": "UK", "UK_TZ": "Not/AZone"})
    scheduler = build_scheduler(_pipeline(cfg), cfg)
    assert scheduler.get_job("uk_close") is None
