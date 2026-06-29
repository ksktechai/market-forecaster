"""APScheduler-based internal scheduler.

One cron job per market (US, NZ, AU, EU, SG, KR, IN, UK), each firing a few minutes
after that market's local cash close -- in the market's OWN timezone, so daylight
saving is handled automatically. Each job forecasts the symbols belonging to that
market and POSTs the result to the Hermes webhook for Telegram relay.

Only markets that actually have a symbol in ``SYMBOLS`` get a job. Entirely
optional: gated by ``ENABLE_SCHEDULER`` so the service can run as pure on-demand HTTP.
"""

from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from market_forecaster.config import Config
from market_forecaster.logging_setup import get_logger
from market_forecaster.pipeline import Pipeline, post_to_hermes

log = get_logger(__name__)


def _symbols_for_market(config: Config, market: str) -> list[str]:
    out = []
    for canonical in config.symbols:
        try:
            if config.spec(canonical).market == market:
                out.append(canonical)
        except KeyError:
            continue
    return out


def _run_market(pipeline: Pipeline, config: Config, market: str) -> None:
    symbols = _symbols_for_market(config, market)
    if not symbols:
        log.info("scheduler.no_symbols", market=market)
        return
    log.info("scheduler.trigger", market=market, symbols=symbols)
    run = pipeline.run(symbols=symbols)
    post_to_hermes(config, run)


def build_scheduler(pipeline: Pipeline, config: Config) -> BackgroundScheduler:
    # The scheduler's own bookkeeping runs in UTC; each job carries its market tz.
    scheduler = BackgroundScheduler(timezone="UTC")
    added: list[dict] = []

    for market, sched in config.market_schedules.items():
        symbols = _symbols_for_market(config, market)
        if not symbols:
            continue  # no configured symbols for this market -> no job
        try:
            trigger = CronTrigger.from_crontab(sched.cron, timezone=sched.timezone)
        except Exception as exc:
            log.error(
                "scheduler.bad_schedule",
                market=market,
                cron=sched.cron,
                timezone=sched.timezone,
                error=str(exc),
            )
            continue
        scheduler.add_job(
            _run_market,
            trigger,
            args=[pipeline, config, market],
            id=f"{market.lower()}_close",
            name=f"{market} market close forecast",
            replace_existing=True,
        )
        added.append({"market": market, "cron": sched.cron, "timezone": sched.timezone, "symbols": symbols})
        log.info(
            "scheduler.job_added",
            market=market,
            cron=sched.cron,
            timezone=sched.timezone,
            symbols=symbols,
        )

    if not added:
        log.warning(
            "scheduler.no_jobs",
            hint="no market in SYMBOLS has a schedule; scheduler will idle",
        )
    log.info("scheduler.configured", jobs=len(added))
    return scheduler
