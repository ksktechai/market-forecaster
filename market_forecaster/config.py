"""Application configuration, loaded entirely from environment variables.

Every tunable is an env var (see ``.env.example``). A small symbol registry maps
the *canonical* names we expose (``NASDAQ``, ``NZX50``) onto provider-specific
tickers, because Finnhub and the Yahoo fallback disagree on how indices are named
-- and because NZX50 is not available on Finnhub, so prices come from Yahoo's real
^NZ50 index (with the ENZL ETF available as a documented override).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, replace

from market_forecaster.logging_setup import get_logger, register_secret

_log = get_logger(__name__)


def _get(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _get_bool(name: str, default: bool) -> bool:
    raw = _get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = _get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class SymbolSpec:
    """How one canonical symbol maps onto each data source.

    ``finnhub``/``yahoo`` are the provider tickers. ``proxy`` is True when the
    series is a documented stand-in (e.g. an ETF) rather than the index itself,
    so the output can say so honestly.
    """

    canonical: str
    display: str
    finnhub: str | None = None
    yahoo: str | None = None
    proxy: bool = False
    proxy_note: str = ""
    # Which market this belongs to, used by the scheduler to pick symbols.
    market: str = "US"


# Default registry. Yahoo Finance (via yfinance) serves both indices keylessly --
# including the real S&P/NZX 50 (^NZ50), so no ETF proxy is needed. Finnhub
# candles require a premium plan, so its tickers are only used when
# ENABLE_FINNHUB_CANDLES is on. (ENZL, the iShares MSCI NZ ETF, remains a handy
# documented override for NZX50 via SYMBOL_MAP_JSON if you ever want US-hours data.)
DEFAULT_SYMBOLS: dict[str, SymbolSpec] = {
    "NASDAQ": SymbolSpec(
        canonical="NASDAQ",
        display="NASDAQ Composite (^IXIC)",
        finnhub="^IXIC",
        yahoo="^IXIC",
        market="US",
    ),
    "NZX50": SymbolSpec(
        canonical="NZX50",
        display="S&P/NZX 50 (^NZ50)",
        finnhub=None,  # not on Finnhub even on premium; Yahoo has the real index
        yahoo="^NZ50",
        market="NZ",
    ),
    "ASX": SymbolSpec(
        canonical="ASX",
        display="S&P/ASX 200 (^AXJO)",
        yahoo="^AXJO",
        market="AU",
    ),
    "EUROPE": SymbolSpec(
        canonical="EUROPE",
        display="EURO STOXX 50 (^STOXX50E)",
        yahoo="^STOXX50E",
        market="EU",
    ),
    "SINGAPORE": SymbolSpec(
        canonical="SINGAPORE",
        display="Straits Times Index (^STI)",
        yahoo="^STI",
        market="SG",
    ),
    "KOREA": SymbolSpec(
        canonical="KOREA",
        display="KOSPI (^KS11)",
        yahoo="^KS11",
        market="KR",
    ),
    "INDIA": SymbolSpec(
        canonical="INDIA",
        display="Nifty 50 (^NSEI)",
        yahoo="^NSEI",
        market="IN",
    ),
    "UK": SymbolSpec(
        canonical="UK",
        display="FTSE 100 (^FTSE)",
        yahoo="^FTSE",
        market="UK",
    ),
}


@dataclass(frozen=True)
class MarketSchedule:
    """When to run the scheduled forecast for a market, in that market's own tz.

    Running each job in the market's local timezone (rather than fixed UTC) means
    daylight-saving shifts are handled automatically -- the job always fires a few
    minutes after that market's actual local close.
    """

    market: str
    cron: str
    timezone: str


# Default close-time schedules, in each market's LOCAL time/timezone, a few minutes
# after the cash-market close. Override per market via {MARKET}_CLOSE_CRON / {MARKET}_TZ
# (e.g. UK_CLOSE_CRON, UK_TZ). US_CLOSE_CRON / NZ_CLOSE_CRON keep working as before.
DEFAULT_MARKET_SCHEDULES: dict[str, tuple[str, str]] = {
    "US": ("5 16 * * 1-5", "America/New_York"),   # 16:05 ET
    "NZ": ("10 17 * * 1-5", "Pacific/Auckland"),  # 17:10 NZ
    "AU": ("15 16 * * 1-5", "Australia/Sydney"),  # 16:15 AET
    "EU": ("40 17 * * 1-5", "Europe/Berlin"),     # 17:40 CET (EURO STOXX close)
    "SG": ("10 17 * * 1-5", "Asia/Singapore"),    # 17:10 SGT
    "KR": ("40 15 * * 1-5", "Asia/Seoul"),        # 15:40 KST
    "IN": ("40 15 * * 1-5", "Asia/Kolkata"),      # 15:40 IST
    "UK": ("40 16 * * 1-5", "Europe/London"),     # 16:40 London
}


@dataclass(frozen=True)
class Config:
    # --- Finnhub ---
    finnhub_api_key: str | None = None
    finnhub_base_url: str = "https://finnhub.io/api/v1"

    # --- Ollama ---
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "qwen3"

    # --- Forecast ---
    symbols: tuple[str, ...] = ("NASDAQ", "NZX50")
    forecast_horizon: int = 5
    lookback_days: int = 365
    enable_backtest: bool = True

    # --- Data source behaviour ---
    # Yahoo (yfinance) is the working keyless price source. Finnhub candles need a
    # premium plan, so they are OFF by default (free keys 403). Turn on only if you
    # have a premium Finnhub plan -- it will then be tried first, Yahoo as fallback.
    enable_yahoo_source: bool = True
    enable_finnhub_candles: bool = False

    # --- Scheduler / delivery ---
    hermes_webhook_url: str | None = None
    enable_scheduler: bool = False
    # market code -> when to run that market's scheduled forecast (local tz).
    market_schedules: dict[str, MarketSchedule] = field(default_factory=dict)

    # --- HTTP ---
    http_host: str = "0.0.0.0"
    http_port: int = 8000

    # --- Misc ---
    log_level: str = "INFO"
    hf_home: str | None = None

    symbol_registry: dict[str, SymbolSpec] = field(default_factory=lambda: dict(DEFAULT_SYMBOLS))

    def spec(self, canonical: str) -> SymbolSpec:
        """Look up a symbol spec, case-insensitively, or raise KeyError."""
        key = canonical.strip().upper()
        if key not in self.symbol_registry:
            raise KeyError(f"Unknown symbol '{canonical}'. Known: {sorted(self.symbol_registry)}")
        return self.symbol_registry[key]

    def known_symbols(self) -> list[str]:
        return list(self.symbol_registry)


def load_config(env: dict | None = None) -> Config:
    """Build a Config from the environment.

    Also registers the Finnhub API key as a secret so it is redacted from logs.
    """
    if env is not None:
        # Allow tests to inject an explicit environment.
        old = dict(os.environ)
        os.environ.clear()
        os.environ.update({k: str(v) for k, v in env.items()})
        try:
            return load_config(env=None)
        finally:
            os.environ.clear()
            os.environ.update(old)

    finnhub_key = _get("FINNHUB_API_KEY")
    if finnhub_key:
        register_secret(finnhub_key)

    registry = dict(DEFAULT_SYMBOLS)
    # Optional JSON override: {"NASDAQ": {"finnhub": "...", "yahoo": "..."}, ...}
    raw_map = _get("SYMBOL_MAP_JSON")
    if raw_map:
        try:
            overrides = json.loads(raw_map)
            for canon, fields in overrides.items():
                base = registry.get(canon.upper()) or SymbolSpec(canonical=canon.upper(), display=canon)
                registry[canon.upper()] = replace(base, **fields)
        except (ValueError, TypeError):
            pass  # bad override JSON: keep defaults rather than crash

    symbols_raw = _get("SYMBOLS")
    if symbols_raw:
        symbols = tuple(s.strip().upper() for s in symbols_raw.split(",") if s.strip())
    else:
        symbols = ("NASDAQ", "NZX50")

    schedules: dict[str, MarketSchedule] = {}
    for market, (default_cron, default_tz) in DEFAULT_MARKET_SCHEDULES.items():
        schedules[market] = MarketSchedule(
            market=market,
            cron=_get(f"{market}_CLOSE_CRON", default_cron),
            timezone=_get(f"{market}_TZ", default_tz),
        )

    return Config(
        finnhub_api_key=finnhub_key,
        finnhub_base_url=_get("FINNHUB_BASE_URL", "https://finnhub.io/api/v1"),
        ollama_base_url=_get("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
        ollama_model=_get("OLLAMA_MODEL", "qwen3"),
        symbols=symbols,
        forecast_horizon=_get_int("FORECAST_HORIZON", 5),
        lookback_days=_get_int("LOOKBACK_DAYS", 365),
        enable_backtest=_get_bool("ENABLE_BACKTEST", True),
        enable_yahoo_source=_get_bool("ENABLE_YAHOO_SOURCE", True),
        enable_finnhub_candles=_get_bool("ENABLE_FINNHUB_CANDLES", False),
        hermes_webhook_url=_get("HERMES_WEBHOOK_URL"),
        enable_scheduler=_get_bool("ENABLE_SCHEDULER", False),
        market_schedules=schedules,
        http_host=_get("HTTP_HOST", "0.0.0.0"),
        http_port=_get_int("HTTP_PORT", 8000),
        log_level=_get("LOG_LEVEL", "INFO"),
        hf_home=_get("HF_HOME"),
        symbol_registry=registry,
    )


def resolve_hf_home(hf_home: str | None) -> str | None:
    """Ensure a writable Hugging Face cache dir and export HF_HOME to it.

    The Docker default (``/models/hf``) is read-only when the app runs locally on
    macOS, which crashes the TimesFM load. We try the configured path first, then
    fall back to the user cache and finally a temp dir, exporting HF_HOME so
    huggingface_hub / timesfm pick it up. Returns the resolved path (or None if
    even the temp dir failed, in which case the HF library default is used).
    """
    candidates: list[str] = []
    if hf_home:
        candidates.append(os.path.expanduser(hf_home))
    candidates.append(os.path.join(os.path.expanduser("~/.cache"), "huggingface"))
    candidates.append(os.path.join(tempfile.gettempdir(), "market-forecaster-hf"))

    for candidate in candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
            probe = os.path.join(candidate, ".write_test")
            with open(probe, "w") as fh:
                fh.write("ok")
            os.remove(probe)
        except OSError:
            continue
        os.environ["HF_HOME"] = candidate
        if hf_home and os.path.expanduser(hf_home) != candidate:
            _log.warning(
                "hf_home.fallback",
                requested=hf_home,
                using=candidate,
                hint="configured HF_HOME was not writable (container path on a local run?)",
            )
        else:
            _log.info("hf_home.ready", path=candidate)
        return candidate

    _log.warning("hf_home.unresolved", hint="using huggingface library default")
    return None
