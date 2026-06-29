# market-forecaster

Scheduled and on-demand market forecasts for **NASDAQ** and **NZX50**, combining a
**TimesFM** numeric forecast (point + quantile bands) with **LLM-generated news
sentiment** (Ollama / qwen3), exposed over HTTP for an external Telegram
orchestrator (HermesTechBot / OpenClaw) to deliver.

Two model types sit deliberately side by side so their difference is explicit:

| | TimesFM | qwen3 (Ollama) |
|---|---|---|
| Input → Output | numeric array → point + quantiles | text → text |
| Style | no prompt, no tokens | prompt-based |
| Module | `forecast/timesfm_forecaster.py` | `narrative/ollama.py` |

Every forecast **always** includes a naive-baseline comparison (MASE / beats-naive
on a held-out backtest) — a forecast is never shipped without showing whether it
beats naive.

See **[docs/architecture.md](docs/architecture.md)** for the deployment + sequence
diagrams (Mermaid, rendered on GitHub).

---

## ⚠️ Data-source caveat (read this first)

**Finnhub's free tier no longer serves historical price candles** —
`/stock/candle` returns HTTP 403 (it moved to premium). News still works on the
free tier. So prices come from **Yahoo Finance via the `yfinance` library** — the
working, keyless default:

- **Why not Stooq?** Stooq's CSV endpoint is now behind a JavaScript anti-bot
  challenge, and Yahoo's raw chart API rate-limits (429) un-authenticated callers.
  We use a **recent `yfinance` driven through a `curl_cffi` browser-impersonation
  session** (both pinned in `requirements.txt`) with retries — older yfinance hits
  a now-dead Yahoo endpoint and fails with `Expecting value`, and a non-browser
  client gets 429'd. This combination is the robust keyless option.
- **NASDAQ** = `^IXIC`, **NZX50** = `^NZ50` — Yahoo carries the **real S&P/NZX 50
  index**, so no ETF proxy is needed. (If you ever want US-hours NZ data, point
  NZX50 at the ENZL ETF via `SYMBOL_MAP_JSON` — see `.env.example`.)
- **Other markets** (all keyless via Yahoo): `ASX` (S&P/ASX 200 `^AXJO`),
  `EUROPE` (EURO STOXX 50 `^STOXX50E`), `SINGAPORE` (STI `^STI`), `KOREA`
  (KOSPI `^KS11`), `INDIA` (Nifty 50 `^NSEI`), `UK` (FTSE 100 `^FTSE`). Add/remove
  via the `SYMBOLS` env var; remap tickers via `SYMBOL_MAP_JSON`.
- **Finnhub candles** are tried only if you set `ENABLE_FINNHUB_CANDLES=true`
  (premium plan). When on, the chain is `[Finnhub → Yahoo]` and the output records
  which source actually served each symbol.

The price-data source is behind an interface (`data/base.py`), so swapping in a
premium/paid source is a drop-in change.

---

## Deploy (target: Apple-silicon M2 Max — NOT the Raspberry Pi)

TimesFM + torch are too heavy for the Pi. The Pi only runs HermesTechBot/OpenClaw
and calls this service across the LAN.

```bash
git clone https://github.com/ksktechai/market-forecaster.git
cd market-forecaster
cp .env.example .env        # fill FINNHUB_API_KEY, OLLAMA_BASE_URL, etc.
docker compose up -d --build
```

Update flow:

```bash
git pull && docker compose up -d --build
```

Notes:

- **Ollama runs on the host Mac**, outside the container. Point `OLLAMA_BASE_URL`
  at `http://host.docker.internal:11434`. Compose already sets the
  `host.docker.internal` host mapping. Make sure the model is pulled on the host:
  `ollama pull qwen3`.
- **macOS has no systemd.** For the container to survive a reboot, set **Docker
  Desktop → Settings → General → Start Docker Desktop when you log in**.
  `restart: unless-stopped` then keeps the service up.
- **First run is slow**: TimesFM weights download from Hugging Face and torch
  compiles. The `hf_cache` named volume persists weights so later starts are fast.
- The M2 Max uses CPU/MPS (no CUDA needed).

---

## Local run (no Docker) — handy for the first model download

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # core service
pip install -r requirements-model.txt    # torch + timesfm (heavy, from GitHub)
cp .env.example .env
python -m market_forecaster              # serves on :8000
```

> Use Python 3.11 (not 3.14): torch/timesfm lag the newest CPython.
> Running without `requirements-model.txt` works for everything **except** live
> TimesFM inference (the import is lazy). The test suite needs only
> `requirements-dev.txt`.

---

## Hitting the API

```bash
# Full run (defaults to SYMBOLS + FORECAST_HORIZON)
curl -s http://localhost:8000/forecast -X POST \
  -H 'content-type: application/json' \
  -d '{"symbols": ["NASDAQ", "NZX50"], "horizon": 5}' | jq

# Single symbol
curl -s "http://localhost:8000/forecast/NASDAQ?horizon=5" | jq

# Readiness (model loaded? data source reachable? ollama reachable?)
curl -s http://localhost:8000/health | jq
```

Each per-symbol result contains: `point_forecast`, `quantiles` (10th–90th),
`baseline` (MASE + beats-naive), `news` (sentiment + narrative), and a
`telegram_text` field. The run also carries a combined `telegram_text`.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/forecast` | `{symbols?, horizon?}` → full `RunResult` JSON |
| `GET` | `/forecast/{symbol}` | single-symbol convenience |
| `GET` | `/health` | readiness probe |

---

## Scheduler & delivery

Set `ENABLE_SCHEDULER=true` to run **one cron job per market** — each fires a few
minutes after that market's local cash close, in the market's **own timezone** (so
daylight saving is handled automatically), forecasts that market's symbols, and
`POST`s the result to `HERMES_WEBHOOK_URL`. Only markets that have a symbol in
`SYMBOLS` get a job. HermesTechBot relays it to Telegram — this service **never
talks to Telegram directly**. Leave the scheduler off to run as a pure on-demand
service.

Built-in local-close defaults (override any with `{MARKET}_CLOSE_CRON` /
`{MARKET}_TZ`, e.g. `UK_CLOSE_CRON`, `UK_TZ`):

| Market | Default | Market | Default |
|---|---|---|---|
| US | 16:05 America/New_York | SG | 17:10 Asia/Singapore |
| NZ | 17:10 Pacific/Auckland | KR | 15:40 Asia/Seoul |
| AU | 16:15 Australia/Sydney | IN | 15:40 Asia/Kolkata |
| EU | 17:40 Europe/Berlin | UK | 16:40 Europe/London |

The webhook payload is `{correlation_id, text, result}` where `text` is the
Telegram-ready message and `result` is the full JSON.

---

## Logging

Structured **JSON lines** to stdout (Docker captures them). Each run gets a
**correlation id**, with `>>>` on entry / `<<<` on exit around every external hop
(Finnhub, TimesFM, Ollama, Hermes), so one run is traceable end to end. Every
Finnhub/Ollama/TimesFM call logs request + response summary (status, rows/shape,
latency). **Secrets are redacted** — the API key and any `token=`/`Bearer` value
are scrubbed from all output. `LOG_LEVEL` is configurable.

---

## Configuration

All via env vars (see [`.env.example`](.env.example)):

`FINNHUB_API_KEY`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `SYMBOLS`,
`FORECAST_HORIZON`, `LOOKBACK_DAYS`, `HERMES_WEBHOOK_URL`, `ENABLE_SCHEDULER`,
`{MARKET}_CLOSE_CRON` / `{MARKET}_TZ` (per-market schedule overrides),
`LOG_LEVEL`, `HF_HOME`, plus `ENABLE_YAHOO_SOURCE`, `ENABLE_FINNHUB_CANDLES`,
`ENABLE_BACKTEST`, `SYMBOL_MAP_JSON`, `HTTP_PORT`.

---

## Tests

Offline — Finnhub, Ollama and TimesFM are all mocked.

```bash
pip install -r requirements-dev.txt
pytest
```

Includes a test asserting the baseline comparison is **always** present, a test for
the Finnhub-403 → Yahoo fallback, and a secret-redaction test.

---

## Module layout

```
market_forecaster/
  config.py                  env config + symbol registry (canonical → provider tickers)
  logging_setup.py           JSON logging, correlation ids, secret redaction
  models.py                  pydantic result schema (also the HTTP response)
  data/
    base.py                  PriceDataSource / NewsSource interfaces + types
    finnhub.py               Finnhub prices (premium) + news (free tier)
    yahoo.py                 keyless Yahoo/yfinance price source (the working default)
    chained.py               try-in-order fallback chain
  forecast/
    base.py                  Forecaster interface
    timesfm_forecaster.py    TimesFM 2.5 (numeric in → point+quantiles out)
    baseline.py              naive/drift baselines + MASE comparison
  narrative/
    ollama.py                qwen3 sentiment + narrative (text in → text out)
  formatting.py              Telegram-ready text
  pipeline.py                one run = one correlation id; orchestration + Hermes POST
  scheduler.py               APScheduler US/NZ close crons
  app.py                     FastAPI endpoints + lifespan
  __main__.py                `python -m market_forecaster`
```
