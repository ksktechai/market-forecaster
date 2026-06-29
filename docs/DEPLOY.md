# Deploying market-forecaster (Apple-silicon M2 Max)

Deploy target is an **Apple-silicon M2 Max** (or any x86/arm host with enough RAM).
**Not** the Raspberry Pi — TimesFM + torch are too heavy for the Pi. The Pi only
runs HermesTechBot/OpenClaw and calls this service across the LAN.

Two ways to run it:

- **Docker** — recommended; this is what survives reboots and persists the model cache.
- **Local (no Docker)** — handy for the first model download / debugging.

---

## Prerequisites

- **Docker Desktop** for Mac (for the Docker path), or **Python 3.11** (for the
  local path — use 3.11, *not* 3.14: torch/timesfm lag newer CPython).
- **Ollama** running and reachable, with the narrative model pulled:
  ```bash
  ollama pull qwen3:8b
  ```
- A **Finnhub API key** (free tier is fine — it's used for news only).

---

## 1. Clone & configure

```bash
git clone https://github.com/ksktechai/market-forecaster.git
cd market-forecaster
cp .env.example .env
```

Edit `.env`. The values that matter:

| Var | Set it to | Notes |
|---|---|---|
| `FINNHUB_API_KEY` | your Finnhub key | **News only.** Free tier is fine (candles aren't used). |
| `OLLAMA_BASE_URL` | `http://192.168.1.4:11434` | Where Ollama runs (your LAN IP works from inside Docker too). If Ollama is on *this same Mac*, `http://host.docker.internal:11434` also works. |
| `OLLAMA_MODEL` | `qwen3:8b` | Must be pulled on the Ollama host. |
| `SYMBOLS` | `NASDAQ,NZX50,ASX,EUROPE,SINGAPORE,KOREA,INDIA,UK` | Comma-separated canonical names. |
| `ENABLE_SCHEDULER` | `true` to auto-run at each market close, else `false` | If `true`, set `HERMES_WEBHOOK_URL`. |
| `HERMES_WEBHOOK_URL` | your HermesBot webhook | Where scheduled results are POSTed. Blank = on-demand only. |
| `HF_HOME` | leave `/models/hf` | Docker volume path. Locally the app auto-falls back to `~/.cache/huggingface` if this isn't writable. |

Sensible defaults you can leave as-is: `FORECAST_HORIZON=5`, `LOOKBACK_DAYS=365`,
`ENABLE_BACKTEST=true`, `ENABLE_YAHOO_SOURCE=true`, `ENABLE_FINNHUB_CANDLES=false`,
`HTTP_PORT=8000`, `LOG_LEVEL=INFO`.

> **Per-market schedule overrides** (optional): `{MARKET}_CLOSE_CRON` / `{MARKET}_TZ`,
> e.g. `UK_CLOSE_CRON="40 16 * * 1-5"`, `UK_TZ="Europe/London"`. Built-in defaults
> already fire a few minutes after each market's local close, in that market's tz.

---

## 2a. Run with Docker (recommended)

```bash
docker compose up -d --build
```

- **macOS has no systemd**, so for the container to survive a reboot:
  Docker Desktop → Settings → General → **"Start Docker Desktop when you log in"**.
  `restart: unless-stopped` then keeps it up.
- **First run is slow**: TimesFM weights download from Hugging Face and torch
  compiles. The `hf_cache` named volume persists weights, so it's only slow once.
- Follow logs: `docker compose logs -f` (structured JSON lines).
- Stop: `docker compose down`.

## 2b. Run locally (no Docker)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # core service
pip install -r requirements-model.txt    # torch + timesfm (heavy, from GitHub — needs git)
python -m market_forecaster              # serves on :8000
```

The M2 Max uses CPU/MPS (no CUDA needed).

---

## 3. Verify

```bash
# readiness (data source + ollama reachable? model loaded?)
curl -s http://localhost:8000/health | jq

# a single market — NOTE: TimesFM loads lazily here, so the FIRST call is slow
curl -s "http://localhost:8000/forecast/UK?horizon=5" | jq

# full run, all symbols
curl -s http://localhost:8000/forecast -X POST \
  -H 'content-type: application/json' \
  -d '{"symbols":["NASDAQ","NZX50","ASX","EUROPE","SINGAPORE","KOREA","INDIA","UK"],"horizon":5}' | jq
```

Each per-symbol result has `point_forecast`, `quantiles` (10th–90th), `baseline`
(MASE / beats-naive), `news` (sentiment + narrative), and `telegram_text`.

---

## 4. Update flow

```bash
git pull && docker compose up -d --build
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| First `/forecast` takes a long time | Expected — TimesFM downloads/compiles on the first request (not at startup). `/health` stays fast; later calls are quick. The HF cache volume avoids re-downloads. |
| `news.error` set / sentiment `unknown` | Ollama not reachable or `qwen3:8b` not pulled. Forecast still succeeds (graceful). Check `OLLAMA_BASE_URL` and `ollama pull qwen3:8b`. |
| `no price data` for a symbol | Yahoo is intermittently flaky; the source retries. Re-run; check LAN/internet. Other symbols are unaffected (per-symbol isolation). |
| `Read-only file system: '/models'` (local run) | `HF_HOME` points at the container path. The app auto-falls back to `~/.cache/huggingface`; ensure you're on current code. |
| `forecast model failed: No module named 'timesfm'` (local run) | Install the model deps: `pip install -r requirements-model.txt`. |
| Port 8000 already in use | Change `HTTP_PORT` in `.env` (Docker maps `${HTTP_PORT}:8000`). |

For the architecture and request flow, see [architecture.md](architecture.md).
