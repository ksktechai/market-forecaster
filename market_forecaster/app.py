"""FastAPI HTTP layer + app wiring.

Endpoints:
  POST /forecast            -> {symbols?, horizon?} -> full RunResult JSON
  GET  /forecast/{symbol}   -> convenience single-symbol forecast
  GET  /health              -> readiness (model loaded? data source reachable?)

The scheduler (if enabled) is started on app startup and shut down on exit.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from market_forecaster.config import Config, load_config, resolve_hf_home
from market_forecaster.logging_setup import configure_logging, get_logger
from market_forecaster.models import RunResult, SymbolForecast
from market_forecaster.pipeline import Pipeline, build_pipeline

log = get_logger(__name__)


class ForecastRequest(BaseModel):
    symbols: list[str] | None = Field(default=None, description="Canonical symbols, e.g. ['NASDAQ','NZX50']")
    horizon: int | None = Field(default=None, ge=1, le=60)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_source_reachable: bool
    ollama_reachable: bool
    known_symbols: list[str]


def create_app(config: Config | None = None, pipeline: Pipeline | None = None) -> FastAPI:
    config = config or load_config()
    configure_logging(config.log_level)
    # Make sure the HF cache is writable before anything tries to load TimesFM
    # (the Docker default /models/hf is read-only on a local macOS run).
    resolve_hf_home(config.hf_home)
    pipeline = pipeline or build_pipeline(config)

    scheduler = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal scheduler
        if config.enable_scheduler:
            from market_forecaster.scheduler import build_scheduler

            scheduler = build_scheduler(pipeline, config)
            scheduler.start()
            log.info("app.scheduler_started")
        else:
            log.info("app.scheduler_disabled")
        yield
        if scheduler is not None:
            scheduler.shutdown(wait=False)
            log.info("app.scheduler_stopped")

    app = FastAPI(title="market-forecaster", version="0.1.0", lifespan=lifespan)
    app.state.config = config
    app.state.pipeline = pipeline

    @app.post("/forecast", response_model=RunResult)
    def forecast(req: ForecastRequest) -> RunResult:
        symbols = req.symbols
        if symbols:
            unknown = [s for s in symbols if s.strip().upper() not in config.symbol_registry]
            if unknown:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown symbols {unknown}. Known: {config.known_symbols()}",
                )
        return pipeline.run(symbols=symbols, horizon=req.horizon)

    @app.get("/forecast/{symbol}", response_model=SymbolForecast)
    def forecast_one(symbol: str, horizon: int | None = None) -> SymbolForecast:
        if symbol.strip().upper() not in config.symbol_registry:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown symbol '{symbol}'. Known: {config.known_symbols()}",
            )
        run = pipeline.run(symbols=[symbol], horizon=horizon)
        return run.results[0]

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        model_loaded = pipeline.forecaster.is_ready()
        data_ok = _safe(pipeline.price_source.ping)
        ollama_ok = _safe(pipeline.narrator.is_ready) if pipeline.narrator else False
        status = "ok" if data_ok else "degraded"
        return HealthResponse(
            status=status,
            model_loaded=model_loaded,
            data_source_reachable=data_ok,
            ollama_reachable=ollama_ok,
            known_symbols=config.known_symbols(),
        )

    return app


def _safe(fn) -> bool:
    try:
        return bool(fn())
    except Exception:
        return False


# Uvicorn entrypoint: `uvicorn market_forecaster.app:app`
app = create_app() if __name__ != "__main__" else None
