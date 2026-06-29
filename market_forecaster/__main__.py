"""Run the service: ``python -m market_forecaster``.

Loads .env (if present), configures logging, and starts uvicorn. For production
under Docker we run uvicorn directly (see Dockerfile), but this entrypoint is handy
for local first-run model downloads.
"""

from __future__ import annotations


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    import uvicorn

    from market_forecaster.config import load_config
    from market_forecaster.logging_setup import configure_logging, get_logger

    config = load_config()
    configure_logging(config.log_level)
    log = get_logger(__name__)
    log.info(
        "service.start",
        host=config.http_host,
        port=config.http_port,
        symbols=list(config.symbols),
        scheduler=config.enable_scheduler,
    )
    uvicorn.run(
        "market_forecaster.app:app",
        host=config.http_host,
        port=config.http_port,
        log_config=None,  # keep our JSON logging, don't let uvicorn override it
    )


if __name__ == "__main__":
    main()
