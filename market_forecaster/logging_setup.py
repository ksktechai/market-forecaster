"""Structured (JSON-lines) logging with per-run correlation ids and secret redaction.

Design goals (from the spec):
  * One JSON object per line on stdout, so Docker captures it.
  * A correlation id threaded through a whole forecast run, with ``>>>`` on entry
    and ``<<<`` on exit markers, so one run is traceable across Finnhub / TimesFM /
    Ollama.
  * Secrets (API keys, tokens) are NEVER written, even if a caller accidentally
    passes one into a log field.

Usage::

    configure_logging("INFO")
    log = get_logger(__name__)
    with correlation_scope() as cid:
        log.info(">>> forecast.run start", symbols=["NASDAQ"])
        ...
        log.info("<<< forecast.run done", latency_ms=1234)
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import re
import sys
import uuid
from datetime import datetime, timezone

# --- correlation id ---------------------------------------------------------

_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)


def new_correlation_id() -> str:
    return uuid.uuid4().hex[:12]


def set_correlation_id(cid: str | None) -> None:
    _correlation_id.set(cid)


def get_correlation_id() -> str | None:
    return _correlation_id.get()


@contextlib.contextmanager
def correlation_scope(cid: str | None = None):
    """Bind a correlation id for the duration of a ``with`` block."""
    cid = cid or new_correlation_id()
    token = _correlation_id.set(cid)
    try:
        yield cid
    finally:
        _correlation_id.reset(token)


# --- secret redaction -------------------------------------------------------

_secrets: set[str] = set()

# token=<value> in URLs/query strings, regardless of registered secrets.
_TOKEN_RE = re.compile(r"(token=)([^&\s\"']+)", re.IGNORECASE)
# Common bearer/api-key shapes.
_BEARER_RE = re.compile(r"(Bearer\s+)([A-Za-z0-9._\-]+)", re.IGNORECASE)


def register_secret(value: str) -> None:
    """Register a literal secret string to be scrubbed from all log output."""
    if value and len(value) >= 4:
        _secrets.add(value)


def _redact(text: str) -> str:
    for secret in _secrets:
        if secret:
            text = text.replace(secret, "***REDACTED***")
    text = _TOKEN_RE.sub(r"\1***REDACTED***", text)
    text = _BEARER_RE.sub(r"\1***REDACTED***", text)
    return text


# --- JSON formatter ---------------------------------------------------------

_RESERVED = set(logging.makeLogRecord({}).__dict__)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        cid = get_correlation_id()
        if cid:
            payload["correlation_id"] = cid

        extra = record.__dict__.get("extra_fields")
        if extra:
            payload.update(extra)

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        # Serialize, then redact the whole line so secrets can't slip through any
        # field (including ones a caller passed in by mistake).
        line = json.dumps(payload, default=str, ensure_ascii=False)
        return _redact(line)


class StructLogger:
    """Thin wrapper that turns ``log.info("event", key=value)`` into structured fields."""

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _log(self, level: int, event: str, **fields) -> None:
        if self._logger.isEnabledFor(level):
            self._logger.log(level, event, extra={"extra_fields": fields})

    def debug(self, event: str, **fields) -> None:
        self._log(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields) -> None:
        self._log(logging.INFO, event, **fields)

    def warning(self, event: str, **fields) -> None:
        self._log(logging.WARNING, event, **fields)

    def error(self, event: str, **fields) -> None:
        self._log(logging.ERROR, event, **fields)

    def exception(self, event: str, **fields) -> None:
        if self._logger.isEnabledFor(logging.ERROR):
            self._logger.log(logging.ERROR, event, exc_info=True, extra={"extra_fields": fields})


def get_logger(name: str) -> StructLogger:
    return StructLogger(name)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger, writing to stdout."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Replace any existing handlers so we don't double-log under uvicorn.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)

    # Quiet down noisy third-party loggers but keep them JSON-formatted.
    for noisy in ("uvicorn.access", "apscheduler", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
