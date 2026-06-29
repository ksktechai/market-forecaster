"""Secrets must never appear in logs."""

import json
import logging

from market_forecaster.logging_setup import JsonFormatter, register_secret


def _format(msg, **fields):
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )
    record.__dict__["extra_fields"] = fields
    return JsonFormatter().format(record)


def test_registered_secret_is_redacted():
    register_secret("SUPER-SECRET-API-KEY-123")
    line = _format("calling api", api_key="SUPER-SECRET-API-KEY-123")
    assert "SUPER-SECRET-API-KEY-123" not in line
    assert "***REDACTED***" in line
    # Still valid JSON.
    assert json.loads(line)["msg"] == "calling api"


def test_token_query_param_is_redacted():
    line = _format("fetch", url="https://finnhub.io/api/v1/quote?symbol=AAPL&token=abcd1234secret")
    assert "abcd1234secret" not in line
    assert "token=***REDACTED***" in line


def test_bearer_token_is_redacted():
    line = _format("auth", header="Authorization: Bearer sk-test-9999")
    assert "sk-test-9999" not in line
