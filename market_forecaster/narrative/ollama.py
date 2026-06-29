"""Ollama / qwen3 narrative + sentiment.

This is the PROMPT-based side of the two-model contrast: text (headlines + the
forecast numbers) goes in, text (a sentiment read + short narrative) comes out.
Compare with ``forecast/timesfm_forecaster.py``, which is numeric-array in /
numeric-array out.

We ask qwen3 for strict JSON so the result is structured, but degrade gracefully
to plain text if the model wanders. The full prompt and the raw completion are
logged (with latency + model name) under the run's correlation id.
"""

from __future__ import annotations

import json
import re
import time

import requests

from market_forecaster.data.base import Headline
from market_forecaster.logging_setup import get_logger
from market_forecaster.models import NewsSentiment

log = get_logger(__name__)

# qwen3 emits chain-of-thought inside <think>...</think>; strip it before parsing.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _extract_json(text: str) -> dict | None:
    text = _strip_think(text)
    # Find the first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except ValueError:
        return None


class OllamaNarrator:
    def __init__(
        self,
        base_url: str,
        model: str = "qwen3",
        session: requests.Session | None = None,
        timeout: int = 120,
    ):
        self._base = base_url.rstrip("/")
        self._model = model
        self._session = session or requests.Session()
        self._timeout = timeout

    def is_ready(self) -> bool:
        try:
            resp = self._session.get(f"{self._base}/api/tags", timeout=5)
            return resp.status_code < 400
        except requests.RequestException:
            return False

    def build_prompt(
        self,
        *,
        display: str,
        current_price: float | None,
        horizon: int,
        point_forecast: list[float],
        band: tuple[list[float], list[float]] | None,
        headlines: list[Headline],
    ) -> str:
        last_point = point_forecast[-1] if point_forecast else None
        band_desc = "not available"
        if band and band[0] and band[1]:
            lo = band[0][-1]
            hi = band[1][-1]
            band_desc = f"{lo:.2f} to {hi:.2f} (10th-90th percentile)"
        headline_block = "\n".join(f"- {h.headline}" for h in headlines[:10]) or "- (no headlines available)"

        cur = f"{current_price:.2f}" if current_price is not None else "unknown"
        end_pt = f"{last_point:.2f}" if last_point is not None else "unknown"

        return (
            "You are a markets analyst. Given recent headlines and a numeric price "
            "forecast, judge the news sentiment and write a SHORT plain-English "
            "narrative (2-3 sentences) that explicitly references the forecast number "
            "and the width of the uncertainty band.\n\n"
            f"Instrument: {display}\n"
            f"Current level: {cur}\n"
            f"Model point forecast at +{horizon} steps: {end_pt}\n"
            f"Uncertainty band at the horizon: {band_desc}\n\n"
            "Recent headlines:\n"
            f"{headline_block}\n\n"
            "Respond with STRICT JSON only, no markdown, in exactly this shape:\n"
            '{"sentiment": "bullish|bearish|neutral|mixed", '
            '"confidence": 0.0-1.0, '
            '"narrative": "<2-3 sentences referencing the forecast number and band width>"}'
        )

    def narrate(
        self,
        *,
        display: str,
        current_price: float | None,
        horizon: int,
        point_forecast: list[float],
        band: tuple[list[float], list[float]] | None,
        headlines: list[Headline],
    ) -> NewsSentiment:
        prompt = self.build_prompt(
            display=display,
            current_price=current_price,
            horizon=horizon,
            point_forecast=point_forecast,
            band=band,
            headlines=headlines,
        )
        log.info(">>> ollama.request", model=self._model, prompt=prompt, headlines=len(headlines))
        t0 = time.monotonic()
        try:
            resp = self._session.post(
                f"{self._base}/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": False},
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            log.error("<<< ollama.error", model=self._model, error=str(exc))
            return NewsSentiment(
                sentiment="unknown",
                narrative="",
                headlines_considered=len(headlines),
                model=self._model,
                error=f"Ollama request failed: {exc}",
            )
        latency_ms = round((time.monotonic() - t0) * 1000, 1)

        if resp.status_code >= 400:
            log.warning("<<< ollama.http_error", model=self._model, status=resp.status_code, latency_ms=latency_ms)
            return NewsSentiment(
                sentiment="unknown",
                narrative="",
                headlines_considered=len(headlines),
                model=self._model,
                error=f"Ollama HTTP {resp.status_code}",
            )

        raw_completion = resp.json().get("response", "") if resp.headers.get("content-type", "").startswith("application/json") else resp.text
        log.info(
            "<<< ollama.response",
            model=self._model,
            latency_ms=latency_ms,
            completion=raw_completion,
        )

        parsed = _extract_json(raw_completion)
        if parsed:
            sentiment = str(parsed.get("sentiment", "unknown")).lower().strip()
            try:
                confidence = float(parsed.get("confidence")) if parsed.get("confidence") is not None else None
            except (TypeError, ValueError):
                confidence = None
            narrative = str(parsed.get("narrative", "")).strip()
        else:
            sentiment = "unknown"
            confidence = None
            narrative = _strip_think(raw_completion)[:600]

        return NewsSentiment(
            sentiment=sentiment or "unknown",
            confidence=confidence,
            narrative=narrative,
            headlines_considered=len(headlines),
            model=self._model,
        )
