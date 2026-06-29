"""Forecaster interface shared by TimesFM and any fake used in tests.

This is the NUMERIC side of the two-model contrast: a numeric array goes in, a
point + quantile forecast comes out. No prompt, no tokens. Compare with
``narrative/ollama.py``, which is text-in / text-out.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass
class ForecastOutput:
    point: list[float]
    # quantile level (string, e.g. "0.1") -> per-horizon values
    quantiles: dict[str, list[float]] = field(default_factory=dict)


class Forecaster(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def forecast(self, history: list[float], horizon: int) -> ForecastOutput:
        """Forecast ``horizon`` steps ahead from a 1-D numeric ``history``."""

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """True once the model is loaded and ready to serve (for /health)."""
