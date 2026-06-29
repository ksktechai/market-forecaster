"""TimesFM 2.5 (google/timesfm-2.5-200m-pytorch) wrapper.

This is the numeric forecaster: a 1-D array of closes goes in, a point forecast +
quantile bands come out. There is NO prompt and NO tokenisation here -- that is the
whole point of keeping it separate from the Ollama narrative module.

``timesfm`` + ``torch`` are imported lazily inside ``load()`` so the rest of the
app (and the test suite) can run without the heavy model installed. The model is
loaded once and reused; weights are cached under HF_HOME so they are not
re-downloaded on every container start.

NOTE: TimesFM's Python API has shifted between releases. This targets the 2.5
``from_pretrained`` / ``ForecastConfig`` / ``forecast`` surface. If you pin a
different timesfm build, adjust ``load()`` / ``_run`` accordingly -- everything
else in the app depends only on the ``Forecaster`` interface, not on timesfm.
"""

from __future__ import annotations

import threading
import time

from market_forecaster.forecast.base import Forecaster, ForecastOutput
from market_forecaster.logging_setup import get_logger

log = get_logger(__name__)

# TimesFM 2.5 exposes 10 quantile heads at these levels.
_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class TimesFMForecaster(Forecaster):
    name = "timesfm-2.5-200m"

    def __init__(
        self,
        repo_id: str = "google/timesfm-2.5-200m-pytorch",
        max_context: int = 1024,
        max_horizon: int = 256,
    ):
        self._repo_id = repo_id
        self._max_context = max_context
        self._max_horizon = max_horizon
        self._model = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    # --- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Load + compile the model once. Safe to call repeatedly."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            t0 = time.monotonic()
            log.info(">>> timesfm.load", repo_id=self._repo_id)
            try:
                import timesfm  # noqa: F401  (heavy, lazy)

                model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self._repo_id)
                model.compile(
                    timesfm.ForecastConfig(
                        max_context=self._max_context,
                        max_horizon=self._max_horizon,
                        normalize_inputs=True,
                        use_continuous_quantile_head=True,
                    )
                )
                self._model = model
                self._load_error = None
                log.info(
                    "<<< timesfm.load_done",
                    latency_ms=round((time.monotonic() - t0) * 1000, 1),
                )
            except Exception as exc:  # pragma: no cover - exercised only with real model
                self._load_error = str(exc)
                log.exception("<<< timesfm.load_failed", error=str(exc))
                raise

    def is_ready(self) -> bool:
        return self._model is not None

    # --- inference ---------------------------------------------------------

    def forecast(self, history: list[float], horizon: int) -> ForecastOutput:
        if not history:
            raise ValueError("history is empty")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.load()

        # Trim to the model's max context window.
        series = list(history)[-self._max_context:]
        t0 = time.monotonic()
        log.info(
            ">>> timesfm.forecast",
            input_len=len(series),
            horizon=horizon,
        )
        point, quantiles = self._run(series, horizon)
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        log.info(
            "<<< timesfm.forecast_done",
            input_len=len(series),
            horizon=horizon,
            output_shape=[len(point)],
            quantile_levels=list(quantiles.keys()),
            latency_ms=latency_ms,
        )
        return ForecastOutput(point=point, quantiles=quantiles)

    def _run(self, series: list[float], horizon: int) -> tuple[list[float], dict[str, list[float]]]:
        """Call the real model and normalise its output to plain Python lists."""
        import numpy as np

        point_forecast, quantile_forecast = self._model.forecast(
            horizon=horizon,
            inputs=[np.asarray(series, dtype=np.float32)],
        )
        # point_forecast: [n_series, horizon]
        point = [float(x) for x in np.asarray(point_forecast)[0][:horizon]]
        quantiles = _extract_quantiles(quantile_forecast, horizon)
        return point, quantiles


def _extract_quantiles(quantile_forecast, horizon: int) -> dict[str, list[float]]:
    """Map TimesFM's quantile output to {level: per-horizon values}.

    TimesFM's quantile head returns one column per output, shaped
    [n_series, horizon, n_cols]. CRUCIAL detail: when the full head is present the
    columns are ``[mean, q0.1, q0.2, ..., q0.9]`` -- column 0 is the MEAN, so the
    deciles start at column 1. Mapping level 0.1 -> column 0 (as a naive
    ``enumerate`` would) is an off-by-one that yields non-monotonic, mislabeled
    bands. We detect the mean column by width and offset past it.
    """
    import numpy as np

    q = np.asarray(quantile_forecast, dtype=float)
    if q.ndim == 2:  # [horizon, n_cols] -> add the series axis
        q = q[None, ...]
    if q.ndim != 3:
        return {}

    series0 = q[0]  # [horizon, n_cols]
    n_cols = series0.shape[-1]
    # 10+ columns => a leading mean column; deciles begin at column 1.
    # Exactly 9 columns => the deciles themselves, starting at column 0.
    offset = 1 if n_cols >= len(_QUANTILE_LEVELS) + 1 else 0

    out: dict[str, list[float]] = {}
    for idx, level in enumerate(_QUANTILE_LEVELS):
        col = idx + offset
        if col < n_cols:
            out[str(level)] = [float(v) for v in series0[:horizon, col]]
    return out
