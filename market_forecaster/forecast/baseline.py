"""Naive baselines + the comparison metrics.

The spec is firm: never ship a forecast without showing whether it beats naive.
We compute this on a held-out backtest window (the last ``H`` points), comparing
the model's forecast of that window against:
  * the naive "last value" forecast, and
  * the in-sample naive one-step error (the MASE scale).

MASE = model_MAE / in_sample_naive_MAE. < 1.0 means the model beats the naive
benchmark on the backtest.
"""

from __future__ import annotations

from market_forecaster.models import BaselineComparison


def naive_last_value(history: list[float], horizon: int) -> list[float]:
    """Repeat the last observed value -- the canonical naive forecast."""
    if not history:
        return [0.0] * horizon
    return [history[-1]] * horizon


def drift_forecast(history: list[float], horizon: int) -> list[float]:
    """Random-walk-with-drift: extrapolate the average per-step change."""
    if len(history) < 2:
        return naive_last_value(history, horizon)
    drift = (history[-1] - history[0]) / (len(history) - 1)
    last = history[-1]
    return [last + drift * (i + 1) for i in range(horizon)]


def mae(actual: list[float], predicted: list[float]) -> float:
    n = min(len(actual), len(predicted))
    if n == 0:
        return float("nan")
    return sum(abs(actual[i] - predicted[i]) for i in range(n)) / n


def in_sample_naive_mae(train: list[float]) -> float:
    """Mean absolute one-step change over the training series (the MASE scale)."""
    if len(train) < 2:
        return float("nan")
    return sum(abs(train[i] - train[i - 1]) for i in range(1, len(train))) / (len(train) - 1)


def compare(
    *,
    test_actual: list[float],
    model_pred: list[float],
    naive_pred: list[float],
    train: list[float],
) -> BaselineComparison:
    """Build a BaselineComparison from a backtest window."""
    model_mae = mae(test_actual, model_pred)
    naive_mae = mae(test_actual, naive_pred)
    scale = in_sample_naive_mae(train)

    mase = model_mae / scale if scale and scale == scale and scale != 0 else None
    skill = None
    beats = None
    if naive_mae and naive_mae == naive_mae and naive_mae != 0:
        skill = 1.0 - (model_mae / naive_mae)
        beats = model_mae < naive_mae

    note = ""
    if beats is True:
        note = "Model beat the naive last-value baseline on the backtest window."
    elif beats is False:
        note = "Model did NOT beat the naive last-value baseline on the backtest window."

    return BaselineComparison(
        method="holdout-backtest",
        backtest_horizon=len(test_actual),
        model_mae=round(model_mae, 4),
        naive_mae=round(naive_mae, 4),
        mase=round(mase, 4) if mase is not None else None,
        skill_vs_naive=round(skill, 4) if skill is not None else None,
        beats_naive=beats,
        note=note,
    )


def no_backtest(reason: str) -> BaselineComparison:
    """Used when the history is too short to hold out a backtest window.

    A comparison object is still ALWAYS returned -- it just records why metrics
    are absent -- so the invariant 'baseline comparison is always present' holds.
    """
    return BaselineComparison(method="none", note=reason)
