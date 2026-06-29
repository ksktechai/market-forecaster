"""TimesFM quantile-column mapping (no model needed -- pure array reshaping)."""

import numpy as np

from market_forecaster.forecast.timesfm_forecaster import _extract_quantiles


def _synthetic_with_mean_column(horizon=5):
    """Shape [1, horizon, 10] = [mean, q0.1, ..., q0.9], deciles strictly increasing."""
    deciles = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)  # q0.1..q0.9
    mean_col = np.array([55.0])  # a mean that sits mid-pack (the trap)
    row = np.concatenate([mean_col, deciles])  # 10 cols
    arr = np.tile(row, (horizon, 1))[None, ...]  # [1, horizon, 10]
    return arr


def test_mean_column_is_skipped_and_bands_are_monotonic():
    out = _extract_quantiles(_synthetic_with_mean_column(), horizon=5)
    # 0.1 must be the real low (10), not the mean (55).
    assert out["0.1"][0] == 10.0
    assert out["0.9"][0] == 90.0
    # Monotonic non-decreasing across levels at each step.
    levels = [f"0.{i}" for i in range(1, 10)]
    first_step = [out[lv][0] for lv in levels]
    assert first_step == sorted(first_step)


def test_nine_column_output_has_no_mean_offset():
    # [1, horizon, 9] = deciles directly, no mean column.
    deciles = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)
    arr = np.tile(deciles, (5, 1))[None, ...]
    out = _extract_quantiles(arr, horizon=5)
    assert out["0.1"][0] == 10.0
    assert out["0.9"][0] == 90.0


def test_handles_2d_input():
    deciles = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)
    arr = np.tile(deciles, (5, 1))  # [horizon, 9], no series axis
    out = _extract_quantiles(arr, horizon=5)
    assert out["0.5"][0] == 50.0
