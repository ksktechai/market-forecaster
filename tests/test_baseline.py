from market_forecaster.forecast import baseline


def test_naive_last_value():
    assert baseline.naive_last_value([1, 2, 3], 3) == [3, 3, 3]


def test_drift_forecast_extrapolates():
    out = baseline.drift_forecast([10, 12, 14], 2)
    assert out == [16, 18]


def test_mae():
    assert baseline.mae([1, 2, 3], [1, 2, 3]) == 0
    assert baseline.mae([1, 2, 3], [2, 3, 4]) == 1


def test_compare_detects_model_beating_naive():
    # Actuals rise; model predicts the rise, naive repeats the last train value.
    train = list(range(1, 51))  # 1..50
    test = [51, 52, 53]
    model_pred = [51, 52, 53]      # perfect
    naive_pred = [50, 50, 50]      # last train value
    cmp = baseline.compare(test_actual=test, model_pred=model_pred, naive_pred=naive_pred, train=train)
    assert cmp.beats_naive is True
    assert cmp.model_mae == 0
    assert cmp.mase == 0
    assert cmp.skill_vs_naive == 1.0


def test_no_backtest_still_returns_comparison():
    cmp = baseline.no_backtest("too short")
    assert cmp.method == "none"
    assert cmp.beats_naive is None
    assert "too short" in cmp.note
