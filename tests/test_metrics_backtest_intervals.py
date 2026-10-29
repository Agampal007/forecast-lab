import numpy as np
import pytest

from forecastlab import backtest as bt
from forecastlab.data import make_demand
from forecastlab.intervals import calibration_errors, forecast_with_interval, interval_widths
from forecastlab.metrics import coverage, mae, mase, rmse, smape
from forecastlab.models import Forecaster, make_model


def test_point_metrics_known_values():
    y, p = np.array([10.0, 20.0, 30.0]), np.array([12.0, 18.0, 30.0])
    assert mae(y, p) == pytest.approx(4 / 3)
    assert rmse(y, p) == pytest.approx(np.sqrt(8 / 3))
    expected = 100 * np.mean([2 / 11, 2 / 19, 0.0])
    assert smape(y, p) == pytest.approx(expected)
    assert smape(np.zeros(2), np.zeros(2)) == 0.0


def test_mase_is_scaled_by_the_seasonal_naive_training_error():
    train = np.array([1.0, 2.0, 4.0, 7.0])  # lag-1 errors: 1, 2, 3 -> mean 2
    assert mase(np.array([10.0]), np.array([8.0]), train, m=1) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        mase(np.array([1.0]), np.array([1.0]), np.array([1.0]), m=1)
    assert np.isnan(mase(np.array([1.0]), np.array([2.0]), np.ones(5), m=1))


def test_coverage():
    assert coverage(np.array([1.0, 5.0]), np.array([0.0, 0.0]), np.array([2.0, 2.0])) == 0.5


def test_origins_cover_the_series_without_overrunning_it():
    folds = bt.origins(n=100, horizon=10, initial=60, step=10)
    assert folds == [60, 70, 80, 90]
    assert max(folds) + 10 <= 100
    with pytest.raises(ValueError, match="too short"):
        bt.origins(50, 10, 45, 5)


def test_backtest_never_trains_on_the_test_window():
    seen = []

    class Spy(Forecaster):
        name = "spy"

        def _fit(self, y):
            seen.append(len(y))
            self.last = y[-1]

        def _predict(self, h):
            return np.full(h, self.last)

    import forecastlab.backtest as module

    original = module.make_model
    module.make_model = lambda name, m=7: Spy(season_length=m)
    try:
        y = np.arange(100.0)
        bt.backtest_model("spy", y, horizon=10, initial=60, step=10)
    finally:
        module.make_model = original
    assert seen == [60, 70, 80, 90]


def test_leaderboard_is_sorted_and_ranks_seasonal_models_above_naive():
    y = make_demand(500, seed=0)
    rows = bt.leaderboard(y, ["naive", "seasonal-naive", "ridge-lags"], horizon=14, initial=350, step=28)
    assert [r["model"] for r in rows][0] == "ridge-lags"
    assert [r["mase"] for r in rows] == sorted(r["mase"] for r in rows)
    assert "| Model |" in bt.to_markdown(rows)


def test_calibration_returns_none_when_series_is_too_short():
    assert calibration_errors(lambda: make_model("holt-winters"), np.arange(28.0) + 1, h=14) is None
    point, lo, hi = forecast_with_interval(lambda: make_model("holt-winters"), np.arange(28.0) + 1, 14)
    assert lo is None and hi is None and len(point) == 14


def test_interval_widths_use_a_conservative_empirical_quantile():
    errors = np.tile(np.arange(1.0, 11.0), (3, 1))  # errors 1..10 at every one of 10 steps
    widths = interval_widths(errors, 0.5)
    assert widths.shape == (10,)
    assert np.all(widths >= 1) and np.all(widths <= 10)
    assert np.all(interval_widths(errors, 0.99) >= widths)


def test_interval_coverage_is_near_nominal_on_stationary_noise():
    rng = np.random.default_rng(0)
    hits, total = 0, 0
    for _ in range(30):
        y = 100 + rng.normal(0, 5, 160)
        train, test = y[:-14], y[-14:]
        _, lo, hi = forecast_with_interval(lambda: make_model("moving-average", 7), train, 14, level=0.8)
        hits += int(np.sum((test >= lo) & (test <= hi)))
        total += 14
    assert 0.70 <= hits / total <= 0.90


def test_invalid_level_is_rejected():
    with pytest.raises(ValueError, match="level"):
        forecast_with_interval(lambda: make_model("naive"), np.arange(50.0), 5, level=1.2)
