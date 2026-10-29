import numpy as np
import pytest

from forecastlab.data import load_csv, make_demand
from forecastlab.models import MODELS, HoltWinters, make_model


def test_naive_family_forecasts():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    assert make_model("naive").fit(y).predict(3).tolist() == [8.0, 8.0, 8.0]
    assert make_model("seasonal-naive", 3).fit(y).predict(5).tolist() == [6.0, 7.0, 8.0, 6.0, 7.0]
    assert make_model("drift").fit(y).predict(2).tolist() == [9.0, 10.0]
    assert make_model("moving-average", 4).fit(y).predict(2).tolist() == [6.5, 6.5]


def test_models_reject_bad_input():
    with pytest.raises(ValueError, match="NaN"):
        make_model("naive").fit(np.array([1.0, np.nan, 2.0]))
    with pytest.raises(ValueError, match="at least"):
        make_model("holt-winters", 7).fit(np.arange(10.0))
    with pytest.raises(ValueError, match="one-dimensional"):
        make_model("naive").fit(np.ones((3, 2)))
    with pytest.raises(ValueError, match="horizon"):
        make_model("naive").fit(np.ones(5)).predict(0)
    with pytest.raises(KeyError, match="choose from"):
        make_model("prophet")


def test_holt_winters_recovers_a_clean_seasonal_pattern():
    pattern = np.array([10.0, 12.0, 15.0, 11.0, 9.0, 20.0, 18.0])
    y = np.tile(pattern, 20)
    pred = HoltWinters(season_length=7).fit(y).predict(14)
    assert np.max(np.abs(pred - np.tile(pattern, 2))) < 0.5


@pytest.mark.parametrize("name", ["ridge-lags", "gbm-lags"])
def test_lag_models_beat_naive_on_trending_seasonal_data(name):
    y = make_demand(500, seed=3)
    train, test = y[:-28], y[-28:]
    naive = np.mean(np.abs(test - make_model("naive").fit(train).predict(28)))
    model = np.mean(np.abs(test - make_model(name).fit(train).predict(28)))
    assert model < 0.7 * naive


def test_lag_models_do_not_peek_at_future_values():
    """Changing observations after the forecast origin must not change the forecast."""
    y = make_demand(300, seed=1)
    other = y.copy()
    other[250:] *= 10
    for name in ("ridge-lags", "gbm-lags", "holt-winters"):
        a = make_model(name).fit(y[:250]).predict(14)
        b = make_model(name).fit(other[:250]).predict(14)
        assert np.allclose(a, b), name


def test_all_registered_models_handle_a_constant_series():
    y = np.full(120, 5.0)
    for name in MODELS:
        assert np.allclose(make_model(name).fit(y).predict(7), 5.0, atol=1e-6), name


def test_synthetic_data_is_deterministic_and_positive():
    assert np.array_equal(make_demand(100, seed=2), make_demand(100, seed=2))
    assert not np.array_equal(make_demand(100, seed=2), make_demand(100, seed=3))
    assert make_demand(1096).min() > 0


def test_load_csv(tmp_path):
    good = tmp_path / "s.csv"
    good.write_text("date,sales\n2026-01-01,10\n2026-01-02,12.5\n")
    assert load_csv(good).tolist() == [10.0, 12.5]
    assert load_csv(good, "sales").tolist() == [10.0, 12.5]
    with pytest.raises(ValueError, match="no column"):
        load_csv(good, "price")
    bad = tmp_path / "b.csv"
    bad.write_text("date,sales\n2026-01-01,abc\n")
    with pytest.raises(ValueError, match=r"b.csv:2"):
        load_csv(bad)
