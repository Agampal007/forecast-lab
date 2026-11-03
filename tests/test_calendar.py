from datetime import date, timedelta

import numpy as np
import pytest

from forecastlab import cli
from forecastlab.calendar import extend_dates, holiday_flags, in_year_end_break
from forecastlab.data import load_dates
from forecastlab.models import make_model


def test_nsw_flags_mark_public_holidays_and_the_year_end_break():
    days = [date(2025, 1, 27), date(2025, 6, 9), date(2025, 6, 10), date(2025, 12, 25), date(2025, 12, 29)]
    flags = holiday_flags(days, region="NSW")
    # Australia Day observed, King's Birthday, an ordinary Tuesday, Christmas, and a break day.
    assert flags.tolist() == [[1, 0], [1, 0], [0, 0], [1, 1], [0, 1]]


def test_year_end_break_runs_from_24_december_to_2_january():
    assert not in_year_end_break(date(2025, 12, 23))
    assert in_year_end_break(date(2025, 12, 24))
    assert in_year_end_break(date(2026, 1, 2))
    assert not in_year_end_break(date(2026, 1, 3))


def test_extend_dates_adds_the_forecast_days():
    assert extend_dates([date(2025, 12, 30), date(2025, 12, 31)], 2)[-2:] == [
        date(2026, 1, 1),
        date(2026, 1, 2),
    ]


def test_load_dates_needs_consecutive_days(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("date,v\n2025-01-01,1\n2025-01-03,2\n")
    with pytest.raises(ValueError, match="consecutive"):
        load_dates(path)
    path.write_text("day,v\n1,1\n")
    with pytest.raises(ValueError, match="'date' column"):
        load_dates(path)


def _weekly_series_with_holidays(n: int = 400, seed: int = 0):
    """A weekly pattern where flagged days run 30% low, like public holidays."""
    rng = np.random.default_rng(seed)
    flags = np.zeros((n + 14, 2))
    flags[rng.choice(np.arange(60, n), size=30, replace=False), 0] = 1
    flags[[n + 3, n + 10], 0] = 1  # two holidays inside the 14-day forecast window
    pattern = np.tile([100.0, 104, 106, 105, 103, 90, 85], (n + 14) // 7 + 1)[: n + 14]
    y = pattern * np.where(flags[:, 0] == 1, 0.7, 1.0) + rng.normal(0, 1, n + 14)
    return y, flags


def test_ridge_learns_the_holiday_dip_from_the_calendar():
    y, flags = _weekly_series_with_holidays()
    n = 400
    future_holidays = np.flatnonzero(flags[n:, 0])
    with_cal = make_model("ridge-lags", calendar=flags).fit(y[:n]).predict(14)
    without = make_model("ridge-lags").fit(y[:n]).predict(14)
    actual = y[n : n + 14]
    err_with = np.abs(with_cal - actual)[future_holidays].mean()
    err_without = np.abs(without - actual)[future_holidays].mean()
    assert err_with < err_without / 2


def test_models_check_that_the_calendar_covers_the_forecast():
    y, flags = _weekly_series_with_holidays()
    with pytest.raises(ValueError, match="calendar has"):
        make_model("ridge-lags", calendar=flags[:300]).fit(y[:400])
    model = make_model("ridge-lags", calendar=flags[:405]).fit(y[:400])
    with pytest.raises(ValueError, match="forecasting 14 steps needs 414"):
        model.predict(14)
    with pytest.raises(ValueError, match="2-D"):
        make_model("ridge-lags", calendar=np.ones(10))


def test_models_without_lags_ignore_the_calendar():
    y, flags = _weekly_series_with_holidays()
    plain = make_model("seasonal-naive").fit(y[:400]).predict(7)
    with_cal = make_model("seasonal-naive", calendar=flags).fit(y[:400]).predict(7)
    assert np.array_equal(plain, with_cal)


def test_cli_forecast_with_holidays(tmp_path, capsys):
    start = date(2024, 1, 1)
    path = tmp_path / "demand.csv"
    rows = [f"{start + timedelta(days=i)},{100 + 5 * (i % 7)}" for i in range(400)]
    path.write_text("date,demand\n" + "\n".join(rows) + "\n")
    args = ["forecast", "--csv", str(path), "--model", "ridge-lags", "--horizon", "7", "--holidays", "NSW"]
    assert cli.main(args) == 0
    assert len(capsys.readouterr().out.splitlines()) == 8


def test_cli_holidays_need_a_csv():
    with pytest.raises(SystemExit, match="needs --csv"):
        cli.main(["forecast", "--holidays", "NSW"])
