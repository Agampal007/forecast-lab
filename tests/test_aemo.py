from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from forecastlab.aemo import daily_energy, interval_energy, month_url
from forecastlab.data import load_csv

DATASET = Path(__file__).resolve().parent.parent / "datasets" / "aemo-nsw-daily.csv"
HEADER = "REGION,SETTLEMENTDATE,TOTALDEMAND,RRP,PERIODTYPE\n"


def _file(start: datetime, minutes: int, count: int, mw: float = 6000.0) -> str:
    rows = [
        f"NSW1,{(start + timedelta(minutes=minutes * (i + 1))):%Y/%m/%d %H:%M:%S},{mw},50.0,TRADE"
        for i in range(count)
    ]
    return HEADER + "\n".join(rows) + "\n"


def test_month_url():
    assert month_url(2025, 1).endswith("/PRICE_AND_DEMAND_202501_NSW1.csv")


def test_interval_energy_reads_the_interval_length_from_the_timestamps():
    five = interval_energy(_file(datetime(2025, 1, 1), 5, 3, mw=1200.0))
    thirty = interval_energy(_file(datetime(2020, 1, 1), 30, 3, mw=1200.0))
    assert five[0] == (datetime(2025, 1, 1, 0, 5), pytest.approx(100.0))
    assert thirty[0][1] == pytest.approx(600.0)


def test_interval_energy_rejects_uneven_spacing():
    text = HEADER + "NSW1,2025/01/01 00:05:00,1,1,TRADE\nNSW1,2025/01/01 00:10:00,1,1,TRADE\n"
    text += "NSW1,2025/01/01 00:30:00,1,1,TRADE\n"
    with pytest.raises(ValueError, match="uneven"):
        interval_energy(text)


def test_midnight_interval_belongs_to_the_previous_day():
    # Two full days of 5-minute data at a constant 6,000 MW is 144 GWh per day.
    days = daily_energy([_file(datetime(2025, 1, 1), 5, 2 * 288)])
    assert days == [(date(2025, 1, 1), pytest.approx(144.0)), (date(2025, 1, 2), pytest.approx(144.0))]


def test_partial_days_at_the_edges_are_dropped():
    days = daily_energy([_file(datetime(2025, 1, 1, 12), 30, 4 * 48)])
    assert [d for d, _ in days] == [date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 4)]


def test_files_with_different_interval_lengths_combine():
    old = _file(datetime(2021, 9, 30), 30, 48)
    new = _file(datetime(2021, 10, 1), 5, 288)
    days = daily_energy([old, new])
    assert [d for d, _ in days] == [date(2021, 9, 30), date(2021, 10, 1)]
    assert days[0][1] == pytest.approx(days[1][1])


def test_a_gap_inside_the_range_is_an_error():
    first = _file(datetime(2025, 1, 1), 30, 48)
    third = _file(datetime(2025, 1, 3), 30, 48)
    with pytest.raises(ValueError, match="missing"):
        daily_energy([first, third])


def test_committed_nsw_dataset_is_complete_and_plausible():
    y = load_csv(DATASET)
    assert len(y) == 1461  # 2022-01-01 to 2025-12-31
    # NSW uses roughly 180 GWh a day; anything far outside means a unit or parsing error.
    assert 100 < y.min() and y.max() < 300 and 150 < np.mean(y) < 210
