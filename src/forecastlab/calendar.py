"""Holiday flags for daily series (needs `pip install -e .[calendar]`).

Holidays are known in advance, so a model may use them for the days it forecasts without leaking
anything from the future. The flags are an array aligned with the series index and extended past
its end for the forecast horizon.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np

FLAG_NAMES = ("public_holiday", "year_end_break")


def in_year_end_break(day: date) -> bool:
    """24 December to 2 January: the Christmas and New Year shutdown, fixed by convention."""
    return (day.month == 12 and day.day >= 24) or (day.month == 1 and day.day <= 2)


def holiday_flags(dates: Sequence[date], region: str = "NSW", country: str = "AU") -> np.ndarray:
    """(len(dates), 2) array of 0/1: public holiday in `region`, and the year-end break."""
    try:
        import holidays
    except ImportError as error:
        raise ImportError("holiday features need the optional dependency: pip install holidays") from error
    years = range(min(dates).year, max(dates).year + 1)
    public = holidays.country_holidays(country, subdiv=region, years=years)
    return np.array([[float(d in public), float(in_year_end_break(d))] for d in dates])


def extend_dates(dates: Sequence[date], horizon: int) -> list[date]:
    """`dates` followed by the `horizon` days after the last one (daily series only)."""
    last = dates[-1]
    return list(dates) + [last + timedelta(days=i) for i in range(1, horizon + 1)]
