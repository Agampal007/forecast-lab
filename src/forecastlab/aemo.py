"""Daily electricity demand from AEMO's aggregated price and demand files.

AEMO publishes one CSV per region and month with operational demand in MW for each dispatch
interval (5 minutes since October 2021, 30 minutes before). `SETTLEMENTDATE` is the *end* of the
interval, so the interval ending at 00:00 belongs to the previous day.

Source: Australian Energy Market Operator (AEMO), Aggregated price and demand data,
https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from datetime import date, datetime, timedelta
from itertools import pairwise

MONTH_URL = "https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{yyyymm}_{region}.csv"


def month_url(year: int, month: int, region: str = "NSW1") -> str:
    return MONTH_URL.format(yyyymm=f"{year}{month:02d}", region=region)


def interval_energy(csv_text: str) -> list[tuple[datetime, float]]:
    """(interval end, energy in MWh) for each row of one AEMO monthly file.

    The interval length is read from the spacing of the timestamps, so 5- and 30-minute files both
    work. A file whose spacing is not uniform is rejected rather than guessed at.
    """
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    if len(rows) < 2:
        raise ValueError("AEMO file has fewer than two intervals")
    ends = [datetime.strptime(r["SETTLEMENTDATE"], "%Y/%m/%d %H:%M:%S") for r in rows]
    gaps = {b - a for a, b in pairwise(ends)}
    if len(gaps) != 1:
        raise ValueError(f"AEMO file has uneven interval spacing: {sorted(gaps)}")
    hours = gaps.pop() / timedelta(hours=1)
    return [(end, float(r["TOTALDEMAND"]) * hours) for end, r in zip(ends, rows, strict=True)]


def daily_energy(csv_texts: list[str]) -> list[tuple[date, float]]:
    """Total demand per calendar day in GWh, oldest first.

    Only complete days are returned: every interval of the day must be present. Days cut off at
    the start or end of the downloaded range are dropped; a gap in the middle raises an error.
    """
    energy: dict[date, float] = defaultdict(float)
    count: dict[date, int] = defaultdict(int)
    per_day: dict[date, int] = {}
    for text in csv_texts:
        intervals = interval_energy(text)
        step = intervals[1][0] - intervals[0][0]
        for end, mwh in intervals:
            day = (end - step).date()
            energy[day] += mwh
            count[day] += 1
            per_day[day] = round(timedelta(days=1) / step)

    days = sorted(energy)
    complete = [d for d in days if count[d] == per_day[d]]
    if not complete:
        raise ValueError("no complete days in the AEMO files")
    first, last = complete[0], complete[-1]
    expected = (last - first).days + 1
    if len(complete) != expected:
        missing = sorted(set(first + timedelta(i) for i in range(expected)) - set(complete))
        raise ValueError(f"incomplete or missing days inside the range: {missing[:5]}")
    return [(d, energy[d] / 1000.0) for d in complete]
