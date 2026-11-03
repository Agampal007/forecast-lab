"""Download AEMO's NSW demand for 2022-2025 and write datasets/aemo-nsw-daily.csv.

    python scripts/fetch_aemo_nsw.py

Monthly files are cached in data/aemo/ (not committed). The output has one row per day: the date
and total NSW operational demand in GWh.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from forecastlab.aemo import daily_energy, month_url

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "aemo"
OUT = ROOT / "datasets" / "aemo-nsw-daily.csv"
YEARS = range(2022, 2026)
USER_AGENT = "forecast-lab (+https://github.com/Agampal007/forecast-lab)"


def fetch(year: int, month: int) -> str:
    path = CACHE / f"PRICE_AND_DEMAND_{year}{month:02d}_NSW1.csv"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        # AEMO's server rejects urllib's default user agent.
        request = urllib.request.Request(month_url(year, month), headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
    return path.read_text(encoding="utf-8")


def main() -> None:
    texts = [fetch(year, month) for year in YEARS for month in range(1, 13)]
    days = daily_energy(texts)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("date,demand_gwh\n")
        for day, gwh in days:
            handle.write(f"{day.isoformat()},{gwh:.3f}\n")
    print(f"{len(days)} days, {days[0][0]} to {days[-1][0]} -> {OUT}")


if __name__ == "__main__":
    main()
