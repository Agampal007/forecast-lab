# Datasets

## `aemo-nsw-daily.csv`

Daily operational electricity demand for New South Wales, 1 January 2022 to 31 December 2025
(1,461 days). Columns: `date` and `demand_gwh`, the total demand for that day in gigawatt-hours.

Built by `python scripts/fetch_aemo_nsw.py` from AEMO's monthly 5-minute price and demand files for
region `NSW1`. Each interval's demand in MW is multiplied by the interval length and summed by day.
`SETTLEMENTDATE` marks the end of an interval, so the interval ending at midnight counts towards the
previous day.

**Source and attribution:** Australian Energy Market Operator (AEMO), *Aggregated price and demand
data*,
<https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data>.
AEMO is the author of the underlying data. AEMO permits use of its published material for any
purpose with accurate attribution
([copyright permissions](https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions)).
This file is a daily aggregation of that data. The MIT licence of this repository does not apply
to it.
