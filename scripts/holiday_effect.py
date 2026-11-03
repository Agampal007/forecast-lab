"""Does adding holiday features help? Backtests each lag model with and without them.

    python scripts/holiday_effect.py --csv datasets/aemo-nsw-daily.csv --initial 731 --region NSW

Reports MAE over all forecast days, over flagged days (public holidays and the 24 December to
2 January break) and over the remaining days, plus the paired fold-by-fold MAE difference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from forecastlab.backtest import origins
from forecastlab.calendar import holiday_flags
from forecastlab.data import load_csv, load_dates
from forecastlab.models import make_model

MODELS = ("ridge-lags", "gbm-lags")


def errors_by_day(name, y, calendar, horizon, initial, step) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Absolute errors of every forecast day, the index of that day, and the fold it belongs to."""
    errs, days, folds = [], [], []
    for fold, origin in enumerate(origins(len(y), horizon, initial, step)):
        pred = make_model(name, calendar=calendar).fit(y[:origin]).predict(horizon)
        errs.append(np.abs(y[origin : origin + horizon] - pred))
        days.append(np.arange(origin, origin + horizon))
        folds.append(np.full(horizon, fold))
    return np.concatenate(errs), np.concatenate(days), np.concatenate(folds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--region", default="NSW")
    parser.add_argument("--initial", type=int, default=730)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--step", type=int, default=28)
    args = parser.parse_args()

    y = load_csv(args.csv)
    calendar = holiday_flags(load_dates(args.csv), region=args.region)
    flagged = calendar.any(axis=1)

    print("| Model | Holiday features | MAE, all days | MAE, flagged days | MAE, other days |")
    print("| --- | --- | --- | --- | --- |")
    paired = []
    for name in MODELS:
        fold_mae = {}
        for label, cal in (("no", None), ("yes", calendar)):
            err, day, fold = errors_by_day(name, y, cal, args.horizon, args.initial, args.step)
            on = flagged[day]
            print(
                f"| {name} | {label} | {err.mean():.2f} | {err[on].mean():.2f} (n = {on.sum()}) "
                f"| {err[~on].mean():.2f} |"
            )
            fold_mae[label] = np.array([err[fold == f].mean() for f in np.unique(fold)])
        diff = fold_mae["yes"] - fold_mae["no"]
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        paired.append(
            f"{name}: holiday features change MAE by {diff.mean():+.2f} (paired standard error {se:.2f}); "
            f"better in {int((diff < 0).sum())} of {len(diff)} folds"
        )
    print()
    print("\n".join(paired))


if __name__ == "__main__":
    main()
