"""Paired fold-by-fold MAE comparison of two models in a rolling-origin backtest.

    python scripts/compare_models.py gbm-lags ridge-lags --csv datasets/aemo-nsw-daily.csv --initial 731

Prints the mean MAE difference (first minus second) with its paired standard error, and how many
folds each model won. Folds share their test windows, so the paired error is far smaller than the
spread of either model's MAE on its own.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from forecastlab.backtest import backtest_model
from forecastlab.data import load_csv, make_demand
from forecastlab.models import MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("first", choices=list(MODELS))
    parser.add_argument("second", choices=list(MODELS))
    parser.add_argument("--csv", type=Path, help="default: built-in synthetic demand")
    parser.add_argument("--initial", type=int, default=730)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--step", type=int, default=28)
    parser.add_argument("--season-length", type=int, default=7)
    args = parser.parse_args()

    y = load_csv(args.csv) if args.csv else make_demand()
    runs = [
        backtest_model(name, y, args.horizon, args.initial, args.step, args.season_length)
        for name in (args.first, args.second)
    ]
    diff = np.array([a.mae - b.mae for a, b in zip(*runs, strict=True)])
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    print(
        f"{args.first} minus {args.second}: MAE difference {diff.mean():+.2f} "
        f"(paired standard error {se:.2f}, {len(diff)} folds)"
    )
    print(f"{args.first} won {int((diff < 0).sum())} folds, {args.second} won {int((diff > 0).sum())}")


if __name__ == "__main__":
    main()
