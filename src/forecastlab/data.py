"""Synthetic demand data and a CSV loader."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

WEEKLY = np.array([0.90, 0.95, 1.00, 1.05, 1.15, 1.30, 1.20])  # multiplicative day-of-week effect


def make_demand(n_days: int = 1096, seed: int = 0) -> np.ndarray:
    """Daily demand with a linear trend, weekly and yearly seasonality, promotions and noise.

    This is synthetic data with known structure, so it shows how the methods behave, not how they
    would do on a real business. Use `load_csv` for real series.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    level = 200 + 0.15 * t
    yearly = 1 + 0.15 * np.sin(2 * np.pi * t / 365.25 - 1.0)
    promo = np.where(rng.random(n_days) < 0.03, 1.5, 1.0)
    noise = rng.normal(1.0, 0.06, n_days)
    return level * WEEKLY[t % 7] * yearly * promo * noise


def load_csv(path: str | Path, column: str | None = None) -> np.ndarray:
    """Read one numeric column (default: the last) from a CSV with a header row."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path}: empty file")
        column = column or reader.fieldnames[-1]
        if column not in reader.fieldnames:
            raise ValueError(f"{path}: no column {column!r}; columns are {reader.fieldnames}")
        values = []
        for number, row in enumerate(reader, start=2):
            try:
                values.append(float(row[column]))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{number}: {column!r} is not a number: {row[column]!r}") from error
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{path}: column {column!r} contains NaN or infinite values")
    return arr
