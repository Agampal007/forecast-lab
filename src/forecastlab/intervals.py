"""Prediction intervals from rolling-origin residuals (a split-conformal style calibration)."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from .models import Forecaster

MAX_CALIBRATION_ORIGINS = 6
NEIGHBOUR_STEPS = 3  # pool errors from nearby horizon steps so each quantile has enough samples


def calibration_errors(
    factory: Callable[[], Forecaster], y: np.ndarray, h: int, n_origins: int = MAX_CALIBRATION_ORIGINS
) -> np.ndarray | None:
    """Absolute forecast errors, shape (origins, h), from fitting only on data before each origin.

    Returns None when the series is too short to leave room for the forecast plus one calibration origin.
    """
    min_train = factory().min_length() + 1
    step = min(h, 14)
    origins = []
    for k in range(n_origins):
        origin = len(y) - h - k * step
        if origin >= min_train:
            origins.append(origin)
    if not origins:
        return None
    errors = []
    for origin in origins:
        pred = factory().fit(y[:origin]).predict(h)
        errors.append(np.abs(y[origin : origin + h] - pred))
    return np.asarray(errors)


def interval_widths(errors: np.ndarray, level: float) -> np.ndarray:
    """Half-width per horizon step: an upper empirical quantile of pooled absolute errors."""
    n_origins, h = errors.shape
    widths = np.empty(h)
    for i in range(h):
        window = errors[:, max(0, i - NEIGHBOUR_STEPS) : i + NEIGHBOUR_STEPS + 1].ravel()
        rank = min(len(window), math.ceil((len(window) + 1) * level))
        widths[i] = np.sort(window)[rank - 1]
    return widths


def forecast_with_interval(
    factory: Callable[[], Forecaster], y: np.ndarray, h: int, level: float = 0.8
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Point forecast plus symmetric interval (None, None when there is too little history)."""
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    point = factory().fit(y).predict(h)
    errors = calibration_errors(factory, y, h)
    if errors is None:
        return point, None, None
    half = interval_widths(errors, level)
    return point, point - half, point + half
