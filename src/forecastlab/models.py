"""Forecasting models behind one interface: `fit(y)` then `predict(h)`.

The series is assumed to be regularly spaced and to start at index 0. Seasonal position is the
index modulo the season length, so no calendar is needed.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class Forecaster:
    name = "base"
    description = ""

    def __init__(self, season_length: int = 7):
        if season_length < 1:
            raise ValueError("season_length must be at least 1")
        self.m = season_length

    def min_length(self) -> int:
        return 2

    def fit(self, y: np.ndarray) -> Forecaster:
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or infinite values")
        if len(y) < self.min_length():
            raise ValueError(f"{self.name} needs at least {self.min_length()} observations, got {len(y)}")
        self._fit(y)
        return self

    def predict(self, h: int) -> np.ndarray:
        if h < 1:
            raise ValueError("horizon must be at least 1")
        return np.asarray(self._predict(h), dtype=float)

    def _fit(self, y: np.ndarray) -> None:
        raise NotImplementedError

    def _predict(self, h: int) -> np.ndarray:
        raise NotImplementedError


class Naive(Forecaster):
    name = "naive"
    description = "Repeat the last observation."

    def min_length(self):
        return 1

    def _fit(self, y):
        self.last = y[-1]

    def _predict(self, h):
        return np.full(h, self.last)


class SeasonalNaive(Forecaster):
    name = "seasonal-naive"
    description = "Repeat the last full season."

    def min_length(self):
        return self.m

    def _fit(self, y):
        self.season = y[-self.m :]

    def _predict(self, h):
        return np.resize(self.season, h)


class Drift(Forecaster):
    name = "drift"
    description = "Extend the straight line between the first and last observation."

    def _fit(self, y):
        self.last = y[-1]
        self.slope = (y[-1] - y[0]) / (len(y) - 1)

    def _predict(self, h):
        return self.last + self.slope * np.arange(1, h + 1)


class MovingAverage(Forecaster):
    name = "moving-average"
    description = "Mean of the last season, repeated."

    def min_length(self):
        return self.m

    def _fit(self, y):
        self.mean = float(np.mean(y[-self.m :]))

    def _predict(self, h):
        return np.full(h, self.mean)


class HoltWinters(Forecaster):
    """Additive damped-trend Holt-Winters; smoothing parameters fitted by minimising one-step SSE."""

    name = "holt-winters"
    description = "Additive damped-trend exponential smoothing with seasonality."

    def min_length(self):
        return 2 * self.m

    @staticmethod
    def _run(y, m, alpha, beta, gamma, phi, level, trend, season):
        season = season.copy()
        sse = 0.0
        for t, obs in enumerate(y):
            s = season[t % m]
            pred = level + phi * trend + s
            err = obs - pred
            sse += err * err
            new_level = level + phi * trend + alpha * err
            trend = phi * trend + alpha * beta * err
            season[t % m] = s + gamma * (1 - alpha) * err
            level = new_level
        return sse, level, trend, season

    def _fit(self, y):
        m = self.m
        level0 = float(np.mean(y[:m]))
        trend0 = float((np.mean(y[m : 2 * m]) - np.mean(y[:m])) / m)
        season0 = y[:m] - level0

        def loss(p):
            return self._run(y, m, p[0], p[1], p[2], p[3], level0, trend0, season0)[0]

        result = minimize(
            loss,
            x0=[0.2, 0.05, 0.1, 0.95],
            bounds=[(0.01, 0.99), (0.0, 0.5), (0.0, 0.99), (0.8, 0.999)],
            method="L-BFGS-B",
        )
        alpha, beta, gamma, phi = result.x
        _, self.level, self.trend, self.season = self._run(
            y, m, alpha, beta, gamma, phi, level0, trend0, season0
        )
        self.phi, self.n = phi, len(y)

    def _predict(self, h):
        steps = np.arange(1, h + 1)
        damp = np.cumsum(self.phi**steps)
        seasonal = self.season[(self.n + steps - 1) % self.m]
        return self.level + damp * self.trend + seasonal


class _LagRegressor(Forecaster):
    """Recursive regression on lagged values and Fourier calendar terms.

    Every target and lag is expressed relative to the mean of the preceding season, which removes
    the level so that tree models can handle a trending series.
    """

    lags: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 14, 21, 28)
    yearly_period = 365.25

    def min_length(self):
        return max(self.max_lag, self.m) + max(self.m, 7) + 30

    @property
    def max_lag(self):
        return max(max(self.lags), self.m)

    def _make_estimator(self):
        raise NotImplementedError

    def _calendar(self, t: int) -> list[float]:
        feats = []
        for k in range(1, min(3, self.m // 2) + 1):
            angle = 2 * math.pi * k * (t % self.m) / self.m
            feats += [math.sin(angle), math.cos(angle)]
        if self.use_yearly:
            for k in (1, 2):
                angle = 2 * math.pi * k * t / self.yearly_period
                feats += [math.sin(angle), math.cos(angle)]
        return feats

    def _row(self, history: np.ndarray, t: int) -> tuple[list[float], float]:
        base = float(np.mean(history[t - self.m : t]))
        feats = [history[t - lag] - base for lag in self.lags]
        return feats + self._calendar(t), base

    def _fit(self, y):
        self.use_yearly = len(y) >= 2 * self.yearly_period
        self.history = y.copy()
        rows, targets = [], []
        for t in range(self.max_lag, len(y)):
            feats, base = self._row(y, t)
            rows.append(feats)
            targets.append(y[t] - base)
        self.scaler = StandardScaler().fit(rows)
        self.estimator = self._make_estimator().fit(self.scaler.transform(rows), targets)

    def _predict(self, h):
        history = list(self.history)
        out = []
        for _ in range(h):
            arr = np.asarray(history)
            feats, base = self._row(arr, len(arr))
            delta = float(self.estimator.predict(self.scaler.transform([feats]))[0])
            out.append(base + delta)
            history.append(base + delta)
        return np.asarray(out)


class RidgeLags(_LagRegressor):
    name = "ridge-lags"
    description = "Ridge regression on lags and Fourier calendar terms, forecast recursively."

    def _make_estimator(self):
        return Ridge(alpha=1.0)


class GradientBoostingLags(_LagRegressor):
    name = "gbm-lags"
    description = "Histogram gradient boosting on the same features as ridge-lags."

    def _make_estimator(self):
        return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_depth=4, random_state=0)


MODELS: dict[str, Callable[..., Forecaster]] = {
    cls.name: cls
    for cls in (Naive, SeasonalNaive, Drift, MovingAverage, HoltWinters, RidgeLags, GradientBoostingLags)
}


def make_model(name: str, season_length: int = 7) -> Forecaster:
    try:
        return MODELS[name](season_length=season_length)
    except KeyError:
        raise KeyError(f"unknown model {name!r}; choose from {', '.join(MODELS)}") from None
