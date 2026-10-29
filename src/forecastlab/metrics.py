"""Point-forecast and interval metrics."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE in percent; points where both values are zero count as zero error."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    err = np.where(denom == 0, 0.0, np.abs(y_true - y_pred) / np.where(denom == 0, 1.0, denom))
    return float(100 * np.mean(err))


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, m: int = 1) -> float:
    """MAE scaled by the in-sample MAE of the seasonal-naive forecast. Below 1 beats seasonal naive."""
    if len(y_train) <= m:
        raise ValueError("training series must be longer than the season length")
    scale = float(np.mean(np.abs(y_train[m:] - y_train[:-m])))
    if scale == 0:
        return float("nan")
    return mae(y_true, y_pred) / scale


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))
