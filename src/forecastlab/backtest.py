"""Rolling-origin backtesting: refit at each origin, forecast the next window, score it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .intervals import forecast_with_interval
from .metrics import coverage, mae, mase, rmse, smape
from .models import MODELS, make_model


@dataclass
class FoldScore:
    origin: int
    mae: float
    rmse: float
    smape: float
    mase: float
    coverage: float | None = None
    width: float | None = None


def origins(n: int, horizon: int, initial: int, step: int) -> list[int]:
    """Training-set sizes of each fold; every fold forecasts y[origin : origin + horizon]."""
    if initial + horizon > n:
        raise ValueError(f"series of {n} points is too short for initial={initial} and horizon={horizon}")
    return list(range(initial, n - horizon + 1, step))


def backtest_model(
    name: str,
    y: np.ndarray,
    horizon: int = 28,
    initial: int = 730,
    step: int = 28,
    season_length: int = 7,
    level: float | None = None,
) -> list[FoldScore]:
    """Backtest one model. Pass `level` (for example 0.8) to also score prediction intervals."""
    folds = []
    for origin in origins(len(y), horizon, initial, step):
        train, test = y[:origin], y[origin : origin + horizon]

        def factory(name=name):
            return make_model(name, season_length)

        if level is None:
            pred = factory().fit(train).predict(horizon)
            cov = width = None
        else:
            pred, lower, upper = forecast_with_interval(factory, train, horizon, level)
            cov = width = None
            if lower is not None:
                cov, width = coverage(test, lower, upper), float(np.mean(upper - lower))
        folds.append(
            FoldScore(
                origin,
                mae(test, pred),
                rmse(test, pred),
                smape(test, pred),
                mase(test, pred, train, season_length),
                cov,
                width,
            )
        )
    return folds


def summarise(folds: list[FoldScore]) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "folds": len(folds),
        "mae": float(np.mean([f.mae for f in folds])),
        "rmse": float(np.mean([f.rmse for f in folds])),
        "smape": float(np.mean([f.smape for f in folds])),
        "mase": float(np.mean([f.mase for f in folds])),
    }
    covered = [f.coverage for f in folds if f.coverage is not None]
    out["coverage"] = float(np.mean(covered)) if covered else None
    widths = [f.width for f in folds if f.width is not None]
    out["width"] = float(np.mean(widths)) if widths else None
    return out


def leaderboard(
    y: np.ndarray,
    models: list[str] | None = None,
    horizon: int = 28,
    initial: int = 730,
    step: int = 28,
    season_length: int = 7,
    level: float | None = None,
) -> list[dict]:
    rows = []
    for name in models or list(MODELS):
        folds = backtest_model(name, y, horizon, initial, step, season_length, level)
        rows.append({"model": name, **summarise(folds)})
    return sorted(rows, key=lambda r: r["mase"])


def to_markdown(rows: list[dict], level: float | None = None) -> str:
    cols = ["model", "folds", "mae", "rmse", "smape", "mase"] + (["coverage", "width"] if level else [])
    head = {
        "smape": "sMAPE %",
        "mase": "MASE",
        "mae": "MAE",
        "rmse": "RMSE",
        "model": "Model",
        "folds": "Folds",
        "coverage": f"Coverage of {int((level or 0) * 100)}% interval",
        "width": "Mean width",
    }
    lines = ["| " + " | ".join(head[c] for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            if v is None:
                cells.append("n/a")
            elif c == "model":
                cells.append(str(v))
            elif c == "folds":
                cells.append(str(int(v)))
            elif c == "coverage":
                cells.append(f"{v:.3f}")
            else:
                cells.append(f"{v:.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
