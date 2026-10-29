"""Optional plots (needs `pip install -e .[plots]`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .intervals import forecast_with_interval
from .models import make_model


def forecast_plot(
    y: np.ndarray, model: str, horizon: int, out: Path, level: float = 0.8, history: int = 120
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train, test = y[:-horizon], y[-horizon:]
    point, lower, upper = forecast_with_interval(lambda: make_model(model), train, horizon, level)
    t_hist = np.arange(len(train) - history, len(train))
    t_fut = np.arange(len(train), len(train) + horizon)
    fig, ax = plt.subplots(figsize=(9, 4), dpi=120)
    ax.plot(t_hist, train[-history:], color="#475569", lw=1.2, label="history")
    ax.plot(t_fut, test, color="#0f172a", lw=1.2, ls="--", label="actual")
    if lower is not None:
        ax.fill_between(
            t_fut, lower, upper, color="#2563eb", alpha=0.18, label=f"{int(level * 100)}% interval"
        )
    ax.plot(t_fut, point, color="#2563eb", lw=1.8, label=f"{model} forecast")
    ax.set_xlabel("day")
    ax.set_ylabel("demand (synthetic)")
    ax.set_title(f"Held-out {horizon}-day forecast")
    ax.legend(frameon=False, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
