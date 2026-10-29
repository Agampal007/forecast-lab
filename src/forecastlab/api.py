"""FastAPI service: POST a series, get a forecast with a prediction interval."""

from __future__ import annotations

import math

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .backtest import leaderboard
from .intervals import forecast_with_interval
from .models import MODELS, make_model

MAX_POINTS = 5000
MAX_HORIZON = 365

app = FastAPI(
    title="forecastlab",
    version=__version__,
    description="Forecast a regularly spaced series with calibrated prediction intervals.",
)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Report what is wrong without echoing the input back (it can hold NaN, which JSON cannot encode)."""
    errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


class Series(BaseModel):
    values: list[float] = Field(
        ..., min_length=2, max_length=MAX_POINTS, description="Regularly spaced observations, oldest first"
    )
    season_length: int = Field(
        7,
        ge=1,
        le=366,
        description="Observations per seasonal cycle (7 for daily data with a weekly pattern)",
    )

    @field_validator("values")
    @classmethod
    def finite(cls, values: list[float]) -> list[float]:
        if not all(math.isfinite(v) for v in values):
            raise ValueError("values must all be finite numbers")
        return values


class ForecastRequest(Series):
    horizon: int = Field(14, ge=1, le=MAX_HORIZON)
    model: str = "ridge-lags"
    level: float = Field(0.8, gt=0, lt=1, description="Nominal coverage of the prediction interval")


class ForecastResponse(BaseModel):
    model: str
    horizon: int
    level: float
    forecast: list[float]
    lower: list[float] | None
    upper: list[float] | None
    note: str | None = None


class BacktestRequest(Series):
    models: list[str] | None = None
    horizon: int = Field(14, ge=1, le=90)
    initial: int | None = Field(None, ge=10)
    step: int = Field(14, ge=1)


def _check_model(name: str) -> None:
    if name not in MODELS:
        raise HTTPException(status_code=422, detail=f"unknown model {name!r}; choose from {sorted(MODELS)}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/models")
def models() -> list[dict]:
    return [{"name": name, "description": cls.description} for name, cls in MODELS.items()]


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    _check_model(req.model)
    y = np.asarray(req.values)

    def factory():
        return make_model(req.model, req.season_length)

    need = factory().min_length()
    if len(y) < need:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{req.model} needs at least {need} observations "
                f"with season_length={req.season_length}; got {len(y)}"
            ),
        )
    point, lower, upper = forecast_with_interval(factory, y, req.horizon, req.level)
    note = None if lower is not None else "Series too short to calibrate an interval for this horizon."
    return ForecastResponse(
        model=req.model,
        horizon=req.horizon,
        level=req.level,
        forecast=point.tolist(),
        lower=None if lower is None else lower.tolist(),
        upper=None if upper is None else upper.tolist(),
        note=note,
    )


@app.post("/backtest")
def backtest(req: BacktestRequest) -> list[dict]:
    names = req.models or list(MODELS)
    for name in names:
        _check_model(name)
    y = np.asarray(req.values)
    initial = req.initial or max(len(y) // 2, 10)
    try:
        return leaderboard(y, names, req.horizon, initial, req.step, req.season_length)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
