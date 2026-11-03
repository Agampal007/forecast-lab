"""Command line: forecastlab backtest | forecast | serve | models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .backtest import leaderboard, to_markdown
from .data import load_csv, load_dates, make_demand
from .intervals import forecast_with_interval
from .models import MODELS, make_model


def _series(args: argparse.Namespace) -> np.ndarray:
    return load_csv(args.csv, args.column) if args.csv else make_demand()


def _calendar(args: argparse.Namespace, n: int, horizon: int = 0) -> np.ndarray | None:
    """Holiday flags for the series plus `horizon` future days, or None without --holidays."""
    if not args.holidays:
        return None
    if not args.csv:
        raise SystemExit("--holidays needs --csv with a 'date' column")
    from .calendar import extend_dates, holiday_flags

    dates = load_dates(args.csv)
    if len(dates) != n:
        raise SystemExit(f"{args.csv}: {len(dates)} dates for {n} values")
    return holiday_flags(extend_dates(dates, horizon), region=args.holidays)


def _series_label(args: argparse.Namespace) -> str:
    """Name of the forecast column, matching load_csv's default of the last column."""
    if not args.csv:
        return "demand (synthetic)"
    if args.column:
        return args.column
    with args.csv.open(encoding="utf-8") as handle:
        return handle.readline().strip().split(",")[-1]


def cmd_backtest(args: argparse.Namespace) -> int:
    y = _series(args)
    initial = args.initial or max(len(y) * 2 // 3, 10)
    calendar = _calendar(args, len(y))
    rows = leaderboard(
        y, args.models, args.horizon, initial, args.step, args.season_length, args.level, calendar
    )
    table = to_markdown(rows, args.level)
    source = args.csv or "synthetic demand (seed 0)"
    if args.holidays:
        source = f"{source} with {args.holidays} holiday features"
    print(
        f"Rolling-origin backtest on {source}: {len(y)} points, horizon {args.horizon}, "
        f"initial {initial}, step {args.step}\n"
    )
    print(table)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "leaderboard.md").write_text(table + "\n", encoding="utf-8")
    if args.plot:
        from .plots import forecast_plot

        best = rows[0]["model"]
        path = forecast_plot(y, best, args.horizon, args.plot, ylabel=_series_label(args), calendar=calendar)
        print(f"\nPlot of the best model ({best}) written to {path}")
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    y = _series(args)
    calendar = _calendar(args, len(y), args.horizon)
    point, lower, upper = forecast_with_interval(
        lambda: make_model(args.model, args.season_length, calendar), y, args.horizon, args.level
    )
    print("step,forecast,lower,upper")
    for i, value in enumerate(point):
        lo = "" if lower is None else f"{lower[i]:.2f}"
        hi = "" if upper is None else f"{upper[i]:.2f}"
        print(f"{i + 1},{value:.2f},{lo},{hi}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("forecastlab.api:app", host=args.host, port=args.port)
    return 0


def cmd_models(_: argparse.Namespace) -> int:
    for name, cls in MODELS.items():
        print(f"{name:16s} {cls.description}")
    return 0


def _add_series_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--csv", type=Path, help="CSV with a header row; default is built-in synthetic demand")
    p.add_argument("--column", help="column to forecast (default: last column)")
    p.add_argument("--season-length", type=int, default=7)
    p.add_argument("--horizon", type=int, default=28)
    p.add_argument(
        "--holidays",
        metavar="REGION",
        help="add public-holiday and year-end features for an Australian state, e.g. NSW "
        "(needs a 'date' column and pip install holidays)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forecastlab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("backtest", help="rolling-origin leaderboard of all models")
    _add_series_args(p)
    p.add_argument("--models", nargs="*", choices=list(MODELS))
    p.add_argument("--initial", type=int, help="size of the first training window (default: two thirds)")
    p.add_argument("--step", type=int, default=28)
    p.add_argument(
        "--level", type=float, default=None, help="also score prediction intervals at this level, e.g. 0.8"
    )
    p.add_argument("--out", type=Path)
    p.add_argument("--plot", type=Path, help="write a PNG of the best model's last window (needs matplotlib)")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("forecast", help="print a forecast as CSV")
    _add_series_args(p)
    p.add_argument("--model", default="ridge-lags", choices=list(MODELS))
    p.add_argument("--level", type=float, default=0.8)
    p.set_defaults(func=cmd_forecast)

    p = sub.add_parser("serve", help="run the API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("models", help="list models")
    p.set_defaults(func=cmd_models)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
