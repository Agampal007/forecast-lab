import numpy as np
import pytest
from fastapi.testclient import TestClient

from forecastlab import cli
from forecastlab.api import app
from forecastlab.data import make_demand

client = TestClient(app)
SERIES = [round(v, 2) for v in make_demand(200, seed=0)]


def test_health_and_models():
    assert client.get("/health").json()["status"] == "ok"
    names = [m["name"] for m in client.get("/models").json()]
    assert "ridge-lags" in names and "seasonal-naive" in names


def test_forecast_returns_point_and_interval():
    r = client.post("/forecast", json={"values": SERIES, "horizon": 10, "model": "seasonal-naive"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["forecast"]) == len(body["lower"]) == len(body["upper"]) == 10
    assert all(
        lo <= f <= hi for lo, f, hi in zip(body["lower"], body["forecast"], body["upper"], strict=True)
    )
    assert body["note"] is None


def test_forecast_is_deterministic():
    payload = {"values": SERIES, "horizon": 7, "model": "ridge-lags"}
    assert client.post("/forecast", json=payload).json() == client.post("/forecast", json=payload).json()


def test_short_series_gives_forecast_without_interval():
    r = client.post("/forecast", json={"values": SERIES[:15], "horizon": 14, "model": "naive"})
    assert r.status_code == 200
    assert r.json()["lower"] is None and "too short" in r.json()["note"]


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ({"values": SERIES, "model": "prophet"}, "unknown model"),
        ({"values": SERIES[:10], "model": "holt-winters"}, "needs at least"),
        ({"values": SERIES, "horizon": 0}, "greater than or equal to 1"),
        ({"values": SERIES, "horizon": 1000}, "less than or equal to 365"),
        ({"values": SERIES, "level": 1.5}, "less than 1"),
        ({"values": [1.0]}, "at least 2"),
        ({"values": SERIES, "season_length": 0}, "greater than or equal to 1"),
    ],
)
def test_validation_errors_are_422_with_a_reason(payload, fragment):
    r = client.post("/forecast", json=payload)
    assert r.status_code == 422
    assert fragment in r.text


def test_non_finite_values_are_rejected():
    r = client.post(
        "/forecast", content='{"values": [1, 2, NaN, 4]}', headers={"content-type": "application/json"}
    )
    assert r.status_code == 422


def test_backtest_endpoint():
    r = client.post(
        "/backtest", json={"values": SERIES, "models": ["naive", "seasonal-naive"], "horizon": 14, "step": 14}
    )
    assert r.status_code == 200
    rows = r.json()
    assert {row["model"] for row in rows} == {"naive", "seasonal-naive"}
    assert rows[0]["mase"] <= rows[1]["mase"]
    bad = client.post("/backtest", json={"values": SERIES[:30], "horizon": 30, "initial": 20})
    assert bad.status_code == 422


def test_cli_backtest_and_forecast(tmp_path, capsys):
    assert (
        cli.main(
            [
                "backtest",
                "--models",
                "naive",
                "seasonal-naive",
                "--initial",
                "900",
                "--step",
                "98",
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert "seasonal-naive" in (tmp_path / "leaderboard.md").read_text()
    assert cli.main(["forecast", "--model", "seasonal-naive", "--horizon", "3"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-4] == "step,forecast,lower,upper" and len(lines[-3:]) == 3


def test_cli_reads_csv(tmp_path, capsys):
    path = tmp_path / "d.csv"
    path.write_text(
        "day,units\n" + "\n".join(f"{i},{v}" for i, v in enumerate(np.tile([5, 6, 7, 8, 9, 10, 11], 12)))
    )
    assert cli.main(["forecast", "--csv", str(path), "--model", "seasonal-naive", "--horizon", "7"]) == 0
    first = capsys.readouterr().out.splitlines()[1].split(",")
    assert float(first[1]) == 5.0
