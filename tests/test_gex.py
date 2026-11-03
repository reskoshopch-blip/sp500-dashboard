"""GEX: API-Client (gemockt), Merkmale, Kostenschutz, Lernen ohne Zukunftsblick, Seite."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.gex import collect as COL
from src.gex import learn as L
from src.gex.client import AFXClient, AFXError
from src.gex.features import features, profile

EXAMPLE = {"apiVersion": "v1", "data": {
    "root": "SPY", "spotPrice": 590.25, "strikes": [560, 565, 570, 575, 580, 585, 590, 595, 600, 605],
    "callGex": [1e7, 2e7, 3e7, 5e7, 6e7, 8e7, 9e7, 1.2e8, 7e7, 3e7],
    "putGex": [-5e7, -9e7, -6e7, -4e7, -3e7, -2e7, -1e7, -5e6, -2e6, -1e6],
    "totalGex": [-4e7, -7e7, -3e7, 1e7, 3e7, 6e7, 8e7, 1.15e8, 6.8e7, 2.9e7],
    "keyLevels": {"maxGamma": 595, "gammaFlip": 572, "spotPrice": 590.25},
    "expirations": ["20261106", "20261113"]},
    "meta": {"requestId": "x", "credits": {"cost": 2, "remaining": 1800}}}
ZETA = {"apiVersion": "v1", "data": {"asset": "SPY", "computedAt": "2026-11-02T21:00:00Z",
                                    "data": {"now": {"regime": "Bull", "drift": 0.6, "risk": 0.2}}},
        "meta": {"credits": {"cost": 1, "remaining": 1799}}}


def mock_client(calls):
    def handler(req: httpx.Request):
        calls.append(req)
        assert req.headers["Authorization"] == "Bearer afx_test"
        if "/zeta/" in req.url.path:
            return httpx.Response(200, json=ZETA)
        if req.url.path.endswith("/BAD"):
            return httpx.Response(400, json={"error": {"message": "no chain"}})
        sym = req.url.path.rsplit("/", 1)[-1]
        e = json.loads(json.dumps(EXAMPLE))
        e["data"]["root"] = sym
        return httpx.Response(200, json=e)
    return AFXClient("afx_test", transport=httpx.MockTransport(handler))


def test_features_levels_and_regime():
    f = features(EXAMPLE, "SPY", datetime(2026, 11, 2, 21, 30, tzinfo=timezone.utc))
    assert f["flip"] == 572 and f["max_gamma"] == 595 and f["call_wall"] == 595 and f["put_wall"] == 565
    assert f["regime"] == "positiv" and 0 < f["wall_pos"] < 1
    assert f["days_to_exp"] == 4
    assert len(profile(EXAMPLE)) == 10
    no_flip = json.loads(json.dumps(EXAMPLE))
    no_flip["data"]["keyLevels"] = {}
    f2 = features(no_flip, "SPY", datetime(2026, 11, 2, tzinfo=timezone.utc))
    assert 560 < f2["flip"] < 600        # aus dem Profil berechnet


def test_client_errors_and_key_required(monkeypatch):
    monkeypatch.delenv("ANALYTICSFX_API_KEY", raising=False)
    with pytest.raises(AFXError):
        AFXClient()
    c = mock_client([])
    with pytest.raises(AFXError):
        c.gex("BAD")


def test_collect_guard_saves_raw_and_snapshots(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GEX_SYMBOLS", "SPY,SPX")
    calls = []
    now = datetime(2026, 11, 2, 21, 30, tzinfo=timezone.utc)
    r = COL.collect(client=mock_client(calls), now=now)
    assert r["done"] == ["SPY", "SPX"] and r["credits_remaining"] == 1800
    assert len(calls) == 3                                   # 2× GEX + Zeta nur für SPY
    r2 = COL.collect(client=mock_client(calls), now=now + timedelta(hours=1))
    assert r2["skipped"] == ["SPY", "SPX"] and len(calls) == 3   # 6-Stunden-Sperre
    snaps = COL.load_snapshots()
    assert len(snaps) == 2 and snaps.loc[snaps["symbol"] == "SPY", "zeta_regime"].iloc[0] == "Bull"
    assert any((tmp_path / "gex" / "raw" / "SPY").glob("*.json"))
    assert "afx_test" not in (tmp_path / "gex" / "status.json").read_text()


def _synthetic_history(tmp_path, days=60, seed=0):
    """Snapshots + Kurse, bei denen negative Gamma grössere Bewegungen bringt (geplanter Effekt)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-05-01", periods=days + 40)
    px, rows, p = [], [], 500.0
    regimes = rng.integers(0, 2, len(dates))
    for i, d in enumerate(dates):
        vol = 0.016 if (i > 0 and regimes[i - 1] == 0) else 0.006
        p *= 1 + rng.normal(0, vol)
        px.append(p)
        if i >= 30:
            flip = p * (0.98 if regimes[i] == 1 else 1.02)
            rows.append({"symbol": "SPY", "taken_at": pd.Timestamp(d).tz_localize("UTC") + pd.Timedelta(hours=21, minutes=30),
                         "spot": p, "flip": flip, "call_wall": p * 1.02, "put_wall": p * 0.98, "max_gamma": p * 1.005,
                         "regime": "positiv" if regimes[i] == 1 else "negativ", "net_ratio": 0.3 if regimes[i] else -0.3,
                         "dist_flip": p / flip - 1, "concentration": 0.4, "wall_pos": 0.5, "wall_width": 0.04,
                         "dist_call_wall": -0.02, "dist_put_wall": 0.02, "dist_max_gamma": -0.005})
    d = tmp_path / "prices_long"
    d.mkdir(exist_ok=True)
    pd.DataFrame({"date": dates, "close": px}).to_parquet(d / "SPY.parquet", index=False)
    (tmp_path / "gex").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(tmp_path / "gex" / "snapshots.parquet", index=False)


def test_learning_walk_forward_and_page(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ANALYTICSFX_API_KEY", "afx_test")
    _synthetic_history(tmp_path, days=150)
    res = L.run(fetch=False)
    assert res["n_done"] >= 50
    ds = pd.read_parquet(tmp_path / "gex" / "dataset.parquet")
    # erste MIN_SAMPLES Tage ohne KI-Vorhersage (nichts zum Lernen), danach mit
    assert ds["ml_move_ratio"].iloc[: L.MIN_SAMPLES].isna().all()
    assert ds["ml_move_ratio"].iloc[-5:].notna().all()
    assert res["eval"]["ml_move_corr"] > 0.15                  # geplanter Effekt gefunden
    stats = {r["Regime"]: r for r in res["regimes"]}
    assert stats["negative Gamma"]["Bewegung vs. normal"] > stats["positive Gamma"]["Bewegung vs. normal"]
    assert "ml_exp_move" in res["current"]["SPY"]
    at = AppTest.from_string("from src.pages import gex\ngex.page()\n", default_timeout=120)
    at.run()
    assert not at.exception


def test_page_without_key(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ANALYTICSFX_API_KEY", "")
    at = AppTest.from_string("from src.pages import gex\ngex.page()\n", default_timeout=60)
    at.run()
    assert not at.exception and at.warning


def test_learning_first_day_without_results(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    (tmp_path / "gex").mkdir()
    f = features(EXAMPLE, "SPY", datetime(2026, 11, 2, 22, 0, tzinfo=timezone.utc))
    pd.DataFrame([f]).assign(taken_at=lambda x: pd.to_datetime(x["taken_at"], utc=True)).to_parquet(
        tmp_path / "gex" / "snapshots.parquet", index=False)
    res = L.run(fetch=False)
    assert res["n_done"] == 0 and res["current"]["SPY"]["regime"] == "positiv"
