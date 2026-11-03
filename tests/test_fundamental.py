"""Fundamental-Terminal: Gewinne aus SEC-Facts, Point-in-time-Verzug, Lernen ohne Zukunftsblick, Seite."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.fundamental import earnings as E
from src.fundamental.features import build, cot_features, known_on
from src.fundamental.learn import apply_weights, calibrate, learn_weights, percentile_scores, run_market
from src.fundamental.sources import _normalize_cot, parse_fred_csv
from tests.fixtures import fund_synth as FS


def _cf(rows_ni, shares=None):
    facts = {"us-gaap": {"NetIncomeLoss": {"units": {"USD": rows_ni}}}}
    if shares:
        facts["dei"] = {"EntityCommonStockSharesOutstanding": {"units": {"shares": shares}}}
    return {"cik": 1, "entityName": "Test", "facts": facts}


def _q(start, end, val, filed, form="10-Q"):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form, "accn": "x", "fy": 2020, "fp": "Q"}


def test_quarters_q4_derived_and_ttm_point_in_time():
    rows = [_q("2020-01-01", "2020-03-31", 10, "2020-05-01"), _q("2020-04-01", "2020-06-30", 20, "2020-08-01"),
            _q("2020-07-01", "2020-09-30", 30, "2020-11-01"),
            _q("2020-01-01", "2020-09-30", 60, "2020-11-01"),
            _q("2020-01-01", "2020-12-31", 100, "2021-02-15", "10-K"),
            _q("2021-01-01", "2021-03-31", 15, "2021-05-01"),
            # spätere Korrektur des 1. Quartals darf den ersten Wert nicht ersetzen
            _q("2020-01-01", "2020-03-31", 999, "2021-05-01")]
    q = E.quarterly(E._duration_facts(_cf(rows), ["NetIncomeLoss"]))
    assert q.loc[q["end"] == "2020-12-31", "val"].iloc[0] == 40           # 100 - 60
    assert q.loc[q["end"] == "2020-12-31", "known"].iloc[0] == pd.Timestamp("2021-02-15")
    assert q.loc[q["end"] == "2020-03-31", "val"].iloc[0] == 10           # zuerst gemeldet
    t = E.ttm(q)
    assert t.iloc[0]["ttm"] == 100 and t.iloc[0]["known"] == pd.Timestamp("2021-02-15")
    assert t.iloc[1]["ttm"] == 105                                          # 20+30+40+15


def test_aggregate_like_for_like_growth():
    rows = []
    for cik, base in ((1, 100.0), (2, 50.0)):
        for y in (2019, 2020):
            for k, (s, e) in enumerate((("01-01", "03-31"), ("04-01", "06-30"), ("07-01", "09-30"), ("10-01", "12-31"))):
                v = base / 4 * (1.1 if y == 2020 else 1.0)
                rows.append({"cik": cik, "kind": "ni", "end": pd.Timestamp(f"{y}-{e}"), "ttm": np.nan,
                             "known": pd.Timestamp(f"{y}-{e}") + pd.Timedelta(days=40), "q": v})
    # TTM je Firma bauen
    t = []
    for cik in (1, 2):
        q = pd.DataFrame([{"end": r["end"], "val": r["q"], "known": r["known"]} for r in rows if r["cik"] == cik])
        t.append(E.ttm(q).assign(cik=cik, kind="ni"))
    t = pd.concat(t)
    a = E.aggregate(t, pd.DatetimeIndex([pd.Timestamp("2021-03-31")]), min_firms=2)
    assert a["ni_growth"].iloc[0] == pytest.approx(0.10)
    # vor Veröffentlichung kein Wert
    a2 = E.aggregate(t, pd.DatetimeIndex([pd.Timestamp("2019-12-31")]), min_firms=2)
    assert "ni_growth" not in a2 or a2["ni_growth"].isna().all()


def test_current_pe_skips_multi_class():
    t = pd.DataFrame({"cik": [1, 2], "kind": "ni", "end": pd.Timestamp("2026-06-30"), "ttm": [10.0, 10.0],
                      "known": pd.Timestamp("2026-08-01")})
    shares = pd.DataFrame({"cik": [1, 2], "shares": [100.0, 100.0], "classes": [1, 2]})
    px = {"A": pd.Series([2.0], index=[pd.Timestamp("2026-10-20")]), "B": pd.Series([5.0], index=[pd.Timestamp("2026-10-20")])}
    t = pd.concat([t] * 30, ignore_index=True)
    t["cik"] = list(range(60))
    shares = pd.DataFrame({"cik": range(60), "shares": 100.0, "classes": [1] * 55 + [2] * 5})
    px = {f"T{i}": pd.Series([2.0], index=[pd.Timestamp("2026-10-20")]) for i in range(60)}
    r = E.current_pe(t, shares, px, {i: f"T{i}" for i in range(60)}, pd.Timestamp("2026-10-23"))
    assert r["n"] == 55 and r["pe"] == pytest.approx(20.0)


def test_known_on_respects_lag_and_staleness():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-02-01"]))
    idx = pd.bdate_range("2020-01-01", "2020-06-30")
    k = known_on(s, idx, lag=45, max_age=60)
    assert k[: pd.Timestamp("2020-02-14")].isna().all()            # erst ab 15. Februar bekannt
    assert k[pd.Timestamp("2020-02-17")] == 1.0
    assert k[pd.Timestamp("2020-03-17")] == 2.0
    assert k[pd.Timestamp("2020-06-30")] != k[pd.Timestamp("2020-06-30")]  # veraltet -> NaN


def test_parse_fred_and_cot_formats():
    s = parse_fred_csv("observation_date,DGS10\n2020-01-02,1.88\n2020-01-03,.\n2020-01-06,1.81\n")
    assert len(s) == 2 and s.iloc[-1] == 1.81
    s2 = parse_fred_csv("DATE,X\n2020-01-01,5\n")
    assert s2.iloc[0] == 5
    zipfmt = pd.DataFrame({"Market and Exchange Names": ["NIKKEI"], "As of Date in Form YYYY-MM-DD": ["2020-01-07"],
                           "CFTC Contract Market Code": ["240741"], "Open Interest (All)": [100],
                           "Noncommercial Positions-Long (All)": [60], "Noncommercial Positions-Short (All)": [20],
                           "Commercial Positions-Long (All)": [10], "Commercial Positions-Short (All)": [50]})
    n = _normalize_cot(zipfmt)
    assert n.loc[0, "nc_long"] == 60 and n.loc[0, "c_long"] == 10 and n.loc[0, "code"] == "240741"
    f = cot_features(FS.cot_frame(), ["240741"])
    assert f["cot_index"].dropna().between(0, 100).all()


def test_learning_uses_only_completed_periods():
    idx = pd.bdate_range("1995-01-02", "2012-12-31")
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx)))), index=idx)
    fwd = close.shift(-126) / close - 1
    # Indikator = zukünftige Rendite (perfekt) – darf nur aus Vergangenem gelernt werden
    raw = pd.DataFrame({"us10y_chg": fwd.shift(0)}, index=idx)
    sc = percentile_scores(raw.fillna(0))
    L1 = learn_weights(sc, fwd, [2005])
    # Zukunftsrenditen nach dem Lerntag manipulieren -> Gelerntes darf sich nicht ändern
    fwd2 = fwd.copy()
    cut = idx.searchsorted(pd.Timestamp("2005-01-01")) - 126
    fwd2.iloc[cut:] = -fwd2.iloc[cut:]
    L2 = learn_weights(sc, fwd2, [2005])
    assert L1["ic"].iloc[0] == pytest.approx(L2["ic"].iloc[0])
    o, w, comp = apply_weights(sc, L1)
    assert comp["F_gesamt"][: pd.Timestamp("2004-12-31")].isna().all()


def test_calibration_pulls_noise_to_base_rate():
    rng = np.random.default_rng(0)
    ev = pd.DataFrame({"p": rng.uniform(0.6, 0.99, 600), "y": (rng.uniform(size=600) < 0.6).astype(float)})
    assert abs(calibrate(ev, 0.95) - 0.6) < 0.08


def test_run_market_and_page(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from src.config import get_settings
    from src.fundamental import run as R
    from src.fundamental import sources
    s = get_settings()
    sources.fetch_fred(s, list(R.FRED), downloader=FS.fred_downloader)
    sources.fetch_cot(s, ["13874A", "240741"], downloader=FS.cot_downloader)
    idx = pd.bdate_range("1990-01-02", "2026-10-23")
    rng = np.random.default_rng(3)
    d = tmp_path / "prices_long"
    d.mkdir()
    for name in ("SPY", "IDX_N225", "IDX_GSPC", "HG_F", "GC_F", "CL_F"):
        c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, len(idx))))
        pd.DataFrame({"date": idx, "close": c}).to_parquet(d / f"{name}.parquet", index=False)
    meta = R.run(offline=True)
    assert set(meta["markets"]) == {"SPY", "N225"}
    for k in ("SPY", "N225"):
        m = meta["markets"][k]
        assert "error" not in m and "3M" in m["ml"]
        daily = pd.read_parquet(tmp_path / "fundamental" / f"fund_{k}_daily.parquet")
        assert daily["F_gesamt"].notna().sum() > 1000
    at = AppTest.from_string("from src.pages import fundamental\nfundamental.page()\n", default_timeout=180)
    at.run()
    assert not at.exception
    at.radio(key="fund_mkt").set_value("N225").run()
    assert not at.exception
    at.radio(key="fund_per").set_value("Alles").run()
    at.radio(key="fund_bt_h").set_value(252).run()
    assert not at.exception


def test_page_without_results(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    at = AppTest.from_string("from src.pages import fundamental\nfundamental.page()\n", default_timeout=60)
    at.run()
    assert not at.exception
    assert any("Noch keine Fundamentalanalyse" in i.value for i in at.info)


def test_build_features_market_specific():
    idx = pd.bdate_range("2000-01-03", "2010-12-31")
    fred = {sid: FS.fred_series(sid) for sid in FS.FRED}
    spy = build("SPY", idx, fred, FS.cot_frame())
    jp = build("N225", idx, fred, FS.cot_frame())
    assert "us_jp_spread" in jp and "us_jp_spread" not in spy
    assert "pe" not in spy  # ohne Gewinndaten kein KGV
    assert spy["cot_index"].notna().any() and jp["usdjpy_chg"].notna().any()
    # Rohstoffe: ohne Yahoo-Reihen aus FRED (Öl täglich, Kupfer monatlich), mit Yahoo auch Kupfer/Gold
    assert spy["oil_3m"].notna().any() and spy["copper_yoy"].notna().any() and "copper_gold" not in spy
    cidx = pd.bdate_range("2000-01-03", "2010-12-31")
    com = {k: pd.Series(np.linspace(1, 3, len(cidx)), index=cidx) for k in ("copper", "gold", "oil")}
    spy2 = build("SPY", idx, fred, None, commodities=com)
    assert spy2["copper_gold"].notna().sum() > 2000 and spy2["gold_yoy"].notna().any()
