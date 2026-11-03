"""COT-Modell: Zeitpunkte, Lernen ohne Zukunftsblick, Signale, Backtest, Seite."""
from __future__ import annotations

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.cot import model as C


def synth(codes=("088691",), years=(1995, 2026), seed=0, edge=0.004):
    """COT-Daten, bei denen die Commercials-Position die Rendite der Folgewochen beeinflusst (geplanter Effekt)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{years[0]}-01-03", f"{years[1]}-10-27", freq="W-TUE")
    n = len(dates)
    c = np.zeros(n)
    for i in range(1, n):
        c[i] = 0.97 * c[i - 1] + rng.normal(0, 0.03)
    rows = []
    for code in codes:
        for d, x in zip(dates, c):
            oi = 100000.0
            rows.append({"date": d, "code": code, "oi": oi, "c_long": 40000 + x * oi / 2, "c_short": 40000 - x * oi / 2,
                         "nc_long": 30000 - x * oi / 3, "nc_short": 30000 + x * oi / 3, "nr_long": 10000.0,
                         "nr_short": 10000.0})
    cot = pd.DataFrame(rows)
    # Kurs: Wochenrendite ab Handelstag (Di + 6) hängt von c der Vorwoche ab
    wk = 0.001 + edge * (c / c.std()) + rng.normal(0, 0.02, n)
    trade = dates + pd.Timedelta(days=6)
    px = pd.Series(100 * np.exp(np.cumsum(np.r_[0, wk[:-1]])), index=trade)
    daily = px.reindex(pd.bdate_range(trade[0], trade[-1] + pd.Timedelta(days=7)), method="ffill")
    return cot, daily


def test_trade_price_is_monday_after_release():
    cot, px = synth()
    f = C.weekly_frame(cot, ["088691"], px)
    assert (f["trade_date"] - f.index == pd.Timedelta(days=6)).all()
    d0 = f.index[10]
    assert f.loc[d0, "price"] == px[px.index <= d0 + pd.Timedelta(days=6)].iloc[-1]


def test_learning_finds_planted_effect_without_lookahead():
    cot, px = synth()
    res = C.run_market(cot, ["088691"], px)
    L = res["learned"]
    last = L[L["year"] == L["year"].max()].set_index("key")
    assert last.loc["c_idx156", "sign"] == 1 and last.loc["c_idx156", "source"] == "gelernt"
    assert last.loc["nc_net", "sign"] == -1
    # Zukunft manipulieren -> Lernen für 2010 unverändert
    w = res["weekly"]
    sc = C.percentiles(w)
    fwd = C.fwd_returns(w["price"], 13)
    a = C.learn(sc, fwd, [2010])
    fwd2 = fwd.copy()
    fwd2[fwd2.index >= "2009-09-27"] *= -1  # Ergebnis erst nach dem 1.1.2010 bekannt
    b = C.learn(sc, fwd2, [2010])
    pd.testing.assert_series_equal(a["ic"], b["ic"])
    sig = C.signal(w["score"], 60, 40)
    bt = C.backtest(w, sig, start=res["start"])
    assert bt["cagr"] > bt["cagr_bh"]
    bs = C.by_signal(w, sig, res["start"]).set_index("Signal")
    assert bs.loc["Long", "Ø 13 W."] > bs.loc["Short", "Ø 13 W."]


def test_signal_thresholds_and_backtest_long_only():
    s = pd.Series([70, 50, 30, np.nan], index=pd.date_range("2020-01-07", periods=4, freq="W-TUE"))
    assert C.signal(s, 60, 40).tolist()[:3] == [1, 0, -1]
    cot, px = synth(seed=2, edge=0.0)
    res = C.run_market(cot, ["088691"], px)
    w = res["weekly"]
    bt = C.backtest(w, C.signal(w["score"], 60, 40), start=res["start"], allow_short=False)
    assert bt["time_short"] == 0


def test_run_and_page(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from src.config import get_settings
    from src.cot import run as R
    s = get_settings()
    frames, prices = [], {}
    for i, (k, m) in enumerate(C.MARKETS.items()):
        cot, px = synth(tuple(m["codes"]), seed=i)
        frames.append(cot)
        prices[m["price"]] = px
    (tmp_path / "fundamental").mkdir()
    pd.concat(frames).to_parquet(tmp_path / "fundamental" / R.COT_FILE, index=False)
    d = tmp_path / "prices_long"
    d.mkdir()
    for t, px in prices.items():
        safe = t.replace("^", "IDX_").replace("=", "_")
        pd.DataFrame({"date": px.index, "close": px.values}).to_parquet(d / f"{safe}.parquet", index=False)
    meta = R.run(offline=True)
    assert all("error" not in v for v in meta["markets"].values())
    at = AppTest.from_string("from src.pages import cot\ncot.page()\n", default_timeout=120)
    at.run()
    assert not at.exception
    at.radio(key="cot_mkt").set_value("NDX").run()
    at.radio(key="cot_method").set_value("classic").run()
    at.toggle(key="cot_short").set_value(False).run()
    at.radio(key="cot_per").set_value("Alles").run()
    assert not at.exception


def test_page_without_results(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    at = AppTest.from_string("from src.pages import cot\ncot.page()\n", default_timeout=60)
    at.run()
    assert not at.exception and at.info
