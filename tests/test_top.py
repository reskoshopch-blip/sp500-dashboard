import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.ml.market_top import (drawdown_episodes, forward_labels, market_features, red_phase_stats, run_top,
                               run_top_market, walk_forward_top, TOP_FEATURES, ampel, available_signs)
from src.ml.signals import exit_signals, perf_stats, signal_stats, simulate
from tests.fixtures.ml_synth import spy_series


def synth_prices(n=60, seed=0):
    spy = spy_series()
    rng = np.random.default_rng(seed)
    r = np.log(spy).diff().fillna(0).values
    prices = {"SPY": spy}
    for i in range(n):
        beta = rng.uniform(0.6, 1.4)
        prices[f"S{i}"] = pd.Series(50 * np.exp(np.cumsum(beta * r + rng.normal(0, 0.012, len(r)))), index=spy.index)
    vol = pd.Series(r, index=spy.index).rolling(21).std().bfill()
    prices["^VIX"] = 12 + 400 * vol
    prices["HYG"] = spy * 0.4 + 40
    prices["IEF"] = pd.Series(100.0, index=spy.index)
    return prices, spy


def test_forward_labels_only_use_future():
    idx = pd.bdate_range("2020-01-01", periods=100)
    s = pd.Series(100.0, index=idx)
    s.iloc[50] = 85.0                                    # einmaliger Einbruch an Tag 50
    lab = forward_labels(s, horizon=10)
    assert lab["y_top"].iloc[45] == 1 and lab["y_top"].iloc[39] == 0
    assert lab["y_top"].iloc[50] == 0                    # der Tag selbst zählt nicht
    assert lab["y_top"].iloc[-5:].isna().all()           # Zukunft unbekannt
    assert lab["label_end"].iloc[0] == idx[10]


def test_drawdown_episodes():
    idx = pd.bdate_range("2020-01-01", periods=8)
    s = pd.Series([100, 110, 95, 98, 112, 105, 99, 100], index=idx, dtype=float)
    ep = drawdown_episodes(s)
    assert len(ep) == 2
    assert ep.iloc[0]["peak"] == idx[1] and ep.iloc[0]["max_dd"] == pytest.approx(95 / 110 - 1)
    assert pd.isna(ep.iloc[1]["recovered"])               # zweiter Einbruch noch nicht erholt


def test_red_phases_merge_short_gaps():
    idx = pd.bdate_range("2020-01-01", periods=400)
    f = pd.DataFrame({"close": np.linspace(100, 140, 400), "regime": "Bull"}, index=idx)
    f.iloc[100:110, 1] = "Bear"
    f.iloc[115:120, 1] = "Bear"                           # Lücke 5 Tage -> gleiche Phase
    f.iloc[200:205, 1] = "Bear"
    r = red_phase_stats(f)
    assert len(r) == 2
    assert r["Rendite 1 Monat"].iloc[0] > 0


def test_walk_forward_top_is_purged_and_runs():
    prices, spy = synth_prices()
    f = market_features(spy, prices).join(ampel(market_features(spy, prices))).join(forward_labels(spy))
    assert f["breadth_200"].dropna().between(0, 1).all()
    oos, folds = walk_forward_top(f, list(TOP_FEATURES))
    assert len(folds) >= 8
    assert oos["prob"].between(0, 1).all()
    assert (oos.index.year >= 2014).all()


def test_run_top_end_to_end_and_page(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from src.config import get_settings
    from src.ml.run import save_top
    prices, spy = synth_prices()
    res = run_top(prices, spy)
    assert {"daily", "episodes", "coef", "red", "meta"} <= set(res)
    assert res["daily"]["prob"].dropna().between(0, 1).all()
    assert res["meta"]["has_vix"] and res["meta"]["has_credit"]
    save_top(get_settings(), prices, spy)
    for name in ("daily", "episodes", "coef", "red"):
        assert (tmp_path / "ml" / f"top_SPY_{name}.parquet").exists()
    at = AppTest.from_string("from src.pages import ml_top\nml_top.page()\n", default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.slider(key="top_thr").set_value(1.4).run()
    assert not at.exception, [e.value for e in at.exception]


def test_run_top_without_vix_and_credit():
    prices, spy = synth_prices(n=40)
    for k in ("^VIX", "HYG", "IEF"):
        prices.pop(k)
    res = run_top(prices, spy)
    assert not res["meta"]["has_vix"] and res["daily"]["prob"].notna().any()


def test_top_page_without_results(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    at = AppTest.from_string("from src.pages import ml_top\nml_top.page()\n", default_timeout=60)
    at.run()
    assert not at.exception and at.info


def test_top_page_flags_unreliable_model(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from src.config import get_settings
    from src.ml.run import save_top
    prices, spy = synth_prices(n=40)
    save_top(get_settings(), prices, spy)
    mp = tmp_path / "ml" / "top_SPY_meta.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    m["auc_oos"] = 0.39
    mp.write_text(json.dumps(m), encoding="utf-8")
    at = AppTest.from_string("from src.pages import ml_top\nml_top.page()\n", default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("nicht verlässlich" in e.value for e in at.error)



def test_zigzag_counts_drops_inside_long_bear_market():
    idx = pd.bdate_range("2000-01-01", periods=9)
    s = pd.Series([100, 80, 95, 75, 90, 70, 72, 85, 60], index=idx, dtype=float)   # nie neues Hoch
    ep = drawdown_episodes(s)
    assert len(ep) == 4                                      # jeder Rückgang ≥ 10 % nach Erholung ≥ 10 % zählt
    assert ep.iloc[0]["max_dd"] == pytest.approx(-0.20)


def test_japan_market_with_fx_and_no_vol_index():
    prices, spy = synth_prices(n=40)
    stocks = {k: v for k, v in prices.items() if k.startswith("S")}
    fx = pd.Series(110.0, index=spy.index)
    fx.iloc[-30:] = 100.0                                   # Yen wertet stark auf
    res = run_top_market(spy, stocks, None, prices["HYG"], prices["IEF"], fx, first_test_year=2014)
    d = res["daily"]
    assert res["meta"]["has_fx"] and res["meta"]["vol_source"] == "realisiert"
    assert "a_fx" in res["meta"]["ampel_signs"]
    assert d["a_fx"].iloc[-25] == 1
    assert d["vix"].isna().all() and d["vix_vs_min63"].notna().any()


def test_signals_and_simulation():
    prices, spy = synth_prices(n=30)
    res = run_top(prices, spy)
    d = res["daily"]
    sigs = exit_signals(d, 50, 2, 20)
    assert len(sigs) == 3
    eps = drawdown_episodes(d["close"])
    for sg in sigs.values():
        st = signal_stats(d, sg, eps, pd.Timestamp(2014, 1, 1))
        assert st["signals"] >= 0
    tech = list(sigs.values())[1]
    sim = simulate(d, tech, 50, 0.001, pd.Timestamp(2014, 1, 1))
    assert sim["buy_hold"].iloc[0] == 1 and sim["invested"].isin([0, 1]).all()
    assert sim["trade"].sum() > 0
    # ohne Signale = Kaufen und Halten
    none = simulate(d, tech & False, 50, 0.0, pd.Timestamp(2014, 1, 1))
    assert none["strategy"].iloc[-1] == pytest.approx(none["buy_hold"].iloc[-1])
    ps = perf_stats(sim["strategy"], sim["invested"], sim["trade"])
    assert -1 <= ps["Max. Rückgang"] <= 0


def test_market_pipeline_with_mock_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SEC_USER_AGENT", "Pytest Runner pytest@testlauf.ch")
    import src.ml.markets as mk
    from src.config import get_settings
    from src.ml.run import save_top_markets
    prices, spy = synth_prices(n=30)
    rng = np.random.default_rng(3)

    def dl(tickers, start):
        out = {}
        for t in tickers:
            if t == "^JNIV":
                continue                                    # Vol-Index nicht verfügbar -> Ersatz
            out[t] = spy * rng.uniform(0.5, 2) if t not in ("JPY=X",) else pd.Series(110.0, index=spy.index)
        return out
    jp_codes = [f"{1300 + i}.T" for i in range(30)]
    monkeypatch.setattr(mk, "fetch_nikkei_constituents", lambda s: (jp_codes, "Test"))
    st = get_settings()
    st.data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"code": [c[:4] for c in jp_codes]}).to_csv(st.data_dir / "nikkei225_constituents.csv", index=False)
    info = mk.fetch_market_data(st, downloader=dl)
    assert "^JNIV" in info["long"]["failed"] and info["nikkei_n"] == 30
    metas = save_top_markets(st, [])
    assert {"SPY", "N225", "EWJ"} <= set(metas)
    assert "error" not in metas["N225"] and metas["N225"]["has_fx"]
    at = AppTest.from_string("from src.pages import ml_top\nml_top.page()\n", default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.radio(key="top_mkt").set_value("N225").run()
    assert not at.exception, [e.value for e in at.exception]
    at.selectbox(key="sig_ma").set_value(200).run()
    assert not at.exception, [e.value for e in at.exception]
    at.slider(key="top_ampn").set_value(3).run()
    assert not at.exception, [e.value for e in at.exception]
    assert (tmp_path / "ml" / "health_SPY_daily.parquet").exists()
    at = AppTest.from_string("from src.pages import ml_terminal\nml_terminal.page()\n", default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.radio(key="term_mkt").set_value("N225").run()
    assert not at.exception, [e.value for e in at.exception]
    at.radio(key="term_bt_h").set_value("lang").run()
    at.selectbox(key="term_comp").set_value(at.selectbox(key="term_comp").options[-1]).run()
    assert not at.exception, [e.value for e in at.exception]


def test_nikkei_code_extraction():
    from src.data_sources.nikkei import extract_codes
    wiki = "".join(f'<li>Firma {i} (<a href="x">TYO</a>: <a href="y">{1300 + i * 30}</a>)</li>' for i in range(225))
    wiki = wiki.replace("<a href=\"x\">TYO</a>", "TYO")
    assert len(extract_codes(wiki)) == 225
    table = "<table>" + "".join(f"<tr><td><a>{2000 + i * 20}</a></td><td>Name</td></tr>" for i in range(225)) + "</table>"
    assert len(extract_codes(table)) == 225
    assert extract_codes("<p>TYO: 7203</p>") == []          # zu wenige -> unplausibel


def test_market_without_any_stocks():
    prices, spy = synth_prices(n=5)
    res = run_top_market(spy, {}, prices["^VIX"], None, None, None, 2014)
    assert res["meta"]["n_stocks_breadth"] == 0 and "a_breadth" not in res["meta"]["ampel_signs"]


def test_nikkei_official_table_format_with_alnum_codes():
    from src.data_sources.nikkei import extract_codes
    codes = ["285A"] + [str(1301 + i * 37) for i in range(224)]
    sectors = [codes[i:i + 45] for i in range(0, 225, 45)]
    html = "".join("<h3>Sektor</h3><table><tr><th>Code</th><th>Company Name</th></tr>"
                   + "".join(f"<tr><td>{c}</td><td>FIRMA {c}</td></tr>" for c in sec) + "</table>" for sec in sectors)
    got = extract_codes(html)
    assert len(got) == 225 and "285A" in got



def test_health_scores_have_no_lookahead_and_valid_range():
    from src.ml.health import compute_health
    prices, spy = synth_prices(n=40)
    stocks = {k: v for k, v in prices.items() if k.startswith("S")}
    ex = {"RSP": spy * 1.01, "^VIX": prices["^VIX"], "^VIX3M": prices["^VIX"] * 1.1}
    f = run_top_market(spy, stocks, prices["^VIX"], prices["HYG"], prices["IEF"], None, 2014, ex)["daily"]
    d1, bt, comps = compute_health(f)
    cut = f.index[3000]
    f2 = f.copy()
    f2.loc[f2.index > cut, ["close", "breadth_50", "spy_vs_sma50"]] *= 1.5     # Zukunft verändern
    d2, _, _ = compute_health(f2)
    pd.testing.assert_series_equal(d1.loc[:cut, "H_mittel"], d2.loc[:cut, "H_mittel"])
    for h in ("H_kurz", "H_mittel", "H_lang"):
        assert d1[h].dropna().between(0, 100).all()
    assert set(bt["horizon"]) == {"kurz", "mittel", "lang"} and len(comps) >= 10


def test_rule_trades_dip_and_turn():
    import numpy as np
    import pandas as pd
    from src.ml.health import rule_trades, trade_stats
    idx = pd.bdate_range("2020-01-01", periods=60)
    kurz = np.full(60, 50.0)
    kurz[10:13] = 10          # Dip an Tag 10, Drehung an Tag 13
    kurz[40:42] = 10          # zweiter Dip, Trade bleibt offen (zu wenig Tage)
    d = pd.DataFrame({"close": np.linspace(100, 160, 60), "H_kurz": kurz, "H_lang": 50.0}, index=idx)
    t = rule_trades(d, "dip", 20, 40, 21)
    assert list(t["Einstieg"]) == [idx[10], idx[40]]
    assert t["Status"].tolist() == ["abgeschlossen", "offen"]
    assert t["Rendite"].iloc[0] > 0
    t2 = rule_trades(d, "dreh", 20, 40, 21)
    assert t2["Einstieg"].iloc[0] == idx[13]
    # Langfrist-Filter blockiert alles
    assert rule_trades(d.assign(H_lang=30.0), "dip", 20, 40, 21).empty
    s = trade_stats(t)
    assert s["n"] == 1 and s["winrate"] == 1.0
