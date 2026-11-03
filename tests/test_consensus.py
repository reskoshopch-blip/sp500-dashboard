"""Modell-Konsens: Zählung, Trefferquoten, Trades, Seite."""
from __future__ import annotations

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src import consensus as K


def _write(tmp_path, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    (tmp_path / "ml").mkdir()
    (tmp_path / "fundamental").mkdir()
    (tmp_path / "cot").mkdir()
    h = pd.DataFrame({"date": idx, "close": close, "H_kurz": rng.uniform(0, 100, n), "H_mittel": rng.uniform(0, 100, n),
                      "H_lang": rng.uniform(0, 100, n)})
    h.to_parquet(tmp_path / "ml" / "health_SPY_daily.parquet", index=False)
    pd.DataFrame({"date": idx, "F_gesamt": rng.uniform(0, 100, n), "p_3M": rng.uniform(0, 1, n)}).to_parquet(
        tmp_path / "fundamental" / "fund_SPY_daily.parquet", index=False)
    w = pd.date_range(idx[0], idx[-1], freq="W-TUE")
    pd.DataFrame({"date": w, "trade_date": w + pd.Timedelta(days=6), "score": rng.uniform(0, 100, len(w))}).to_parquet(
        tmp_path / "cot" / "SPX_weekly.parquet", index=False)
    pd.DataFrame({"date": idx, "ampel_score": rng.integers(0, 3, n)}).to_parquet(
        tmp_path / "ml" / "top_SPY_daily.parquet", index=False)


def test_load_and_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _write(tmp_path)
    d = K.load_frame()
    assert set(K.DEFAULT) <= set(d.columns) and d["TOP"].max() == 100
    cnt = K.by_count(d, K.DEFAULT, 60, 21)
    assert cnt[cnt["k"] >= 0]["Tage"].sum() == cnt[cnt["k"] == -1]["Tage"].iloc[0]
    # COT erst ab dem Montag nach dem Stichtag
    first_cot = d["COT"].first_valid_index()
    assert first_cot >= pd.Timestamp("2005-01-04") + pd.Timedelta(days=6)


def test_trades_do_not_overlap():
    idx = pd.bdate_range("2020-01-01", periods=100)
    d = pd.DataFrame({"close": np.linspace(100, 200, 100)}, index=idx)
    t = K.trades(d, pd.Series(True, index=idx), 10)
    assert len(t) == 9 and (t["Einstieg"].diff().dropna() >= pd.Timedelta(days=10)).all()
    assert K.trade_stats(t)["winrate"] == 1.0


def test_page(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _write(tmp_path)
    at = AppTest.from_string("from src.pages import consensus\nconsensus.page()\n", default_timeout=180)
    at.run()
    assert not at.exception
    at.slider(key="cons_th").set_value(70).run()
    at.radio(key="cons_h").set_value(5).run()
    assert not at.exception


def test_page_without_data(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    at = AppTest.from_string("from src.pages import consensus\nconsensus.page()\n", default_timeout=60)
    at.run()
    assert not at.exception and at.info
