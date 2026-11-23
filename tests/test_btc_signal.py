"""BTC-Spot-Absorption-Signal: Logik, Backtest und Seitenaufbau (ohne echte Netzwerkzugriffe)."""
from __future__ import annotations

import httpx
import numpy as np
import pandas as pd
import pytest

from src.crypto import btc_signal as bs

H = bs.HOUR_MS
T0 = 1_700_000_000_000 - (1_700_000_000_000 % H)


def _rows(closes, buy_share=0.5, highs=None, lows=None):
    rows = []
    for i, c in enumerate(closes):
        bsh = buy_share[i] if isinstance(buy_share, (list, np.ndarray)) else buy_share
        qv = 1_000_000.0
        hi = highs[i] if highs is not None else c * 1.001
        lo = lows[i] if lows is not None else c * 0.999
        rows.append([T0 + i * H, str(c), str(hi), str(lo), str(c), "10", T0 + (i + 1) * H - 1, str(qv), 100,
                     "5", str(qv * bsh), "0"])
    return rows


def _frame(closes, **kw):
    return bs.to_frame(_rows(closes, **kw), now_ms=T0 + (len(closes) + 1) * H)


def test_delta_from_taker_volume():
    df = _frame([100.0] * 3, buy_share=0.7)
    assert df["delta"].iloc[0] == pytest.approx(400_000)   # 700k Kauf − 300k Verkauf


def test_signal_requires_drop_and_positive_cvd():
    closes = [100.0] * 30 + [96.0]                       # −4 % gegenüber 24 Kerzen zuvor
    sig_buy = bs.compute(_frame(closes, buy_share=0.6))
    sig_sell = bs.compute(_frame(closes, buy_share=0.4))
    assert sig_buy["signal"].iloc[-1]
    assert not sig_sell["signal"].iloc[-1]
    flat = bs.compute(_frame([100.0] * 31, buy_share=0.6))
    assert not flat["signal"].any()


def test_open_candle_never_triggers():
    closes = [100.0] * 30 + [96.0]
    df = bs.to_frame(_rows(closes, buy_share=0.6), now_ms=T0 + 30 * H + 10)   # letzte Kerze noch offen
    assert not bs.compute(df)["signal"].iloc[-1]


def test_backtest_take_profit_stop_loss_and_fees():
    rule = bs.Rule(tp_pct=2, sl_pct=6, fee_pct=0.05)
    closes = [100.0] * 30 + [96.0, 96.5, 98.5]
    highs = [c * 1.001 for c in closes]
    lows = [c * 0.999 for c in closes]
    d = bs.compute(_frame(closes, buy_share=0.6, highs=highs, lows=lows), rule)
    trades, open_trade = bs.backtest(d, rule)
    assert len(trades) == 1 and open_trade is None
    t = trades.iloc[0]
    assert t["Grund"] == "Take Profit" and t["PnL %"] == pytest.approx(2 - 0.1)
    # Stop: Tief bricht 6 % unter Einstieg
    lows2 = lows[:-1] + [80.0]
    d2 = bs.compute(_frame(closes, buy_share=0.6, highs=highs, lows=lows2), rule)
    t2, _ = bs.backtest(d2, rule)
    assert t2.iloc[0]["Grund"] == "Stop Loss" and t2.iloc[0]["PnL %"] == pytest.approx(-6 - 0.1)
    s = bs.stats(t2)
    assert s["n"] == 1 and s["winrate"] == 0 and s["max_losses"] == 1


def test_open_trade_and_status():
    closes = [100.0] * 30 + [96.0, 96.2]
    d = bs.compute(_frame(closes, buy_share=0.6))
    trades, open_trade = bs.backtest(d, bs.Rule())
    assert trades.empty and open_trade["Einstiegspreis"] == 96.0
    st_ = bs.status(d)
    assert st_["cvd_ok"] and "missing_drop" in st_


def test_fetch_paginates():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        start = int(req.url.params["startTime"])
        calls.append(start)
        n = 1000 if len(calls) == 1 else 5
        return httpx.Response(200, json=[[start + i * H, "1", "1", "1", "1", "1",
                                                                 start + (i + 1) * H - 1, "1", 1, "1", "0.5", "0"]
                                   for i in range(n)])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    df = bs.fetch_klines(days=60, client=client, now_ms=T0 + 2000 * H)
    assert len(calls) == 2 and len(df) == 1005


def test_page_renders(monkeypatch):
    from streamlit.testing.v1 import AppTest
    rng = np.random.default_rng(1)
    closes = list(60000 * np.exp(np.cumsum(rng.normal(0, 0.01, 1500))))
    share = list(rng.uniform(0.4, 0.62, 1500))
    df = _frame(closes, buy_share=share)
    monkeypatch.setattr(bs, "fetch_klines", lambda *a, **k: df)
    at = AppTest.from_string("from src.pages import btc_signals\nbtc_signals.page()\n", default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Trefferquote" in m.label for m in at.metric)


def test_regime_bull_then_bear_and_filter():
    up = list(np.linspace(100, 200, 24 * 20))
    down = list(np.linspace(200, 90, 24 * 20))
    d = bs.compute(_frame(up + down, buy_share=0.6), bs.Rule(regime_days=5))
    assert d["regime"].iloc[24 * 15] == "Bull"
    assert d["regime"].iloc[-1] == "Bär"
    assert d["regime"].iloc[:24 * 4].isna().all()          # Vorlauf ohne Regime
    runs = bs.regime_runs(d)
    assert [r[2] for r in runs] == ["Bull", "Bär"]
    # starker Rückgang mit Spot-Käufen: Signale nur im Bär-Teil -> mit Bull-Filter keine
    assert d["signal"].any()
    filt = bs.compute(_frame(up + down, buy_share=0.6), bs.Rule(regime_days=5, bull_only=True))
    assert not filt["signal"].any()
    trades, _ = bs.backtest(d, bs.Rule(regime_days=5))
    by = bs.stats_by_regime(trades)
    assert list(by["Regime"]) == ["Bull", "Bär"]


def test_timeframes_window_in_bars():
    assert bs.Rule(timeframe="15m", window_h=4).window == 16
    assert bs.Rule(timeframe="1h", window_h=24).window == 24
    assert bs.Rule(timeframe="4h", window_h=72).window == 18
    for tf, pr in bs.PRESETS.items():
        assert tf in bs.TF_MIN and pr["days"] in pr["days_opts"]


def test_trade_duration_in_hours_for_15m():
    rule = bs.Rule(timeframe="15m", window_h=4, drop_pct=1.5, tp_pct=1, sl_pct=3)
    closes = [100.0] * 30 + [98.0, 98.3, 99.2]
    d = bs.compute(_frame(closes, buy_share=0.6), rule)
    trades, _ = bs.backtest(d, rule)
    assert len(trades) == 1 and trades.iloc[0]["Dauer h"] == pytest.approx(0.5)


def test_cvd_percentile_mode_and_multi_window():
    rng = np.random.default_rng(5)
    n = 24 * 120
    closes = list(60000 * np.exp(np.cumsum(rng.normal(0, 0.006, n))))
    df = _frame(closes, buy_share=list(rng.uniform(0.4, 0.6, n)))
    d0 = bs.compute(df, bs.Rule())
    d1 = bs.compute(df, bs.Rule(cvd_mode="pct", cvd_pct=80))
    assert d1["cvd_rank"].dropna().between(0, 100).all()
    assert (d1["cvd_ok"] == (d1["cvd_rank"] > 80)).all()
    assert d0["cvd_ok"].equals(d0["cvd"] > 0)
    mw = bs.multi_window(df, bs.Rule())
    assert list(mw["Fenster"]) == ["4 h", "24 h", "3 T", "7 T"]


def test_strategy_tables():
    from src.crypto.hyblock_strategies import scan_frame, strategies_frame
    s = strategies_frame()
    assert {"Name", "Regel", "Trades", "Summe %", "Jahr 1", "Jahr 2"} <= set(s.columns)
    assert len(scan_frame()) >= 6
