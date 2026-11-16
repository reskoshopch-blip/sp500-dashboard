"""Nasdaq-IPOs: Kalender tolerant lesen, Monate cachen, Performance ab Ausgabepreis/1. Schluss, Seite."""
from __future__ import annotations

import json
from datetime import date

import httpx
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.ipo import nasdaq as N


def cal(month):
    if month == "2026-11":
        return {"data": {"upcoming": {"upcomingTable": {"rows": [
            {"dealID": "u1", "proposedTickerSymbol": "NEWX", "companyName": "NewX Inc.", "proposedExchange": "NASDAQ Global",
             "proposedSharePrice": "14.00-16.00", "sharesOffered": "10,000,000", "expectedPriceDate": "11/20/2026",
             "dollarValueOfSharesOffered": "$160,000,000"}]}},
            "priced": {"rows": []}, "filed": {"rows": [{"dealID": "f1", "companyName": "Later Corp",
                                                        "filedDate": "11/02/2026", "dollarValueOfSharesOffered": "$10,000,000"}]}}}
    if month == "2025-03":
        return {"data": {"priced": {"rows": [
            {"dealID": "p1", "proposedTickerSymbol": "AAA", "companyName": "Alpha", "proposedExchange": "NASDAQ Global Select",
             "proposedSharePrice": "20.00", "sharesOffered": "5,000,000", "pricedDate": "03/10/2025",
             "dollarValueOfSharesOffered": "$100,000,000"},
            {"dealID": "p2", "proposedTickerSymbol": "NYS", "companyName": "NYSE Co", "proposedExchange": "NYSE",
             "proposedSharePrice": "10.00", "pricedDate": "03/12/2025", "dollarValueOfSharesOffered": "$50,000,000"}]},
            "upcoming": {"upcomingTable": {"rows": None}}}}
    return {"data": {"priced": {"rows": None}, "upcoming": {"upcomingTable": None}, "filed": {"rows": []}}}


def client(calls):
    def h(req):
        m = req.url.params["date"]
        calls.append(m)
        return httpx.Response(200, json=cal(m))
    return httpx.Client(transport=httpx.MockTransport(h))


def downloader(tickers, start):
    idx = pd.bdate_range("2025-03-10", "2026-11-13")
    out = {}
    for t in tickers:
        if t == "AAA":
            out[t] = pd.Series(np.linspace(30, 45, len(idx)), index=idx)      # 1. Tag +50 %, danach steigend
        if t == "QQQ":
            out[t] = pd.Series(np.linspace(400, 440, len(idx)), index=idx)
    return out


def test_parse_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    calls = []
    df = N.collect(months=24, client=client(calls), today=date(2026, 11, 16))
    assert set(df["status"]) == {"priced", "upcoming", "filed"}
    up = df[df["status"] == "upcoming"].iloc[0]
    assert (up["price_lo"], up["price_hi"], up["volume_usd"], up["date"]) == (14, 16, 160e6, "2026-11-20")
    n1 = len(calls)
    calls.clear()
    N.collect(months=24, client=client(calls), today=date(2026, 11, 16))
    assert calls == ["2026-11", "2026-12"] and n1 == 26           # vergangene Monate aus dem Cache


def test_run_performance_and_page(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(N, "months_back", lambda n, today=None: ["2025-03", "2026-11", "2026-12"])
    m = N.run(months=24, client=client([]), downloader=downloader)
    perf = pd.read_parquet(tmp_path / "ipo" / "performance.parquet")
    a = perf[perf["symbol"] == "AAA"].iloc[0]
    assert abs(a["r_1 Tag"] - 0.5) < 1e-9 and a["r_3 Monate"] > 0.5 and a["x_3 Monate"] > 0
    assert "NYS" not in set(perf["symbol"])                      # nur Nasdaq
    assert m["n_upcoming"] == 1 and m["stats"]["horizons"][0]["IPOs"] == 1
    at = AppTest.from_string("from src.pages import ipo\nipo.page()\n", default_timeout=60)
    at.run()
    assert not at.exception and at.metric[0].value == "1"


def test_page_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    at = AppTest.from_string("from src.pages import ipo\nipo.page()\n", default_timeout=60)
    at.run()
    assert not at.exception and at.info


def test_fred_yahoo_proxy(tmp_path, monkeypatch):
    from src.config import get_settings
    from src.fundamental import sources as S
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    def bad(sid):
        raise RuntimeError("getaddrinfo failed")
    def yd(tickers, start):
        idx = pd.bdate_range("2020-01-01", periods=300)
        return {t: pd.Series(42.0, index=idx) for t in tickers}
    r = S.fetch_fred(get_settings(), ["DGS10", "UNRATE"], downloader=bad, proxy_downloader=yd)
    assert r["proxy"] == ["DGS10"] and "UNRATE" in r["failed"]
    assert S.load_fred(get_settings(), ["DGS10"])["DGS10"].iloc[-1] == 4.2          # ^TNX ×10 korrigiert
