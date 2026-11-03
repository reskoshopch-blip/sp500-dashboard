"""Champagner-Index: Korb, Score ohne Zukunftsdaten, gelernte Richtung, Seite und Konsens."""
from __future__ import annotations

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.geo import champagne as C


def downloader(tickers, start):
    idx = pd.bdate_range("2008-01-01", "2026-10-30")
    rng = np.random.default_rng(1)
    out = {}
    for i, t in enumerate(tickers):
        if t == "VRAP.PA":
            continue                                  # fehlender Titel darf nicht stören
        out[t] = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, len(idx)))), index=idx)
    return out


def test_run_and_no_lookahead(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    r = C.run(downloader=downloader)
    assert 0 <= r["score"] <= 100 and "VRAP.PA" in r["missing"] and r["stats"]
    df = pd.read_parquet(tmp_path / "geo" / "champagne_daily.parquet").set_index("date")
    # Score bis 2015 darf sich nicht ändern, wenn spätere Kurse wegfallen
    prices = C.load_prices(C.get_settings(), [*C.HOUSES, *C.MARKET, C.SPY], subdir=C.SUBDIR, include_benchmark=False)
    cut = {k: v[v.index <= "2015-12-31"] for k, v in prices.items()}
    short = C.build(cut)
    common = short.index[short.index <= "2015-12-31"]
    assert np.allclose(short.loc[common, "score"].fillna(-1), df.loc[common, "score"].fillna(-1))
    assert df["score_learned"].notna().sum() > 1000


def test_page_and_consensus(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    at = AppTest.from_string("from src.pages import champagne\nchampagne.page()\n", default_timeout=60)
    at.run()
    assert not at.exception and at.info
    C.run(downloader=downloader)
    at = AppTest.from_string("from src.pages import champagne\nchampagne.page()\n", default_timeout=60)
    at.run()
    assert not at.exception and len(at.metric) == 4
