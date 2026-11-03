"""Märkte für Regime, Top-Warnung und technische Signale.

  SPY   S&P 500 über den ETF SPY (inkl. Dividenden), Yahoo ab 1993
  N225  JP225 / Nikkei 225 Index (^N225, ohne Dividenden – wie der JP225-CFD), Yahoo ab 1965
  EWJ   iShares MSCI Japan ETF (USD, inkl. Dividenden), Yahoo ab 1996

Zusatzreihen (Yahoo): ^VIX (US-Volatilität), ^JNIV (Nikkei Volatility Index, falls verfügbar – sonst realisierte
Volatilität), HYG/IEF (US-Kreditmarkt als globaler Risikoappetit), JPY=X (USD/JPY).
Marktbreite: S&P 500 aus den Mitgliedern des Dashboards, Japan aus den Nikkei-225-Mitgliedern (Code.T).
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from src.config import Settings
from src.data_sources.nikkei import fetch_nikkei_constituents
from src.ml.prices import fetch_prices, load_prices

LONG_START = "1960-01-01"
JP_STOCK_START = "2000-01-01"
LONG_DIR, JP_DIR = "prices_long", "prices_jp"

MARKETS: dict[str, dict] = {
    "SPY": {"label": "S&P 500 (SPY, inkl. Dividenden)", "short": "S&P 500", "main": "SPY", "vol": "^VIX",
            "credit": ("HYG", "IEF"), "fx": None, "stocks": "sp500", "first_test_year": 2001},
    "N225": {"label": "JP225 / Nikkei 225 (Index ^N225)", "short": "JP225", "main": "^N225", "vol": "^JNIV",
             "credit": ("HYG", "IEF"), "fx": "JPY=X", "stocks": "nikkei", "first_test_year": 1990},
    "EWJ": {"label": "Japan-ETF EWJ (USD, inkl. Dividenden)", "short": "EWJ", "main": "EWJ", "vol": "^JNIV",
            "credit": ("HYG", "IEF"), "fx": "JPY=X", "stocks": "nikkei", "first_test_year": 2005},
}
# Intermarket/Makro: gleichgewichteter S&P (RSP), Small Caps (IWM), zyklischer vs. defensiver Konsum (XLY/XLP),
# VIX-Terminstruktur (^VIX3M), Zinskurve (10 J. ^TNX minus 3 M. ^IRX), Kupfer/Gold (HG=F / GC=F)
US_EXTRAS = ["RSP", "IWM", "XLY", "XLP", "^VIX", "^VIX3M"]
GLOBAL_EXTRAS = ["^TNX", "^IRX", "HG=F", "GC=F"]
LONG_TICKERS = sorted({m["main"] for m in MARKETS.values()} | {"^VIX", "^JNIV", "HYG", "IEF", "JPY=X"}
                      | set(US_EXTRAS) | set(GLOBAL_EXTRAS))


def fetch_market_data(settings: Settings, downloader=None, force: bool = False,
                      progress: Callable[[str], None] | None = None) -> dict:
    info = {}
    if progress:
        progress("Indizes, Volatilität, Kredit, USD/JPY (lange Historie)")
    info["long"] = fetch_prices(settings, LONG_TICKERS, force=force, downloader=downloader, start=LONG_START,
                                subdir=LONG_DIR, require_benchmark=False)
    if progress:
        progress("Nikkei-225-Mitglieder")
    jp, source = fetch_nikkei_constituents(settings)
    info["nikkei_source"] = source
    info["nikkei_n"] = len(jp)
    if jp:
        if progress:
            progress(f"{len(jp)} japanische Aktien (Marktbreite)")
        info["jp"] = fetch_prices(settings, jp, force=force, downloader=downloader, start=JP_STOCK_START,
                                  subdir=JP_DIR, require_benchmark=False)
    return info


def market_inputs(settings: Settings, key: str, sp500_tickers: list[str]) -> dict | None:
    m = MARKETS[key]
    long = load_prices(settings, LONG_TICKERS, subdir=LONG_DIR, include_benchmark=False)
    main = long.get(m["main"])
    if main is None and key == "SPY":   # Rückfall: SPY aus dem Bilanz-ML-Ordner (ab 2008)
        main = load_prices(settings, []).get("SPY")
    if main is None or len(main) < 400:
        return None
    if m["stocks"] == "sp500":
        stocks = load_prices(settings, sp500_tickers, include_benchmark=False)
    else:
        csv = settings.data_dir / "nikkei225_constituents.csv"
        codes = pd.read_csv(csv, dtype=str)["code"].tolist() if csv.exists() else []
        stocks = load_prices(settings, [f"{c}.T" for c in codes], subdir=JP_DIR, include_benchmark=False)
    vol = long.get(m["vol"])
    if vol is None and key == "SPY":
        vol = load_prices(settings, ["^VIX"], include_benchmark=False).get("^VIX")
    hyg, ief = long.get(m["credit"][0]), long.get(m["credit"][1])
    names = GLOBAL_EXTRAS + (US_EXTRAS if key == "SPY" else [])
    extras = {n: long[n] for n in names if n in long}
    if key == "SPY" and "^VIX" not in extras and vol is not None:
        extras["^VIX"] = vol
    return {"main": main, "stocks": stocks, "vol": vol, "hyg": hyg, "ief": ief,
            "fx": long.get(m["fx"]) if m["fx"] else None, "first_test_year": m["first_test_year"], "extras": extras}
