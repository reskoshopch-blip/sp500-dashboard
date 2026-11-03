"""Tägliche Schlusskurse (bereinigt um Splits und Dividenden) von Yahoo Finance über das Paket yfinance.

TradingView bietet keine freie Schnittstelle für Massendownloads historischer Kurse; Yahoo Finance ist die
beste kostenlose Quelle für ~500 US-Aktien. Die Kurse sind 'auto_adjust' (Dividenden reinvestiert), damit
Renditen Total Returns sind. Benchmark ist der ETF SPY (S&P 500 inkl. Dividenden).

Gespeichert wird je Ticker eine Parquet-Datei data/prices/<TICKER>.parquet (Spalten date, close).
Dateien jünger als 1 Tag werden nicht neu geladen; abgebrochene Abrufe setzen fort.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from src.config import Settings

log = logging.getLogger(__name__)
BENCHMARK = "SPY"
START = "2008-06-01"   # 1 Jahr Vorlauf vor den ersten Bilanzen (Momentum, Volatilität, SMA200)

Downloader = Callable[[list[str], str], dict[str, pd.Series]]


def price_dir(settings: Settings, subdir: str = "prices") -> Path:
    d = settings.data_dir / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(settings: Settings, ticker: str, subdir: str = "prices") -> Path:
    safe = ticker.replace("/", "_").replace("^", "IDX_").replace("=", "_")
    return price_dir(settings, subdir) / f"{safe}.parquet"


def yfinance_downloader(tickers: list[str], start: str) -> dict[str, pd.Series]:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Paket yfinance fehlt – bitte 'pip install -r requirements.txt' ausführen.") from exc
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False, group_by="ticker", threads=True)
    out: dict[str, pd.Series] = {}
    if df is None or df.empty:
        return out
    for t in tickers:
        try:
            if isinstance(df.columns, pd.MultiIndex):
                if t not in df.columns.get_level_values(0):
                    continue
                s = df[t]["Close"]
            else:
                s = df["Close"]
        except KeyError:
            continue
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s):
            s.index = pd.to_datetime(s.index).tz_localize(None) if getattr(s.index, "tz", None) else pd.to_datetime(s.index)
            out[t] = s
    return out


def fetch_prices(settings: Settings, tickers: list[str], force: bool = False, batch_size: int = 40,
                 downloader: Downloader | None = None, progress: Callable[[int, int], None] | None = None,
                 max_age_hours: float = 20, start: str = START, subdir: str = "prices",
                 require_benchmark: bool = True) -> dict:
    """Lädt fehlende/veraltete Kurse. Gibt {'ok': [...], 'failed': [...], 'skipped': [...]} zurück."""
    downloader = downloader or yfinance_downloader
    tickers = sorted(set(tickers) | ({BENCHMARK} if require_benchmark else set()))
    todo, skipped = [], []
    for t in tickers:
        p = _path(settings, t, subdir)
        if not force and p.exists() and (time.time() - p.stat().st_mtime) < max_age_hours * 3600:
            skipped.append(t)
        else:
            todo.append(t)
    ok, failed = [], []
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        got: dict[str, pd.Series] = {}
        for attempt in range(4):
            try:
                got = downloader(batch, start)
                break
            except Exception as exc:  # noqa: BLE001 – Rate Limit / Netzwerk: Backoff
                wait = 10 * (attempt + 1)
                log.warning("Kursabruf fehlgeschlagen (%s) – neuer Versuch in %ss", exc, wait)
                time.sleep(wait)
        for t in batch:
            s = got.get(t)
            if s is None or s.empty:
                failed.append(t)
                continue
            pd.DataFrame({"date": pd.to_datetime(s.index), "close": s.values.astype(float)}) \
                .to_parquet(_path(settings, t, subdir), index=False)
            ok.append(t)
        if progress:
            progress(min(i + batch_size, len(todo)), len(todo))
    if require_benchmark and BENCHMARK in failed:
        raise RuntimeError("Benchmark SPY konnte nicht geladen werden – ohne Benchmark ist kein Backtest möglich. "
                           "Internetverbindung prüfen und erneut versuchen.")
    return {"ok": ok, "failed": failed, "skipped": skipped}


def load_prices(settings: Settings, tickers: list[str], subdir: str = "prices",
                include_benchmark: bool = True) -> dict[str, pd.Series]:
    out = {}
    for t in set(tickers) | ({BENCHMARK} if include_benchmark else set()):
        p = _path(settings, t, subdir)
        if not p.exists() and subdir == "prices":
            legacy = price_dir(settings) / f"{t.replace('/', '_')}.parquet"   # ältere Dateinamen (z. B. ^VIX)
            p = legacy
        if p.exists():
            df = pd.read_parquet(p)
            s = pd.Series(df["close"].values, index=pd.to_datetime(df["date"])).sort_index()
            s = s[~s.index.duplicated(keep="last")]
            s = s[s > 0]
            if len(s):
                out[t] = s
    return out
