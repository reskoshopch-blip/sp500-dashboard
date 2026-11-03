"""Abruf von Makro- und Positionierungsdaten (gratis, ohne API-Schlüssel).

* FRED (Federal Reserve Bank of St. Louis): https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIE>
  Zinsen, Notenbankbilanzen, Kreditspreads, Konjunktur, Inflation, Wechselkurse – USA und Japan (OECD-Reihen).
* CFTC Commitments of Traders (Legacy, nur Futures): https://publicreporting.cftc.gov (Socrata, Datensatz
  6dca-aqww), Rückfall auf die Jahresdateien https://www.cftc.gov/files/dea/history/deacot<JAHR>.zip.

Jede Reihe wird als Parquet unter data/fundamental/ zwischengespeichert. Fehlt eine Reihe (Quelle nicht
erreichbar, Serie eingestellt), wird sie übersprungen und als fehlend gemeldet – es wird nichts erfunden.
"""
from __future__ import annotations

import io
import time
import zipfile
from datetime import date
from pathlib import Path
from typing import Callable

import httpx
import pandas as pd

from src.config import Settings

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
COT_SOCRATA = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
COT_ZIP = "https://www.cftc.gov/files/dea/history/deacot{year}.zip"
# FRED blockt teils Programme, die sich als Browser ausgeben (Verbindung wird getrennt). Darum zuerst ein
# ehrlicher Programm-User-Agent, bei Fehlern der Browser-Header.
FRED_HEADERS = [{"User-Agent": "sp500-dashboard/1.0 (python-httpx; private research)", "Accept": "text/csv,*/*"}]
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/124.0 Safari/537.36", "Accept": "text/csv,application/json,*/*"}
COT_FIELDS = {"date": "report_date_as_yyyy_mm_dd", "code": "cftc_contract_market_code",
              "name": "market_and_exchange_names", "oi": "open_interest_all",
              "nc_long": "noncomm_positions_long_all", "nc_short": "noncomm_positions_short_all",
              "c_long": "comm_positions_long_all", "c_short": "comm_positions_short_all",
              "nr_long": "nonrept_positions_long_all", "nr_short": "nonrept_positions_short_all"}


def fund_dir(settings: Settings, sub: str = "") -> Path:
    d = settings.data_dir / "fundamental" / sub if sub else settings.data_dir / "fundamental"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------------------------------------------ FRED
def parse_fred_csv(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    if df.shape[1] < 2:
        raise ValueError("unerwartetes FRED-Format")
    d = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    v = pd.to_numeric(df.iloc[:, 1].replace(".", None), errors="coerce")
    s = pd.Series(v.values, index=d).dropna()
    s = s[s.index.notna()]
    return s[~s.index.duplicated()].sort_index()


def fred_downloader(sid: str, client: httpx.Client | None = None) -> pd.Series:
    last = None
    for headers in FRED_HEADERS + [HEADERS]:
        for attempt in range(3):
            try:
                with httpx.Client(headers=headers, timeout=httpx.Timeout(30, connect=15), follow_redirects=True,
                                  http2=False) as c:
                    r = c.get(FRED_URL.format(sid=sid))
                if r.status_code == 200 and "," in r.text[:200]:
                    return parse_fred_csv(r.text)
                last = f"HTTP {r.status_code}"
            except httpx.HTTPError as exc:
                last = str(exc) or type(exc).__name__
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"FRED {sid}: {last}")


def fetch_fred(settings: Settings, ids: list[str], force: bool = False, downloader: Callable | None = None,
               max_age_hours: float = 18, progress: Callable[[str], None] | None = None) -> dict:
    d = fund_dir(settings, "fred")
    ok, failed, cached = [], {}, []
    client = None
    try:
        for n, sid in enumerate(ids):
            p = d / f"{sid}.parquet"
            if p.exists() and not force and time.time() - p.stat().st_mtime < max_age_hours * 3600:
                cached.append(sid)
                continue
            if progress:
                progress(f"FRED {sid} ({n + 1}/{len(ids)})")
            try:
                if not ok and len(failed) >= 3:      # FRED blockiert -> nicht jede Reihe einzeln abwarten
                    raise RuntimeError("FRED nicht erreichbar (übersprungen nach 3 Fehlversuchen)")
                s = downloader(sid) if downloader else fred_downloader(sid)
                if not downloader:
                    time.sleep(0.5)                   # FRED schonend abfragen
                if s is None or s.empty:
                    raise RuntimeError("leer")
                s.rename("value").rename_axis("date").reset_index().to_parquet(p, index=False)
                ok.append(sid)
            except Exception as exc:  # noqa: BLE001 - eine Reihe darf nie den ganzen Lauf abbrechen
                failed[sid] = str(exc)[:200]
    finally:
        if client:
            client.close()
    return {"ok": ok, "cached": cached, "failed": failed}


def load_fred(settings: Settings, ids: list[str]) -> dict[str, pd.Series]:
    d = fund_dir(settings, "fred")
    out = {}
    for sid in ids:
        p = d / f"{sid}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            out[sid] = pd.Series(df["value"].values, index=pd.to_datetime(df["date"]), name=sid).sort_index()
    return out


# ------------------------------------------------------------------------------------------------------- COT
ZIP_COLS = {"market and exchange names": "name", "as of date in form yyyy-mm-dd": "date",
            "cftc contract market code": "code", "open interest (all)": "oi",
            "noncommercial positions-long (all)": "nc_long", "noncommercial positions-short (all)": "nc_short",
            "commercial positions-long (all)": "c_long", "commercial positions-short (all)": "c_short",
            "nonreportable positions-long (all)": "nr_long", "nonreportable positions-short (all)": "nr_short"}


def _normalize_cot(df: pd.DataFrame) -> pd.DataFrame:
    """Socrata- oder Zip-Format → Spalten date, code, name, oi, nc_long, … (numerisch)."""
    if COT_FIELDS["date"] in df.columns:
        cols = {v: k for k, v in COT_FIELDS.items() if v in df.columns}
    else:  # Zip-Format mit Klartext-Spaltennamen (exakter Vergleich, "Commercial" ist Teil von "Noncommercial")
        cols = {c: ZIP_COLS[c.lower().strip()] for c in df.columns if c.lower().strip() in ZIP_COLS}
    out = df[list(cols)].rename(columns=cols)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["code"] = out["code"].astype(str).str.strip()
    for c in ("oi", "nc_long", "nc_short", "c_long", "c_short", "nr_long", "nr_short"):
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["date"])


def cot_downloader(codes: list[str], client: httpx.Client | None = None) -> pd.DataFrame:
    own = client is None
    client = client or httpx.Client(headers=HEADERS, timeout=120, follow_redirects=True)
    try:
        where = "cftc_contract_market_code in(" + ",".join(f"'{c}'" for c in codes) + ")"
        try:
            r = client.get(COT_SOCRATA, params={"$where": where, "$limit": 50000, "$order": COT_FIELDS["date"]})
            if r.status_code == 200 and r.json():
                df = pd.DataFrame(r.json())
                if {COT_FIELDS["date"], COT_FIELDS["oi"], COT_FIELDS["nc_long"]} <= set(df.columns):
                    return _normalize_cot(df)
        except (httpx.HTTPError, ValueError):
            pass
        frames = []                                  # Rückfall: Jahresdateien
        for y in range(1986, date.today().year + 1):
            try:
                r = client.get(COT_ZIP.format(year=y))
                if r.status_code != 200:
                    continue
                z = zipfile.ZipFile(io.BytesIO(r.content))
                with z.open(z.namelist()[0]) as fh:
                    df = _normalize_cot(pd.read_csv(fh, low_memory=False))
                frames.append(df[df["code"].isin(codes)])
            except Exception:  # noqa: BLE001
                continue
        if not frames:
            raise RuntimeError("CFTC nicht erreichbar")
        return pd.concat(frames, ignore_index=True)
    finally:
        if own:
            client.close()


def fetch_cot(settings: Settings, codes: list[str], force: bool = False, downloader: Callable | None = None,
              max_age_hours: float = 24, name: str = "cot.parquet") -> dict:
    p = fund_dir(settings) / name
    if p.exists() and not force and time.time() - p.stat().st_mtime < max_age_hours * 3600:
        return {"ok": False, "cached": True}
    try:
        df = (downloader or cot_downloader)(codes)
        df = df.drop_duplicates(["date", "code"]).sort_values("date")
        df.to_parquet(p, index=False)
        return {"ok": True, "rows": len(df)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:200]}


def load_cot(settings: Settings, name: str = "cot.parquet") -> pd.DataFrame:
    p = fund_dir(settings) / name
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()
