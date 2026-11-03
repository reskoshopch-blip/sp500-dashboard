"""S&P-500-Mitgliederliste.

Die offizielle Liste von S&P Dow Jones Indices ist lizenzpflichtig und nicht frei abrufbar.
Als nachvollziehbare, öffentlich zugängliche Quelle wird die Wikipedia-Liste
"List of S&P 500 companies" verwendet (Tabellen 'constituents' und 'changes'), deren Einträge
auf Pressemitteilungen von S&P DJI verweisen. Die CIK aus Wikipedia wird gegen die offizielle
SEC-Ticker-Datei geprüft.

Fallback: eine lokale CSV unter data/sp500_constituents.csv mit den Spalten
  ticker,name,cik,sector,sub_industry   (sector/sub_industry optional)
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone

import httpx
import pandas as pd

from src.config import Settings

log = logging.getLogger(__name__)


@dataclass
class ConstituentData:
    constituents: pd.DataFrame          # ticker, name, cik, sector, sub_industry, date_added, source
    changes: pd.DataFrame               # date, added_ticker, removed_ticker, reason
    source: str
    retrieved_at: str


def normalize_ticker(t: str) -> str:
    return str(t).strip().upper().replace(".", "-")


def fetch_constituents(settings: Settings, sec_tickers: dict | None = None) -> ConstituentData:
    errors: list[str] = []
    try:
        data = _from_wikipedia(settings)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Wikipedia: {exc}")
        log.warning("Wikipedia-Liste nicht abrufbar: %s", exc)
        data = None
    if data is None:
        if settings.local_constituents_csv.exists():
            data = _from_csv(settings)
        else:
            raise RuntimeError(
                "Die S&P-500-Liste konnte nicht geladen werden ("
                + "; ".join(errors)
                + f"). Lege alternativ eine CSV unter {settings.local_constituents_csv} ab "
                "(Spalten: ticker,name,cik,sector,sub_industry)."
            )
    if sec_tickers:
        data.constituents = _reconcile_with_sec(data.constituents, sec_tickers)
    return data


def _from_wikipedia(settings: Settings) -> ConstituentData:
    resp = httpx.get(
        settings.constituents_url,
        headers={"User-Agent": settings.require_user_agent()},
        timeout=settings.http_timeout,
        follow_redirects=True,
    )
    resp.raise_for_status()
    html = resp.text
    cons = pd.read_html(io.StringIO(html), attrs={"id": "constituents"})[0]
    cons = cons.rename(columns={
        "Symbol": "ticker", "Security": "name", "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry", "Date added": "date_added", "CIK": "cik",
    })
    cons = cons[["ticker", "name", "cik", "sector", "sub_industry", "date_added"]].copy()
    cons["ticker"] = cons["ticker"].map(normalize_ticker)
    cons["cik"] = pd.to_numeric(cons["cik"], errors="coerce").astype("Int64")
    cons["source"] = "Wikipedia: List of S&P 500 companies"
    try:
        ch = pd.read_html(io.StringIO(html), attrs={"id": "changes"})[0]
        changes = parse_changes_table(ch)
    except Exception as exc:  # noqa: BLE001
        log.warning("Änderungstabelle nicht lesbar: %s", exc)
        changes = pd.DataFrame(columns=["date", "added_ticker", "removed_ticker", "reason"])
    return ConstituentData(cons, changes, settings.constituents_url, _now())


def parse_changes_table(ch: pd.DataFrame) -> pd.DataFrame:
    if isinstance(ch.columns, pd.MultiIndex):
        ch.columns = [" ".join(dict.fromkeys(str(c) for c in col)).strip() for col in ch.columns]
    cols = {c.lower(): c for c in ch.columns}

    def pick(*keys):
        for k, orig in cols.items():
            if all(x in k for x in keys):
                return orig
        return None

    out = pd.DataFrame({
        "date": pd.to_datetime(ch[pick("date")], errors="coerce"),
        "added_ticker": ch[pick("added", "ticker")] if pick("added", "ticker") else None,
        "removed_ticker": ch[pick("removed", "ticker")] if pick("removed", "ticker") else None,
        "reason": ch[pick("reason")] if pick("reason") else None,
    })
    for c in ("added_ticker", "removed_ticker"):
        out[c] = out[c].where(out[c].notna(), None).map(lambda x: normalize_ticker(x) if x else None)
    return out.dropna(subset=["date"]).reset_index(drop=True)


def members_at(current_tickers: set[str], changes: pd.DataFrame, at: date) -> set[str]:
    """Rekonstruiert die Indexmitglieder zu einem Stichtag, indem alle Änderungen nach dem
    Stichtag rückwärts aufgelöst werden. Ergebnis ist nur so vollständig wie die Änderungstabelle."""
    members = set(current_tickers)
    ts = pd.Timestamp(at)
    later = changes[changes["date"] > ts].sort_values("date", ascending=False)
    for row in later.itertuples(index=False):
        if isinstance(row.added_ticker, str) and row.added_ticker:
            members.discard(row.added_ticker)
        if isinstance(row.removed_ticker, str) and row.removed_ticker:
            members.add(row.removed_ticker)
    return members


def _from_csv(settings: Settings) -> ConstituentData:
    df = pd.read_csv(settings.local_constituents_csv)
    missing = {"ticker", "name", "cik"} - set(df.columns)
    if missing:
        raise RuntimeError(f"{settings.local_constituents_csv} fehlt Spalten: {sorted(missing)}")
    for c in ("sector", "sub_industry", "date_added"):
        if c not in df.columns:
            df[c] = None
    df["ticker"] = df["ticker"].map(normalize_ticker)
    df["cik"] = pd.to_numeric(df["cik"], errors="coerce").astype("Int64")
    df["source"] = f"Lokale Datei {settings.local_constituents_csv.name}"
    changes = pd.DataFrame(columns=["date", "added_ticker", "removed_ticker", "reason"])
    return ConstituentData(df[["ticker", "name", "cik", "sector", "sub_industry", "date_added", "source"]],
                           changes, str(settings.local_constituents_csv), _now())


def _reconcile_with_sec(cons: pd.DataFrame, sec_tickers: dict) -> pd.DataFrame:
    """CIK fehlt -> aus SEC-Tickerdatei ergänzen; Abweichungen werden markiert."""
    sec_map = {normalize_ticker(v["ticker"]): int(v["cik_str"]) for v in sec_tickers.values()}
    cons = cons.copy()
    notes = []
    for i, row in cons.iterrows():
        sec_cik = sec_map.get(row["ticker"])
        note = ""
        if pd.isna(row["cik"]) and sec_cik:
            cons.at[i, "cik"] = sec_cik
            note = "CIK aus SEC-Tickerdatei ergänzt"
        elif sec_cik and not pd.isna(row["cik"]) and int(row["cik"]) != sec_cik:
            note = f"CIK-Abweichung: Liste {int(row['cik'])}, SEC {sec_cik} (Listenwert verwendet)"
        elif not sec_cik:
            note = "Ticker nicht in SEC-Tickerdatei"
        notes.append(note)
    cons["cik_note"] = notes
    return cons


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
