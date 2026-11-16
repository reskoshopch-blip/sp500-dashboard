"""Nasdaq-IPO-Kalender: python -m src.ipo.nasdaq [--months 36] [--force]

Quelle: der öffentliche IPO-Kalender von nasdaq.com (https://api.nasdaq.com/api/ipo/calendar?date=JJJJ-MM) mit den
Tabellen „priced“ (erfolgte Börsengänge), „upcoming“ (anstehend), „filed“ (eingereicht) und „withdrawn“.
Keine offizielle Schnittstelle: Felder werden tolerant gelesen. Vergangene Monate werden einmal gespeichert
(data/ipo/months/JJJJ-MM.json) und nicht erneut abgefragt; aktueller und nächster Monat bei jedem Lauf.

Danach: Kurse ab dem ersten Handelstag (Yahoo Finance) und Auswertung, wie sich Nasdaq-IPOs nach 1 Tag, 1 Woche,
1/3/6/12 Monaten gegenüber dem Ausgabepreis und dem Nasdaq 100 (QQQ) entwickelt haben → data/ipo/.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date, datetime, timezone

import httpx
import numpy as np
import pandas as pd

from src.config import get_settings
from src.utils import atomic_write_text

URL = "https://api.nasdaq.com/api/ipo/calendar"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/130.0 Safari/537.36",
           "Accept": "application/json, text/plain, */*", "Accept-Language": "de-CH,de;q=0.9,en;q=0.8",
           "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/market-activity/ipos"}
BENCH = "QQQ"
HORIZONS = {"1 Tag": 0, "1 Woche": 5, "1 Monat": 21, "3 Monate": 63, "6 Monate": 126, "12 Monate": 252}
PRICE_DIR = "ipo_prices"


def ipo_dir(sub: str = ""):
    d = get_settings().data_dir / "ipo" / sub if sub else get_settings().data_dir / "ipo"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------------------------- Abruf/Parsen
def _num(v):
    if v is None:
        return None
    s = re.sub(r"[^0-9.\-]", "", str(v))
    try:
        return float(s) if s not in ("", "-", ".") else None
    except ValueError:
        return None


def _range(v):
    """'14.00-16.00' -> (14, 16); '15.00' -> (15, 15)."""
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(v or ""))]
    return (min(nums), max(nums)) if nums else (None, None)


def _rows(js, *path):
    o = js
    for k in path:
        o = (o or {}).get(k) if isinstance(o, dict) else None
    return o if isinstance(o, list) else []


def _date(v):
    d = pd.to_datetime(v, errors="coerce")
    return None if pd.isna(d) else d.strftime("%Y-%m-%d")


def parse_month(js: dict) -> pd.DataFrame:
    data = (js or {}).get("data") or {}
    out = []
    tables = {"priced": _rows(data, "priced", "rows"),
              "upcoming": _rows(data, "upcoming", "upcomingTable", "rows") or _rows(data, "upcoming", "rows"),
              "filed": _rows(data, "filed", "rows"), "withdrawn": _rows(data, "withdrawn", "rows")}
    for status, rows in tables.items():
        for r in rows:
            lo, hi = _range(r.get("proposedSharePrice"))
            out.append({
                "status": status, "deal_id": str(r.get("dealID") or ""),
                "symbol": (r.get("proposedTickerSymbol") or "").strip().upper() or None,
                "company": (r.get("companyName") or "").strip(),
                "exchange": (r.get("proposedExchange") or "").strip(),
                "price_lo": lo, "price_hi": hi,
                "shares": _num(r.get("sharesOffered")),
                "volume_usd": _num(r.get("dollarValueOfSharesOffered")),
                "date": _date(r.get("pricedDate") or r.get("expectedPriceDate") or r.get("filedDate")
                              or r.get("withdrawDate")),
            })
    return pd.DataFrame(out)


def fetch_month(month: str, client: httpx.Client) -> dict:
    last = None
    for attempt in range(3):
        try:
            r = client.get(URL, params={"date": month})
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
        except (httpx.HTTPError, ValueError) as exc:
            last = str(exc) or type(exc).__name__
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Nasdaq {month}: {last}")


def months_back(n: int, today: date | None = None) -> list[str]:
    today = today or date.today()
    p = pd.Period(today, "M")
    return [str(p + k) for k in range(-n, 2)]          # n Monate zurück bis nächster Monat


def collect(months: int = 36, force: bool = False, client: httpx.Client | None = None,
            today: date | None = None) -> pd.DataFrame:
    md = ipo_dir("months")
    cur = str(pd.Period(today or date.today(), "M"))
    own = client is None
    client = client or httpx.Client(headers=HEADERS, timeout=httpx.Timeout(30, connect=15), follow_redirects=True)
    frames, errors = [], {}
    try:
        for m in months_back(months, today):
            p = md / f"{m}.json"
            if p.exists() and not force and m < cur:
                js = json.loads(p.read_text(encoding="utf-8"))
            else:
                try:
                    js = fetch_month(m, client)
                    atomic_write_text(p, json.dumps(js))
                    time.sleep(0.4)
                except Exception as exc:  # noqa: BLE001
                    errors[m] = str(exc)[:150]
                    if not p.exists():
                        continue
                    js = json.loads(p.read_text(encoding="utf-8"))
            f = parse_month(js)
            if len(f):
                f["month"] = m
                frames.append(f)
    finally:
        if own:
            client.close()
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(df):
        # ein Deal kann in mehreren Monaten auftauchen (z.B. upcoming → priced): den „weitesten“ Status behalten
        rank = {"filed": 0, "upcoming": 1, "withdrawn": 2, "priced": 3}
        df["_r"] = df["status"].map(rank)
        key = df["deal_id"].where(df["deal_id"] != "", df["company"])
        df = df.assign(_k=key).sort_values(["_k", "_r", "month"]).drop_duplicates("_k", keep="last")
        df = df.drop(columns=["_r", "_k"]).reset_index(drop=True)
    df.attrs["errors"] = errors
    return df


# --------------------------------------------------------------------------------------------- Auswertung
def performance(deals: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    """Je erfolgtem IPO: Rendite ab Ausgabepreis nach jedem Horizont und gegenüber QQQ im selben Zeitraum."""
    bench = prices.get(BENCH)
    rows = []
    for _, d in deals[deals["status"] == "priced"].iterrows():
        s = prices.get(d["symbol"]) if d["symbol"] else None
        ipo_px = d["price_hi"] if d["price_hi"] == d["price_lo"] else np.nanmean([d["price_lo"], d["price_hi"]])
        row = d.to_dict()
        row["ipo_price"] = ipo_px
        if s is None or s.empty or not ipo_px or pd.isna(ipo_px) or not d["date"]:
            rows.append(row)
            continue
        s = s[s.index >= pd.Timestamp(d["date"])]
        if s.empty:
            rows.append(row)
            continue
        first = s.index[0]
        if (first - pd.Timestamp(d["date"])).days > 10:          # Kurse beginnen viel später -> unbrauchbar
            rows.append(row)
            continue
        row["first_day"] = f"{first:%Y-%m-%d}"
        for lab, k in HORIZONS.items():
            if len(s) > k:
                row[f"r_{lab}"] = s.iloc[k] / ipo_px - 1
                if bench is not None and lab != "1 Tag":
                    b0, b1 = bench.asof(first), bench.asof(s.index[k])
                    # Überrendite ab Schlusskurs des 1. Tages (realistischer Einstieg für Privatanleger)
                    row[f"x_{lab}"] = (s.iloc[k] / s.iloc[0] - 1) - (b1 / b0 - 1) if b0 and b1 else None
                    row[f"d_{lab}"] = s.iloc[k] / s.iloc[0] - 1
        row["r_heute"] = s.iloc[-1] / ipo_px - 1
        row["last_price"] = float(s.iloc[-1])
        rows.append(row)
    return pd.DataFrame(rows)


def stats(perf: pd.DataFrame) -> dict:
    out = {"horizons": [], "pop": [], "size": []}
    p = perf.dropna(subset=["r_1 Tag"]) if "r_1 Tag" in perf else perf.iloc[0:0]
    for lab in HORIZONS:
        r = p.get(f"r_{lab}", pd.Series(dtype=float)).dropna()
        d = p.get(f"d_{lab}", pd.Series(dtype=float)).dropna()
        x = p.get(f"x_{lab}", pd.Series(dtype=float)).dropna()
        out["horizons"].append({
            "Horizont": lab, "IPOs": int(len(r)),
            "Ø ab Ausgabepreis": float(r.mean()) if len(r) else None,
            "Median ab Ausgabepreis": float(r.median()) if len(r) else None,
            "Anteil über Ausgabepreis": float((r > 0).mean()) if len(r) else None,
            "Ø ab 1. Schlusskurs": float(d.mean()) if len(d) else None,
            "Anteil im Plus ab 1. Schluss": float((d > 0).mean()) if len(d) else None,
            "Anteil besser als QQQ": float((x > 0).mean()) if len(x) else None,
            "Median vs. QQQ": float(x.median()) if len(x) else None})
    if len(p):
        bins = [-np.inf, 0, 0.25, 0.75, np.inf]
        labels = ["unter Ausgabepreis", "0–25 %", "25–75 %", "über 75 %"]
        p = p.assign(pop=pd.cut(p["r_1 Tag"], bins, labels=labels))
        for lab, g in p.groupby("pop", observed=True):
            x = g.get("x_3 Monate", pd.Series(dtype=float)).dropna()
            d = g.get("d_3 Monate", pd.Series(dtype=float)).dropna()
            out["pop"].append({"1. Tag": str(lab), "IPOs": int(len(g)),
                               "Ø nach 3 M. ab 1. Schluss": float(d.mean()) if len(d) else None,
                               "Anteil im Plus": float((d > 0).mean()) if len(d) else None,
                               "Anteil besser als QQQ": float((x > 0).mean()) if len(x) else None})
        v = p["volume_usd"]
        q = pd.cut(v, [0, 50e6, 200e6, 1e9, np.inf], labels=["< 50 Mio.", "50–200 Mio.", "200 Mio.–1 Mrd.",
                                                            "> 1 Mrd."])
        for lab, g in p.assign(sz=q).groupby("sz", observed=True):
            d = g.get("d_3 Monate", pd.Series(dtype=float)).dropna()
            out["size"].append({"Volumen": str(lab), "IPOs": int(len(g)),
                                "Ø 1. Tag": float(g["r_1 Tag"].mean()),
                                "Ø nach 3 M. ab 1. Schluss": float(d.mean()) if len(d) else None,
                                "Anteil im Plus": float((d > 0).mean()) if len(d) else None})
    return out


def run(months: int = 36, force: bool = False, client=None, downloader=None, only_nasdaq: bool = True) -> dict:
    from src.ml.prices import fetch_prices, load_prices
    s = get_settings()
    deals = collect(months, force, client)
    errors = deals.attrs.get("errors", {})
    if deals.empty:
        raise RuntimeError("Keine IPO-Daten von nasdaq.com erhalten" + (f" ({list(errors.values())[0]})"
                                                                         if errors else ""))
    if only_nasdaq:
        deals = deals[deals["exchange"].str.upper().str.contains("NASDAQ") | (deals["exchange"] == "")]
    priced = deals[(deals["status"] == "priced") & deals["symbol"].notna()]
    tickers = sorted(set(priced["symbol"]) | {BENCH})
    start = (pd.Timestamp(priced["date"].min()) - pd.Timedelta(days=10)).strftime("%Y-%m-%d") if len(priced) \
        else "2020-01-01"
    info = fetch_prices(s, tickers, start=start, subdir=PRICE_DIR, require_benchmark=False, downloader=downloader,
                        max_age_hours=20)
    prices = load_prices(s, tickers, subdir=PRICE_DIR, include_benchmark=False)
    perf = performance(deals, prices)
    deals.to_parquet(ipo_dir() / "deals.parquet", index=False)
    perf.to_parquet(ipo_dir() / "performance.parquet", index=False)
    st = stats(perf)
    meta = {"updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "months": months,
            "n_priced": int((deals["status"] == "priced").sum()), "n_upcoming": int((deals["status"] == "upcoming").sum()),
            "n_filed": int((deals["status"] == "filed").sum()), "n_with_prices": int(perf.get("r_1 Tag", pd.Series(dtype=float)).notna().sum()),
            "fetch_errors": errors, "price_failed": len(info.get("failed", [])), "stats": st}
    atomic_write_text(ipo_dir() / "meta.json", json.dumps(meta, ensure_ascii=False, default=float))
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Nasdaq-IPO-Kalender und Auswertung")
    ap.add_argument("--months", type=int, default=36)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    try:
        m = run(a.months, a.force)
    except Exception as exc:  # noqa: BLE001
        print(f"IPO-Auswertung fehlgeschlagen: {exc}", file=sys.stderr)
        return 1
    print(f"IPOs: {m['n_priced']} erfolgt ({m['n_with_prices']} mit Kursen), {m['n_upcoming']} anstehend, "
          f"{m['n_filed']} eingereicht. Fehler: {len(m['fetch_errors'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
