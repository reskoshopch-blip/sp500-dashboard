"""Fundamental-Lauf: Makro (FRED), Positionierung (CFTC), S&P-Gewinne (SEC) → lernender Score + ML-Prognose.

Aufruf:
  python -m src.fundamental.run            # Daten aktualisieren und alles berechnen
  python -m src.fundamental.run --offline  # nur mit vorhandenen Daten neu rechnen
Ergebnis: data/fundamental/fund_<MARKT>_*.parquet und fund_<MARKT>_meta.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import ConfigError, Settings, get_settings
from src.database.store import connect_readonly
from src.fundamental import earnings as E
from src.fundamental.catalog import COMMODITY_TICKERS, COT_CODES, FRED, IND_BY_KEY, for_market
from src.fundamental.features import build, known_on
from src.fundamental.learn import CAT_COL, run_market
from src.fundamental.sources import fetch_cot, fetch_fred, fund_dir, load_cot, load_fred
from src.ml.markets import LONG_DIR, LONG_START, MARKETS
from src.ml.prices import fetch_prices, load_prices
from src.utils import atomic_write_text

log = logging.getLogger("fundamental")
FUND_MARKETS = {"SPY": {"start": "1993-01-01", "first_test_year": 2001},
                "N225": {"start": "1990-01-01", "first_test_year": 1998}}
INDEX_TICKER = "^GSPC"


def status(settings: Settings, **kw) -> None:
    kw["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    atomic_write_text(fund_dir(settings) / "status.json", json.dumps(kw, ensure_ascii=False))


def _companies(settings: Settings) -> pd.DataFrame:
    if not settings.db_path.exists():
        return pd.DataFrame(columns=["cik", "primary_ticker"])
    con = connect_readonly(settings.db_path)
    try:
        c = con.execute("SELECT cik, primary_ticker FROM companies WHERE fetch_status = 'ok'").df()
    finally:
        con.close()
    c["cik"] = c["cik"].astype(int)
    return c


def build_earnings(settings: Settings, progress=None) -> dict:
    """Gewinnaggregate aus dem SEC-Cache + KGV-Anker. Rückgabe {agg, anchor} (leer, wenn keine Daten)."""
    comps = _companies(settings)
    if comps.empty:
        return {"agg": pd.DataFrame(), "anchor": {}}
    t, shares = E.collect(settings.raw_dir, comps["cik"].tolist(), progress)
    if t.empty:
        return {"agg": pd.DataFrame(), "anchor": {}}
    end = pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(0)
    dates = pd.date_range("2010-01-31", end, freq="ME")
    agg = E.aggregate(t, dates)
    cik_t = dict(zip(comps["cik"], comps["primary_ticker"]))
    prices = load_prices(settings, list(cik_t.values()), include_benchmark=False)
    anchor = E.current_pe(t, shares, prices, cik_t, pd.Timestamp.today().normalize())
    agg.reset_index().to_parquet(fund_dir(settings) / "earnings.parquet", index=False)
    return {"agg": agg, "anchor": anchor, "n_firms_ttm": int(t["cik"].nunique())}


def pe_daily(agg: pd.DataFrame, gspc: pd.Series | None, anchor: dict) -> pd.Series | None:
    """Tägliches KGV: S&P-500-Kursindex / verketteter Gewinnindex (bekannt zum Monatsende), auf das heutige
    Gesamt-KGV skaliert."""
    if agg is None or agg.empty or "e_chain" not in agg or gspc is None or gspc.empty:
        return None
    idx = gspc.index[gspc.index >= agg.index.min()]
    e = known_on(agg["e_chain"], idx, 0, 45)
    raw = gspc.reindex(idx) / e
    raw = raw.dropna()
    if raw.empty:
        return None
    pe = anchor.get("pe")
    return raw / raw.iloc[-1] * pe if pe else None


def _fmt_now(daily: pd.DataFrame, keys: list[str]) -> list[dict]:
    last = daily.iloc[-1]
    prev = daily.iloc[-64] if len(daily) > 64 else None
    rows = []
    for k in keys:
        rows.append({"key": k, "raw": last.get(f"raw_{k}"), "score": last.get(f"s_{k}"),
                     "oriented": last.get(f"o_{k}"), "weight": last.get(f"w_{k}"),
                     "oriented_3m": prev.get(f"o_{k}") if prev is not None else None})
    return rows


def run(offline: bool = False, force: bool = False) -> dict:
    settings = get_settings()
    out_dir = fund_dir(settings)
    try:
        info: dict = {}
        if not offline:
            status(settings, state="läuft", step="Makrodaten (FRED)")
            info["fred"] = fetch_fred(settings, list(FRED), force=force,
                                      progress=lambda m: status(settings, state="läuft", step=m))
            status(settings, state="läuft", step="Positionierung (CFTC COT)")
            info["cot"] = fetch_cot(settings, sorted({c for v in COT_CODES.values() for c in v}), force=force)
            status(settings, state="läuft", step="S&P-500-Kursindex")
            info["index"] = fetch_prices(settings, [INDEX_TICKER] + [m["main"] for m in MARKETS.values()]
                                         + list(COMMODITY_TICKERS.values()),
                                         force=force, start=LONG_START, subdir=LONG_DIR, require_benchmark=False)
        fred = load_fred(settings, list(FRED))
        cot = load_cot(settings)
        long = load_prices(settings, [INDEX_TICKER, "SPY", "^N225", *COMMODITY_TICKERS.values()], subdir=LONG_DIR,
                           include_benchmark=False)
        commodities = {k: long[t] for k, t in COMMODITY_TICKERS.items() if t in long}
        status(settings, state="läuft", step="S&P-500-Gewinne aus SEC-Daten")
        ern = build_earnings(settings)
        pe = pe_daily(ern["agg"], long.get(INDEX_TICKER), ern["anchor"])
        metas = {}
        for key, cfg in FUND_MARKETS.items():
            close = long.get(MARKETS[key]["main"])
            if close is None or len(close) < 1500:
                metas[key] = {"error": f"keine Kurse für {MARKETS[key]['main']}"}
                continue
            status(settings, state="läuft", step=f"{MARKETS[key]['short']}: Indikatoren und Lernen")
            close = close[close.index >= cfg["start"]]
            raw = build(key, close.index, fred, cot, ern["agg"], pe, commodities)
            raw = raw.loc[:, raw.notna().sum() > 0]
            res = run_market(raw, close, cfg["first_test_year"])
            d = res["daily"]
            d.to_parquet(out_dir / f"fund_{key}_daily.parquet", index=False)
            res["learned"].to_parquet(out_dir / f"fund_{key}_learned.parquet", index=False)
            res["bt"].to_parquet(out_dir / f"fund_{key}_bt.parquet", index=False)
            res["bt_div"].to_parquet(out_dir / f"fund_{key}_btdiv.parquet", index=False)
            res["effects"].to_parquet(out_dir / f"fund_{key}_effects.parquet", index=False)
            keys = [i.key for i in for_market(key)]
            missing = [IND_BY_KEY[k].label for k in keys if k not in raw.columns or raw[k].dropna().empty]
            metas[key] = {"ml": res["ml"], "start": res["start"], "missing": missing,
                          "available": [k for k in keys if k in raw.columns],
                          "now": _fmt_now(d.set_index("date"), [k for k in keys if k in raw.columns]),
                          "categories": {c: col for c, col in CAT_COL.items()},
                          "as_of": str(d["date"].iloc[-1].date())}
        meta = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "markets": metas, "pe_anchor": ern.get("anchor", {}), "n_firms_ttm": ern.get("n_firms_ttm"),
                "fred_failed": info.get("fred", {}).get("failed", {}), "cot": info.get("cot", {}),
                "fred_available": sorted(fred), "cot_rows": int(len(cot))}
        atomic_write_text(out_dir / "meta.json", json.dumps(meta, ensure_ascii=False, indent=1, default=_json))
        ok = [k for k, v in metas.items() if "error" not in v]
        status(settings, state="fertig", step=f"fertig ({', '.join(ok) or 'keine Märkte'})",
               fred_ok=len(fred), fred_failed=len(meta["fred_failed"]), cot_rows=int(len(cot)))
        return meta
    except Exception as exc:
        status(settings, state="Fehler", step=str(exc)[:300])
        raise


def _json(o):
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return str(o)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fundamental-Terminal berechnen")
    ap.add_argument("--offline", action="store_true", help="Keine Downloads, nur vorhandene Daten")
    ap.add_argument("--force", action="store_true", help="Alle Reihen neu laden")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        meta = run(a.offline, a.force)
    except ConfigError as exc:
        print(f"KONFIGURATIONSFEHLER: {exc}", file=sys.stderr)
        return 2
    print(f"FRED-Reihen: {len(meta['fred_available'])} vorhanden, {len(meta['fred_failed'])} fehlgeschlagen"
          + (f" ({', '.join(meta['fred_failed'])})" if meta["fred_failed"] else ""))
    print(f"COT-Zeilen: {meta['cot_rows']} · KGV-Anker: {meta['pe_anchor']}")
    for k, m in meta["markets"].items():
        if "error" in m:
            print(f"{k}: FEHLER {m['error']}")
            continue
        r3, r12 = m["ml"].get("3M", {}), m["ml"].get("12M", {})
        print(f"{k}: AUC 3M {r3.get('auc', float('nan')):.3f}, 12M {r12.get('auc', float('nan')):.3f}; "
              f"fehlend: {', '.join(m['missing']) or '–'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
