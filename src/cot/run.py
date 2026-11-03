"""COT-Modell berechnen: python -m src.cot.run [--offline] [--force]

Lädt die CFTC-COT-Daten (Gold, Öl, Kupfer, E-mini S&P 500, E-mini Nasdaq 100) und die Kurse (Yahoo: GC=F, CL=F,
HG=F, ^GSPC, ^NDX; Öl vor 2000 von FRED), lernt je Markt den COT-Score und speichert alles unter data/cot/.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

from src.config import ConfigError, Settings, get_settings
from src.cot.model import ALL_CODES, MARKETS, run_market
from src.fundamental.sources import fetch_cot, fetch_fred, load_cot, load_fred
from src.ml.markets import LONG_DIR, LONG_START
from src.ml.prices import fetch_prices, load_prices
from src.utils import atomic_write_text

COT_FILE = "cot_markets.parquet"


def cot_dir(settings: Settings):
    d = settings.data_dir / "cot"
    d.mkdir(parents=True, exist_ok=True)
    return d


def status(settings: Settings, **kw) -> None:
    kw["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    atomic_write_text(cot_dir(settings) / "status.json", json.dumps(kw, ensure_ascii=False))


def run(offline: bool = False, force: bool = False) -> dict:
    s = get_settings()
    out = cot_dir(s)
    try:
        info = {}
        if not offline:
            status(s, state="läuft", step="CFTC-COT-Daten")
            info["cot"] = fetch_cot(s, ALL_CODES, force=force, name=COT_FILE, max_age_hours=12)
            status(s, state="läuft", step="Kurse (Yahoo, FRED)")
            info["prices"] = fetch_prices(s, [m["price"] for m in MARKETS.values()], force=force, start=LONG_START,
                                          subdir=LONG_DIR, require_benchmark=False)
            fetch_fred(s, ["DCOILWTICO"], force=force)
        cot = load_cot(s, COT_FILE)
        if cot.empty:
            raise RuntimeError("Keine COT-Daten – CFTC nicht erreichbar? Später erneut versuchen.")
        prices = load_prices(s, [m["price"] for m in MARKETS.values()], subdir=LONG_DIR, include_benchmark=False)
        oil_fred = load_fred(s, ["DCOILWTICO"]).get("DCOILWTICO")
        metas = {}
        for key, m in MARKETS.items():
            status(s, state="läuft", step=f"{m['label']}: lernen und testen")
            px = prices.get(m["price"])
            if key == "OIL" and oil_fred is not None:
                fred_px = oil_fred[oil_fred > 0]
                px = fred_px if px is None else px.combine_first(fred_px[fred_px.index < px.index.min()])
            res = run_market(cot, m["codes"], px)
            if res is None:
                metas[key] = {"error": "zu wenige COT-Daten"}
                continue
            w = res["weekly"]
            w.reset_index().rename(columns={"index": "date"}).to_parquet(out / f"{key}_weekly.parquet", index=False)
            res["learned"].to_parquet(out / f"{key}_learned.parquet", index=False)
            last = w.iloc[-1]
            metas[key] = {"start": str(res["start"].date()) if res["start"] is not None else None,
                          "report_date": str(w.index[-1].date()), "score": _num(last["score"]),
                          "nc_net": _num(last["nc_net"]), "nc_contracts": _num(last["nc_contracts"]),
                          "has_price": bool(w["price"].notna().any())}
        meta = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "markets": metas,
                "cot_rows": int(len(cot)), "fetch": {k: v for k, v in info.items() if k == "cot"}}
        atomic_write_text(out / "meta.json", json.dumps(meta, ensure_ascii=False, indent=1, default=str))
        ok = [k for k, v in metas.items() if "error" not in v]
        status(s, state="fertig", step=f"fertig ({', '.join(ok) or 'keine Märkte'})", n=len(ok))
        return meta
    except Exception as exc:
        status(s, state="Fehler", step=str(exc)[:300])
        raise


def _num(v):
    return None if v is None or pd.isna(v) else float(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="COT-Modell berechnen")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        meta = run(a.offline, a.force)
    except ConfigError as exc:
        print(f"KONFIGURATIONSFEHLER: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        return 3
    for k, m in meta["markets"].items():
        print(f"{k}: " + (f"FEHLER {m['error']}" if "error" in m else
                          f"COT-Score {m['score']:.0f} (Stichtag {m['report_date']})" if m["score"] is not None
                          else "kein Score"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
