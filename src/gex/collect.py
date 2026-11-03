"""GEX-Schnappschüsse sammeln: python -m src.gex.collect [--force]

Speichert je Abruf die Rohantwort (data/gex/raw/<SYMBOL>/<UTC-Zeit>.json) und eine Merkmalszeile
(data/gex/snapshots.parquet). Schutz vor unnötigen Kosten: Ein Symbol wird höchstens alle 6 Stunden abgerufen
(ausser --force), und bei weniger als 20 verbleibenden Credits wird abgebrochen.
Empfohlen: einmal täglich nach US-Börsenschluss (22:15 Schweizer Zeit) – siehe gex_zeitplan.bat.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import pandas as pd

from src.config import get_settings
from src.gex.client import ZETA_ASSETS, AFXClient, AFXError, symbols
from src.gex.features import features, zeta_features
from src.utils import atomic_write_text

MIN_HOURS = 6
MIN_CREDITS = 20


def gex_dir():
    d = get_settings().data_dir / "gex"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_snapshots() -> pd.DataFrame:
    p = gex_dir() / "snapshots.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["taken_at"] = pd.to_datetime(df["taken_at"], utc=True)
    return df


def status(**kw) -> None:
    kw["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    atomic_write_text(gex_dir() / "status.json", json.dumps(kw, ensure_ascii=False))


def collect(force: bool = False, client: AFXClient | None = None, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    snaps = load_snapshots()
    own = client is None
    client = client or AFXClient()
    done, skipped, errors, remaining = [], [], {}, None
    rows = []
    try:
        for sym in symbols():
            if not force and not snaps.empty:
                last = snaps.loc[snaps["symbol"] == sym, "taken_at"].max()
                if pd.notna(last) and (pd.Timestamp(now) - last).total_seconds() < MIN_HOURS * 3600:
                    skipped.append(sym)
                    continue
            if remaining is not None and remaining < MIN_CREDITS:
                errors[sym] = f"nur noch {remaining} Credits – abgebrochen"
                continue
            try:
                resp = client.gex(sym)
            except AFXError as exc:
                errors[sym] = str(exc)
                continue
            remaining = ((resp.get("meta") or {}).get("credits") or {}).get("remaining", remaining)
            raw_dir = gex_dir() / "raw" / sym
            raw_dir.mkdir(parents=True, exist_ok=True)
            stamp = now.strftime("%Y%m%d_%H%M%S")
            atomic_write_text(raw_dir / f"{stamp}.json", json.dumps(resp))
            f = features(resp, sym, now)
            zeta_off = gex_dir() / "zeta_disabled"
            if sym in ZETA_ASSETS and not zeta_off.exists():
                try:
                    z = client.zeta(sym)
                    atomic_write_text(raw_dir / f"{stamp}_zeta.json", json.dumps(z))
                    f.update(zeta_features(z))
                    remaining = ((z.get("meta") or {}).get("credits") or {}).get("remaining", remaining)
                except AFXError as exc:
                    if "403" in str(exc) or "401" in str(exc):   # nicht im Plan enthalten -> nicht mehr versuchen
                        zeta_off.write_text("Zeta im AnalyticsFX-Plan nicht freigeschaltet (HTTP 403).", encoding="utf-8")
                    else:
                        errors[f"{sym} Zeta"] = str(exc)
            rows.append(f)
            done.append(sym)
    finally:
        if own:
            client.close()
    if rows:
        new = pd.DataFrame(rows)
        new["taken_at"] = pd.to_datetime(new["taken_at"], utc=True)
        allr = pd.concat([snaps, new], ignore_index=True) if not snaps.empty else new
        allr.to_parquet(gex_dir() / "snapshots.parquet", index=False)
    res = {"done": done, "skipped": skipped, "errors": errors, "credits_remaining": remaining}
    status(state="fertig" if not errors or done else "Fehler", **res)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GEX-Schnappschüsse von AnalyticsFX sammeln")
    ap.add_argument("--force", action="store_true", help="auch wenn der letzte Abruf < 6 Stunden her ist")
    a = ap.parse_args(argv)
    try:
        r = collect(a.force)
    except AFXError as exc:
        status(state="Fehler", errors={"": str(exc)})
        print(f"FEHLER: {exc}", file=sys.stderr)
        return 2
    print(f"Abgerufen: {', '.join(r['done']) or '–'} · übersprungen (<{MIN_HOURS} h): {', '.join(r['skipped']) or '–'}"
          f" · Credits übrig: {r['credits_remaining']}")
    for k, v in r["errors"].items():
        print(f"  {k}: {v}")
    if r["done"]:
        from src.gex.learn import run as learn_run
        learn_run()
        print("Modell aktualisiert.")
    return 0 if not r["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
