"""Datenabruf-Pipeline: S&P-500-Liste -> SEC Company Facts -> Normalisierung -> Parquet -> DuckDB.

Aufruf:
  python -m src.pipeline                      # alle Unternehmen, inkrementell (überspringt frische)
  python -m src.pipeline --tickers AAPL,MSFT  # nur ausgewählte Unternehmen
  python -m src.pipeline --limit 20           # erste 20 (zum Testen)
  python -m src.pipeline --force              # Cache ignorieren, alles neu laden
  python -m src.pipeline --build-only         # nur Datenbank aus vorhandenen Daten neu aufbauen
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd

from src.config import ConfigError, Settings, get_settings
from src.utils import atomic_write_text
from src.data_sources.sec_client import SecClient, SecError, SecNotFound
from src.data_sources.sp500 import fetch_constituents, normalize_ticker
from src.database.store import build_database, read_company_status, write_company_result, write_company_status
from src.normalization.facts import normalize_company

log = logging.getLogger("pipeline")


def write_progress(settings: Settings, **kw) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    kw["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    atomic_write_text(settings.status_path, json.dumps(kw, ensure_ascii=False))


def load_constituents(settings: Settings, client: SecClient) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    snap = settings.data_dir / "constituents_snapshot.csv"
    snap_changes = settings.data_dir / "constituents_changes_snapshot.csv"
    try:
        tickers = client.company_tickers()
    except SecError as exc:
        log.warning("SEC-Tickerdatei nicht abrufbar: %s", exc)
        tickers = None
    try:
        data = fetch_constituents(settings, tickers)
        data.constituents.to_csv(snap, index=False)
        data.changes.to_csv(snap_changes, index=False)
        return data.constituents, data.changes, f"{data.source} (abgerufen {data.retrieved_at})"
    except Exception as exc:  # noqa: BLE001
        if snap.exists():
            log.warning("Mitgliederliste nicht abrufbar (%s) – verwende letzten Snapshot %s", exc, snap)
            cons = pd.read_csv(snap)
            changes = pd.read_csv(snap_changes, parse_dates=["date"]) if snap_changes.exists() else pd.DataFrame(
                columns=["date", "added_ticker", "removed_ticker", "reason"])
            ts = datetime.fromtimestamp(snap.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
            return cons, changes, f"Snapshot vom {ts} (Live-Abruf fehlgeschlagen: {exc})"
        raise


def process_company(settings: Settings, client: SecClient, cik: int, force: bool, reprocess: bool = False) -> str:
    """force: neu herunterladen. reprocess: aus dem lokalen Cache neu normalisieren (kein Download, falls Cache frisch).
    Ändert sich der Jahresbereich (z. B. ML_YEARS), wird automatisch neu verarbeitet."""
    st = read_company_status(settings, cik)
    same_years = st is not None and (st["status"] != "ok" or st.get("years") == list(settings.load_years))
    if st and not force and not reprocess and same_years and st["status"] in ("ok", "keine Daten") and \
            time.time() - st.get("fetched_ts", 0) < settings.max_cache_age_days * 86400:
        return "übersprungen (aktuell)"
    try:
        facts = client.company_facts(cik, force=force)
        res = normalize_company(facts, settings.load_years)
        write_company_result(settings, res, settings.load_years)
        return "ok"
    except SecNotFound:
        write_company_status(settings, cik, "keine Daten", "SEC liefert keine XBRL Company Facts (404)")
        return "keine Daten"
    except Exception as exc:  # noqa: BLE001 – Fehler pro Unternehmen isolieren, Lauf fortsetzen
        log.exception("Fehler bei CIK %s", cik)
        write_company_status(settings, cik, "Fehler", f"{type(exc).__name__}: {exc}")
        return "Fehler"


def run(tickers: list[str] | None = None, limit: int | None = None, force: bool = False,
        build_only: bool = False, workers: int = 4, reprocess: bool = False) -> dict:
    settings = get_settings()
    client = SecClient(settings)
    try:
        write_progress(settings, state="läuft", step="Mitgliederliste", done=0, total=0)
        cons, changes, list_source = load_constituents(settings, client)
        cons = cons.dropna(subset=["cik"]).copy()
        cons["cik"] = cons["cik"].astype(int)
        if tickers:
            wanted = {normalize_ticker(t) for t in tickers}
            unknown = wanted - set(cons["ticker"])
            if unknown:
                log.warning("Nicht im S&P 500 gefunden: %s", ", ".join(sorted(unknown)))
            run_ciks = sorted(set(cons.loc[cons["ticker"].isin(wanted), "cik"]))
        else:
            run_ciks = sorted(set(cons["cik"]))
        if limit:
            run_ciks = run_ciks[:limit]

        counts: dict[str, int] = {}
        if not build_only:
            total = len(run_ciks)
            done = 0
            write_progress(settings, state="läuft", step="SEC-Abruf", done=0, total=total)
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futs = {pool.submit(process_company, settings, client, c, force, reprocess): c for c in run_ciks}
                for fut in as_completed(futs):
                    r = fut.result()
                    counts[r] = counts.get(r, 0) + 1
                    done += 1
                    if done % 5 == 0 or done == total:
                        write_progress(settings, state="läuft", step="SEC-Abruf", done=done, total=total, counts=counts)
                        log.info("%d/%d verarbeitet %s", done, total, counts)

        write_progress(settings, state="läuft", step="Datenbank aufbauen", done=len(run_ciks), total=len(run_ciks))
        # Die Datenbank enthält immer die gesamte Liste; nicht abgerufene Unternehmen erscheinen als Lücke.
        meta = build_database(settings, cons, changes, {
            "constituents_source": list_source,
            "last_run_scope": "alle Unternehmen" if not tickers and not limit else f"Teilmenge ({len(run_ciks)} Unternehmen)",
            "fiscal_years": f"{min(settings.fiscal_years)}-{max(settings.fiscal_years)}",
            "membership_method": "rekonstruiert aus Änderungshistorie" if not changes.empty
            else "aktuelle Zusammensetzung rückwirkend verwendet",
            "last_run_counts": json.dumps(counts, ensure_ascii=False),
        })
        write_progress(settings, state="fertig", step="fertig", done=len(run_ciks), total=len(run_ciks), counts=counts,
                       loaded=meta["n_loaded"], universe=meta["n_companies"])
        return meta
    except Exception as exc:
        write_progress(settings, state="Fehler", step=str(exc), done=0, total=0)
        raise
    finally:
        client.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S&P-500-Bilanzdaten von der SEC laden")
    ap.add_argument("--tickers", help="Kommagetrennte Ticker, z. B. AAPL,MSFT")
    ap.add_argument("--limit", type=int, help="Nur die ersten N Unternehmen (nach CIK)")
    ap.add_argument("--force", action="store_true", help="Cache ignorieren und neu laden")
    ap.add_argument("--build-only", action="store_true", help="Nur Datenbank neu aufbauen")
    ap.add_argument("--reprocess", action="store_true", help="Aus lokalem Cache neu normalisieren (ohne Neu-Download)")
    ap.add_argument("--workers", type=int, default=4, help="Parallele Downloads (Rate Limit gilt global)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        meta = run([t for t in (args.tickers or "").split(",") if t.strip()] or None, args.limit, args.force,
                   args.build_only, args.workers, args.reprocess)
    except ConfigError as exc:
        print(f"\nKONFIGURATIONSFEHLER:\n{exc}\n", file=sys.stderr)
        return 2
    except SecError as exc:
        print(f"\nSEC-FEHLER:\n{exc}\n", file=sys.stderr)
        return 3
    print(f"\nFertig: {meta['n_loaded']} von {meta['n_companies']} Unternehmen mit Daten. "
          f"Datenbank aufgebaut {meta['built_at']}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
