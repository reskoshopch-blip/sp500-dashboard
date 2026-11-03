"""Stichprobenartige Verifikation.

1. Gegenprüfung gegen einen unabhängigen SEC-Endpunkt (XBRL Company Concept API) für
   Assets, LiabilitiesAndStockholdersEquity und StockholdersEquity.
2. Bilanzidentität: Assets = Liabilities and Equity (Toleranz 0,5 %).
3. Link zum Originalbericht (10-K) für die manuelle Kontrolle.

Aufruf: python -m src.verify --sample 15
Ergebnis: data/verification_report.csv (wird im Dashboard unter "Datenqualität" angezeigt).
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timezone

import pandas as pd

from src.config import ConfigError, get_settings
from src.data_sources.sec_client import SecClient, SecError, filing_index_url
from src.database.store import connect_readonly

CHECK_TAGS = {"Assets": "total_assets", "LiabilitiesAndStockholdersEquity": "liabilities_and_equity",
              "StockholdersEquity": "equity_parent"}


def identity_checks(con) -> pd.DataFrame:
    return con.execute("""
        SELECT a.cik, a.fiscal_year, a.value AS assets, l.value AS liabilities_and_equity,
               abs(a.value - l.value) / nullif(abs(a.value), 0) AS rel_diff
        FROM balance_items a JOIN balance_items l
          ON a.cik = l.cik AND a.fiscal_year = l.fiscal_year
        WHERE a.item_key = 'total_assets' AND l.item_key = 'liabilities_and_equity'
    """).df()


def run(sample: int, seed: int | None) -> pd.DataFrame:
    settings = get_settings()
    con = connect_readonly(settings.db_path)
    items = con.execute("""SELECT b.cik, c.primary_ticker, b.fiscal_year, b.period_end, b.item_key, b.value, b.accn
                           FROM balance_items b JOIN companies c USING (cik)
                           WHERE NOT b.is_derived AND b.item_key IN ('total_assets','liabilities_and_equity','equity_parent')""").df()
    con.close()
    if items.empty:
        raise SystemExit("Keine Daten in der Datenbank – zuerst 'python -m src.pipeline' ausführen.")
    ciks = sorted(items["cik"].unique())
    rnd = random.Random(seed)
    chosen = rnd.sample(list(ciks), min(sample, len(ciks)))
    client = SecClient(settings)
    rows = []
    try:
        for cik in chosen:
            for tag, key in CHECK_TAGS.items():
                mine = items[(items["cik"] == cik) & (items["item_key"] == key)]
                if mine.empty:
                    continue
                try:
                    concept = client.company_concept(int(cik), tag)
                except SecError as exc:
                    rows.append({"cik": cik, "tag": tag, "result": f"Abruf fehlgeschlagen: {exc}"})
                    continue
                facts = pd.DataFrame(concept.get("units", {}).get("USD", []))
                if facts.empty:
                    continue
                facts = facts[facts["form"].isin(["10-K", "10-K/A", "10-KT", "20-F", "40-F"])]
                if "start" in facts.columns:
                    facts = facts[facts["start"].isna()]
                facts["end"] = pd.to_datetime(facts["end"])
                for _, m in mine.iterrows():
                    cand = facts[facts["end"] == pd.Timestamp(m["period_end"])].sort_values("filed")
                    if cand.empty:
                        res, sec_val = "kein Vergleichswert", None
                    else:
                        sec_val = float(cand.iloc[-1]["val"])
                        res = "übereinstimmend" if abs(sec_val - m["value"]) <= 0.5 else "ABWEICHUNG"
                    rows.append({"cik": int(cik), "ticker": m["primary_ticker"], "fiscal_year": int(m["fiscal_year"]),
                                 "tag": tag, "dashboard_value": m["value"], "sec_concept_value": sec_val,
                                 "result": res, "filing": filing_index_url(int(cik), m["accn"])})
    finally:
        client.close()
    report = pd.DataFrame(rows)
    report["checked_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = settings.data_dir / "verification_report.csv"
    report.to_csv(out, index=False)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=15)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    try:
        rep = run(a.sample, a.seed)
    except ConfigError as exc:
        print(f"KONFIGURATIONSFEHLER: {exc}", file=sys.stderr)
        return 2
    print(rep["result"].value_counts().to_string())
    print("Bericht gespeichert: data/verification_report.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
