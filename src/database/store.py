"""Speicherung.

Zwischenstufe: pro Unternehmen Parquet-Dateien unter data/processed/<art>/CIK##########.parquet
und eine Statusdatei. Das macht den Abruf wiederaufnehmbar und inkrementell – bereits
verarbeitete Unternehmen werden übersprungen, solange sie nicht älter als MAX_CACHE_AGE_DAYS sind.

Endstufe: eine DuckDB-Datei, die in einem kurzen Schritt aus den Parquet-Dateien neu aufgebaut
wird. Die Oberfläche öffnet die Datenbank nur lesend.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from src.analytics.metrics import compute_metrics
from src.config import Settings
from src.utils import atomic_write_text
from src.data_sources.sp500 import members_at
from src.normalization.facts import CompanyResult

KINDS = ("periods", "annual_all", "annual_items", "latest_items")


def company_status_path(settings: Settings, cik: int) -> Path:
    p = settings.processed_dir / "status"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"CIK{cik:010d}.json"


def read_company_status(settings: Settings, cik: int) -> dict | None:
    p = company_status_path(settings, cik)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def write_company_result(settings: Settings, res: CompanyResult, years: tuple[int, ...] = ()) -> None:
    for kind in KINDS:
        d = settings.processed_dir / kind
        d.mkdir(parents=True, exist_ok=True)
        df: pd.DataFrame = getattr(res, kind)
        path = d / f"CIK{res.cik:010d}.parquet"
        if df is None or df.empty:
            path.unlink(missing_ok=True)
        else:
            df = df.copy()
            for c in df.columns:
                if df[c].dtype != object:
                    continue
                nonnull = df[c].dropna()
                if c in ("value", "original_value", "n_versions", "tag_rank", "fy_reported"):
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
                elif len(nonnull) and nonnull.map(lambda x: isinstance(x, bool)).all():
                    df[c] = df[c].astype("boolean")
                elif len(nonnull) and nonnull.map(lambda x: isinstance(x, pd.Timestamp)).all():
                    df[c] = pd.to_datetime(df[c])
                else:
                    df[c] = df[c].astype("string")
            df.to_parquet(path, index=False)
    write_company_status(settings, res.cik, "ok", entity_name=res.entity_name, issues=res.issues, years=years)


def write_company_status(settings: Settings, cik: int, status: str, error: str = "", entity_name: str = "",
                         issues: list[str] | None = None, years: tuple[int, ...] = ()) -> None:
    payload = {"cik": cik, "status": status, "error": error, "entity_name": entity_name, "years": list(years),
               "issues": issues or [], "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "fetched_ts": time.time()}
    atomic_write_text(company_status_path(settings, cik), json.dumps(payload, ensure_ascii=False))


def _load_kind(settings: Settings, kind: str, ciks: set[int]) -> pd.DataFrame:
    d = settings.processed_dir / kind
    if not d.exists():
        return pd.DataFrame()
    frames = []
    for p in sorted(d.glob("CIK*.parquet")):
        if int(p.stem[3:]) in ciks:
            frames.append(pd.read_parquet(p))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_database(settings: Settings, constituents: pd.DataFrame, changes: pd.DataFrame, meta: dict) -> dict:
    """Baut die DuckDB-Datei atomar neu auf (erst temporäre Datei, dann Umbenennen)."""
    companies = (constituents.sort_values("ticker")
                 .groupby("cik", as_index=False)
                 .agg(ticker=("ticker", lambda s: "/".join(s)), primary_ticker=("ticker", "first"),
                      name=("name", "first"), sector=("sector", "first"), sub_industry=("sub_industry", "first"),
                      date_added=("date_added", "first"), list_source=("source", "first"),
                      cik_note=("cik_note", "first") if "cik_note" in constituents else ("source", lambda s: "")))
    companies["cik"] = companies["cik"].astype(int)
    statuses = []
    for cik in companies["cik"]:
        st = read_company_status(settings, int(cik)) or {"status": "nicht abgerufen", "error": "", "issues": [],
                                                         "fetched_at": None, "entity_name": ""}
        statuses.append({"cik": int(cik), "fetch_status": st["status"], "fetch_error": st.get("error", ""),
                         "fetched_at": st.get("fetched_at"), "entity_name": st.get("entity_name", ""),
                         "issues": " | ".join(st.get("issues", []))})
    companies = companies.merge(pd.DataFrame(statuses), on="cik", how="left")
    ciks = set(companies["cik"].astype(int))

    data = {k: _load_kind(settings, k, ciks) for k in KINDS}
    items = data["annual_items"]
    if not items.empty:
        wide = items.pivot_table(index=["cik", "fiscal_year"], columns="item_key", values="value",
                                 aggfunc="first").reset_index()
        metrics = compute_metrics(wide)
    else:
        metrics = pd.DataFrame(columns=["cik", "fiscal_year", "metric_key", "value", "status", "note"])

    membership = _membership(companies, data["periods"], changes, settings)

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    tmp = settings.db_path.with_suffix(".building.duckdb")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    try:
        tables = {"companies": companies, "periods": data["periods"], "annual_facts": data["annual_all"],
                  "balance_items": items, "latest_items": data["latest_items"], "metrics": metrics,
                  "sp500_changes": changes, "membership": membership}
        for name, df in tables.items():
            df = df if df is not None and not df.empty else _empty(name)
            con.register("tmp_df", df)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM tmp_df")
            con.unregister("tmp_df")
        meta = dict(meta, built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    n_companies=len(companies),
                    n_loaded=int((companies["fetch_status"] == "ok").sum()))
        con.execute("CREATE TABLE meta (key VARCHAR, value VARCHAR)")
        con.executemany("INSERT INTO meta VALUES (?, ?)", [(k, str(v)) for k, v in meta.items()])
    finally:
        con.close()
    _replace_with_retry(tmp, settings.db_path)
    return meta


def _replace_with_retry(src: Path, dst: Path, attempts: int = 30) -> None:
    # Unter Windows kann die Datei kurz von der Oberfläche geöffnet sein.
    for i in range(attempts):
        try:
            src.replace(dst)
            return
        except PermissionError:
            time.sleep(1)
    raise RuntimeError(f"Datenbank {dst} ist gesperrt. Bitte das Dashboard kurz schließen und erneut versuchen.")


def _membership(companies: pd.DataFrame, periods: pd.DataFrame, changes: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    rows = []
    has_changes = changes is not None and not changes.empty
    current = set()
    for t in companies["ticker"]:
        current.update(t.split("/"))
    fy_end = (periods.groupby(["cik", "fiscal_year"])["period_end"].max().reset_index()
              if not periods.empty else pd.DataFrame(columns=["cik", "fiscal_year", "period_end"]))
    cache: dict = {}
    for _, r in fy_end.iterrows():
        if int(r["fiscal_year"]) not in settings.load_years:
            continue
        tick = companies.loc[companies["cik"] == r["cik"], "ticker"]
        tickers = set(tick.iloc[0].split("/")) if len(tick) else set()
        if has_changes:
            key = pd.Timestamp(r["period_end"])
            if key not in cache:
                cache[key] = members_at(current, changes, key.date())
            rows.append((int(r["cik"]), int(r["fiscal_year"]), bool(tickers & cache[key]),
                         "rekonstruiert aus Änderungshistorie"))
        else:
            rows.append((int(r["cik"]), int(r["fiscal_year"]), None, "aktuelle Zusammensetzung rückwirkend"))
    return pd.DataFrame(rows, columns=["cik", "fiscal_year", "was_member", "method"])


def _empty(name: str) -> pd.DataFrame:
    schemas = {
        "periods": ["cik", "accn", "form", "filed", "period_end", "fy_reported", "fp", "fiscal_year", "fy_source"],
        "annual_facts": ["cik", "fiscal_year", "period_end", "taxonomy", "tag", "label", "unit", "value", "accn",
                         "form", "filed", "original_value", "original_accn", "original_filed", "n_versions",
                         "restated", "conflict"],
        "balance_items": ["cik", "fiscal_year", "period_end", "item_key", "value", "unit", "tag", "accn", "form",
                          "filed", "original_value", "original_filed", "n_versions", "restated", "conflict", "is_derived",
                          "derivation", "tag_rank"],
        "latest_items": ["cik", "period_end", "item_key", "value", "unit", "tag", "accn", "form", "filed",
                         "original_value", "original_filed", "n_versions", "restated", "conflict", "is_derived", "derivation",
                         "tag_rank", "source_form"],
        "metrics": ["cik", "fiscal_year", "metric_key", "value", "status", "note"],
        "sp500_changes": ["date", "added_ticker", "removed_ticker", "reason"],
        "membership": ["cik", "fiscal_year", "was_member", "method"],
    }
    num = {"cik", "fiscal_year", "value", "original_value", "n_versions", "fy_reported", "tag_rank"}
    return pd.DataFrame({c: pd.Series(dtype="float64" if c in num else "string") for c in schemas[name]})


def connect_readonly(db_path: Path, attempts: int = 10) -> duckdb.DuckDBPyConnection:
    last = None
    for _ in range(attempts):
        try:
            return duckdb.connect(str(db_path), read_only=True)
        except duckdb.IOException as exc:
            last = exc
            time.sleep(0.5)
    raise RuntimeError(f"Datenbank nicht lesbar ({last}).")
