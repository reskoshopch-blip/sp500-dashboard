"""Lesender Datenzugriff für die Oberfläche (DuckDB, nur lesend, gecacht bis zur nächsten Neuerstellung)."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.config import get_settings
from src.database.store import connect_readonly
from src.normalization.taxonomy import ITEM_BY_KEY, ITEMS, SECTION_ORDER


def db_version() -> float:
    s = get_settings()
    return s.db_path.stat().st_mtime if s.db_path.exists() else 0.0


def db_exists() -> bool:
    return get_settings().db_path.exists()


def _query(sql: str, params: list | None = None) -> pd.DataFrame:
    con = connect_readonly(get_settings().db_path)
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def load_all(version: float) -> dict:
    companies = _query("SELECT * FROM companies ORDER BY name")
    items = _query("SELECT cik, fiscal_year, period_end, item_key, value, unit, tag, accn, form, filed, "
                   "original_value, n_versions, restated, conflict, is_derived, derivation FROM balance_items")
    metrics = _query("SELECT * FROM metrics")
    latest = _query("SELECT * FROM latest_items")
    membership = _query("SELECT * FROM membership")
    periods = _query("SELECT * FROM periods")
    changes = _query("SELECT * FROM sp500_changes")
    meta = dict(_query("SELECT key, value FROM meta").itertuples(index=False, name=None))
    for df in (items, metrics, membership, periods):
        if "fiscal_year" in df and len(df):
            df["fiscal_year"] = df["fiscal_year"].astype(int)
        if "cik" in df and len(df):
            df["cik"] = df["cik"].astype(int)
    if len(latest):
        latest["cik"] = latest["cik"].astype(int)
    companies["cik"] = companies["cik"].astype(int)
    companies["sector"] = companies["sector"].fillna("Unbekannt")
    companies["label"] = companies["primary_ticker"] + " · " + companies["name"]
    items = items.merge(companies[["cik", "primary_ticker", "name", "sector"]], on="cik", how="left")
    item_meta = pd.DataFrame([{"item_key": i.key, "item_label": i.label_de, "item_en": i.label_en,
                               "section": i.section, "level": i.level} for i in ITEMS])
    items = items.merge(item_meta, on="item_key", how="left")
    return {"companies": companies, "items": items, "metrics": metrics, "latest": latest,
            "membership": membership, "periods": periods, "changes": changes, "meta": meta}


@st.cache_data(show_spinner=False, max_entries=64)
def load_company_facts(version: float, cik: int) -> pd.DataFrame:
    return _query("SELECT * FROM annual_facts WHERE cik = ? ORDER BY tag, fiscal_year", [cik])


def pipeline_status() -> dict | None:
    p = get_settings().status_path
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def item_options(section_filter: list[str] | None = None) -> list[str]:
    return [i.key for s in SECTION_ORDER for i in ITEMS if i.section == s and (not section_filter or s in section_filter)]


def item_label(key: str) -> str:
    return ITEM_BY_KEY[key].label_de if key in ITEM_BY_KEY else key
