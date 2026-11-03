"""Zugriff auf die Ergebnisse des Fundamental-Terminals (data/fundamental) für die Oberfläche."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.config import get_settings


def fund_dir():
    return get_settings().data_dir / "fundamental"


def fund_version() -> float:
    p = fund_dir() / "meta.json"
    return p.stat().st_mtime if p.exists() else 0.0


def fund_status() -> dict | None:
    p = fund_dir() / "status.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def load_fund(version: float) -> dict | None:
    d = fund_dir()
    if not (d / "meta.json").exists():
        return None
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    out = {"meta": meta, "markets": {}}
    for key, m in meta.get("markets", {}).items():
        if "error" in m:
            continue
        p = d / f"fund_{key}_daily.parquet"
        if not p.exists():
            continue
        daily = pd.read_parquet(p)
        daily["date"] = pd.to_datetime(daily["date"])
        tabs = {}
        for t in ("learned", "bt", "btdiv", "effects"):
            q = d / f"fund_{key}_{t}.parquet"
            tabs[t] = pd.read_parquet(q) if q.exists() else pd.DataFrame()
        out["markets"][key] = {"daily": daily, **tabs, "meta": m}
    e = d / "earnings.parquet"
    out["earnings"] = pd.read_parquet(e) if e.exists() else pd.DataFrame()
    return out
